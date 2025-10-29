#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "math_ops.h"
#include "transformer.h"
#include <time.h>

typedef struct
{
  // Model hyperparameters
  int vocab_size;
  int d_model;
  int num_layers;
  int num_heads;
  int ffn_hidden_dim;

  // Embeddings
  Matrix *embedding_table;

  // Decoder layers (arrays of pointers, one per layer)
  Matrix **W_Q_layers;
  Matrix **W_K_layers;
  Matrix **W_V_layers;
  Matrix **W_O_layers;

  Matrix **W1_layers;
  Matrix **b1_layers;
  Matrix **W2_layers;
  Matrix **b2_layers;

  float **ln1_gain_layers;
  float **ln1_bias_layers;
  float **ln2_gain_layers;
  float **ln2_bias_layers;

  // Output projection
  Matrix *W_out;
  Matrix *b_out;

} TransformerModel;
typedef struct
{
  // Add these fields:
  Matrix *input; // Original input to attention (needed for weight gradients)

  Matrix *result;
  Matrix *concat_result;
  Matrix **head_outputs;
  Matrix **head_scores;
  Matrix **Q_heads;
  Matrix **K_heads;
  Matrix **V_heads;
  Matrix *Q;
  Matrix *K;
  Matrix *V;
  int num_of_heads;
  int head_dim;
} MultiHeadAttentionCache;

typedef struct
{
  // ADD THIS:
  Matrix *input; // Original decoder_input (needed for first residual gradient)

  MultiHeadAttentionCache attention_cache;
  Matrix *residual_1;
  LayerNormCache *ln1_caches;

  // ADD THIS:
  Matrix *ln1_input; // Input to first LayerNorm (= residual_1, but save pointer)

  Matrix *norm1;
  Matrix *ffn_pre_gelu;
  Matrix *ffn_post_gelu;
  // REMOVE ffn_hidden if it's redundant
  Matrix *ffn_output;

  Matrix *residual_2;
  LayerNormCache *ln2_caches;

  // ADD THIS:
  Matrix *ln2_input; // Input to second LayerNorm (= residual_2)

  int num_rows;
} DecoderLayerCache;
char *load_text_file(char *filepath, int *output_length)
{
  FILE *file = fopen(filepath, "r");
  if (file == NULL)
  {
    printf("Could not open file%s\n", filepath);
    exit(1);
  }
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *text = malloc(file_size + 1);
  if (text == NULL)
  {
    printf("Error : Allocating Memort for File");
    exit(1);
  }
  size_t bytes_read = fread(text, 1, file_size, file);
  text[bytes_read] = '\0';
  fclose(file);

  *output_length = bytes_read;
  return text;
}
Vocabulary *vocab_alloc(char *text)
{
  Vocabulary *vocab = malloc(sizeof(Vocabulary));
  if (vocab == NULL)
  {
    printf("Error: Error allocating memory for vocab");
    exit(1);
  }

  for (int i = 0; i < 256; i++)
  {
    vocab->char_to_id[i] = -1;
  }
  int seen[256] = {0};
  int unique_count = 0;
  for (const char *p = text; *p; p++)
  {
    unsigned char c = (unsigned char)*p;
    if (seen[c] == 0)
    {
      seen[c] = 1;
      unique_count++;
    }
  }
  vocab->vocab_size = unique_count;
  vocab->chars = malloc(unique_count + 1);
  if (!vocab->chars)
  {
    free(vocab);
    printf("Error : Allocating Memory for chars");
    exit(1);
  }
  int idx = 0;
  for (int i = 0; i < 256; i++)
  {
    if (seen[i])
    {
      vocab->chars[idx] = (char)i;
      vocab->char_to_id[i] = idx;
      idx++;
    }
  }
  vocab->chars[idx] = '\0';
  if (vocab->chars == NULL)
  {
    printf("Error: Allocating memory for chars in vocab");
    exit(1);
  }
  printf("seen: [");
  for (int i = 0; i < 256; i++)
  {
    printf("%d", seen[i]);
    if (i < 256 - 1)
    {
      printf(", ");
    }
  }
  printf("]\n");
  return vocab;
}
void vocab_free(Vocabulary *vocabulary)
{
  if (vocabulary)
  {
    free(vocabulary->chars);
    free(vocabulary);
  }
}
int *tokenize(char *text, Vocabulary *vocab, int *out_length)
{
  int text_len = strlen(text);
  int *token_ids = malloc(text_len * sizeof(int));
  if (token_ids == NULL)
  {
    printf("Error allocating memory for token_ids\n");
    exit(1);
  }

  for (int i = 0; i < text_len; i++)
  {
    unsigned char c = (unsigned char)text[i];
    int id = vocab->char_to_id[c];
    if (id == -1)
    {
      printf("Error: character '%c' (ASCII %d) not in vocabulary\n", c, c);
      exit(1);
    }
    token_ids[i] = id;
  }
  *out_length = text_len;
  return token_ids;
}

char *detokenize(int *token_ids, int length, Vocabulary *vocab)
{
  char *arr = malloc(length + 1);
  if (arr == NULL)
  {
    printf("Error : Allocating memory for char array in detokenize");
    exit(1);
  }
  for (int i = 0; i < length; i++)
  {
    if (token_ids[i] < 0 || token_ids[i] > vocab->vocab_size)
    {
      printf("In Valid id in Detokenize");
      exit(1);
    }
    arr[i] = vocab->chars[token_ids[i]];
  }
  arr[length + 1] = '\0';
  return arr;
}
void add_positional_encoding(Matrix *embeddings, int seq_len, int d_model)
{
  for (int row = 0; row < embeddings->rows; row++)
  {
    for (int col = 0; col < embeddings->cols; col++)
    {
      float positional_value = 0;
      int i = col / 2; // This is the key change!
      if (col % 2 == 0)
      {
        positional_value = sin(row / pow(10000.0, (2.0 * (float)i) / (float)d_model));
      }
      else
      {
        positional_value = cos(row / pow(10000.0, (2.0 * (float)i) / (float)d_model));
      }
      embeddings->data[row * embeddings->cols + col] += positional_value;
    }
  }
}
// Why normalise? it basically reduces the activity of the neuron

void feed_forward(Matrix *input, Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2, Matrix *output, Matrix *hidden_post_gelu, Matrix *hidden_pre_gelu)
{
  Matrix *matrix_multiply_result = matrix_alloc(input->rows, W1->cols);
  Matrix *second_multiply_result = matrix_alloc(input->rows, W2->cols);

  matmul(input, W1, matrix_multiply_result);
  add_bias(matrix_multiply_result, b1, hidden_pre_gelu);
  for (int i = 0; i < hidden_pre_gelu->capacity; i++)
  {
    hidden_post_gelu->data[i] = hidden_pre_gelu->data[i]; // we are bascially keeping 1 copy for backprop thus the extra storage hidden_pre_gelu->data[i];
  }
  GELU(hidden_post_gelu);
  matmul(hidden_post_gelu, W2, second_multiply_result);
  add_bias(second_multiply_result, b2, output);

  matrix_free(matrix_multiply_result);
  matrix_free(second_multiply_result);
}
float findMax(float *data, int length)
{
  float max = 0;
  for (int i = 0; i < length; i++)
  {
    float curVal = data[i];
    if (curVal > max)
    {
      max = curVal;
    }
  }
  return max;
}

AttentionOutput masked_attention(Matrix *Q, Matrix *K, Matrix *V)
{
  Matrix *T = matrix_alloc(K->cols, K->rows);
  matrix_transpose(K, T);
  Matrix *scores = matrix_alloc(Q->rows, K->rows);
  for (int query_row = 0; query_row < Q->rows; query_row++)
  {
    for (int key_col = 0; key_col < T->cols; key_col++)
    {
      float dot_product = 0.0f;
      for (int i = 0; i < Q->cols; i++)
      {

        dot_product += T->data[i * T->cols + key_col] * Q->data[query_row * Q->cols + i];
      }
      scores->data[query_row * scores->cols + key_col] = dot_product;
    }
  }
  float scale = sqrt(Q->cols);
  for (int i = 0; i < scores->rows; i++)
  {
    for (int j = 0; j < scores->cols; j++)
    {
      scores->data[i * scores->cols + j] /= scale;
      if (j > i)
      {
        scores->data[i * scores->cols + j] = MASK_VALUE;
      }
    }
  }
  mat_softmax_rows(scores);
  Matrix *result = matrix_alloc(Q->rows, V->cols);
  matmul(scores, V, result);
  matrix_free(T);
  AttentionOutput output;
  output.result = result;
  output.score = scores;
  return output;
}
void masked_attention_backward(
    Matrix *d_result, // gradient flowing back from next layer
    Matrix *Q,        // stored from forward pass
    Matrix *K,        // stored from forward pass
    Matrix *V,        // stored from forward pass
    Matrix *scores,   // stored attention scores (post-softmax) from forward
    Matrix *d_Q,      // output: gradient w.r.t. Q
    Matrix *d_K,      // output: gradient w.r.t. K
    Matrix *d_V       // output: gradient w.r.t. V
)
{
  Matrix *transposed_V = matrix_alloc(V->cols, V->rows);
  matrix_transpose(V, transposed_V);
  Matrix *d_scores = matrix_alloc(Q->rows, K->rows);
  matmul(d_result, transposed_V, d_scores);
  Matrix *transposed_scores = matrix_alloc(scores->cols, scores->rows);
  matrix_transpose(scores, transposed_scores);
  matmul(transposed_scores, d_result, d_V);
  matrix_free(transposed_scores);
  Matrix *d_scores_pre_softmax = matrix_alloc(scores->rows, scores->cols);
  for (int i = 0; i < scores->rows; i++)
  {
    float sum = 0.0;
    for (int k = 0; k < scores->cols; k++)
    {
      sum += d_scores->data[i * d_scores->cols + k] * scores->data[i * scores->cols + k];
    }
    for (int j = 0; j < scores->cols; j++)
    {
      d_scores_pre_softmax->data[i * d_scores_pre_softmax->cols + j] = scores->data[i * scores->cols + j] * (d_scores->data[i * d_scores->cols + j] - sum);
    }
  }
  for (int row = 0; row < d_scores_pre_softmax->rows; row++)
  {
    for (int col = 0; col < d_scores_pre_softmax->cols; col++)
    {
      if (col > row)
      {
        d_scores_pre_softmax->data[row * d_scores_pre_softmax->cols + col] = 0.0f;
      }
    }
  }
  for (int row = 0; row < d_scores_pre_softmax->rows; row++)
  {
    for (int col = 0; col < d_scores_pre_softmax->cols; col++)
    {
      d_scores_pre_softmax->data[row * d_scores_pre_softmax->cols + col] = d_scores_pre_softmax->data[row * d_scores_pre_softmax->cols + col] / sqrt(Q->cols);
    }
  }
  matmul(d_scores_pre_softmax, K, d_Q);
  Matrix *d_scores_pre_softmax_T = matrix_alloc(d_scores_pre_softmax->cols, d_scores_pre_softmax->rows);
  matrix_transpose(d_scores_pre_softmax, d_scores_pre_softmax_T);
  matmul(d_scores_pre_softmax_T, Q, d_K);
  matrix_free(transposed_V);
  matrix_free(d_scores);
  matrix_free(d_scores_pre_softmax);
  matrix_free(d_scores_pre_softmax_T);
};
void write_data_to_matrix(float *data, Matrix *output)
{
  for (int i = 0; i < output->rows; i++)
  {
    for (int j = 0; j < output->cols; j++)
    {
      if (j > i)
      {
        output->data[i * output->cols + j] = data[i * output->cols + j];
      }
    }
  }
}
float train_step(
    // Input data
    int *input_tokens,  // The input sequence
    int *target_tokens, // What the model should predict
    int seq_len,        // Length of sequences

    // Model architecture params
    int num_layers,
    int num_heads,
    int d_model,
    int vocab_size,

    // All the model weights (these will be updated)
    Matrix *embedding_table,
    Matrix **W_Q_layers,
    Matrix **W_K_layers,
    Matrix **W_V_layers,
    Matrix **W_O_layers,
    Matrix **W1_layers,
    Matrix **b1_layers,
    Matrix **W2_layers,
    Matrix **b2_layers,
    float **ln1_gain_layers,
    float **ln1_bias_layers,
    float **ln2_gain_layers,
    float **ln2_bias_layers,
    Matrix *W_out,
    Matrix *b_out,

    // Training hyperparameter
    float learning_rate)
{
  // Convert tokens to embeddings
  Matrix *input_embeddings = embed_tokens(input_tokens, seq_len, embedding_table);

  // Add positional encoding
  add_positional_encoding(input_embeddings, seq_len, d_model);

  // Run through decoder
  Matrix *decoder_output = matrix_alloc(seq_len, d_model);
  decoder(input_embeddings,
          W_Q_layers, W_K_layers, W_V_layers, W_O_layers,
          W1_layers, b1_layers, W2_layers, b2_layers,
          ln1_gain_layers, ln1_bias_layers, ln2_gain_layers, ln2_bias_layers,
          num_layers, num_heads,
          decoder_output);

  // Project to vocabulary
  Matrix *logits = output_projection(decoder_output, W_out, b_out);
  Matrix *dLogits = matrix_alloc(logits->rows, logits->cols);
  mat_softmax_rows(logits);

  // Compute loss
  float loss = cross_entropy_loss(logits, target_tokens, seq_len, vocab_size);
  for (int i = 0; i < logits->rows; i++)
  {
    float correct_token = target_tokens[i];
    for (int j = 0; j < logits->cols; j++)
    {
      dLogits->data[i * dLogits->cols + j] = logits->data[i * logits->cols + j];

      if (j == correct_token)
      {
        dLogits->data[i * dLogits->cols + j] -= 1.0;
      }
    }
  }
  Matrix *decoder_output_T = matrix_alloc(decoder_output->cols, decoder_output->rows);
  matrix_transpose(decoder_output, decoder_output_T);
  Matrix *dW_out = matrix_alloc(W_out->rows, W_out->cols);
  matmul(decoder_output_T, dLogits, dW_out);
  Matrix *dB_out = matrix_alloc(b_out->rows, b_out->cols);
  matrix_sum_rows(dLogits, dB_out);

  Matrix *dDecoder_output_T = matrix_alloc(W_out->cols, W_out->rows);
  matrix_transpose(W_out, dDecoder_output_T);
  Matrix *dDecoder_output = matrix_alloc(dDecoder_output_T->rows, dDecoder_output_T->cols);
  matmul(dLogits, dDecoder_output_T, dDecoder_output);
};

MultiHeadAttentionCache multi_head_masked_attention(Matrix *input, Matrix *W_Q, Matrix *W_K, Matrix *W_V, Matrix *W_0, int num_of_heads)
{
  MultiHeadAttentionCache cache;
  cache.input = input;
  cache.Q = matrix_alloc(input->rows, W_Q->cols);
  cache.K = matrix_alloc(input->rows, W_K->cols);
  cache.V = matrix_alloc(input->rows, W_V->cols);

  matmul(input, W_Q, cache.Q);
  matmul(input, W_K, cache.K);
  matmul(input, W_V, cache.V);
  int head_dim = cache.Q->cols / num_of_heads;
  cache.head_dim = head_dim;

  cache.Q_heads = malloc(num_of_heads * sizeof(Matrix *));
  cache.K_heads = malloc(num_of_heads * sizeof(Matrix *));
  cache.V_heads = malloc(num_of_heads * sizeof(Matrix *));

  for (int h = 0; h < num_of_heads; h++)
  {
    cache.Q_heads[h] = matrix_alloc(cache.Q->rows, head_dim);
    cache.K_heads[h] = matrix_alloc(cache.K->rows, head_dim);
    cache.V_heads[h] = matrix_alloc(cache.V->rows, head_dim);
  }
  for (int i = 0; i < num_of_heads; i++)
  {
    int start_index = i * head_dim;
    int end_index = start_index + head_dim;
    slice_array(cache.Q->data, start_index, end_index, cache.Q->rows, cache.Q->cols, cache.Q_heads[i]->data);
    slice_array(cache.K->data, start_index, end_index, cache.K->rows, cache.K->cols, cache.K_heads[i]->data);
    slice_array(cache.V->data, start_index, end_index, cache.V->rows, cache.V->cols, cache.V_heads[i]->data);
  }
  cache.concat_result = matrix_alloc(input->rows, W_V->cols);
  cache.head_outputs = malloc(num_of_heads * sizeof(Matrix *));
  cache.head_scores = malloc(num_of_heads * sizeof(Matrix *)); // STORE THESE!

  for (int i = 0; i < num_of_heads; i++)
  {
    AttentionOutput attn_out = masked_attention(cache.Q_heads[i], cache.K_heads[i], cache.V_heads[i]);
    cache.head_outputs[i] = attn_out.result;
    cache.head_scores[i] = attn_out.score;
  }
  for (int row = 0; row < input->rows; row++)
  {
    for (int h = 0; h < num_of_heads; h++)
    {
      for (int col = 0; col < head_dim; col++)
      {
        int concat_col = h * head_dim + col;
        cache.concat_result->data[row * cache.concat_result->cols + concat_col] = cache.head_outputs[h]->data[row * head_dim + col];
      }
    }
  }

  cache.result = matrix_alloc(input->rows, W_0->cols);
  matmul(cache.concat_result, W_0, cache.result);

  cache.num_of_heads = num_of_heads;

  return cache;
}
void multi_head_masked_attention_backward(
    Matrix *d_final_result, // gradient flowing back from next layer
    Matrix *input,          // original input (from forward)
    Matrix *W_Q,            // weight matrices (from forward)
    Matrix *W_K,
    Matrix *W_V,
    Matrix *W_0,
    MultiHeadAttentionCache *cache, // all the stored activations from forward
    Matrix *d_input,                // output: gradient w.r.t. input
    Matrix *d_W_Q,                  // output: gradient w.r.t. W_Q
    Matrix *d_W_K,                  // output: gradient w.r.t. W_K
    Matrix *d_W_V,                  // output: gradient w.r.t. W_V
    Matrix *d_W_0                   // output: gradient w.r.t. W_0
)
{
  Matrix *Q = cache->Q;
  Matrix *K = cache->K;
  Matrix *V = cache->V;
  int head_dim = Q->cols / cache->num_of_heads;
  Matrix **Q_heads = cache->Q_heads;
  Matrix **K_heads = cache->K_heads;
  Matrix **V_heads = cache->V_heads;

  Matrix *W_0_T = matrix_alloc(W_0->cols, W_0->rows);
  Matrix *concat_result_T = matrix_alloc(cache->concat_result->cols, cache->concat_result->rows);
  matrix_transpose(W_0, W_0_T);
  matrix_transpose(cache->concat_result, concat_result_T);
  Matrix *d_concat_result = matrix_alloc(cache->concat_result->rows, cache->concat_result->cols);
  matmul(d_final_result, W_0_T, d_concat_result);
  matmul(concat_result_T, d_concat_result, d_W_0);
  Matrix **d_head_outputs = malloc(cache->num_of_heads * sizeof(Matrix *));
  for (int i = 0; i < cache->num_of_heads; i++)
  {
    d_head_outputs[i] = matrix_alloc(input->rows, head_dim);
  }
  for (int row = 0; row < d_concat_result->rows; row++)
  {
    for (int h = 0; h < cache->num_of_heads; h++)
    {

      for (int col = 0; col < head_dim; col++)
      {
        int concat_col = h * head_dim + col;
        d_head_outputs[h]->data[row * head_dim + col] = d_concat_result->data[row * cache->concat_result->cols + concat_col];
      }
    }
  }
  Matrix **d_Q_heads = malloc(cache->num_of_heads * sizeof(Matrix *));
  if (d_Q_heads == NULL)
  {
    printf("Could not allocate memory for D Q Heads");
    exit(1);
  }
  Matrix **d_K_heads = malloc(cache->num_of_heads * sizeof(Matrix *));
  if (d_K_heads == NULL)
  {
    free(d_Q_heads);
    printf("Could not allocate memory for D K Heads");
    exit(1);
  }
  Matrix **d_V_heads = malloc(cache->num_of_heads * sizeof(Matrix *));
  if (d_V_heads == NULL)
  {
    free(d_Q_heads);
    free(d_K_heads);
    printf("Could not allocate memory for D V Heads");
    exit(1);
  }
  for (int i = 0; i < cache->num_of_heads; i++)
  {
    d_Q_heads[i] = matrix_alloc(Q_heads[i]->rows, Q_heads[i]->cols);
    d_K_heads[i] = matrix_alloc(K_heads[i]->rows, K_heads[i]->cols);
    d_V_heads[i] = matrix_alloc(V_heads[i]->rows, V_heads[i]->cols);
  }
  for (int i = 0; i < cache->num_of_heads; i++)
  {
    masked_attention_backward(d_head_outputs[i], Q_heads[i], K_heads[i], V_heads[i], cache->head_scores[i], d_Q_heads[i], d_K_heads[i], d_V_heads[i]);
  }
  Matrix *d_Q = matrix_alloc(Q->rows, Q->cols);
  Matrix *d_K = matrix_alloc(K->rows, K->cols);
  Matrix *d_V = matrix_alloc(V->rows, V->cols);
  for (int h = 0; h < cache->num_of_heads; h++)
  {
    for (int row = 0; row < d_Q_heads[h]->rows; row++)
    {
      for (int col = 0; col < d_Q_heads[h]->cols; col++)
      {
        d_Q->data[row * d_Q->cols + h * head_dim + col] = d_Q_heads[h]->data[row * head_dim + col];
      }
    }
  }
  for (int h = 0; h < cache->num_of_heads; h++)
  {
    for (int row = 0; row < d_K_heads[h]->rows; row++)
    {
      for (int col = 0; col < d_K_heads[h]->cols; col++)
      {
        d_K->data[row * d_K->cols + h * head_dim + col] = d_K_heads[h]->data[row * head_dim + col];
      }
    }
  }
  for (int h = 0; h < cache->num_of_heads; h++)
  {
    for (int row = 0; row < d_V_heads[h]->rows; row++)
    {
      for (int col = 0; col < d_V_heads[h]->cols; col++)
      {
        d_V->data[row * d_V->cols + h * head_dim + col] = d_V_heads[h]->data[row * head_dim + col];
      }
    }
  }
  Matrix *d_input_from_Q = matrix_alloc(input->rows, input->cols);
  Matrix *d_input_from_K = matrix_alloc(input->rows, input->cols);
  Matrix *d_input_from_V = matrix_alloc(input->rows, input->cols);
  Matrix *W_Q_T = matrix_alloc(W_Q->cols, W_Q->rows);
  Matrix *W_K_T = matrix_alloc(W_K->cols, W_K->rows);
  Matrix *W_V_T = matrix_alloc(W_V->cols, W_V->rows);
  matrix_transpose(W_Q, W_Q_T);
  matrix_transpose(W_K, W_K_T);
  matrix_transpose(W_V, W_V_T);
  matmul(d_Q, W_Q_T, d_input_from_Q);
  matmul(d_K, W_K_T, d_input_from_K);
  matmul(d_V, W_V_T, d_input_from_V);
  for (int i = 0; i < input->rows; i++)
  {
    for (int j = 0; j < input->cols; j++)
    {
      int idx = i * input->cols + j;
      d_input->data[idx] = d_input_from_Q->data[idx] + d_input_from_K->data[idx] + d_input_from_V->data[idx];
    }
  }
  Matrix *input_T = matrix_alloc(input->cols, input->rows);
  for (int i = 0; i < cache->num_of_heads; i++)
  {
    matrix_free(d_Q_heads[i]);
    matrix_free(d_V_heads[i]);
    matrix_free(d_K_heads[i]);
  }
  matrix_transpose(input, input_T);
  matmul(input_T, d_Q, d_W_Q);
  matmul(input_T, d_K, d_W_K);
  matmul(input_T, d_V, d_W_V);
  matrix_free(d_input_from_Q);
  matrix_free(d_input_from_K);
  matrix_free(d_input_from_V);
  matrix_free(concat_result_T);
  matrix_free(d_concat_result);
  matrix_free(W_Q_T);
  matrix_free(W_K_T);
  matrix_free(W_V_T);
  matrix_free(W_0_T);
  matrix_free(d_Q);
  matrix_free(d_K);
  matrix_free(d_V);
  free(d_Q_heads);
  free(d_K_heads);
  free(d_V_heads);

  for (int i = 0; i < cache->num_of_heads; i++)
  {
    matrix_free(d_head_outputs[i]);
  }
  free(d_head_outputs);
}
void decoder_layer_backward(
    Matrix *dOutput,

    DecoderLayerCache *cache,

    // Weights
    Matrix *W_Q_self,
    Matrix *W_K_self,
    Matrix *W_V_self,
    Matrix *W_O_self,
    Matrix *W1,
    Matrix *W2,
    float *ln1_gain,
    float *ln1_bias, // ← ADD THIS
    float *ln2_gain,
    float *ln2_bias, // ← ADD THIS
    int num_heads,

    // Outputs
    Matrix *dInput,
    Matrix *dW_Q_self,
    Matrix *dW_K_self,
    Matrix *dW_V_self,
    Matrix *dW_O_self,
    Matrix *dW1,
    Matrix *dB1,
    Matrix *dW2,
    Matrix *dB2,
    float *dLn1_gain,
    float *dLn1_bias,
    float *dLn2_gain,
    float *dLn2_bias)
{
  Matrix *d_residual_2 = matrix_alloc(dOutput->rows, dOutput->cols);
  Matrix *d_residual_1 = matrix_alloc(dOutput->rows, dOutput->cols);

  for (int i = 0; i < dOutput->rows; i++)
  {
    int start_index = i * dOutput->cols;
    int end_index = start_index + dOutput->cols;
    Matrix *d_norm_row = matrix_alloc(1, dOutput->cols);
    for (int j = 0; j < dOutput->cols; j++)
    {
      d_norm_row->data[j] = dOutput->data[i * dOutput->cols + j];
    }
    Matrix *d_residual2_row = matrix_alloc(1, dOutput->cols);

    layer_normalize_backward(
        d_norm_row,            // gradient for this row
        &cache->ln2_caches[i], // cache for this row
        ln2_gain,              // gain from forward
        ln2_bias,              // bias from forward  ← NOW YOU HAVE IT!
        dOutput->cols,
        d_residual2_row, // output: gradient for unnormalized row
        dLn2_gain,       // output: accumulate gain gradient
        dLn2_bias        // output: accumulate bias gradient
    );
    for (int j = 0; j < dOutput->cols; j++)
    {
      d_residual_2->data[i * dOutput->cols + j] = d_residual2_row->data[j];
    }
    matrix_free(d_norm_row);
    matrix_free(d_residual2_row);
  }
  Matrix *grad_ffn_output = matrix_alloc(d_residual_2->rows, d_residual_2->cols);
  Matrix *grad_norm1 = matrix_alloc(d_residual_2->rows, d_residual_2->cols);
  matrix_copy(d_residual_2, grad_ffn_output);
  matrix_copy(d_residual_2, grad_norm1);
  Matrix *d_norm1_from_ffn = matrix_alloc(cache->norm1->rows, cache->norm1->cols);
  feed_forward_backward(grad_ffn_output, cache->norm1, cache->ffn_pre_gelu, cache->ffn_post_gelu, W1, W2, d_norm1_from_ffn, dW1, dB1, dW2, dB2);
  for (int i = 0; i < d_norm1_from_ffn->capacity; i++)
  {
    grad_norm1->data[i] = grad_norm1->data[i] + d_norm1_from_ffn->data[i];
  }
  // The correct way could be either of these but i think first one is more correct idk
  for (int i = 0; i < cache->norm1->rows; i++)
  {
    Matrix *d_norm_row = matrix_alloc(1, cache->residual_1->cols);
    for (int j = 0; j < cache->norm1->cols; j++)
    {
      d_norm_row->data[j] = grad_norm1->data[i * grad_norm1->cols + j];
    }
    Matrix *d_residual2_row = matrix_alloc(1, grad_norm1->cols);
    layer_normalize_backward(d_norm_row, &cache->ln1_caches[i], ln1_gain, ln1_bias, dOutput->cols, d_residual2_row, dLn1_gain, dLn1_bias);
    for (int j = 0; j < d_residual_1->cols; j++)
    {
      d_residual_1->data[i * d_residual_1->cols + j] = d_residual2_row->data[j];
    }
    matrix_free(d_norm_row);
    matrix_free(d_residual2_row);
  }
  Matrix *d_attention_result = matrix_alloc(cache->residual_1->rows, cache->residual_1->cols);
  Matrix *d_decoder_input_from_residual = matrix_alloc(cache->residual_1->rows, cache->residual_1->cols);
  for (int i = 0; i < cache->residual_1->capacity; i++)
  {
    d_attention_result->data[i] = d_residual_1->data[i];
    d_decoder_input_from_residual->data[i] = d_residual_1->data[i];
  }
  Matrix *d_decoder_input_from_attention = matrix_alloc(cache->attention_cache.input->rows, cache->attention_cache.input->cols);
  multi_head_masked_attention_backward(d_attention_result, cache->attention_cache.input, W_Q_self, W_K_self, W_V_self, W_O_self, &cache->attention_cache, d_decoder_input_from_attention, dW_Q_self, dW_K_self, dW_V_self, dW_O_self);
  for (int i = 0; i < dInput->capacity; i++)
  {
    dInput->data[i] = d_decoder_input_from_attention->data[i] + d_decoder_input_from_residual->data[i];
  }
  matrix_free(d_residual_1);
  matrix_free(d_residual_2);
  matrix_free(grad_ffn_output);
  matrix_free(grad_norm1);
  matrix_free(d_norm1_from_ffn);
  matrix_free(d_attention_result);
  matrix_free(d_decoder_input_from_residual);
  matrix_free(d_decoder_input_from_attention);
}

DecoderLayerCache decoder_layer(
    Matrix *decoder_input,
    Matrix *W_Q_self, Matrix *W_K_self, Matrix *W_V_self, Matrix *W_O_self,
    int num_heads,
    Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2,
    float *ln1_gain, float *ln1_bias,
    float *ln2_gain, float *ln2_bias,
    Matrix *output)
{
  DecoderLayerCache cache;
  cache.input = decoder_input;
  cache.attention_cache = multi_head_masked_attention(decoder_input, W_Q_self, W_K_self, W_V_self, W_O_self, num_heads);

  cache.residual_1 = matrix_alloc(decoder_input->rows, decoder_input->cols);
  cache.ln1_caches = malloc(decoder_input->rows * sizeof(LayerNormCache));
  matrix_add(cache.attention_cache.result, decoder_input, cache.residual_1);

  float *normalised_rows_for_self_attention_output = malloc(cache.residual_1->capacity * sizeof(float));

  for (int i = 0; i < cache.attention_cache.result->rows; i++)
  {
    int start_index = i * cache.residual_1->cols;
    int end_index = start_index + cache.residual_1->cols;
    float *layer = get_row(cache.residual_1->data, start_index, end_index);

    layer_normalize(layer, &cache.ln1_caches[i], ln1_gain, ln1_bias, cache.residual_1->cols);

    for (int j = 0; j < (end_index - start_index); j++)
    {
      normalised_rows_for_self_attention_output[start_index + j] = cache.ln1_caches[i].normalized[j];
    }
    free(layer);
  }
  cache.norm1 = matrix_alloc(decoder_input->rows, decoder_input->cols);

  for (int i = 0; i < cache.norm1->capacity; i++)
  {
    cache.norm1->data[i] = normalised_rows_for_self_attention_output[i];
  }

  cache.ffn_output = matrix_alloc(decoder_input->rows, decoder_input->cols);
  cache.ffn_pre_gelu = matrix_alloc(cache.norm1->rows, W1->cols);
  cache.ffn_post_gelu = matrix_alloc(cache.norm1->rows, W1->cols);
  feed_forward(cache.norm1, W1, b1, W2, b2, cache.ffn_output, cache.ffn_post_gelu, cache.ffn_pre_gelu);
  cache.residual_2 = matrix_alloc(decoder_input->rows, decoder_input->cols);
  matrix_add(cache.ffn_output, cache.norm1, cache.residual_2);
  float *normalised_rows_for_feed_forward_output = malloc(cache.residual_2->capacity * sizeof(float));
  cache.ln2_caches = malloc(decoder_input->rows * sizeof(LayerNormCache));
  for (int i = 0; i < cache.attention_cache.result->rows; i++)
  {
    int start_index = i * cache.residual_2->cols;
    int end_index = start_index + cache.residual_2->cols;
    float *layer = get_row(cache.residual_2->data, start_index, end_index);
    layer_normalize(layer, &cache.ln2_caches[i], ln2_gain, ln2_bias, cache.residual_2->cols);

    for (int j = 0; j < (end_index - start_index); j++)
    {
      normalised_rows_for_feed_forward_output[start_index + j] = cache.ln2_caches[i].normalized[j];
    }
    free(layer);
  }
  for (int i = 0; i < cache.norm1->capacity; i++)
  {
    output->data[i] = normalised_rows_for_feed_forward_output[i];
  }
  free(normalised_rows_for_feed_forward_output);
  free(normalised_rows_for_self_attention_output);
  return cache;
}
void decoder_backward(Matrix *d_output,           // gradient from output projection
                      DecoderLayerCache **caches, // caches from forward pass

                      // All weights
                      Matrix **W_Q_self_layers,
                      Matrix **W_K_self_layers,
                      Matrix **W_V_self_layers,
                      Matrix **W_O_self_layers,
                      Matrix **W1_layers,
                      Matrix **W2_layers,
                      float **ln1_gain_layers,
                      float **ln1_bias_layers,
                      float **ln2_gain_layers,
                      float **ln2_bias_layers,

                      int num_layers,
                      int num_heads,

                      // Outputs (gradients)
                      Matrix *d_input, // gradient w.r.t. decoder input
                      Matrix **dW_Q_self_layers,
                      Matrix **dW_K_self_layers,
                      Matrix **dW_V_self_layers,
                      Matrix **dW_O_self_layers,
                      Matrix **dW1_layers,
                      Matrix **dB1_layers,
                      Matrix **dW2_layers,
                      Matrix **dB2_layers,
                      float **dLn1_gain_layers,
                      float **dLn1_bias_layers,
                      float **dLn2_gain_layers,
                      float **dLn2_bias_layers)

{
  Matrix *current_gradient = d_output;
  for (int layer = num_layers - 1; layer > 0; layer--)
  {
    Matrix *layer_input_grad = matrix_alloc(d_output->rows, d_output->cols);
    decoder_layer_backward(current_gradient, caches[layer], W_Q_self_layers[layer], W_K_self_layers[layer], W_V_self_layers[layer], W_O_self_layers[layer], W1_layers[layer], W2_layers[layer], ln1_gain_layers[layer], ln1_bias_layers[layer], ln2_gain_layers[layer], ln2_bias_layers[layer], num_heads, layer_input_grad, dW_Q_self_layers[layer], dW_K_self_layers[layer], dW_V_self_layers[layer], dW_O_self_layers[layer], dW1_layers[layer], dB1_layers[layer], dW2_layers[layer], dB2_layers[layer], dLn1_gain_layers[layer], dLn1_bias_layers[layer], dLn2_gain_layers, dLn2_bias_layers[layer]);
    if (layer < num_layers - 1)
    {
      matrix_free(current_gradient);
    }
    current_gradient = layer_input_grad;
  }
  for (int i = 0; i < d_input->capacity; i++)
  {
    d_input->data[i] = current_gradient->data[i];
  }
  matrix_free(current_gradient);
}
void decoder(
    Matrix *decoder_input, // Target sequence embeddings (with positional encoding)
    // Self-attention weights (masked) for each layer
    Matrix **W_Q_self_layers,
    Matrix **W_K_self_layers,
    Matrix **W_V_self_layers,
    Matrix **W_O_self_layers,

    // Feed-forward weights for each layer
    Matrix **W1_layers,
    Matrix **b1_layers,
    Matrix **W2_layers,
    Matrix **b2_layers,

    float **ln1_gain_layers, // After masked self-attention
    float **ln1_bias_layers,
    float **ln2_gain_layers, // After feed-forward (renamed from ln3)
    float **ln2_bias_layers,

    int num_layers, // Number of decoder layers (6 in paper)
    int num_heads,  // Number of attention heads

    Matrix *output // Output matrix (same shape as decoder_input)
)
{
  Matrix *current = decoder_input;
  for (int i = 0; i < num_layers; i++)
  {
    Matrix *next_output = matrix_alloc(decoder_input->rows, decoder_input->cols);
    decoder_layer(current,
                  W_Q_self_layers[i], W_K_self_layers[i], W_V_self_layers[i], W_O_self_layers[i],
                  num_heads,
                  W1_layers[i], b1_layers[i], W2_layers[i], b2_layers[i],
                  ln1_gain_layers[i], ln1_bias_layers[i],
                  ln2_gain_layers[i], ln2_bias_layers[i], // Changed from ln3
                  next_output);
    if (i > 0)
    {
      matrix_free(current);
    }
    current = next_output;
  }
  for (int i = 0; i < output->capacity; i++)
  {
    output->data[i] = current->data[i];
  }
  matrix_free(current);
}

Matrix *embed_tokens(int *token_ids, int seq_len, Matrix *embedding_table)
{
  Matrix *matrix = matrix_alloc(seq_len, embedding_table->cols);
  for (int row = 0; row < seq_len; row++)
  {
    int token_id = token_ids[row];
    for (int col = 0; col < embedding_table->cols; col++)
    {
      matrix->data[row * embedding_table->cols + col] = embedding_table->data[token_id * embedding_table->cols + col];
    }
  }
  return matrix;
}

Matrix *project(Matrix *Q, Matrix *W_Q)
{
  Matrix *result = matrix_alloc(Q->rows, W_Q->cols);
  matmul(Q, W_Q, result);
  return result;
}

Matrix *create_embedding_table(int vocab_size, int d_model)
{
  Matrix *matrix = matrix_alloc(vocab_size, d_model);
  double lower = -0.01;
  double upper = 0.01;
  for (int i = 0; i < matrix->capacity; i++)
  {
    matrix->data[i] = lower + (rand() / (double)RAND_MAX) * (upper - lower);
  }
  return matrix;
}
Matrix *output_projection(Matrix *decoder_output, Matrix *W_out, Matrix *b_out)
{
  Matrix *projected_output = matrix_alloc(decoder_output->rows, W_out->cols);
  matmul(decoder_output, W_out, projected_output);
  add_bias(projected_output, b_out, projected_output);
  return projected_output;
}
int sample_token(Matrix *logits, int vocab_size)
{
  float r = rand() / (double)RAND_MAX;
  mat_softmax_rows(logits);

  float cummulative = 0.0;
  int last_row_start = (logits->rows - 1) * logits->cols;
  for (int i = 0; i < vocab_size; i++)
  {
    cummulative += logits->data[last_row_start + i];
    if (cummulative >= r)
    {
      return i;
    }
  }
  return vocab_size - 1;
}
float cross_entropy_loss(Matrix *logist, int *target_token, int seq_len, int vocab_size)
{
  float accumulated_loss = 0.0;
  mat_softmax_rows(logist);
  for (int i = 0; i < seq_len; i++)
  {
    int target_token_id = target_token[i];
    float probability = logist->data[i * vocab_size + target_token_id];
    probability = fmaxf(probability, 1e-10);
    accumulated_loss += -log(probability);
  }
  return accumulated_loss / seq_len;
}
float compute_gelu_derivative(float x)
{
  // Constants
  float sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/pi)
  float coeff = 0.044715;

  // Compute intermediate values
  float x_cubed = x * x * x;
  float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
  float tanh_val = tanh(tanh_arg);

  // First term: 0.5 * (1 + tanh(...))
  float term1 = 0.5 * (1.0 + tanh_val);

  // Second term: x * derivative of the tanh part
  // d/dx[tanh(arg)] = sech²(arg) * d(arg)/dx
  // sech²(arg) = 1 - tanh²(arg)
  float sech_squared = 1.0 - tanh_val * tanh_val;

  // d(arg)/dx = sqrt_2_over_pi * (1 + 3 * coeff * x²)
  float d_arg_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);

  float term2 = 0.5 * x * sech_squared * d_arg_dx;

  return term1 + term2;
}
void feed_forward_backward(
    Matrix *d_output,         // gradient from next layer
    Matrix *input,            // from cache
    Matrix *hidden_pre_gelu,  // from cache (your naming)
    Matrix *hidden_post_gelu, // from cache (your naming)
    Matrix *W1,
    Matrix *W2,
    Matrix *d_input, // OUTPUT
    Matrix *d_W1,    // OUTPUT
    Matrix *d_b1,    // OUTPUT
    Matrix *d_W2,    // OUTPUT
    Matrix *d_b2     // OUTPUT
)
{
  matrix_sum_rows(d_output, d_b2);
  Matrix *hidden_post_gelu_T = matrix_alloc(hidden_post_gelu->cols, hidden_post_gelu->rows);
  Matrix *W2_T = matrix_alloc(W2->cols, W2->rows);
  matrix_transpose(W2, W2_T);
  Matrix *d_hidden_post_gelu = matrix_alloc(d_output->rows, W2->rows);
  matmul(d_output, W2_T, d_hidden_post_gelu);
  matrix_transpose(hidden_post_gelu, hidden_post_gelu_T);
  matmul(hidden_post_gelu_T, d_output, d_W2);
  matrix_free(hidden_post_gelu_T);
  Matrix *d_hidden_pre_gelu = matrix_alloc(hidden_pre_gelu->rows, hidden_pre_gelu->cols);
  for (int i = 0; i < hidden_pre_gelu->capacity; i++)
  {
    float x = hidden_pre_gelu->data[i];
    float gelu_derivative = compute_gelu_derivative(x);
    d_hidden_pre_gelu->data[i] = d_hidden_post_gelu->data[i] * gelu_derivative;
  }

  matrix_sum_rows(d_hidden_pre_gelu, d_b1);

  Matrix *W1_T = matrix_alloc(W1->cols, W1->rows);
  matrix_transpose(W1, W1_T);

  matmul(d_hidden_pre_gelu, W1_T, d_input);
  Matrix *input_T = matrix_alloc(input->cols, input->rows);
  matrix_transpose(input, input_T);

  matmul(input_T, d_hidden_pre_gelu, d_W1);
  matrix_free(input_T);
  matrix_free(W1_T);
  matrix_free(W2_T);
  matrix_free(hidden_post_gelu_T);
  matrix_free(d_hidden_pre_gelu);
}

void embedding_backward(
    Matrix *d_embeddings, // gradient flowing back from decoder
    int *token_ids,       // original input tokens
    int seq_len,
    Matrix *d_embedding_table // OUTPUT: gradient for embedding table (accumulate)
)
{
  for (int i = 0; i < seq_len; i++)
  {
    int token_id = token_ids[i];
    for (int j = 0; j < seq_len; j++)
    {
      d_embedding_table->data[token_id * d_embedding_table->cols + j] += d_embeddings->data[i * d_embeddings->cols + j];
    }
  }
};
update_weights(Matrix *weights, Matrix *gradients, float learning_rate)
{
  for (int i = 0; i < weights->capacity; i++)
  {
    int index = i;
    weights->data[index] = weights->data[index] - (learning_rate * gradients->data[index]);
  }
};
update_weights_float(float *weights, float *gradients, int size, float learning_rate)
{
  for (int i = 0; i < size; i++)
  {
    weights[i] = weights[i] - (gradients[i] * learning_rate);
  }
}
// 4. Complete training step with backward pass
float train_step_with_gradients(
    int *input_tokens,
    int *target_tokens,
    int seq_len,
    int num_layers,
    int num_heads,
    int d_model,
    int vocab_size,
    // All model weights (to be updated)
    Matrix *embedding_table,
    Matrix **W_Q_layers,
    Matrix **W_K_layers,
    Matrix **W_V_layers,
    Matrix **W_O_layers,
    Matrix **W1_layers,
    Matrix **b1_layers,
    Matrix **W2_layers,
    Matrix **b2_layers,
    float **ln1_gain_layers,
    float **ln1_bias_layers,
    float **ln2_gain_layers,
    float **ln2_bias_layers,
    Matrix *W_out,
    Matrix *b_out,
    DecoderLayerCache **layer_caches,
    float learning_rate) {

};
TransformerModel *init_model(int vocab_size, int d_model, int num_layers, int num_heads)
{
  int ffn_hidden_dim = 4 * d_model;
  TransformerModel *model = malloc(sizeof(TransformerModel));
  model->vocab_size = vocab_size;
  model->d_model = d_model;
  model->num_heads = num_heads;
  model->num_layers = num_layers;
  model->ffn_hidden_dim = ffn_hidden_dim;

  // Initialize embedding table - each value is different!
  model->embedding_table = matrix_alloc(vocab_size, d_model);
  float embed_scale = sqrt(2.0f / d_model);
  for (int i = 0; i < model->embedding_table->capacity; i++)
  {
    // Generate NEW random number for EACH weight
    model->embedding_table->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * embed_scale;
  }

  model->W_Q_layers = malloc(num_layers * sizeof(Matrix *));
  model->W_K_layers = malloc(num_layers * sizeof(Matrix *));
  model->W_V_layers = malloc(num_layers * sizeof(Matrix *));
  model->W_O_layers = malloc(num_layers * sizeof(Matrix *));
  model->W1_layers = malloc(num_layers * sizeof(Matrix *));
  model->b1_layers = malloc(num_layers * sizeof(Matrix *));
  model->W2_layers = malloc(num_layers * sizeof(Matrix *));
  model->b2_layers = malloc(num_layers * sizeof(Matrix *));
  model->ln1_gain_layers = malloc(num_layers * sizeof(float *));
  model->ln1_bias_layers = malloc(num_layers * sizeof(float *));
  model->ln2_gain_layers = malloc(num_layers * sizeof(float *));
  model->ln2_bias_layers = malloc(num_layers * sizeof(float *));

  for (int i = 0; i < num_layers; i++)
  {
    // Attention Matrices
    model->W_Q_layers[i] = matrix_alloc(d_model, d_model);
    model->W_K_layers[i] = matrix_alloc(d_model, d_model);
    model->W_V_layers[i] = matrix_alloc(d_model, d_model);
    model->W_O_layers[i] = matrix_alloc(d_model, d_model);

    float attn_scale = sqrt(2.0f / d_model);
    for (int j = 0; j < d_model * d_model; j++)
    {
      // NEW random number for each weight!
      model->W_Q_layers[i]->data[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * attn_scale;
      model->W_K_layers[i]->data[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * attn_scale;
      model->W_V_layers[i]->data[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * attn_scale;
      model->W_O_layers[i]->data[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * attn_scale;
    }

    // FFN matrices
    model->W1_layers[i] = matrix_alloc(d_model, ffn_hidden_dim);
    model->b1_layers[i] = matrix_alloc(1, ffn_hidden_dim);
    model->W2_layers[i] = matrix_alloc(ffn_hidden_dim, d_model);
    model->b2_layers[i] = matrix_alloc(1, d_model);

    float w1_scale = sqrt(2.0f / d_model);
    float w2_scale = sqrt(2.0f / ffn_hidden_dim);

    for (int j = 0; j < model->W1_layers[i]->capacity; j++)
    {
      model->W1_layers[i]->data[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * w1_scale;
    }
    for (int j = 0; j < model->b1_layers[i]->capacity; j++)
    {
      model->b1_layers[i]->data[j] = 0.0f;
    }
    for (int j = 0; j < model->W2_layers[i]->capacity; j++)
    {
      model->W2_layers[i]->data[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * w2_scale;
    }
    for (int j = 0; j < model->b2_layers[i]->capacity; j++)
    {
      model->b2_layers[i]->data[j] = 0.0f;
    }

    // Layer norm params
    model->ln1_gain_layers[i] = malloc(d_model * sizeof(float));
    model->ln1_bias_layers[i] = malloc(d_model * sizeof(float));
    model->ln2_gain_layers[i] = malloc(d_model * sizeof(float));
    model->ln2_bias_layers[i] = malloc(d_model * sizeof(float));

    for (int j = 0; j < d_model; j++)
    {
      model->ln1_gain_layers[i][j] = 1.0f;
      model->ln1_bias_layers[i][j] = 0.0f;
      model->ln2_gain_layers[i][j] = 1.0f;
      model->ln2_bias_layers[i][j] = 0.0f;
    }
  }

  model->W_out = matrix_alloc(d_model, vocab_size);
  model->b_out = matrix_alloc(1, vocab_size);

  float out_scale = sqrt(2.0f / d_model);
  for (int i = 0; i < model->W_out->capacity; i++)
  {
    model->W_out->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * out_scale;
  }
  for (int i = 0; i < model->b_out->capacity; i++)
  {
    model->b_out->data[i] = 0.0f;
  }

  return model;
}
char *generate(TransformerModel *model, Vocabulary *vocab, char *prompt, int max_len)
{
  int prompt_token_length;
  int *prompt_token_ids = tokenize(prompt, vocab, &prompt_token_length);
  int *sequence = malloc((prompt_token_length + max_len) * sizeof(int));
  for (int i = 0; i < prompt_token_length; i++)
  {
    sequence[i] = prompt_token_ids[i];
  }
  int current_length = prompt_token_length;
  for (int i = 0; i < max_len; i++)
  {
    Matrix *embeddings = embed_tokens(sequence, current_length, model->embedding_table);
    add_positional_encoding(embeddings, current_length, model->d_model);
    Matrix *decoder_output = matrix_alloc(current_length, model->d_model);
    decoder(
        embeddings,
        model->W_Q_layers,      // <-- from model
        model->W_K_layers,      // <-- from model
        model->W_V_layers,      // <-- from model
        model->W_O_layers,      // <-- from model
        model->W1_layers,       // <-- from model
        model->b1_layers,       // <-- from model
        model->W2_layers,       // <-- from model
        model->b2_layers,       // <-- from model
        model->ln1_gain_layers, // <-- from model
        model->ln1_bias_layers, // <-- from model
        model->ln2_gain_layers, // <-- from model
        model->ln2_bias_layers, // <-- from model
        model->num_layers,      // <-- from model
        model->num_heads,       // <-- from model
        decoder_output);
    Matrix *projected_output = output_projection(decoder_output, model->W_out, model->b_out);
    int last_row_index = current_length - 1;
    Matrix *last_row_mat = matrix_alloc(1, projected_output->cols);

    for (int col = 0; col < projected_output->cols; col++)
    {
      last_row_mat->data[col] = projected_output->data[last_row_index * projected_output->cols + col];
    }
    int next_token = sample_token(last_row_mat, vocab->vocab_size);
    sequence[current_length] = next_token;
    current_length++;
    matrix_free(decoder_output);
    matrix_free(projected_output);
    matrix_free(last_row_mat);
    matrix_free(embeddings);
  }
  char *generated = detokenize(sequence, current_length, vocab);
  free(sequence);
  free(prompt_token_ids);
  return generated;
}
float train_step_with_gradients(
    TransformerModel *model,
    Vocabulary *vocab,
    int *input_tokens,
    int *target_tokens,
    int seq_len,
    float learning_rate)
{
  // ========== FORWARD PASS ==========
  Matrix *embeddings = embed_tokens(input_tokens, seq_len, model->embedding_table);
  add_positional_encoding(embeddings, seq_len, model->d_model);

  DecoderLayerCache **layer_caches = malloc(model->num_layers * sizeof(DecoderLayerCache *));

  Matrix *decoder_output = matrix_alloc(seq_len, model->d_model);

  // Run through ALL decoder layers using the decoder function
  decoder(embeddings,
          model->W_Q_layers, model->W_K_layers, model->W_V_layers, model->W_O_layers,
          model->W1_layers, model->b1_layers, model->W2_layers, model->b2_layers,
          model->ln1_gain_layers, model->ln1_bias_layers,
          model->ln2_gain_layers, model->ln2_bias_layers,
          model->num_layers, model->num_heads,
          decoder_output);

  Matrix *logits = output_projection(decoder_output, model->W_out, model->b_out);
  float loss = cross_entropy_loss(logits, target_tokens, seq_len, model->vocab_size);

  // ========== BACKWARD PASS ==========

  // Gradient of loss w.r.t. logits
  Matrix *dLogits = matrix_alloc(logits->rows, logits->cols);
  int vocab_size = model->vocab_size;
  for (int row = 0; row < seq_len; row++)
  {
    int correct_token = target_tokens[row];
    for (int col = 0; col < vocab_size; col++)
    {
      int index = row * vocab_size + col;
      dLogits->data[index] = logits->data[index];
      if (col == correct_token) // FIX: was "col == target_tokens"
      {
        dLogits->data[index] -= 1.0;
      }
    }
  }

  // Backprop through output projection
  Matrix *dW_out = matrix_alloc(model->W_out->rows, model->W_out->cols);
  Matrix *db_out = matrix_alloc(model->b_out->rows, model->b_out->cols);
  Matrix *d_decoder_output = matrix_alloc(decoder_output->rows, decoder_output->cols);

  Matrix *decoder_output_T = matrix_alloc(decoder_output->cols, decoder_output->rows);
  matrix_transpose(decoder_output, decoder_output_T);
  matmul(decoder_output_T, dLogits, dW_out); // FIX: compute dW_out correctly
  matrix_sum_rows(dLogits, db_out);

  Matrix *W_out_T = matrix_alloc(model->W_out->cols, model->W_out->rows);
  matrix_transpose(model->W_out, W_out_T);
  matmul(dLogits, W_out_T, d_decoder_output); // FIX: compute d_decoder_output

  // Allocate gradient matrices for all layers
  Matrix **dW_Q_layers = malloc(model->num_layers * sizeof(Matrix *));
  Matrix **dW_K_layers = malloc(model->num_layers * sizeof(Matrix *));
  Matrix **dW_V_layers = malloc(model->num_layers * sizeof(Matrix *));
  Matrix **dW_O_layers = malloc(model->num_layers * sizeof(Matrix *));

  Matrix **dW1_layers = malloc(model->num_layers * sizeof(Matrix *));
  Matrix **db1_layers = malloc(model->num_layers * sizeof(Matrix *));
  Matrix **dW2_layers = malloc(model->num_layers * sizeof(Matrix *));
  Matrix **db2_layers = malloc(model->num_layers * sizeof(Matrix *));

  float **dLn1_gain_layers = malloc(model->num_layers * sizeof(float *));
  float **dLn1_bias_layers = malloc(model->num_layers * sizeof(float *));
  float **dLn2_gain_layers = malloc(model->num_layers * sizeof(float *));
  float **dLn2_bias_layers = malloc(model->num_layers * sizeof(float *));

  for (int i = 0; i < model->num_layers; i++)
  {
    dW_Q_layers[i] = matrix_alloc(model->W_Q_layers[i]->rows, model->W_Q_layers[i]->cols);
    dW_K_layers[i] = matrix_alloc(model->W_K_layers[i]->rows, model->W_K_layers[i]->cols);
    dW_V_layers[i] = matrix_alloc(model->W_V_layers[i]->rows, model->W_V_layers[i]->cols);
    dW_O_layers[i] = matrix_alloc(model->W_O_layers[i]->rows, model->W_O_layers[i]->cols);
    dW1_layers[i] = matrix_alloc(model->W1_layers[i]->rows, model->W1_layers[i]->cols);
    db1_layers[i] = matrix_alloc(model->b1_layers[i]->rows, model->b1_layers[i]->cols);
    dW2_layers[i] = matrix_alloc(model->W2_layers[i]->rows, model->W2_layers[i]->cols);
    db2_layers[i] = matrix_alloc(model->b2_layers[i]->rows, model->b2_layers[i]->cols);

    dLn1_gain_layers[i] = calloc(model->d_model, sizeof(float));
    dLn1_bias_layers[i] = calloc(model->d_model, sizeof(float));
    dLn2_gain_layers[i] = calloc(model->d_model, sizeof(float));
    dLn2_bias_layers[i] = calloc(model->d_model, sizeof(float));
  }

  Matrix *d_embeddings = matrix_alloc(embeddings->rows, embeddings->cols);

  // Backprop through decoder layers
  decoder_backward(d_decoder_output,
                   layer_caches,
                   model->W_Q_layers, model->W_K_layers, model->W_V_layers, model->W_O_layers,
                   model->W1_layers, model->W2_layers,
                   model->ln1_gain_layers, model->ln1_bias_layers, // FIX: was ln2_gain
                   model->ln2_gain_layers, model->ln2_bias_layers,
                   model->num_layers, model->num_heads,
                   d_embeddings,
                   dW_Q_layers, dW_K_layers, dW_V_layers, dW_O_layers,
                   dW1_layers, db1_layers, dW2_layers, db2_layers,
                   dLn1_gain_layers, dLn1_bias_layers,
                   dLn2_gain_layers, dLn2_bias_layers);

  // Backprop through embeddings
  Matrix *d_embedding_table = matrix_alloc(model->embedding_table->rows, model->embedding_table->cols);
  embedding_backward(d_embeddings, input_tokens, seq_len, d_embedding_table);

  // ========== UPDATE WEIGHTS ==========

  // Update output projection
  update_weights(model->W_out, dW_out, learning_rate);
  update_weights(model->b_out, db_out, learning_rate);

  // Update embedding table
  update_weights(model->embedding_table, d_embedding_table, learning_rate);

  // Update all decoder layer weights
  for (int i = 0; i < model->num_layers; i++)
  {
    update_weights(model->W_Q_layers[i], dW_Q_layers[i], learning_rate);
    update_weights(model->W_K_layers[i], dW_K_layers[i], learning_rate);
    update_weights(model->W_V_layers[i], dW_V_layers[i], learning_rate);
    update_weights(model->W_O_layers[i], dW_O_layers[i], learning_rate);
    update_weights(model->W1_layers[i], dW1_layers[i], learning_rate);
    update_weights(model->b1_layers[i], db1_layers[i], learning_rate);
    update_weights(model->W2_layers[i], dW2_layers[i], learning_rate);
    update_weights(model->b2_layers[i], db2_layers[i], learning_rate);

    update_weights_float(model->ln1_gain_layers[i], dLn1_gain_layers[i], model->d_model, learning_rate);
    update_weights_float(model->ln1_bias_layers[i], dLn1_bias_layers[i], model->d_model, learning_rate);
    update_weights_float(model->ln2_gain_layers[i], dLn2_gain_layers[i], model->d_model, learning_rate);
    update_weights_float(model->ln2_bias_layers[i], dLn2_bias_layers[i], model->d_model, learning_rate);
  }

  // ========== CLEANUP ==========

  matrix_free(embeddings);
  matrix_free(decoder_output);
  matrix_free(logits);
  matrix_free(dLogits);
  matrix_free(dW_out);
  matrix_free(db_out);
  matrix_free(d_decoder_output);
  matrix_free(decoder_output_T);
  matrix_free(W_out_T);
  matrix_free(d_embeddings);
  matrix_free(d_embedding_table);

  for (int i = 0; i < model->num_layers; i++)
  {
    matrix_free(dW_Q_layers[i]);
    matrix_free(dW_K_layers[i]);
    matrix_free(dW_V_layers[i]);
    matrix_free(dW_O_layers[i]);
    matrix_free(dW1_layers[i]);
    matrix_free(db1_layers[i]);
    matrix_free(dW2_layers[i]);
    matrix_free(db2_layers[i]);

    free(dLn1_gain_layers[i]);
    free(dLn1_bias_layers[i]);
    free(dLn2_gain_layers[i]);
    free(dLn2_bias_layers[i]);
  }

  free(dW_Q_layers);
  free(dW_K_layers);
  free(dW_V_layers);
  free(dW_O_layers);
  free(dW1_layers);
  free(db1_layers);
  free(dW2_layers);
  free(db2_layers);
  free(dLn1_gain_layers);
  free(dLn1_bias_layers);
  free(dLn2_gain_layers);
  free(dLn2_bias_layers);

  // TODO: Free layer_caches properly (need to free internal matrices in each cache)
  free(layer_caches);

  return loss;
}
