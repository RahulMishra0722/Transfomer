#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "math_ops.h"
#include "transformer.h"
#include <time.h>
typedef struct
{
  Matrix *result;
  Matrix *score;
} AttentionOutput;

typedef struct
{
  Matrix *result; // The final output

  // For backprop through final matmul
  Matrix *concat_result; // Concatenated head outputs

  // For backprop through each head
  Matrix **head_outputs; // Output of each attention head
  Matrix **head_scores;  // Attention scores (post-softmax) for each head

  // For backprop through slicing
  Matrix **Q_heads; // Q split into heads
  Matrix **K_heads; // K split into heads
  Matrix **V_heads; // V split into heads

  // For backprop through initial projections
  Matrix *Q; // Q = input @ W_Q
  Matrix *K; // K = input @ W_K
  Matrix *V; // V = input @ W_V

  // Metadata
  int num_of_heads; // Number of attention heads
  int head_dim;     // Dimension per head
} MultiHeadAttentionCache;

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

void feed_forward(Matrix *input, Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2, Matrix *output)
{
  Matrix *matrix_multiply_result = matrix_alloc(input->rows, W1->cols);
  Matrix *hidden = matrix_alloc(input->rows, W1->cols);
  Matrix *second_multiply_result = matrix_alloc(input->rows, W2->cols);

  matmul(input, W1, matrix_multiply_result);
  add_bias(matrix_multiply_result, b1, hidden);

  GELU(hidden);
  matmul(hidden, W2, second_multiply_result);
  add_bias(second_multiply_result, b2, output);

  matrix_free(matrix_multiply_result);
  matrix_free(hidden);
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
Matrix *attention(Matrix *Q, Matrix *K, Matrix *V)
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
    }
  }
  mat_softmax_rows(scores);
  Matrix *result = matrix_alloc(Q->rows, V->cols);
  matmul(scores, V, result);
  matrix_free(T);
  matrix_free(scores);
  return result;
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
  Matrix *Q = matrix_alloc(input->rows, W_Q->cols);
  Matrix *K = matrix_alloc(input->rows, W_K->cols);
  Matrix *V = matrix_alloc(input->rows, W_V->cols);
  matmul(input, W_Q, Q);
  matmul(input, W_K, K);
  matmul(input, W_V, V);
  int head_dim = Q->cols / num_of_heads;
  Matrix **Q_heads = malloc(num_of_heads * sizeof(Matrix));
  Matrix **K_heads = malloc(num_of_heads * sizeof(Matrix));
  Matrix **V_heads = malloc(num_of_heads * sizeof(Matrix));
  for (int h = 0; h < num_of_heads; h++)
  {
    Q_heads[h] = matrix_alloc(Q->rows, head_dim);
    K_heads[h] = matrix_alloc(K->rows, head_dim);
    V_heads[h] = matrix_alloc(V->rows, head_dim);
  }
  for (int i = 0; i < num_of_heads; i++)
  {
    int start_index = i * head_dim;
    int end_index = start_index + head_dim;
    slice_array(Q->data, start_index, end_index, Q->rows, Q->cols, Q_heads[i]->data);
    slice_array(K->data, start_index, end_index, K->rows, K->cols, K_heads[i]->data);
    slice_array(V->data, start_index, end_index, V->rows, V->cols, V_heads[i]->data);
  }
  Matrix *concat_result = matrix_alloc(input->rows, W_V->cols);
  Matrix **head_outputs = malloc(num_of_heads * sizeof(Matrix));
  Matrix **head_scores = malloc(num_of_heads * sizeof(Matrix *)); // STORE THESE!

  int index = 0;
  for (int i = 0; i < num_of_heads; i++)
  {
    AttentionOutput attn_out = masked_attention(Q_heads[i], K_heads[i], V_heads[i]);
    head_outputs[i] = attn_out.result;
    head_scores[i] = attn_out.score;
  }
  for (int row = 0; row < input->rows; row++)
  {
    for (int h = 0; h < num_of_heads; h++)
    {
      for (int col = 0; col < head_dim; col++)
      {
        int concat_col = h * head_dim + col;
        concat_result->data[row * concat_result->cols + concat_col] = head_outputs[h]->data[row * head_dim + col];
      }
    }
  }
  for (int i = 0; i < num_of_heads; i++)
  {
    matrix_free(head_outputs[i]);
  }
  Matrix *final_result = matrix_alloc(input->rows, W_0->cols);
  matmul(concat_result, W_0, final_result);
  for (int i = 0; i < num_of_heads; i++)
  {
    matrix_free(Q_heads[i]);
    matrix_free(K_heads[i]);
    matrix_free(V_heads[i]);
  }
  MultiHeadAttentionCache cache;
  cache.concat_result = concat_result;
  cache.head_dim = head_dim;
  cache.head_outputs = head_outputs;
  cache.head_scores = head_scores;
  cache.K = K;
  cache.Q = Q;
  cache.V = V;

  cache.num_of_heads = num_of_heads;
  cache.Q_heads = Q_heads;
  cache.result = final_result;
  cache.V_heads = V_heads;

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

    Matrix *decoder_input,
    Matrix *W_Q_self,
    Matrix *W_K_self,
    Matrix *W_V_self,
    Matrix *W_O_self,
    int num_heads,
    Matrix *W1,
    Matrix *b1,
    Matrix *W2,
    Matrix *b2,
    float *ln1_gain,
    float *ln1_bias,
    float *ln2_gain,
    float *ln2_bias,

    // Saved activations from forward pass (you need to save these!)
    Matrix *self_attention_output,
    Matrix *residual_1,
    Matrix *norm1,
    Matrix *feed_forward_result,
    Matrix *residual_2,

    // Output: gradient to pass backward
    Matrix *dInput,

    // Output: gradients for all weights in this layer
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
}

void decoder_layer(
    Matrix *decoder_input,
    Matrix *W_Q_self, Matrix *W_K_self, Matrix *W_V_self, Matrix *W_O_self,
    int num_heads,
    Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2,
    float *ln1_gain, float *ln1_bias,
    float *ln2_gain, float *ln2_bias, // Renamed from ln3
    Matrix *output)
{
  MultiHeadAttentionCache cache = multi_head_masked_attention(decoder_input, W_Q_self, W_K_self, W_V_self, W_O_self, num_heads);
  Matrix *self_attention_output = cache.result;
  Matrix *residual_1 = matrix_alloc(decoder_input->rows, decoder_input->cols);
  matrix_add(self_attention_output, decoder_input, residual_1);

  float *normalised_rows_for_self_attention_output = malloc(residual_1->capacity * sizeof(float));

  for (int i = 0; i < self_attention_output->rows; i++)
  {
    int start_index = i * residual_1->cols;
    int end_index = start_index + residual_1->cols;
    float *layer = get_row(residual_1->data, start_index, end_index);
    float *normalised_row = malloc((end_index - start_index) * sizeof(float));

    layer_normalize(layer, normalised_row, ln1_gain, ln1_bias, residual_1->cols);

    for (int j = 0; j < (end_index - start_index); j++)
    {
      normalised_rows_for_self_attention_output[start_index + j] = normalised_row[j];
    }
    free(normalised_row);
    free(layer);
  }
  Matrix *norm1 = matrix_alloc(decoder_input->rows, decoder_input->cols);

  for (int i = 0; i < norm1->capacity; i++)
  {
    norm1->data[i] = normalised_rows_for_self_attention_output[i];
  }

  Matrix *feed_forward_result = matrix_alloc(decoder_input->rows, decoder_input->cols);

  feed_forward(norm1, W1, b1, W2, b2, feed_forward_result);
  Matrix *residual_3 = matrix_alloc(decoder_input->rows, decoder_input->cols);
  matrix_add(feed_forward_result, norm1, residual_3);
  float *normalised_rows_for_feed_forward_output = malloc(residual_3->capacity * sizeof(float));
  for (int i = 0; i < self_attention_output->rows; i++)
  {
    int start_index = i * residual_3->cols;
    int end_index = start_index + residual_3->cols;
    float *layer = get_row(residual_3->data, start_index, end_index);
    float *normalised_row = malloc((end_index - start_index) * sizeof(float));
    layer_normalize(layer, normalised_row, ln2_gain, ln2_bias, residual_3->cols);

    for (int j = 0; j < (end_index - start_index); j++)
    {
      normalised_rows_for_feed_forward_output[start_index + j] = normalised_row[j];
    }
    free(normalised_row);
    free(layer);
  }
  for (int i = 0; i < norm1->capacity; i++)
  {
    output->data[i] = normalised_rows_for_feed_forward_output[i];
  }
  free(normalised_rows_for_feed_forward_output);
  matrix_free(residual_1);
  matrix_free(norm1);
  matrix_free(feed_forward_result);
  matrix_free(residual_3);
  matrix_free(self_attention_output);
  free(normalised_rows_for_self_attention_output);
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
// Projecting means multiply really? wtf calling it saling would have made more sense
Matrix *multi_head_attention(Matrix *input, Matrix *W_Q, Matrix *W_K, Matrix *W_V, Matrix *W_0, int num_of_heads)
{
  Matrix *Q = matrix_alloc(input->rows, W_Q->cols);
  Matrix *K = matrix_alloc(input->rows, W_K->cols);
  Matrix *V = matrix_alloc(input->rows, W_V->cols);
  matmul(input, W_Q, Q);
  matmul(input, W_K, K);
  matmul(input, W_V, V);
  int head_dim = Q->cols / num_of_heads;
  Matrix **Q_heads = malloc(num_of_heads * sizeof(Matrix));
  Matrix **K_heads = malloc(num_of_heads * sizeof(Matrix));
  Matrix **V_heads = malloc(num_of_heads * sizeof(Matrix));
  for (int h = 0; h < num_of_heads; h++)
  {
    Q_heads[h] = matrix_alloc(Q->rows, head_dim);
    K_heads[h] = matrix_alloc(K->rows, head_dim);
    V_heads[h] = matrix_alloc(V->rows, head_dim);
  }
  for (int i = 0; i < num_of_heads; i++)
  {
    int start_index = i * head_dim;
    int end_index = start_index + head_dim;
    slice_array(Q->data, start_index, end_index, Q->rows, Q->cols, Q_heads[i]->data);
    slice_array(K->data, start_index, end_index, K->rows, K->cols, K_heads[i]->data);
    slice_array(V->data, start_index, end_index, V->rows, V->cols, V_heads[i]->data);
  }
  Matrix *concat_result = matrix_alloc(input->rows, W_V->cols);
  Matrix **head_outputs = malloc(num_of_heads * sizeof(Matrix));

  int index = 0;
  for (int i = 0; i < num_of_heads; i++)
  {
    head_outputs[i] = attention(Q_heads[i], K_heads[i], V_heads[i]);
  }
  for (int row = 0; row < input->rows; row++)
  {
    for (int h = 0; h < num_of_heads; h++)
    {
      for (int col = 0; col < head_dim; col++)
      {
        int concat_col = h * head_dim + col;
        concat_result->data[row * concat_result->cols + concat_col] = head_outputs[h]->data[row * head_dim + col];
      }
    }
  }
  for (int i = 0; i < num_of_heads; i++)
  {
    matrix_free(head_outputs[i]);
  }
  Matrix *final_result = matrix_alloc(input->rows, W_0->cols);
  matmul(concat_result, W_0, final_result);
  for (int i = 0; i < num_of_heads; i++)
  {
    matrix_free(Q_heads[i]);
    matrix_free(K_heads[i]);
    matrix_free(V_heads[i]);
  }
  free(Q_heads);
  free(K_heads);
  free(V_heads);
  matrix_free(Q);
  matrix_free(K);
  matrix_free(V);
  free(head_outputs);
  matrix_free(concat_result);
  return final_result;
}
void encoder_layer(Matrix *input, Matrix *W_Q, Matrix *W_K, Matrix *W_V, Matrix *W_0, int num_of_heads, Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2, float *ln1_gain, float *ln1_bias, float *ln2_gain, float *ln2_bias, Matrix *output)
{
  Matrix *attenion_output = multi_head_attention(input, W_Q, W_K, W_V, W_0, num_of_heads);
  Matrix *residual_connection = matrix_alloc(input->rows, input->cols);
  matrix_add(input, attenion_output, residual_connection);
  float *normalized_rows = malloc(input->capacity * sizeof(float));
  for (int i = 0; i < residual_connection->rows; i++)
  {
    int start_index = i * residual_connection->cols;
    int end_index = start_index + residual_connection->cols;

    float *layer = get_row(residual_connection->data, start_index, end_index);
    float *normalized_row = malloc((end_index - start_index) * sizeof(float));
    layer_normalize(layer, normalized_row, ln1_gain, ln1_bias, residual_connection->cols);
    // It was confusing its weird to think float as arrays lol we are just copying out floats here as normalized_row is a float we still have to walk through all elements and assign to normalized
    for (int j = 0; j < (end_index - start_index); j++)
    {
      normalized_rows[start_index + j] = normalized_row[j];
    }
    free(normalized_row);
    free(layer);
  }
  Matrix *matrix = matrix_alloc(input->rows, input->cols);
  for (int i = 0; i < matrix->capacity; i++)
  {
    matrix->data[i] = normalized_rows[i];
  }
  Matrix *feed_forward_result = matrix_alloc(input->rows, input->cols);
  feed_forward(matrix, W1, b1, W2, b2, feed_forward_result);
  Matrix *residual_connection_2 = matrix_alloc(input->rows, input->cols);
  matrix_add(matrix, feed_forward_result, residual_connection_2);
  float *normalized_rows_2 = malloc(input->capacity * sizeof(float));
  for (int i = 0; i < residual_connection_2->rows; i++)
  {
    int start_index = i * residual_connection->cols;
    int end_index = start_index + residual_connection->cols;
    float *layer = get_row(residual_connection_2->data, start_index, end_index);
    float *normalized_row = malloc((end_index - start_index) * sizeof(float));
    layer_normalize(layer, normalized_row, ln2_gain, ln2_bias, residual_connection_2->cols);
    for (int j = 0; j < (end_index - start_index); j++)
    {
      normalized_rows_2[start_index + j] = normalized_row[j];
    }
    free(layer);
    free(normalized_row);
  }
  for (int i = 0; i < input->capacity; i++)
  {
    output->data[i] = normalized_rows_2[i];
  }
  matrix_free(residual_connection);
  matrix_free(residual_connection_2);
  matrix_free(feed_forward_result);
  matrix_free(attenion_output);
  matrix_free(matrix);
  free(normalized_rows);
  free(normalized_rows_2);
}
void encoder(
    Matrix *input,           // Input embeddings (seq_len × d_model)
    Matrix **W_Q_layers,     // Array of W_Q matrices, one per layer
    Matrix **W_K_layers,     // Array of W_K matrices, one per layer
    Matrix **W_V_layers,     // Array of W_V matrices, one per layer
    Matrix **W_O_layers,     // Array of W_O matrices, one per layer
    Matrix **W1_layers,      // Array of FFN W1 matrices, one per layer
    Matrix **b1_layers,      // Array of FFN b1 biases, one per layer
    Matrix **W2_layers,      // Array of FFN W2 matrices, one per layer
    Matrix **b2_layers,      // Array of FFN b2 biases, one per layer
    float **ln1_gain_layers, // Array of layer norm 1 gain arrays, one per layer
    float **ln1_bias_layers, // Array of layer norm 1 bias arrays, one per layer
    float **ln2_gain_layers, // Array of layer norm 2 gain arrays, one per layer
    float **ln2_bias_layers, // Array of layer norm 2 bias arrays, one per layer
    int num_layers,          // Number of encoder layers (typically 6)
    int num_heads,           // Number of attention heads
    Matrix *output           // Output matrix (same size as input)
)
{
  Matrix *current = input;
  for (int i = 0; i < num_layers; i++)
  {

    Matrix *next = matrix_alloc(input->rows, input->cols);
    encoder_layer(current, W_Q_layers[i], W_K_layers[i], W_V_layers[i], W_O_layers[i], num_heads, W1_layers[i], b1_layers[i], W2_layers[i], b2_layers[i], ln1_gain_layers[i], ln1_bias_layers[i], ln2_gain_layers[i], ln2_bias_layers[i], next);
    if (i > 0)
    {
      matrix_free(current);
    }
    current = next;
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

Matrix *multi_head_cross_attention(
    Matrix *decoder_input,  // Decoder's current state
    Matrix *encoder_output, // Encoder's output (stays same for all decoder layers)
    Matrix *W_Q,            // Projects decoder to queries
    Matrix *W_K,            // Projects encoder to keys
    Matrix *W_V,            // Projects encoder to values
    Matrix *W_O,            // Output projection after concatenating heads
    int num_heads           // Number of attention heads
)
{
  Matrix *Q = matrix_alloc(decoder_input->rows, W_Q->cols);
  Matrix *K = matrix_alloc(encoder_output->rows, W_K->cols);
  Matrix *V = matrix_alloc(encoder_output->rows, W_V->cols);
  matmul(decoder_input, W_Q, Q);
  matmul(encoder_output, W_K, K);
  matmul(encoder_output, W_V, V);
  int head_dim = Q->cols / num_heads;
  Matrix **Q_heads = malloc(num_heads * sizeof(Matrix));
  Matrix **K_heads = malloc(num_heads * sizeof(Matrix));
  Matrix **V_heads = malloc(num_heads * sizeof(Matrix));
  for (int h = 0; h < num_heads; h++)
  {
    Q_heads[h] = matrix_alloc(Q->rows, head_dim);
    K_heads[h] = matrix_alloc(K->rows, head_dim);
    V_heads[h] = matrix_alloc(V->rows, head_dim);
  }
  for (int i = 0; i < num_heads; i++)
  {
    int start_index = i * head_dim;
    int end_index = start_index + head_dim;
    slice_array(Q->data, start_index, end_index, Q->rows, Q->cols, Q_heads[i]->data);
    slice_array(K->data, start_index, end_index, K->rows, K->cols, K_heads[i]->data);
    slice_array(V->data, start_index, end_index, V->rows, V->cols, V_heads[i]->data);
  }
  Matrix *concat_result = matrix_alloc(decoder_input->rows, W_V->cols);
  Matrix **head_outputs = malloc(num_heads * sizeof(Matrix));

  int index = 0;
  for (int i = 0; i < num_heads; i++)
  {
    head_outputs[i] = attention(Q_heads[i], K_heads[i], V_heads[i]);
  }
  for (int row = 0; row < decoder_input->rows; row++)
  {
    for (int h = 0; h < num_heads; h++)
    {
      for (int col = 0; col < head_dim; col++)
      {
        int concat_col = h * head_dim + col;
        concat_result->data[row * concat_result->cols + concat_col] = head_outputs[h]->data[row * head_dim + col];
      }
    }
  }
  for (int i = 0; i < num_heads; i++)
  {
    matrix_free(head_outputs[i]);
  }
  Matrix *final_result = matrix_alloc(decoder_input->rows, W_O->cols);
  matmul(concat_result, W_O, final_result);
  for (int i = 0; i < num_heads; i++)
  {
    matrix_free(Q_heads[i]);
    matrix_free(K_heads[i]);
    matrix_free(V_heads[i]);
  }
  free(Q_heads);
  free(K_heads);
  free(V_heads);
  matrix_free(Q);
  matrix_free(K);
  matrix_free(V);
  free(head_outputs);
  matrix_free(concat_result);
  return final_result;
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
