#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "math_ops.h"
#include "transformer.h"
#include <time.h>
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
Matrix *masked_attention(Matrix *Q, Matrix *K, Matrix *V)
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
  matrix_free(scores);
  return result;
}

void write_data_to_matrix(float *data, Matrix *output)
{
  for (int i = 0; i < output->rows; i++)
  {
    for (int j = 0; j < output->cols; j++)
    {
      output->data[i * output->cols + j] = data[i * output->cols + j];
    }
  }
}
Matrix *multi_head_masked_attention(Matrix *input, Matrix *W_Q, Matrix *W_K, Matrix *W_V, Matrix *W_0, int num_of_heads)
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
    head_outputs[i] = masked_attention(Q_heads[i], K_heads[i], V_heads[i]);
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

void decoder_layer(
    Matrix *decoder_input,
    Matrix *encoder_output,
    Matrix *W_Q_self, Matrix *W_K_self, Matrix *W_V_self, Matrix *W_O_self,
    Matrix *W_Q_cross, Matrix *W_K_cross, Matrix *W_V_cross, Matrix *W_O_cross,
    int num_heads,
    Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2,
    float *ln1_gain, float *ln1_bias, float *ln2_gain, float *ln2_bias, float *ln3_gain, float *ln3_bias, Matrix *output)
{
  Matrix *self_attention_output = multi_head_masked_attention(decoder_input, W_Q_self, W_K_self, W_V_self, W_O_self, num_heads);
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
  Matrix *cross_attention_output = multi_head_cross_attention(norm1, encoder_output, W_Q_cross, W_K_cross, W_V_cross, W_O_cross, num_heads);
  Matrix *residual_2 = matrix_alloc(decoder_input->rows, decoder_input->cols);

  matrix_add(cross_attention_output, norm1, residual_2);
  float *normalised_rows_for_cross_attention_output = malloc(residual_2->capacity * sizeof(float));

  for (int i = 0; i < self_attention_output->rows; i++)
  {
    int start_index = i * residual_2->cols;
    int end_index = start_index + residual_2->cols;
    float *layer = get_row(residual_2->data, start_index, end_index);
    float *normalised_row = malloc((end_index - start_index) * sizeof(float));
    layer_normalize(layer, normalised_row, ln2_gain, ln2_bias, residual_2->cols);

    for (int j = 0; j < (end_index - start_index); j++)
    {
      normalised_rows_for_cross_attention_output[start_index + j] = normalised_row[j];
    }
    free(normalised_row);
    free(layer);
  }
  Matrix *norm2 = matrix_alloc(decoder_input->rows, decoder_input->cols);
  for (int i = 0; i < norm1->capacity; i++)
  {
    norm2->data[i] = normalised_rows_for_cross_attention_output[i];
  }

  Matrix *feed_forward_result = matrix_alloc(decoder_input->rows, decoder_input->cols);

  feed_forward(norm2, W1, b1, W2, b2, feed_forward_result);
  Matrix *residual_3 = matrix_alloc(decoder_input->rows, decoder_input->cols);
  matrix_add(feed_forward_result, norm2, residual_3);
  float *normalised_rows_for_feed_forward_output = malloc(residual_3->capacity * sizeof(float));
  for (int i = 0; i < self_attention_output->rows; i++)
  {
    int start_index = i * residual_3->cols;
    int end_index = start_index + residual_3->cols;
    float *layer = get_row(residual_3->data, start_index, end_index);
    float *normalised_row = malloc((end_index - start_index) * sizeof(float));
    layer_normalize(layer, normalised_row, ln3_gain, ln3_bias, residual_3->cols);

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
  matrix_free(residual_2);
  matrix_free(norm2);
  matrix_free(feed_forward_result);
  matrix_free(residual_3);
  matrix_free(self_attention_output);
  matrix_free(cross_attention_output);
  free(normalised_rows_for_self_attention_output);
  free(normalised_rows_for_cross_attention_output);
}
void decoder(
    Matrix *decoder_input,  // Target sequence embeddings (with positional encoding)
    Matrix *encoder_output, // Output from the encoder stack

    // Self-attention weights (masked) for each layer
    Matrix **W_Q_self_layers,
    Matrix **W_K_self_layers,
    Matrix **W_V_self_layers,
    Matrix **W_O_self_layers,

    // Cross-attention weights for each layer
    Matrix **W_Q_cross_layers,
    Matrix **W_K_cross_layers,
    Matrix **W_V_cross_layers,
    Matrix **W_O_cross_layers,

    // Feed-forward weights for each layer
    Matrix **W1_layers,
    Matrix **b1_layers,
    Matrix **W2_layers,
    Matrix **b2_layers,

    // Layer norm parameters (3 per layer now!)
    float **ln1_gain_layers, // After masked self-attention
    float **ln1_bias_layers,
    float **ln2_gain_layers, // After cross-attention
    float **ln2_bias_layers,
    float **ln3_gain_layers, // After feed-forward
    float **ln3_bias_layers,

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
                  encoder_output,
                  W_Q_self_layers[i], W_K_self_layers[i], W_V_self_layers[i], W_O_self_layers[i], W_Q_cross_layers[i], W_K_cross_layers[i], W_V_cross_layers[i], W_O_cross_layers[i], num_heads, W1_layers[i], b1_layers[i], W2_layers[i], b2_layers[i], ln1_gain_layers[i], ln1_bias_layers[i], ln2_gain_layers[i], ln2_bias_layers[i], ln3_gain_layers[i], ln3_bias_layers[i], next_output);

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
    int start_index = row * embedding_table->cols;
    for (int col = 0; col < embedding_table->cols; col++)
    {
      matrix->data[start_index + col] = embedding_table->data[token_id * embedding_table->cols + col];
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
