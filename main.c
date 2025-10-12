#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "math_ops.h"
#include "transformer.h"

// Helper to create random matrix
Matrix *create_random_matrix(int rows, int cols)
{
  Matrix *m = matrix_alloc(rows, cols);
  double lower = -0.01;
  double upper = 0.01;
  for (int i = 0; i < m->capacity; i++)
  {
    m->data[i] = lower + (rand() / (double)RAND_MAX) * (upper - lower);
  }
  return m;
}

// Helper to create random array (for biases and layer norm)
float *create_random_array(int size)
{
  float *arr = malloc(size * sizeof(float));
  for (int i = 0; i < size; i++)
  {
    arr[i] = (rand() / (double)RAND_MAX) * 0.02 - 0.01;
  }
  return arr;
}

// Helper to create layer norm parameters (gain=1, bias=0)
float *create_ln_gain(int size)
{
  float *arr = malloc(size * sizeof(float));
  for (int i = 0; i < size; i++)
  {
    arr[i] = 1.0f;
  }
  return arr;
}

float *create_ln_bias(int size)
{
  float *arr = malloc(size * sizeof(float));
  for (int i = 0; i < size; i++)
  {
    arr[i] = 0.0f;
  }
  return arr;
}

int main()
{
  srand(time(NULL));

  int d_model = 64;     // was 128
  int num_heads = 2;    // was 4
  int num_layers = 1;   // was 2
  int d_ff = 256;       // was 512
  int context_len = 10; // Only use last 10 tokens

  // Load text and create vocabulary
  int text_len;
  char *text = load_text_file("shakespeare.txt", &text_len);
  Vocabulary *vocab = vocab_alloc(text);
  int vocab_size = vocab->vocab_size;

  printf("Vocab size: %d\n", vocab_size);
  printf("Text length: %d\n", text_len);

  // Create embedding tables
  Matrix *encoder_embedding = create_embedding_table(vocab_size, d_model);
  Matrix *decoder_embedding = create_embedding_table(vocab_size, d_model);

  // Allocate arrays for layer weights
  Matrix **enc_W_Q = malloc(num_layers * sizeof(Matrix *));
  Matrix **enc_W_K = malloc(num_layers * sizeof(Matrix *));
  Matrix **enc_W_V = malloc(num_layers * sizeof(Matrix *));
  Matrix **enc_W_O = malloc(num_layers * sizeof(Matrix *));
  Matrix **enc_W1 = malloc(num_layers * sizeof(Matrix *));
  Matrix **enc_b1 = malloc(num_layers * sizeof(Matrix *));
  Matrix **enc_W2 = malloc(num_layers * sizeof(Matrix *));
  Matrix **enc_b2 = malloc(num_layers * sizeof(Matrix *));
  float **enc_ln1_gain = malloc(num_layers * sizeof(float *));
  float **enc_ln1_bias = malloc(num_layers * sizeof(float *));
  float **enc_ln2_gain = malloc(num_layers * sizeof(float *));
  float **enc_ln2_bias = malloc(num_layers * sizeof(float *));

  // Initialize encoder weights for each layer
  for (int i = 0; i < num_layers; i++)
  {
    enc_W_Q[i] = create_random_matrix(d_model, d_model);
    enc_W_K[i] = create_random_matrix(d_model, d_model);
    enc_W_V[i] = create_random_matrix(d_model, d_model);
    enc_W_O[i] = create_random_matrix(d_model, d_model);
    enc_W1[i] = create_random_matrix(d_model, d_ff);
    enc_b1[i] = create_random_matrix(1, d_ff);
    enc_W2[i] = create_random_matrix(d_ff, d_model);
    enc_b2[i] = create_random_matrix(1, d_model);
    enc_ln1_gain[i] = create_ln_gain(d_model);
    enc_ln1_bias[i] = create_ln_bias(d_model);
    enc_ln2_gain[i] = create_ln_gain(d_model);
    enc_ln2_bias[i] = create_ln_bias(d_model);
  }

  // Allocate decoder weights (same structure)
  Matrix **dec_W_Q_self = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W_K_self = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W_V_self = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W_O_self = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W_Q_cross = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W_K_cross = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W_V_cross = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W_O_cross = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W1 = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_b1 = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_W2 = malloc(num_layers * sizeof(Matrix *));
  Matrix **dec_b2 = malloc(num_layers * sizeof(Matrix *));
  float **dec_ln1_gain = malloc(num_layers * sizeof(float *));
  float **dec_ln1_bias = malloc(num_layers * sizeof(float *));
  float **dec_ln2_gain = malloc(num_layers * sizeof(float *));
  float **dec_ln2_bias = malloc(num_layers * sizeof(float *));
  float **dec_ln3_gain = malloc(num_layers * sizeof(float *));
  float **dec_ln3_bias = malloc(num_layers * sizeof(float *));

  // Initialize decoder weights
  for (int i = 0; i < num_layers; i++)
  {
    dec_W_Q_self[i] = create_random_matrix(d_model, d_model);
    dec_W_K_self[i] = create_random_matrix(d_model, d_model);
    dec_W_V_self[i] = create_random_matrix(d_model, d_model);
    dec_W_O_self[i] = create_random_matrix(d_model, d_model);
    dec_W_Q_cross[i] = create_random_matrix(d_model, d_model);
    dec_W_K_cross[i] = create_random_matrix(d_model, d_model);
    dec_W_V_cross[i] = create_random_matrix(d_model, d_model);
    dec_W_O_cross[i] = create_random_matrix(d_model, d_model);
    dec_W1[i] = create_random_matrix(d_model, d_ff);
    dec_b1[i] = create_random_matrix(1, d_ff);
    dec_W2[i] = create_random_matrix(d_ff, d_model);
    dec_b2[i] = create_random_matrix(1, d_model);
    dec_ln1_gain[i] = create_ln_gain(d_model);
    dec_ln1_bias[i] = create_ln_bias(d_model);
    dec_ln2_gain[i] = create_ln_gain(d_model);
    dec_ln2_bias[i] = create_ln_bias(d_model);
    dec_ln3_gain[i] = create_ln_gain(d_model);
    dec_ln3_bias[i] = create_ln_bias(d_model);
  }

  // Output projection layer
  Matrix *W_out = create_random_matrix(d_model, vocab_size);
  Matrix *b_out = create_random_matrix(1, vocab_size);

  printf("All weights initialized!\n");

  // Simple generation test
  char *prompt = "KING:";
  int prompt_len;
  int *prompt_tokens = tokenize(prompt, vocab, &prompt_len);

  printf("Generating from prompt: '%s'\n", prompt);
  printf("Output: %s", prompt);
  fflush(stdout);

  // Embed and encode the prompt
  Matrix *encoder_input = embed_tokens(prompt_tokens, prompt_len, encoder_embedding);
  add_positional_encoding(encoder_input, prompt_len, d_model);

  Matrix *encoder_output = matrix_alloc(encoder_input->rows, encoder_input->cols);
  encoder(encoder_input, enc_W_Q, enc_W_K, enc_W_V, enc_W_O,
          enc_W1, enc_b1, enc_W2, enc_b2,
          enc_ln1_gain, enc_ln1_bias, enc_ln2_gain, enc_ln2_bias,
          num_layers, num_heads, encoder_output);

  // Generate tokens one by one with sliding window
  int *all_generated = malloc(1000 * sizeof(int)); // Store all for printing
  all_generated[0] = prompt_tokens[prompt_len - 1];
  int total_generated = 1;

  for (int step = 0; step < 50; step++)
  {
    // Use only last context_len tokens
    int use_len = (total_generated < context_len) ? total_generated : context_len;
    int *context = &all_generated[total_generated - use_len];
    printf("\n[Step %d: use_len=%d, total=%d]\n", step, use_len, total_generated);
    // Embed current context
    Matrix *decoder_input = embed_tokens(context, use_len, decoder_embedding);
    printf("decoder_input: %dx%d\n", decoder_input->rows, decoder_input->cols);
    add_positional_encoding(decoder_input, use_len, d_model);
    printf("After pos encoding\n");

    // Run decoder
    Matrix *decoder_output = matrix_alloc(decoder_input->rows, decoder_input->cols);
    printf("About to call decoder...\n");
    decoder(decoder_input, encoder_output,
            dec_W_Q_self, dec_W_K_self, dec_W_V_self, dec_W_O_self,
            dec_W_Q_cross, dec_W_K_cross, dec_W_V_cross, dec_W_O_cross,
            dec_W1, dec_b1, dec_W2, dec_b2,
            dec_ln1_gain, dec_ln1_bias, dec_ln2_gain, dec_ln2_bias,
            dec_ln3_gain, dec_ln3_bias,
            num_layers, num_heads, decoder_output);
    printf("Decoder done\n");
    // Project to vocabulary
    Matrix *logits = output_projection(decoder_output, W_out, b_out);

    // Sample next token
    int next_token = sample_token(logits, vocab_size);

    // Print the character
    printf("%c", vocab->chars[next_token]);
    fflush(stdout);

    // Add to generated sequence
    all_generated[total_generated++] = next_token;

    matrix_free(decoder_input);
    matrix_free(decoder_output);
    matrix_free(logits);
  }

  printf("\n\nGeneration complete!\n");

  // Cleanup
  free(prompt_tokens);
  free(all_generated);
  matrix_free(encoder_input);
  matrix_free(encoder_output);
  free(text);
  vocab_free(vocab);

  return 0;
}
