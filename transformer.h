#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "math_ops.h" // Include math_ops since we use Matrix
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
typedef struct
{
    char *chars;
    int vocab_size;
    int char_to_id[256];
} Vocabulary;
typedef struct
{
    float *input; // ADD THIS - save original input
    float *normalized;
    float mean;
    float variance;
    int size;
} LayerNormCache;

typedef struct
{
    Matrix *result;
    Matrix *score;
} AttentionOutput;
// Vocabulary functions
char *load_text_file(char *filepath, int *output_length);
Vocabulary *vocab_alloc(char *text);
void vocab_free(Vocabulary *vocabulary);
int *tokenize(char *text, Vocabulary *vocab, int *out_length);
char *detokenize(int *token_ids, int length, Vocabulary *vocab);

// Transformer components
void add_positional_encoding(Matrix *embeddings, int seq_len, int d_model);
void add_bias(Matrix *matrix, Matrix *bias, Matrix *result);
void feed_forward(Matrix *input, Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2, Matrix *output, Matrix *hidden_post_gelu, Matrix *hidden_pre_gelu);
AttentionOutput *masked_attention(Matrix *Q, Matrix *K, Matrix *V);
MultiHeadAttentionCache multi_head_masked_attention(Matrix *input, Matrix *W_Q, Matrix *W_K, Matrix *W_V, Matrix *W_0, int num_of_heads);
Matrix *embed_tokens(int *token_ids, int seq_len, Matrix *embedding_table);

// Encoder and decoder
DecoderLayerCache decoder_layer(
    Matrix *decoder_input,
    Matrix *W_Q_self, Matrix *W_K_self, Matrix *W_V_self, Matrix *W_O_self,
    int num_heads,
    Matrix *W1, Matrix *b1, Matrix *W2, Matrix *b2,
    float *ln1_gain, float *ln1_bias,
    float *ln2_gain, float *ln2_bias,
    Matrix *output);

void decoder(
    Matrix *decoder_input,

    Matrix **W_Q_self_layers,
    Matrix **W_K_self_layers,
    Matrix **W_V_self_layers,
    Matrix **W_O_self_layers,

    Matrix **W1_layers,
    Matrix **b1_layers,
    Matrix **W2_layers,
    Matrix **b2_layers,

    float **ln1_gain_layers, // After masked self-attention
    float **ln1_bias_layers,
    float **ln2_gain_layers, // After feed-forward (renamed from ln3)
    float **ln2_bias_layers,

    int num_layers,
    int num_heads,

    Matrix *output);
void decoder_backward(
    Matrix *d_output,           // gradient from output projection
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
    float **dLn2_bias_layers);

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
    float learning_rate);
update_weights(Matrix *weights, Matrix *gradients, float learning_rate)
{
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
    float *dLn2_bias);
// Returns: the loss value for this training step
Matrix *create_embedding_table(int vocab_size, int d_model);
// In transformer.h

Matrix *output_projection(Matrix *decoder_output, Matrix *W_out, Matrix *b_out);
int sample_token(Matrix *logits, int vocab_size);
#endif // TRANSFORMER_H
