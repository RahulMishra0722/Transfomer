#include "math_ops.h"
#include "transformer.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ── Hyperparameters ──────────────────────────────────────────────────────────
#define D_MODEL 256
#define NUM_HEADS 8
#define NUM_LAYERS 6
#define CONTEXT_LEN 256 // tokens per training window
#define LEARNING_RATE 3e-4f
#define NUM_EPOCHS 100
#define STEPS_PER_EPOCH 3000
#define PRINT_EVERY 300   // print loss every N steps
#define GEN_LEN 300        // characters to generate per epoch sample
#define CHECKPOINT_EVERY 5 // save weights every N epochs
#define CHECKPOINT_PATH "model.bin"
// ─────────────────────────────────────────────────────────────────────────────

// Forward-only loss (no backprop) — used for sanity checks
static float forward_loss(TransformerModel *model, int *input_tokens,
                          int *target_tokens, int seq_len) {
  Matrix *embeddings =
      embed_tokens(input_tokens, seq_len, model->embedding_table);
  add_positional_encoding(embeddings, seq_len, model->d_model);

  Matrix *decoder_output = matrix_alloc(seq_len, model->d_model);
  decoder(embeddings, model->W_Q_layers, model->W_K_layers, model->W_V_layers,
          model->W_O_layers, model->W1_layers, model->b1_layers,
          model->W2_layers, model->b2_layers, model->ln1_gain_layers,
          model->ln1_bias_layers, model->ln2_gain_layers,
          model->ln2_bias_layers, model->num_layers, model->num_heads,
          decoder_output);

  Matrix *logits =
      output_projection(decoder_output, model->W_out, model->b_out);
  float loss =
      cross_entropy_loss(logits, target_tokens, seq_len, model->vocab_size);

  matrix_free(embeddings);
  matrix_free(decoder_output);
  matrix_free(logits);
  return loss;
}

// ── Train mode ───────────────────────────────────────────────────────────────
static void run_training(void) {
  int text_len;
  char *text = load_text_file("shakespeare.txt", &text_len);
  if (!text) {
    fprintf(stderr, "Failed to load shakespeare.txt\n");
    exit(1);
  }

  // Sanitize text to remove non-printable characters and fix smart quotes
  int write_idx = 0;
  for (int read_idx = 0; read_idx < text_len; read_idx++) {
    unsigned char c = (unsigned char)text[read_idx];
    
    // Check for right/left single quotation mark (E2 80 99 or E2 80 98)
    if (read_idx + 2 < text_len && c == 0xE2 && (unsigned char)text[read_idx+1] == 0x80 && 
       ((unsigned char)text[read_idx+2] == 0x99 || (unsigned char)text[read_idx+2] == 0x98)) {
      text[write_idx++] = '\'';
      read_idx += 2; // skip the next two bytes
    }
    // Check for double quotation marks (E2 80 9C or E2 80 9D)
    else if (read_idx + 2 < text_len && c == 0xE2 && (unsigned char)text[read_idx+1] == 0x80 && 
            ((unsigned char)text[read_idx+2] == 0x9C || (unsigned char)text[read_idx+2] == 0x9D)) {
      text[write_idx++] = '"';
      read_idx += 2;
    }
    // Check for em-dash (E2 80 94) and replace with standard hyphen
    else if (read_idx + 2 < text_len && c == 0xE2 && (unsigned char)text[read_idx+1] == 0x80 && (unsigned char)text[read_idx+2] == 0x94) {
      text[write_idx++] = '-';
      read_idx += 2;
    }
    // Keep printable ASCII, tabs, and newlines
    else if (isprint(c) || c == '\n' || c == '\t') {
      text[write_idx++] = text[read_idx];
    }
    // Any other unprintable byte is simply skipped (dropped from the array)
  }
  text[write_idx] = '\0';
  text_len = write_idx; // shrink the logical length of the text

  Vocabulary *vocab = vocab_alloc(text);
  int vocab_size = vocab->vocab_size;
  printf("Vocab size : %d\n", vocab_size);
  printf("Text length: %d chars\n\n", text_len);

  int token_len;
  int *tokens = tokenize(text, vocab, &token_len);

  TransformerModel *model =
      init_model(vocab_size, D_MODEL, NUM_LAYERS, NUM_HEADS);
  printf("Model initialised  (d_model=%d, layers=%d, heads=%d, ctx=%d)\n\n",
         D_MODEL, NUM_LAYERS, NUM_HEADS, CONTEXT_LEN);
  fflush(stdout);

  int num_windows = token_len - CONTEXT_LEN - 1;

  AdamState *adam_state = init_adam_state(model);
  int global_step = 1; // Adam bias correction uses 1-indexed t

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    float epoch_loss = 0.0f;

    float current_lr = LEARNING_RATE;
    if (epoch >= 5) {
      current_lr = LEARNING_RATE * (1.0f / (1.0f + 0.03f * (epoch - 5)));
    }

    for (int step = 0; step < STEPS_PER_EPOCH; step++) {
      int start = rand() % num_windows;
      int *input_tokens = &tokens[start];
      int *target_tokens = &tokens[start + 1];

      float loss = train_step_with_gradients(
          model, vocab, input_tokens, target_tokens, CONTEXT_LEN, current_lr,
          adam_state, global_step);

      epoch_loss += loss;
      global_step++;

      if ((step + 1) % PRINT_EVERY == 0) {
        printf("Epoch %2d | Step %3d/%d | avg_loss = %.4f\n", epoch + 1,
               step + 1, STEPS_PER_EPOCH, epoch_loss / (step + 1));
        fflush(stdout);
      }
    }

    float avg_loss = epoch_loss / STEPS_PER_EPOCH;
    printf("\n=== Epoch %d done  avg_loss=%.4f ===\n", epoch + 1, avg_loss);

    // Sanity-check with a forward-only window
    int start = rand() % num_windows;
    float check =
        forward_loss(model, &tokens[start], &tokens[start + 1], CONTEXT_LEN);
    printf("Sanity check loss: %.4f\n", check);

    // Generate a sample
    char *sample = generate(model, vocab, "KING:", GEN_LEN, CONTEXT_LEN);
    printf("Sample:\n%s\n\n", sample);
    free(sample);
    fflush(stdout);

    // Save checkpoint periodically
    if ((epoch + 1) % CHECKPOINT_EVERY == 0)
      save_model(model, vocab, CHECKPOINT_PATH);
  }

  // Ensure we save at the very end regardless of epoch count
  save_model(model, vocab, CHECKPOINT_PATH);
  printf("\nTraining complete! Weights saved to '%s'\n", CHECKPOINT_PATH);
  printf("Run with --chat to start an interactive session.\n\n");

  free_adam_state(adam_state, model->num_layers);
  free(tokens);
  free(text);
  vocab_free(vocab);
}

// ── Chat / inference mode ────────────────────────────────────────────────────
static void run_chat(void) {
  Vocabulary *vocab = NULL;
  TransformerModel *model = load_model("model_epoch75_cpu.bin", &vocab);

  printf("\n=== Interactive mode ===\n");
  printf("Type a prompt and press Enter. Ctrl-C to quit.\n\n");

  char prompt[1024];
  while (1) {
    printf("You: ");
    fflush(stdout);
    if (!fgets(prompt, sizeof(prompt), stdin))
      break;

    // strip trailing newline
    size_t plen = strlen(prompt);
    if (plen > 0 && prompt[plen - 1] == '\n')
      prompt[plen - 1] = '\0';
    if (strlen(prompt) == 0)
      continue;

    printf("\nModel:\n");
    fflush(stdout);
    char *response = generate(model, vocab, prompt, GEN_LEN, CONTEXT_LEN);
    printf("\n\n");
    free(response);
  }

  vocab_free(vocab);
}

// ── Entry point ──────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  srand((unsigned)time(NULL));

  if (argc > 1 && strcmp(argv[1], "--chat") == 0) {
    run_chat();
  } else {
    run_training();
  }

  return 0;
}
