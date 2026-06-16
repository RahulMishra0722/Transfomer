# Backpropagation

## What is backpropagation and what should I care?

Well this is where during training the learning happens. What a feedforward network does is it makes a prediction and during inference or before training it's random or we have to make it not random. Well how can we do that? We use a neat little trick that makes the weights / the model "learn". What is that trick?

## Building from first principles

Suppose I wanted to make a machine that given a person's height and age calculated and predicted their weight.

So let's put some numbers down:

- Actual weight = 70 kg
- Age = 25 years
- Height = 170 cm

Now comes the fun part.

## The forward pass

Suppose this is the function, let's call it `forward`. It won't do anything special other than make a prediction and it will take some arguments.

```c
float forward(int age, int height, float w1, float w2) {
    return (age * w1) + (height * w2);
}
```

It does nothing other than multiply and return the output.

So let's run with some example - say the initial weights were w1 = 0.1 and w2 = 0.1.

```c
float prediction = (25 * 0.1) + (170 * 0.1);  // = 19.5 kg
```

The model will return 19.5 but the actual weight was 70. Way off!

## Here's where the backprop begins

The output of the function gets fed into another function. Let's for now call it backward.

My first idea was something like this:

```c
void backward(float predicted_weight, float real_weight) {
    // normally we would compute loss use fancy formulas but for basic
    // understanding as to what it means to backpropagate this seems ok
    if (predicted_weight < real_weight) {
        // increase the weight
    } else if (predicted_weight == real_weight) {
        // we won't get exact, we stop when we get really close
        // stop
    } else {
        // decrease the weight
    }
}
```

And this runs in a loop - the weight will get adjusted by some amount every time based on if what we predicted was higher than what we expected or lower.

## What I was missing

But there's something crucial here - how much do I adjust? And which weight gets adjusted how much?

Turns out what actually happens is this:

```c
void backward(int age, int height, float *w1, float *w2,
    float actual_weight, float learning_rate){
    float prediction = (age * (*w1) + height * (*w2));
    float error = prediction - actual_weight;
    float gradient_w1 = error * w1;
    float gradient_w2 = error * w2;
    *w1 = *w1 - (learning_rate * gradient_w1);
    *w2 = *w2 - (learning_rate * gradient_w2);
}
```

What is gradient?
its a vector of slope in all direction:
Supposed you are standing on a hill and the gradient of f(x, y) tells you:

- How steep is it if you go east/west (∂f/∂x)
- How steep is it if you go north/south (∂f/∂y)
  Package all of these togather and we get [∂f/∂x, ∂f/∂y]

How to calculate it?
Just take the derivative with respect to each other seperately.
Example: f(x, y) = x² + 3xy + y²
From before [∂f/∂x, ∂f/∂y]
[∂f/∂x] = 2x + 3y (treat y as a constant)
[∂f/∂y] = 3x + 2y (treat x as a constant)
So ∇f = [2x + 3y, 3x + 2y]

## Now lets see how it plays out in Our code and MLPs in general

```c
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
```

### Just look at whats happening start with:

- Look at the last opperation of the forward pass decoder_layer function what is the last operation that we do in there? like other then data writes for return and free allocated memory and all the other basic stuff what do we at the very end? look we `Normalize`

````c
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
```
  Look at what we do in as backwadspass pass first opperation
  ```c
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
  ```
do you see how we undo it? i was follwing the paper attension is all you need and i could never understand why we did any of it like why so many steps until i understood something really important and you see that soon here it goes and this was the break through for me
In Forward Pass
```c
  cache.residual_2 = matrix_alloc(decoder_input->rows, decoder_input->cols);
  matrix_add(cache.ffn_output, cache.norm1, cache.residual_2);
```
In backwards pass
```c
  Matrix *d_attention_result = matrix_alloc(cache->residual_1->rows, cache->residual_1->cols);
  Matrix *d_decoder_input_from_residual = matrix_alloc(cache->residual_1->rows, cache->residual_1->cols);
  for (int i = 0; i < cache->residual_1->capacity; i++)
  {
    d_attention_result->data[i] = d_residual_1->data[i];
    d_decoder_input_from_residual->data[i] = d_residual_1->data[i];
  }

```
its confusing right how is it that we added residual_2 once and we are undoing it twice and if you are wondering how is it undoing when are just copying the data from one place to the other? its because adding doesnt effect the slopes/ derivatives of a function its a constant so we dont consider it but its still contributes to some extent and so we copy it for further calculations and now going back to why we did it twice its because lets take an example if say you and your friend were participating in a tournament of game you play and supposed you lose by say 10 points and to fix your mistakes you have to know how much you and your friend contributed to the error would you say its 5 and 5? both contributed equally to the loss? its highlily unlikely that this is the case so knowing how much we contributed exactly is how you can both get better and fix your mistakes based on what your mistake was and how much you contributed to error and thats why we make two seperate copies for both cache.ffn_output cache.norm1 as cache.residual_2 is made up of these two and d_attention_result is the gradient of cache.ffn_output and d_decoder_input_from_residual is the gradient of cache.norm1 and since adding is a constant operation when it comes to derivatives we just copy it over unlike multiplying which is not a constant operation
Back up back up another confusion has been cleared i thought when we say undo in backprop we somehow undo the opperations but no its how sesitive the input is to the weight or how sensitive is error to the final output and it make sense that during addition we just copy it because if undoing meant to undo wouldnt we just subtract? but we dont we just copy it why because derivatives dont get effected by constants

Now lets understand how do we "undo"/ find the senstivity of multiplying in backwards pass `Multiplication`
// In feed_forward:
matmul(input, W1, matrix_multiply_result);
// result = input × W1
input  = [2, 3]  (1×2 matrix, one row)
W1     = [1, 4]     (2×2 matrix)
         [2, 5]
result = [2, 3] * [1, 4] = [2 * 1 + 3 * 2, 2 * 4 + 3 * 5] =  [8, 23]
                  [2, 5]

BEFORE we move on i wanna clear up the notations here
```c
∂Loss/∂input[0] = ∂Loss/∂result[0] × ∂result[0]/∂input[0] +
                  ∂Loss/∂result[1] × ∂result[1]/∂input[0]
```
I used to think we are deviding by loss when i saw the notation but what it actually means is that its a notation it represents if i change result[0] by a tiny amount how does the loss change and if i change the input by a tiny amount how does the result changes its basically tracking all these senstivities "gradients" throughout the function that do the backwards of any of forward pass they start where the feed_forward part left and pick it and compute gradients and see how sensitive they are to change and how does that change effect the loss and based of it increases or decreases the loss we increase or decrease the weights at position[i] or result[i] and thats basically what the backprop does throughout the backward_pass functions.

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
````
