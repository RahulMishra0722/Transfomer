#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "math_ops.h"

#define MASK_VALUE -INFINITY
#define SQRT_2_OVER_PI 0.7978845608

typedef struct
{
  char *chars;         // Array of unique characters: "abcd...xyz !,."
  int vocab_size;      // How many unique characters (probably 50-80)
  int char_to_id[256]; // Lookup table: char_to_id['a'] = 0, etc.
                       // Use 256 because that's all ASCII values
} Vocabulary;

Matrix *matrix_alloc(int rows, int cols)
{
  Matrix *m = malloc(sizeof(Matrix));
  if (m == NULL)
  {
    printf("Failed to allocate memory");
    exit(1);
  }
  m->cols = cols;
  m->rows = rows;
  m->capacity = rows * cols;
  m->data = malloc(m->capacity * sizeof(float));
  if (m->data == NULL)
  {
    printf("Failed to allocate memory");
    exit(1);
  }
  for (int i = 0; i < m->capacity; i++)
  {
    m->data[i] = 0;
  }

  return m;
}
void matrix_free(Matrix *m)
{
  free(m->data);
  free(m);
}
float matrix_get(Matrix *m, int row, int col)
{
  return m->data[row * m->cols + col];
}
void matrix_set(Matrix *m, int row, int col, float value)
{
  m->data[row * m->cols + col] = value;
}
void matrix_add(Matrix *a, Matrix *b, Matrix *result)
{
  for (int i = 0; i < a->capacity; i++)
  {
    result->data[i] = a->data[i] + b->data[i];
  }
}
float dot_product(Matrix *a, Matrix *b, int row, int col)
{
  float sum = 0;
  for (int i = 0; i < a->cols; i++)
  {
    float fromA = a->data[row * a->cols + i];
    float fromB = b->data[i * b->cols + col];
    sum += fromA * fromB;
  }

  return sum;
}
void matmul(Matrix *a, Matrix *b, Matrix *result)
{
  for (int row = 0; row < a->rows; row++)
  {
    for (int col = 0; col < b->cols; col++)
    {
      result->data[row * result->cols + col] = dot_product(a, b, row, col);
    }
  }
}
void matrix_subtract(Matrix *a, Matrix *b, Matrix *result)
{
  for (int i = 0; i < a->capacity; i++)
  {
    result->data[i] = a->data[i] - b->data[i];
  }
}
void matrix_copy(Matrix *src, Matrix *dest)
{
  for (int i = 0; i < src->capacity; i++)
  {
    dest->data[i] = src->data[i];
  }
}
void matrix_hadamard(Matrix *a, Matrix *b, Matrix *result)
{
  for (int i = 0; i < result->capacity; i++)
  {
    result->data[i] = a->data[i] * b->data[i];
  }
};
void matrix_sum_rows(Matrix *m, Matrix *result)
{
  for (int i = 0; i < m->rows; i++)
  {
    result->data[i] = 0.0;
  }
  for (int row = 0; row < m->rows; row++)
  {
    for (int col = 0; col < m->cols; col++)
    {
      result->data[col] += m->data[row * m->cols + col];
    }
  }
};
void GELU_derivative(Matrix *input, Matrix *output)
{
  float π = 3.14159265359;
  for (int i = 0; i < input->capacity; i++)
  {
    float x = input->data[i];
    float z = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
    float tanh_z = tanh(z);
    float sech_squared = 1 - tanh_z * tanh_z;
    float dz_dx = SQRT_2_OVER_PI * (1 + 0.134145 * x * x);
    float derivative = 0.5 * (1 + tanh_z) + 0.5 * x * sech_squared * dz_dx;
    output->data[i] = derivative;
  }
};
void softmax_derivative(float *softmax_output, float *grad_output, float *grad_input, int length)
{
  for (int i = 0; i < length; i++)
  {
    grad_input[i] = 0.0;
    for (int j = 0; j < length; j++)
    {
      if (i == j)
      {
        grad_input[i] += softmax_output[i] * (1 - softmax_output[i]) * grad_output[j];
      }
      else
      {
        grad_input[i] += -softmax_output[i] * softmax_output[j] * grad_output[j];
      }
    }
  }
};
// Given softmax output and gradient flowing back, compute gradient to input
// Computes derivative of GELU at each point
// You already have GELU, now you need d(GELU)/dx
void matrix_scale(Matrix *m, float scalarValue)
{
  for (int i = 0; i < m->capacity; i++)
  {
    m->data[i] = m->data[i] * scalarValue;
  }
}
void matrix_transpose(Matrix *m, Matrix *result)
{
  for (int row = 0; row < result->rows; row++)
  {
    for (int col = 0; col < result->cols; col++)
    {
      result->data[row * result->cols + col] = m->data[col * m->cols + row];
    }
  }
}

void softmax(float *input, float *output, int length)
{
  float max = input[0];
  float sum = 0;
  float *exponentials = malloc(length * sizeof(float));
  if (exponentials == NULL)
  {
    printf("Failed to allocate memort for Exponentials in Softmax");
    exit(1);
  }
  for (int i = 0; i < length; i++)
  {
    float curValue = input[i];
    if (curValue > max)
    {
      max = curValue;
    }
  }

  for (int i = 0; i < length; i++)
  {
    exponentials[i] = exp(input[i] - max);
    sum += exponentials[i];
  }
  for (int i = 0; i < length; i++)
  {
    output[i] = exponentials[i] / sum;
  }
  free(exponentials);
}

void mat_softmax_rows(Matrix *m)
{
  for (int row = 0; row < m->rows; row++)
  {
    float *start_row = &m->data[row * m->cols];
    softmax(start_row, start_row, m->cols);
  }
}
float *get_row(float *data, int start_index, int end_index)
{
  if (start_index >= end_index)
  {
    printf("Not Allowed Start Index Must Be Less Then End Index");
    exit(1);
  }
  int length = end_index - start_index;
  float *result = malloc(length * sizeof(float));
  for (int i = 0; i < length; i++)
  {
    result[i] = data[start_index + i];
  }
  return result;
}
float *slice_array(float *data, int start_index, int end_index, int number_of_rows, int number_of_cols, float *sliced_array)
{
  if (start_index >= end_index)
  {
    printf("How do you expect me to do this? lol");
    exit(1);
  }
  for (int row = 0; row < number_of_rows; row++)
  {
    int index = 0;
    for (int col = 0; col < end_index - start_index; col++)
    {
      sliced_array[index] = data[row * number_of_cols + start_index + col];
      index++;
    }
  }
  return sliced_array;
}
void GELU(Matrix *matrix)
{
  for (int i = 0; i < matrix->capacity; i++)
  {
    float x = matrix->data[i];
    matrix->data[i] = x * 0.5 * (1 + tanh(SQRT_2_OVER_PI * (x + 0.044715 * (x * x * x))));
  }
}
void add_bias(Matrix *matrix, Matrix *bais, Matrix *result)
{
  for (int row = 0; row < matrix->rows; row++)
  {
    for (int col = 0; col < matrix->cols; col++)
    {
      result->data[row * result->cols + col] = matrix->data[row * matrix->cols + col] + bais->data[col];
    }
  }
}
void layer_normalize(float *input, float *output, float *gain, float *bias, int length)
{
  float mean = 0;
  float varience = 0;
  float epsilon = 1e-5;

  for (int i = 0; i < length; i++)
  {
    mean += input[i];
  }
  mean /= length;
  for (int i = 0; i < length; i++)
  {
    varience += (input[i] - mean) * (input[i] - mean);
  }
  varience /= length;

  for (int i = 0; i < length; i++)
  {
    float norm = (input[i] - mean) / sqrt(varience + epsilon);
    output[i] = gain[i] * norm + bias[i];
  }
}
