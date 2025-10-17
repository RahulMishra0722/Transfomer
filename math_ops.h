#ifndef MATH_OPS_H
#define MATH_OPS_H

typedef struct
{
  float *data;
  int rows;
  int cols;
  int capacity;
} Matrix;

#define MASK_VALUE -INFINITY

// Matrix operations
Matrix *matrix_alloc(int rows, int cols);
void matrix_free(Matrix *m);
float matrix_get(Matrix *m, int row, int col);
void matrix_set(Matrix *m, int row, int col, float value);
void matrix_add(Matrix *a, Matrix *b, Matrix *result);
void matmul(Matrix *a, Matrix *b, Matrix *result);
void matrix_scale(Matrix *m, float scalarValue);
void matrix_transpose(Matrix *m, Matrix *result);
void matrix_sum_rows(Matrix *m, Matrix *result);
// Activation and normalization
void GELU(Matrix *matrix);
void softmax(float *input, float *output, int length);
void mat_softmax_rows(Matrix *m);
void softmax_derivative(float *softmax_output, float *grad_output, float *grad_input, int length);
void layer_normalize(float *input, float *output, float *gain, float *bias, int length);

// Utility functions
float *get_row(float *data, int start_index, int end_index);
float *slice_array(float *data, int start_index, int end_index, int number_of_rows, int number_of_cols, float *sliced_array);

#endif // MATH_OPS_H
