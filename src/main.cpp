#include "main.h"
#include "lambda_exp.h"
#include "matrix.h"
#include "sp.h"

using namespace cpp_fin;

int main(int, char **) {
  float *A, *B, *C;

  const int n = 256;

  int grid_size = 1, block_size = 1;

  A = (float *)malloc(n * n * sizeof(float));
  B = (float *)malloc(n * n * sizeof(float));
  C = (float *)malloc(n * n * sizeof(float));

  set_matrix_value(A, n, 1.0f);
  set_matrix_value(B, n, 2.0f);

  matrixAdd(A, B, C, n, grid_size, block_size);

  std::cout << "Matrix add calculated in GPU" << std::endl;

  float maxError = 0.0f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      maxError = fmax(maxError, fabs(C[i * n + j] - 3.0f));
    }
  }
  std::cout << "Max error: " << maxError << std::endl;

  free(A);
  free(B);
  free(C);
  return 0;
}