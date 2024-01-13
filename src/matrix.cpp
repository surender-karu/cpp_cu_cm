#include "matrix.h"

namespace cpp_fin {

static long num_steps = 10000000;
double step;

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

void mat_mull(double **A, double **B, double **C, const int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i][j] = (double)rand() / (double)RAND_MAX;
      B[i][j] = (double)rand() / (double)RAND_MAX;
      C[i][j] = 0.0;
    }
  }

  struct timeval start, end;

  gettimeofday(&start, NULL);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        C[i][j] = A[i][k] * B[k][j];
      }
    }
  }

  gettimeofday(&end, NULL);
  printf("Time to multiply %i * %i Matrix = %0.6fs\n", n, n,
         tdiff(&start, &end));
}

double calculate_pi_omp() {
  double x, pi, sum = 0.0;
  step = 1.0 / (double)num_steps;
#pragma omp parallel for private(x) reduction(+ : sum)
  for (int i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }
  pi = step * sum;
  return pi;
}

void set_matrix_value(float *m, const int n, const float value){
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      m[i * n + j] = value;
    }
  }
}
} // namespace cpp_fin