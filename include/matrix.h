#ifndef CPP_FIN_MATRIX_H
#define CPP_FIN_MATRIX_H

#include "common.h"

namespace cpp_fin {
void mat_mull(double *A, double *B, double *C, const int n);
double calculate_pi_omp();

void set_matrix_value(float *m, const int n, const float value);
}

#endif