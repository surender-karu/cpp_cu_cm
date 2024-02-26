#ifndef CPP_FIN_MATRIX_H
#define CPP_FIN_MATRIX_H

#include "common.h"

namespace cpp_fin {
void matMultiply(double *A, double *B, double *C, const int n);
void matMultiply(float *A, float *B, float *C, const int n);
double ompCalculatePi();
void setMatrixConstValue(float *m, const int n, const float value);

void cuAddMatrix();
}

void matrixAdd(float *A, float *B, float *C, int n, int nThreads);
void matrixMultiply(float *A, float *B, float *C, int n, int nThreads);

void copyToHostAndClean(float *C, float *d_C, int n, float *d_A, float *d_B);

void allocateMemoryInDevice(int n, float *&d_A, float *&d_B, float *&d_C,
                            float *A, float *B);

#endif