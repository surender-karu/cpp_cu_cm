#ifndef CPP_FIN_MAIN_H
#define CPP_FIN_MAIN_H

#include "lambda_exp.h"
#include "matrix.h"
#include "sp.h"

void add();
void queryDevice();
void matrixAdd(float *A, float *B, float *C, int n, int grid_size, int block_size);

#endif

