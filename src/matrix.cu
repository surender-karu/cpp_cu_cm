#include "cuda_common.cuh"

__global__ void matrixAdd(float *A, float *B, float *C, int n) {
  int tid = threadIdx.x;
  int blkIdx = blockIdx.x;

  C[tid * n + blkIdx] = A[tid * n + blkIdx] + B[tid * n + blkIdx];

  // for(int i = 0; i < n; i++) {
  //   for(int j = 0; j < n; j++) {
  //     C[i * n + j] = A[i * n + j] + B[i * n + j];
  //   }
  // }
}

void matrixAdd(float *A, float *B, float *C, int n, int grid_size,
               int block_size) {
  float *d_A, *d_B, *d_C;

  size_t matrix_size = n * n * sizeof(float);

  cudaMalloc(reinterpret_cast<void **>(&d_A), matrix_size);
  cudaMalloc(reinterpret_cast<void **>(&d_B), matrix_size);
  cudaMalloc(reinterpret_cast<void **>(&d_C), matrix_size);

  cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

  matrixAdd<<<n, n>>>(d_A, d_B, d_C, n);

  cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}