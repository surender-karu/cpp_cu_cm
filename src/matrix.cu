#include "cuda_common.cuh"
#include "matrix.h"

__global__ void matrixAddOptimized(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

__global__ void matrixMultiplyOptimized(float *A, float *B, float *C, int n) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Csub = 0;

    for (int m = 0; m < n / 32; ++m) {
        As[ty][tx] = A[row * n + m * 32 + tx];
        Bs[ty][tx] = B[(m * 32 + ty) * n + col];

        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = Csub;
}

__global__ void matrixMultiply(float *A, float *B, float *C, int n) {
    int tid = threadIdx.x;
    int blkIdx = blockIdx.x;

    C[tid * n + blkIdx] = 0;

    for (int i = 0; i < n; i++) {
        C[tid * n + blkIdx] += A[tid * n + i] * B[i * n + blkIdx];
    }
}

void allocateMemoryInDevice(int n, float *&d_A, float *&d_B, float *&d_C,
                            float *A, float *B) {
    size_t matrix_size = n * n * sizeof(float);

    cudaMalloc(reinterpret_cast<void **>(&d_A), matrix_size);
    cudaMalloc(reinterpret_cast<void **>(&d_B), matrix_size);
    cudaMalloc(reinterpret_cast<void **>(&d_C), matrix_size);

    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
}

void copyToHostAndClean(float *C, float *d_C, int n, float *d_A, float *d_B) {
    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrixAdd(float *A, float *B, float *C, int n, int nThreads) {
    float *d_A, *d_B, *d_C;

    allocateMemoryInDevice(n, d_A, d_B, d_C, A, B);

    dim3 threadsPerBlock(nThreads, nThreads);
    dim3 numBlocks(n / threadsPerBlock.x, n / threadsPerBlock.y);    

    matrixAddOptimized<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    copyToHostAndClean(C, d_C, n, d_A, d_B);
}

void matrixMultiply(float *A, float *B, float *C, int n, int nThreads) {
    float *d_A, *d_B, *d_C;

    allocateMemoryInDevice(n, d_A, d_B, d_C, A, B);

    dim3 threadsPerBlock(nThreads, nThreads);
    dim3 numBlocks(n / threadsPerBlock.x, n / threadsPerBlock.y);    

    matrixMultiplyOptimized<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    copyToHostAndClean(C, d_C, n, d_A, d_B);
}