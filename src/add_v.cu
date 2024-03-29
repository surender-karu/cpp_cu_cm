#include "cuda_common.cuh"

// function to add the elements of two arrays
__global__ void add(int n, const float *x, float *y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

void add() {
  int N = 1 << 20; // 1M elements
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  float *h_x, *h_y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&h_x, N * sizeof(float));
  cudaMallocManaged(&h_y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add<<<numBlocks, blockSize>>>(N, h_x, h_y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(h_y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(h_x);
  cudaFree(h_y);
}