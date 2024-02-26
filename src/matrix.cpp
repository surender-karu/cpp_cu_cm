#include "matrix.h"

namespace cpp_fin {

static long num_steps = 10000000;
double step;

float tdiff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) +
           1e-6 * (end->tv_usec - start->tv_usec);
}

// create a function to time a function that accepts multiple arguments
void timeFunction(void (*function)(float *, float *, float *, const int),
                  float *A, float *B, float *C, const int n) {
    struct timeval start, end;

    gettimeofday(&start, NULL);
    function(A, B, C, n);
    gettimeofday(&end, NULL);

    printf("Time to execute function = %0.6fs\n", tdiff(&start, &end));
}

void timeFunctionDevice(void (*function)(float *, float *, float *, int, int),
                  float *A, float *B, float *C, int n, int nThreads) {
    struct timeval start, end;

    gettimeofday(&start, NULL);
    function(A, B, C, n, nThreads);
    gettimeofday(&end, NULL);

    printf("Time to execute function = %0.6fs\n", tdiff(&start, &end));
}

void matMultiply(double *A, double *B, double *C, const int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int index = i * n + j;
            A[index] = (double)rand() / (double)RAND_MAX;
            B[index] = (double)rand() / (double)RAND_MAX;
            C[index] = 0.0;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void matMultiply(float *A, float *B, float *C, const int n) {
    for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

double ompCalculatePi() {
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

void setMatrixConstValue(float *m, const int n, const float value) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            m[i * n + j] = value;
        }
    }
}

void cuAddMatrix() {
    float *h_A, *h_B, *h_C, *h_D;

    const int n = 2048;
    const float valueToCheck = n;

    int nThreads = 64;

    h_A = new float[n * n];
    h_B = new float[n * n];
    h_C = new float[n * n];
    h_D = new float[n * n];

    setMatrixConstValue(h_A, n, 1.0f);
    setMatrixConstValue(h_B, n, 1.0f);

    timeFunction(matMultiply, h_A, h_B, h_C, n);

    float maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            maxError = fmax(maxError, fabs(h_C[i * n + j] - valueToCheck));
        }
    }
    std::cout << "Max error: " << maxError << std::endl;

    timeFunctionDevice(matrixMultiply, h_A, h_B, h_D, n, nThreads);

    std::cout << "Matrix add calculated in GPU" << std::endl;

    maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            maxError = fmax(maxError, fabs(h_C[i * n + j] - valueToCheck));
        }
    }
    std::cout << "Max error: " << maxError << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;
}

}  // namespace cpp_fin