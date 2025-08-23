#include <cuda_runtime.h>
#include <iostream>
float profile_kernel(void (*kernel)(const float*, const float*, float*, int),
    const float* h_A, const float* h_B, float* h_C, int N) {
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel(h_A, h_B, h_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms=0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}
// Example usage: float ms=profile_kernel(cuda_naive_multiply, h_A, h_B, h_C, N);
