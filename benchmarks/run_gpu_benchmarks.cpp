#include <vector>
#include <iostream>
#include <algorithm> // for std::max
#include <chrono>
#include <cmath>
extern void cuda_naive_multiply(const float*, const float*, float*, int);
extern void cuda_tensorcore_multiply(const float*, const float*, float*, int);
extern void opencl_naive_multiply(const float* A, const float* B, float* C, int N);

// Simple CPU reference
void cpu_naive_multiply(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

float benchmark_cuda_naive(const float* A, const float* B, float* C, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    cuda_naive_multiply(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float benchmark_opencl_naive(const float* A, const float* B, float* C, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    opencl_naive_multiply(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float benchmark_cuda_tensorcore(const float* A, const float* B, float* C, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    cuda_tensorcore_multiply(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

int main() {
    std::vector<int> sizes = {64, 128, 256};
    for (int N : sizes) {
        std::vector<float> A(N*N), B(N*N), C_gpu(N*N), C_cpu(N*N), C_opencl(N*N);
        for (int i = 0; i < N*N; ++i) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Benchmark CPU
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        cpu_naive_multiply(A.data(), B.data(), C_cpu.data(), N);
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        float t_cpu = std::chrono::duration<float, std::milli>(cpu_t1 - cpu_t0).count();
        std::cout << "CPU, Size " << N << ": " << t_cpu << " ms\n";

        // OpenCL naive
        float t_opencl = benchmark_opencl_naive(A.data(), B.data(), C_opencl.data(), N);
        double maxdiff_opencl = 0;
        for (int i = 0; i < N*N; ++i)
            maxdiff_opencl = std::max(maxdiff_opencl, (double)std::abs(C_opencl[i] - C_cpu[i]));
        std::cout << "OpenCL naive, Size " << N << ": " << t_opencl
                  << " ms, Max diff: " << maxdiff_opencl << (maxdiff_opencl<1e-2 ? " PASS" : " FAIL") << "\n";

        // CUDA naive
        float t_naive = benchmark_cuda_naive(A.data(), B.data(), C_gpu.data(), N);
        double maxdiff = 0;
        for (int i = 0; i < N*N; ++i)
            maxdiff = std::max(maxdiff, (double)std::abs(C_gpu[i] - C_cpu[i]));
        std::cout << "CUDA naive, Size " << N << ": " << t_naive
                  << " ms, Max diff: " << maxdiff << (maxdiff<1e-2 ? " PASS" : " FAIL") << "\n";

        // CUDA TensorCore: only for sizes multiple of 16
        if (N % 16 == 0) {
            float t_tensorcore = benchmark_cuda_tensorcore(A.data(), B.data(), C_gpu.data(), N);
            double maxdiff_tensor = 0;
            for (int i = 0; i < N*N; ++i)
                maxdiff_tensor = std::max(maxdiff_tensor, (double)std::abs(C_gpu[i] - C_cpu[i]));
            std::cout << "CUDA tensorcore, Size " << N << ": " << t_tensorcore
                      << " ms, Max diff: " << maxdiff_tensor << (maxdiff_tensor<1e-1 ? " PASS" : " FAIL") << "\n";
        } else {
            std::cout << "CUDA tensorcore not run for N=" << N << " (requires N multiple of 16)\n";
        }
    }
    return 0;
}
