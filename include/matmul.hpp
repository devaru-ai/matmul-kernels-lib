#pragma once
#include "matrix_types.hpp"
#include <string>

namespace matmul {

// Naive matrix multiplication
void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Blocked (tile) matrix multiplication
void blocked_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Cache-friendly matrix multiplication
void cache_friendly_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// SIMD-optimized matrix multiplication
void simd_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Multi-threaded CPU multiplication
void multithreaded_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Strassen matrix multiplication
void strassen_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// GPU/CUDA/OpenCL kernel wrappers 
void cuda_naive_multiply(const float* A, const float* B, float* C, int N);
void cuda_tensorcore_multiply(const float* A, const float* B, float* C, int N);
void opencl_naive_multiply(const float* A, const float* B, float* C, int N);

// Modular interface (optional): good for flexible, runtime selection
class IMatMul {
public:
    virtual void multiply(const Matrix&, const Matrix&, Matrix&) const = 0;
    virtual ~IMatMul() = default;
};

class CudaNaiveMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

class CudaTensorCoreMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

class OpenCLNaiveMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

// Example supported algo_name values:
// "naive", "blocked", "cache_friendly", "simd", "multithreaded", "strassen",
// "cuda_naive", "cuda_tensorcore", "opencl_naive"
IMatMul* create_matmul_algo(const std::string& algo_name); // Factory
}
