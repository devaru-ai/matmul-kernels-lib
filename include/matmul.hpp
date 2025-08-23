#pragma once
#include "matrix_types.hpp"
#include <string>

namespace matmul {

// Function declarations (flat API for direct calls)
void cpu_naive_multiply(const Matrix& A, const Matrix& B, Matrix& C);
void blocked_multiply(const Matrix& A, const Matrix& B, Matrix& C);
void cache_friendly_multiply(const Matrix& A, const Matrix& B, Matrix& C);
void simd_multiply(const Matrix& A, const Matrix& B, Matrix& C);
void multithreaded_multiply(const Matrix& A, const Matrix& B, Matrix& C);
void strassen_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// GPU/CUDA/OpenCL kernel wrappers (pointer interface for device APIs)
void cuda_naive_multiply(const float* A, const float* B, float* C, int N);
void cuda_tensorcore_multiply(const float* A, const float* B, float* C, int N);
void opencl_naive_multiply(const float* A, const float* B, float* C, int N);

// Modular interface (object-oriented API for runtime selection)
class IMatMul {
public:
    virtual void multiply(const Matrix&, const Matrix&, Matrix&) const = 0;
    virtual ~IMatMul() = default;
};

// Concrete algorithm classes (consistent with factory.cpp)
class CpuNaiveMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

class BlockedMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

class CacheFriendlyMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

class SimdMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

class MultithreadedMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
};

class StrassenMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override;
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

// Factory function for selecting the algorithm at runtime
IMatMul* create_matmul_algo(const std::string& algo_name);

// Example supported algo_name values:
// "cpu_naive", "blocked", "cache_friendly", "simd", "multithreaded", "strassen",
// "cuda_naive", "cuda_tensorcore", "opencl_naive"

} // namespace matmul
