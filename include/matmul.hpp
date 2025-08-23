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

// Modular interface (optional): good for flexible, runtime selection
class IMatMul {
public:
    virtual void multiply(const Matrix&, const Matrix&, Matrix&) const = 0;
    virtual ~IMatMul() = default;
};
IMatMul* create_matmul_algo(const std::string& algo_name); // Factory
}
