#pragma once
#include "matrix_types.hpp"
#include <string>

namespace matmul {

// Main API
void strassen_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Modular interface (optional): good for flexible, runtime selection
class IMatMul {
public:
    virtual void multiply(const Matrix&, const Matrix&, Matrix&) const = 0;
    virtual ~IMatMul() = default;
};
IMatMul* create_matmul_algo(const std::string& algo_name); // Factory
}
