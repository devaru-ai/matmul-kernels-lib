#include "matmul.hpp"
#include "matrix_types.hpp"
#include <iostream>
#include <random>

Matrix random_matrix(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    for (auto& x : M.data) x = dist(rng);
    return M;
}

void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t j = 0; j < B.cols; ++j)
            for (size_t k = 0; k < A.cols; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

int main() {
    size_t N = 4;
    Matrix A = random_matrix(N, N);
    Matrix B = random_matrix(N, N);
    Matrix C1(N, N), C2(N, N);

    matmul::strassen_multiply(A, B, C1);
    naive_multiply(A, B, C2);

    if (matrix_equal(C1, C2))
        std::cout << "Strassen test PASSED\n";
    else
        std::cout << "Strassen test FAILED\n";

    return 0;
}
