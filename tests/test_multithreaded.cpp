#include "matmul.hpp"
#include "matrix_types.hpp"
#include <iostream>
#include <random>

Matrix random_matrix(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
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
    size_t N = 128; // Large enough to test threading
    Matrix A = random_matrix(N, N);
    Matrix B = random_matrix(N, N);
    Matrix C1(N, N), C2(N, N);

    matmul::multithreaded_multiply(A, B, C1);
    naive_multiply(A, B, C2);

    if (matrix_equal(C1, C2)) {
        std::cout << "Multithreaded multiplication PASSED.\n";
    } else {
        std::cout << "Multithreaded multiplication FAILED.\n";
        return 1;
    }
    return 0;
}
