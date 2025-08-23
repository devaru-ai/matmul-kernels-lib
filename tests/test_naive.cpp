#include "matmul.hpp"
#include "matrix_types.hpp"
#include <iostream>
#include <random>

Matrix random_matrix(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::mt19937 rng(345);
    std::uniform_real_distribution<float> dist(-15.f, 15.f);
    for (auto& x : M.data) x = dist(rng);
    return M;
}

int main() {
    size_t N = 32;
    Matrix A = random_matrix(N, N);
    Matrix B = random_matrix(N, N);
    Matrix C1(N, N), C2(N, N);

    // Run your naive implementation
    matmul::naive_multiply(A, B, C1);

    // Naive again, as reference
    matmul::naive_multiply(A, B, C2);

    if (matrix_equal(C1, C2)) {
        std::cout << "Naive multiplication PASSED.\n";
    } else {
        std::cout << "Naive multiplication FAILED.\n";
        return 1;
    }
    return 0;
}
