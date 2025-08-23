#include "matmul.hpp"
#include "matrix_types.hpp"
#include <iostream>
#include <random>

Matrix random_matrix(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::mt19937 rng(567);
    std::uniform_real_distribution<float> dist(-20.f, 20.f);
    for (auto& x : M.data) x = dist(rng);
    return M;
}

int main() {
    size_t N = 64;
    Matrix A = random_matrix(N, N);
    Matrix B = random_matrix(N, N);
    Matrix C1(N, N), C2(N, N);

    matmul::blocked_multiply(A, B, C1);
    matmul::naive_multiply(A, B, C2);

    if (matrix_equal(C1, C2)) {
        std::cout << "Blocked multiplication PASSED.\n";
    } else {
        std::cout << "Blocked multiplication FAILED.\n";
        return 1;
    }
    return 0;
}
