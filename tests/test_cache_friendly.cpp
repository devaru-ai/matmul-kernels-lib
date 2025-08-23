#include "matmul.hpp"
#include "matrix_types.hpp"
#include <iostream>
#include <random>
#include <cmath>

Matrix random_matrix(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-20.f, 20.f);
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
    const float EPS = 1e-2; // Make sure threshold is loose enough!
    size_t N = 64;
    Matrix A = random_matrix(N, N);
    Matrix B = random_matrix(N, N);
    Matrix C1(N, N), C2(N, N);

    matmul::cache_friendly_multiply(A, B, C1);
    naive_multiply(A, B, C2);

    int mismatch_count = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float diff = std::abs(C1(i,j) - C2(i,j));
            if (diff > EPS) {
                std::cout << "Diff at (" << i << "," << j << "): cache_friendly=" << C1(i,j)
                          << ", naive=" << C2(i,j) << ", diff=" << diff << "\n";
                ++mismatch_count;
            }
        }
    }

    if (mismatch_count != 0) {
        std::cout << "Cache-friendly multiplication FAILED: " << mismatch_count << " mismatches found.\n";
        return 1;
    } else {
        std::cout << "Cache-friendly multiplication PASSED.\n";
        return 0;
    }
}
