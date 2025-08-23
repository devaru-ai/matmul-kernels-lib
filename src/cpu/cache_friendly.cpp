#include "matmul.hpp"
#include <vector>
#include <algorithm>

// You can tune block size per architecture
constexpr size_t BLOCK_SIZE = 32; // Change to e.g. 16 or 64 for your CPU/cache

namespace matmul {

// Transpose B so that inner loop accesses are contiguous in memory
static Matrix transpose_matrix(const Matrix& B) {
    Matrix Bt(B.cols, B.rows);
    for (size_t i = 0; i < B.rows; ++i)
        for (size_t j = 0; j < B.cols; ++j)
            Bt(j, i) = B(i, j);
    return Bt;
}

// Cache-friendly (blocked) multiplication
void cache_friendly_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t M = A.rows, N = B.cols, K = A.cols;
    Matrix Bt = transpose_matrix(B); // Faster col access!

    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); ++i) {
                    for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); ++j) {
                        float sum = 0.0f;
                        for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                            sum += A(i, k) * Bt(j, k); // Bt(j, k) == B(k, j)
                        }
                        C(i, j) += sum;
                    }
                }
            }
        }
    }
}

}
