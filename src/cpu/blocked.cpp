#include "matmul.hpp"
#include <algorithm>

constexpr size_t BLOCK_SIZE = 32; // Tune for your CPU and matrix sizes

namespace matmul {

// Blocked (tile-based) matrix multiplication
void blocked_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t M = A.rows, N = B.cols, K = A.cols;

    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE)
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE)
            for (size_t kk = 0; kk < K; kk += BLOCK_SIZE)
                for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); ++i)
                    for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); ++j) {
                        float sum = 0.0f;
                        for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k)
                            sum += A(i, k) * B(k, j);
                        C(i, j) += sum;
                    }
}

}
