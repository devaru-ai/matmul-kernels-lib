#include "matmul.hpp"

namespace matmul {

// Standard 3-loop naive multiplication
void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t M = A.rows, N = B.cols, K = A.cols;
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < K; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

}
