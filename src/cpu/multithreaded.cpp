#include "matmul.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace matmul {

void multithreaded_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

} // namespace matmul
