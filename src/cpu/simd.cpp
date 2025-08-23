#include "matmul.hpp"
#include <immintrin.h> // AVX intrinsics

namespace matmul {

void simd_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t M = A.rows, N = B.cols, K = A.cols;
    // Assume row-major layout
    // For AVX, we process 8 floats at a time
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                // Load 8 elements from A and B
                __m256 a_vec = _mm256_loadu_ps(&A(i, k));
                __m256 b_vec;
                for (int x = 0; x < 8; ++x) {
                    // B(k+x, j)
                    ((float*)&b_vec)[x] = B(k + x, j);
                }
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, b_vec));
            }
            // Horizontal sum
            float sum_arr[8];
            _mm256_storeu_ps(sum_arr, sum_vec);
            float sum = 0.0f;
            for (int x = 0; x < 8; ++x) sum += sum_arr[x];

            // Remainder loop (not SIMD)
            for (; k < K; ++k)
                sum += A(i, k) * B(k, j);

            C(i, j) = sum;
        }
    }
}

}
