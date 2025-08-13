#include "matrix_types.hpp"
#include <cmath>

void matrix_add(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t N = A.rows * A.cols;
    for (size_t i = 0; i < N; ++i)
        C.data[i] = A.data[i] + B.data[i];
}
void matrix_sub(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t N = A.rows * A.cols;
    for (size_t i = 0; i < N; ++i)
        C.data[i] = A.data[i] - B.data[i];
}
bool matrix_equal(const Matrix& A, const Matrix& B, float eps) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (size_t i = 0; i < A.data.size(); ++i)
        if (std::abs(A.data[i] - B.data[i]) > eps)
            return false;
    return true;
}
