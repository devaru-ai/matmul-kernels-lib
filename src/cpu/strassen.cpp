#include "matmul.hpp"
#include "matrix_types.hpp"
#include <algorithm>

namespace matmul {

// --- Optionally use the shared helpers from src/common/matrix.cpp ---
using ::matrix_add;
using ::matrix_sub;

static void naive(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t N = A.rows;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < N; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

static Matrix get_submatrix(const Matrix& M, size_t row_off, size_t col_off, size_t n) {
    Matrix sub(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            sub(i, j) = M(i + row_off, j + col_off);
    return sub;
}

static void set_submatrix(Matrix& M, const Matrix& sub, size_t row_off, size_t col_off) {
    for (size_t i = 0; i < sub.rows; ++i)
        for (size_t j = 0; j < sub.cols; ++j)
            M(i + row_off, j + col_off) = sub(i, j);
}

void strassen_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t N = A.rows;
    if (N <= 64) { // Tunable: switch to naive for small submatrices
        naive(A, B, C);
        return;
    }
    size_t n2 = N / 2;

    // Split matrices into quadrants
    Matrix A11 = get_submatrix(A, 0,    0,    n2);
    Matrix A12 = get_submatrix(A, 0,    n2,   n2);
    Matrix A21 = get_submatrix(A, n2,   0,    n2);
    Matrix A22 = get_submatrix(A, n2,   n2,   n2);

    Matrix B11 = get_submatrix(B, 0,    0,    n2);
    Matrix B12 = get_submatrix(B, 0,    n2,   n2);
    Matrix B21 = get_submatrix(B, n2,   0,    n2);
    Matrix B22 = get_submatrix(B, n2,   n2,   n2);

    // Allocate temporary matrices
    Matrix M1(n2, n2), M2(n2, n2), M3(n2, n2), M4(n2, n2),
           M5(n2, n2), M6(n2, n2), M7(n2, n2),
           T1(n2, n2), T2(n2, n2);

    matrix_add(A11, A22, T1); matrix_add(B11, B22, T2); strassen_multiply(T1, T2, M1);
    matrix_add(A21, A22, T1);                          strassen_multiply(T1, B11, M2);
    matrix_sub(B12, B22, T2);                          strassen_multiply(A11, T2, M3);
    matrix_sub(B21, B11, T2);                          strassen_multiply(A22, T2, M4);
    matrix_add(A11, A12, T1);                          strassen_multiply(T1, B22, M5);
    matrix_sub(A21, A11, T1); matrix_add(B11, B12, T2); strassen_multiply(T1, T2, M6);
    matrix_sub(A12, A22, T1); matrix_add(B21, B22, T2); strassen_multiply(T1, T2, M7);

    Matrix C11(n2, n2), C12(n2, n2), C21(n2, n2), C22(n2, n2);
    matrix_add(M1, M4, T1); matrix_sub(T1, M5, T2); matrix_add(T2, M7, C11);
    matrix_add(M3, M5, C12);
    matrix_add(M2, M4, C21);
    matrix_add(M1, M3, T1); matrix_sub(T1, M2, T2); matrix_add(T2, M6, C22);

    // Combine quadrants into C
    set_submatrix(C, C11, 0,    0);
    set_submatrix(C, C12, 0,    n2);
    set_submatrix(C, C21, n2,   0);
    set_submatrix(C, C22, n2,   n2);
}
}
