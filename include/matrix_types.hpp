#pragma once
#include <vector>
#include <cstddef>

// Matrix data structure
struct Matrix {
    std::vector<float> data;
    size_t rows, cols;

    Matrix(size_t r, size_t c)
        : data(r * c, 0.0f), rows(r), cols(c) {}

    // Element access (row, col)
    inline float& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    inline const float& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
};

// Declare matrix helper function signatures
void matrix_add(const Matrix&, const Matrix&, Matrix&);
void matrix_sub(const Matrix&, const Matrix&, Matrix&);
bool matrix_equal(const Matrix&, const Matrix&, float eps=1e-5f);
