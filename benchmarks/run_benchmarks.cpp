#include "matmul.hpp"
#include "matrix_types.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>

// ---- Memory Usage Helper (Linux, Colab) ----
size_t get_memory_usage_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::string kb_str = line.substr(line.find_last_of('\t'));
            return std::stoul(kb_str);
        }
    }
    return 0;
}

// ---- Random Matrix Generator ----
Matrix random_matrix(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    for (auto& x : M.data) x = dist(rng);
    return M;
}

// ---- Reference Naive Multiply ----
void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t j = 0; j < B.cols; ++j)
            for (size_t k = 0; k < A.cols; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

// ---- Timing Wrapper ----
template<typename Func>
double timed_multiply(Func algo, const Matrix& A, const Matrix& B, Matrix& C) {
    auto start = std::chrono::high_resolution_clock::now();
    algo(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    const float EPS = 1e-2;
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024}; // Add more as needed

    std::cout << std::left << std::setw(10) << "Size"
              << std::setw(18) << "Algorithm"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "Mem(KB)"
              << "Result\n";
    std::cout << "---------------------------------------------------------------\n";

    for (size_t N : sizes) {
        Matrix A = random_matrix(N, N);
        Matrix B = random_matrix(N, N);
        Matrix ref(N, N);

        naive_multiply(A, B, ref);

        // ---- Benchmark Each Algorithm ----

        // Naive
        {
            Matrix C(N, N);
            size_t mem_before = get_memory_usage_kb();
            double ms = timed_multiply(naive_multiply, A, B, C);
            size_t mem_after = get_memory_usage_kb();
            bool ok = matrix_equal(C, ref, EPS);
            std::cout << std::setw(10) << N
                      << std::setw(18) << "naive"
                      << std::setw(12) << ms
                      << std::setw(12) << (mem_after - mem_before)
                      << (ok ? "CORRECT" : "MISMATCH") << "\n";
        }
      
        // Blocked
        {
            Matrix C(N, N);
            size_t mem_before = get_memory_usage_kb();
            double ms = timed_multiply(matmul::blocked_multiply, A, B, C);
            size_t mem_after = get_memory_usage_kb();
            bool ok = matrix_equal(C, ref, EPS);
            std::cout << std::setw(10) << N
                      << std::setw(18) << "blocked"
                      << std::setw(12) << ms
                      << std::setw(12) << (mem_after - mem_before)
                      << (ok ? "CORRECT" : "MISMATCH") << "\n";
        }
      
        // Cache-Friendly
        {
            Matrix C(N, N);
            size_t mem_before = get_memory_usage_kb();
            double ms = timed_multiply(matmul::cache_friendly_multiply, A, B, C);
            size_t mem_after = get_memory_usage_kb();
            bool ok = matrix_equal(C, ref, EPS);
            std::cout << std::setw(10) << N
                      << std::setw(18) << "cache_friendly"
                      << std::setw(12) << ms
                      << std::setw(12) << (mem_after - mem_before)
                      << (ok ? "CORRECT" : "MISMATCH") << "\n";
        }

        // SIMD
        {
            Matrix C(N, N);
            size_t mem_before = get_memory_usage_kb();
            double ms = timed_multiply(matmul::simd_multiply, A, B, C);
            size_t mem_after = get_memory_usage_kb();
            bool ok = matrix_equal(C, ref, EPS);
            std::cout << std::setw(10) << N
                      << std::setw(18) << "simd"
                      << std::setw(12) << ms
                      << std::setw(12) << (mem_after - mem_before)
                      << (ok ? "CORRECT" : "MISMATCH") << "\n";
        }

        // Multithreaded
        {
            Matrix C(N, N);
            size_t mem_before = get_memory_usage_kb();
            double ms = timed_multiply(matmul::multithreaded_multiply, A, B, C);
            size_t mem_after = get_memory_usage_kb();
            bool ok = matrix_equal(C, ref, EPS);
            std::cout << std::setw(10) << N
                      << std::setw(18) << "multithreaded"
                      << std::setw(12) << ms
                      << std::setw(12) << (mem_after - mem_before)
                      << (ok ? "CORRECT" : "MISMATCH") << "\n";
        }

        // Strassen
        {
            Matrix C(N, N);
            size_t mem_before = get_memory_usage_kb();
            double ms = timed_multiply(matmul::strassen_multiply, A, B, C);
            size_t mem_after = get_memory_usage_kb();
            bool ok = matrix_equal(C, ref, EPS);
            std::cout << std::setw(10) << N
                      << std::setw(18) << "strassen"
                      << std::setw(12) << ms
                      << std::setw(12) << (mem_after - mem_before)
                      << (ok ? "CORRECT" : "MISMATCH") << "\n";
        }
    }
    return 0;
}
