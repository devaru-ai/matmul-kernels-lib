#include "matmul.hpp"

namespace matmul {

class NaiveMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        naive_multiply(A, B, C);
    }
};

class BlockedMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        blocked_multiply(A, B, C);
    }
};

class CacheFriendlyMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        cache_friendly_multiply(A, B, C);
    }
};

class SimdMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        simd_multiply(A, B, C);
    }
};

class MultithreadedMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        multithreaded_multiply(A, B, C);
    }
};

class StrassenMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        strassen_multiply(A, B, C);
    }
};

IMatMul* create_matmul_algo(const std::string& name) {
    if (name == "naive")
        return new NaiveMul();
    if (name == "blocked")
        return new BlockedMul();
    if (name == "cache_friendly")
        return new CacheFriendlyMul();
    if (name == "simd")
        return new SimdMul();
    if (name == "multithreaded")
        return new MultithreadedMul();
    if (name == "strassen")
        return new StrassenMul();
    return nullptr;
}

}
