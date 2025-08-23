#include "matmul.hpp"

namespace matmul {

// Abstract interface for all algorithms
class IMatMul {
public:
    virtual ~IMatMul() {}
    virtual void multiply(const Matrix& A, const Matrix& B, Matrix& C) const = 0;
};

// CPU algos
class CpuNaiveMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        cpu_naive_multiply(A.data.data(), B.data.data(), C.data.data(), (int)A.rows);
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

// CUDA naive
class CudaNaiveMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        cuda_naive_multiply(A.data.data(), B.data.data(), C.data.data(), (int)A.rows);
    }
};

// CUDA tensorcore (with compatibility check: requires N % 16 == 0)
class CudaTensorCoreMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        int N = (int)A.rows;
        if (N % 16 != 0 || A.cols % 16 != 0 || B.rows % 16 != 0 || B.cols % 16 != 0) {
            throw std::runtime_error("Tensor Core requires all matrix sizes to be multiples of 16.");
        }
        // Optionally: check/convert to FP16 if required by your kernel
        cuda_tensorcore_multiply(A.data.data(), B.data.data(), C.data.data(), N);
    }
};

// OpenCL naive
class OpenCLNaiveMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        opencl_naive_multiply(A.data.data(), B.data.data(), C.data.data(), (int)A.rows);
    }
};


IMatMul* create_matmul_algo(const std::string& name) {
    if (name == "cpu_naive")
        return new CpuNaiveMul();
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
    if (name == "cuda_naive")
        return new CudaNaiveMul();
    if (name == "cuda_tensorcore")
        return new CudaTensorCoreMul();
    if (name == "opencl_naive")
        return new OpenCLNaiveMul();
    return nullptr;
}

}
