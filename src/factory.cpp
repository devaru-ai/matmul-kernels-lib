#include "matmul.hpp"

namespace matmul {

class StrassenMul : public IMatMul {
public:
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) const override {
        strassen_multiply(A, B, C);
    }
};

IMatMul* create_matmul_algo(const std::string& name) {
    if (name == "strassen")
        return new StrassenMul();
    // Add: if (name == "naive") return new NaiveMul(); etc.
    return nullptr;
}

}
