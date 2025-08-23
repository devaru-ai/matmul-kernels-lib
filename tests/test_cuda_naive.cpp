#include <vector>
#include <iostream>
#include <cmath>
extern void cuda_naive_multiply(const float*, const float*, float*, int);

int main() {
    int N = 64;
    std::vector<float> A(N*N), B(N*N), C(N*N), Cref(N*N);

    for (int i = 0; i < N*N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Run GPU
    cuda_naive_multiply(A.data(), B.data(), C.data(), N);

    // Reference CPU
    for (int i=0; i<N; ++i)
        for (int j=0; j<N; ++j) {
            float sum=0.0f;
            for (int k=0;k<N;++k)
                sum += A[i*N+k]*B[k*N+j];
            Cref[i*N+j]=sum;
        }
    double maxdiff = 0;
    for (int i=0; i<N*N; ++i)
        maxdiff = std::max(maxdiff, (double)std::abs(C[i] - Cref[i]));
    std::cout << "Max diff: " << maxdiff << std::endl;
    std::cout << ((maxdiff < 1e-2) ? "PASS" : "FAIL") << std::endl;
    return 0;
}
