#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace nvcuda;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm(const half *A, const half *B, float *C, int N) {
    int tile_i = blockIdx.y * WMMA_M;
    int tile_j = blockIdx.x * WMMA_N;
    if (tile_i + WMMA_M <= N && tile_j + WMMA_N <= N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (int k = 0; k < N; k += WMMA_K) {
            const half *tile_a = A + (tile_i*N + k);
            const half *tile_b = B + (k*N + tile_j);
            wmma::load_matrix_sync(a_frag, tile_a, N);
            wmma::load_matrix_sync(b_frag, tile_b, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(C + tile_i*N + tile_j, c_frag, N, wmma::mem_row_major);
    }
}

// In header: extern void cuda_tensorcore_multiply(const float*, const float*, float*, int);
// API wrapper
void to_half(const float* src, half* dst, int size) {
    for (int i = 0; i < size; ++i) dst[i] = __float2half(src[i]);
}

void cuda_tensorcore_multiply(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t size = N * N;
    size_t bytes_A = size * sizeof(half);
    size_t bytes_B = size * sizeof(half);
    size_t bytes_C = size * sizeof(float);

    std::vector<half> h_A_half(size), h_B_half(size);
    to_half(h_A, h_A_half.data(), size);
    to_half(h_B, h_B_half.data(), size);

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, bytes_A); cudaMalloc(&d_B, bytes_B); cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A_half.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half.data(), bytes_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes_C);

    dim3 grid(N / WMMA_M, N / WMMA_N); dim3 block(32, 1);
    wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
