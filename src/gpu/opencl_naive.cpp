#include <CL/cl.hpp>
#include <vector>
#include <cmath>

const char* kernelSource = R"CLC(
__kernel void matmul(__global const float* A,
                    __global const float* B,
                    __global float* C,
                    const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}
)CLC";

void opencl_naive_multiply(const float* h_A, const float* h_B, float* h_C, int N) {
    // setup OpenCL platform and device
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();
    cl::Context context({device});
    cl::Program program(context, kernelSource);
    program.build({device});

    // Proper buffer constructions
    cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N*N, (void*)h_A);
    cl::Buffer d_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N*N, (void*)h_B);
    cl::Buffer d_C(context, CL_MEM_WRITE_ONLY, sizeof(float)*N*N, nullptr);

    cl::Kernel kernel(program, "matmul");
    kernel.setArg(0, d_A);
    kernel.setArg(1, d_B);
    kernel.setArg(2, d_C);
    kernel.setArg(3, N);
    cl::CommandQueue queue(context, device);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, N));
    queue.enqueueReadBuffer(d_C, CL_TRUE, 0, sizeof(float)*N*N, h_C);
}
