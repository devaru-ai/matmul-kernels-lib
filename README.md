# Matrix Multiplication Optimization Library

This project is a modular, high-performance matrix multiplication library that explores multiple algorithms and hardware accelerations for dense and structured matrices.

## Implemented Algorithms

- **Naive Algorithm:** Classic triple-loop implementation. Baseline reference that works for all matrix sizes.
- **Blocked/Tiled Algorithm:** Divides matrices into smaller blocks to reduce cache misses.
- **Cache-Friendly Algorithm:** Optimizes memory access patterns, including transposed access.
- **SIMD-Optimized Algorithm:** Uses CPU vector instructions (e.g., AVX/NEON) to process multiple elements in parallel.
- **Multithreaded Algorithm:** Exploits multicore CPUs with parallelism across threads.
- **GPU-Accelerated Algorithm:** Uses CUDA to leverage thousands of GPU threads for massive speedups.
- **Tensor Core–Optimized Algorithm:** NVIDIA hardware acceleration with FP16/BF16 for peak performance.
- **Strassen’s Algorithm (Optional):** Divide-and-conquer method with $$O(n^{2.807})$$ complexity.



## Supported Matrix Types

- **Dense:** Standard floating-point matrices.
- **Sparse (planned):** Matrices with mostly zero entries.
- **Structured (Toeplitz, etc.) (planned):** Algorithms that exploit matrix structure.



## Hardware Targets

- **CPU:** Single-core and multi-core
- **CPU:** SIMD vectorization
- **GPU:** CUDA, large parallel workloads
- **GPU:** Tensor Cores (NVIDIA Volta and newer)



## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/devaru-ai/matmul-kernels-lib
cd matrix-optimization
```



### 2. CPU Benchmarking

#### Compile all CPU kernel variants:

```bash
g++ run_benchmarks.cpp \
    src/common/matrix.cpp \
    src/cpu/simd.cpp \
    src/cpu/multithreaded.cpp \
    src/cpu/cache_friendly.cpp \
    src/cpu/strassen.cpp \
    -Iinclude \
    -march=native -O3 -ffast-math \
    -fopenmp \
    -o run_benchmarks
```

#### Run CPU benchmarks:

```bash
./run_benchmarks
```



### 3. GPU Benchmarking

#### Compile CUDA and OpenCL kernels:

```bash
nvcc -arch=sm_75 -c src/gpu/cuda_naive.cu -o cuda_naive.o
nvcc -arch=sm_75 -c src/gpu/cuda_tensorcore.cu -o cuda_tensorcore.o
g++ -c src/gpu/opencl_naive.cpp -o opencl_naive.o -I/usr/local/cuda/include
```

#### Link all GPU components (including OpenCL):

```bash
g++ run_gpu_benchmarks.cpp cuda_naive.o cuda_tensorcore.o opencl_naive.o \
    -o run_gpu_benchmarks \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lOpenCL \
    -std=c++11
```

#### Run GPU benchmarks:

```bash
./run_gpu_benchmarks
```



## Benchmarking & Profiling

Benchmarking and profiling results (timing, cache efficiency, and GPU performance comparisons) will be **added soon**. These will include:

- CPU benchmarks across algorithms
- GPU benchmarks (standard vs Tensor Core kernels)
- Comparisons against cuBLAS and MKL
- Profiling data from `perf` (CPU) and `nsight-compute` (GPU)

Stay tuned for updates with performance graphs, profiling screenshots, and optimization analysis.

