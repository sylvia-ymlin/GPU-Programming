#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

#include "../reduce/reduce.cuh"

#define cudaCheck(err) _cudaCheck((err), __FILE__, __LINE__)
inline void _cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::printf("[CUDA ERROR] at %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

struct PerfMetrics {
    float avg_ms;
    float bandwidth_gb;
    float gflops;
    int blocks;
    int threads_per_block;
};

// CPU baseline implementations
float cpu_reduce_sum(const float* X, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += X[i];
    }
    return sum;
}

float cpu_reduce_max(const float* X, int n) {
    float max_val = X[0];
    for (int i = 1; i < n; ++i) {
        if (X[i] > max_val) max_val = X[i];
    }
    return max_val;
}

void cpu_softmax(const float* input, float* output, int N) {
    float max_val = input[0];
    for (int i = 1; i < N; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += std::exp(input[i] - max_val);
    }
    for (int i = 0; i < N; ++i) {
        output[i] = std::exp(input[i] - max_val) / sum;
    }
}

void cpu_softmax_row(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; ++row) {
        cpu_softmax(input + row * N, output + row * N, N);
    }
}

void cpu_softmax_col(const float* input, float* output, int M, int N) {
    for (int col = 0; col < N; ++col) {
        // Find max in column
        float max_val = input[col];
        for (int row = 1; row < M; ++row) {
            if (input[row * N + col] > max_val) max_val = input[row * N + col];
        }
        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            sum += std::exp(input[row * N + col] - max_val);
        }
        // Normalize
        for (int row = 0; row < M; ++row) {
            output[row * N + col] = std::exp(input[row * N + col] - max_val) / sum;
        }
    }
}

bool close_enough(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::printf("=== Device Information ===\n");
    std::printf("Device: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    std::printf("SM Count: %d\n", prop.multiProcessorCount);
    std::printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    std::printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    std::printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    std::printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    std::printf("Peak Memory Bandwidth: %.2f GB/s\n\n", 
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

void print_metrics(const char* name, const PerfMetrics& m, int n) {
    std::printf("%-25s: %.4f ms/iter, %.2f GB/s", name, m.avg_ms, m.bandwidth_gb);
    if (m.gflops > 0) std::printf(", %.2f GFLOPS", m.gflops);
    std::printf("\n");
    std::printf("%-25s  Grid: %d blocks x %d threads\n", "", m.blocks, m.threads_per_block);
    std::printf("%-25s  Elements: %d\n\n", "", n);
}

// ============================================================================
// Correctness Tests
// ============================================================================

void test_sum_kernels() {
    std::printf("Testing sum_v0 (naive single-block)...\n");
    // sum_v0: one-block sanity
    {
        constexpr int N0 = 256;
        constexpr int BLOCK0 = 256;
        float h_in[N0];
        for (int i = 0; i < N0; ++i) h_in[i] = static_cast<float>(i);

        float h_out = -1.0f;
        float *d_in = nullptr, *d_out = nullptr;
        cudaCheck(cudaMalloc(&d_in, N0 * sizeof(float)));
        cudaCheck(cudaMalloc(&d_out, sizeof(float)));
        cudaCheck(cudaMemcpy(d_in, h_in, N0 * sizeof(float), cudaMemcpyHostToDevice));

        sum_v0<<<1, BLOCK0>>>(d_in, d_out);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        const float expected = (N0 - 1) * N0 / 2.0f;
        if (std::fabs(h_out - expected) > 1e-3f) {
            std::printf("sum_v0 FAILED: got %f expected %f\n", h_out, expected);
            std::exit(EXIT_FAILURE);
        }
        std::printf("sum_v0: PASS (value=%f)\n", h_out);

        cudaCheck(cudaFree(d_in));
        cudaCheck(cudaFree(d_out));
    }

    // Shared setup for the remaining variants
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    const float expected = (N - 1) * N / 2.0f;

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    float *d_in = nullptr, *d_out = nullptr;
    const int grid_max = CEIL_DIV(N, BLOCK);
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, grid_max * sizeof(float)));
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_out(grid_max, 0.0f);

    auto check_result = [&](const char* name, float value) {
        if (std::fabs(value - expected) > 1e-3f) {
            std::printf("%s FAILED: got %f expected %f\n", name, value, expected);
            std::exit(EXIT_FAILURE);
        }
        std::printf("%s: PASS (value=%f)\n", name, value);
    };

    // sum_v2: dynamic shared, per-block partials
    std::printf("Testing sum_v2 (dynamic shared)...\n");
    {
        const int grid = CEIL_DIV(N, BLOCK);
        sum_v2<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));
        float acc = 0.0f;
        for (int i = 0; i < grid; ++i) acc += h_out[i];
        check_result("sum_v2", acc);
    }

    // sum_v3: dynamic shared + atomic to single output
    std::printf("Testing sum_v3 (dynamic shared + atomic)...\n");
    {
        const int grid = CEIL_DIV(N, BLOCK);
        cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
        sum_v3<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        float acc = 0.0f;
        cudaCheck(cudaMemcpy(&acc, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        check_result("sum_v3", acc);
    }

    // sum_v4: warp shuffle producing per-block partials
    std::printf("Testing sum_v4 (warp shuffle)...\n");
    {
        const int grid = CEIL_DIV(N, BLOCK);
        sum_v4<<<grid, BLOCK>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));
        float acc = 0.0f;
        for (int i = 0; i < grid; ++i) acc += h_out[i];
        check_result("sum_v4", acc);
    }

    // sum_v5: float4 load + warp shuffle, per-block partials
    std::printf("Testing sum_v5 (float4 + warp shuffle)...\n");
    {
        const int grid = CEIL_DIV(N, BLOCK * 4);
        sum_v5<<<grid, BLOCK>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));
        float acc = 0.0f;
        for (int i = 0; i < grid; ++i) acc += h_out[i];
        check_result("sum_v5", acc);
    }

    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
}

void test_max_kernel() {
    std::printf("Testing max_kernel...\n");
    constexpr int N = 512;
    constexpr int BLOCK = 256;
    const int GRID = CEIL_DIV(N, BLOCK);

    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = -1000.0f + static_cast<float>(i);
    float h_out = -1.0f;

    float *d_in = nullptr, *d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, sizeof(float)));
    cudaCheck(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize output to -FLT_MAX (required for atomicMax to work correctly)
    float neg_max = -FLT_MAX;
    cudaCheck(cudaMemcpy(d_out, &neg_max, sizeof(float), cudaMemcpyHostToDevice));

    max_kernel<<<GRID, BLOCK>>>(d_in, d_out, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    const float expected = h_in[N - 1];
    if (std::fabs(h_out - expected) > 1e-3f) {
        std::printf("max_kernel FAILED: got %f expected %f\n", h_out, expected);
        std::exit(EXIT_FAILURE);
    }
    std::printf("max_kernel: PASS (value=%f)\n", h_out);

    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
    free(h_in);
}

// GPU Softmax helper (3-pass: max, sum, normalize)
void gpu_softmax_3pass(float* d_input, float* d_output, float* d_max, float* d_sum, int N, int block_size) {
    int grid_size = CEIL_DIV(N, block_size);
    
    // Initialize max to -FLT_MAX and sum to 0
    float neg_max = -FLT_MAX;
    cudaCheck(cudaMemcpy(d_max, &neg_max, sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(float)));
    
    // Find max
    max_kernel<<<grid_size, block_size>>>(d_input, d_max, N);
    cudaCheck(cudaGetLastError());
    
    // Compute sum of exp(x - max)
    sum_kernel<<<grid_size, block_size>>>(d_input, d_sum, d_max, N);
    cudaCheck(cudaGetLastError());
    
    // Normalize
    softmax_kernel<<<grid_size, block_size>>>(d_input, d_output, d_sum, d_max, N);
    cudaCheck(cudaGetLastError());
}

void test_softmax_1d() {
    std::printf("Testing softmax_1d (3-pass)...\n");
    
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    
    std::vector<float> h_input(N);
    std::vector<float> h_output_cpu(N);
    std::vector<float> h_output_gpu(N);
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i - N/2) * 0.01f;
    }
    
    cpu_softmax(h_input.data(), h_output_cpu.data(), N);
    
    float *d_input, *d_output, *d_max, *d_sum;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_max, sizeof(float)));
    cudaCheck(cudaMalloc(&d_sum, sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    gpu_softmax_3pass(d_input, d_output, d_max, d_sum, N, BLOCK);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(h_output_gpu[i], h_output_cpu[i], 1e-4f)) {
            fail_count++;
        }
    }
    
    if (fail_count > 0) {
        std::printf("softmax_1d FAILED: %d mismatches, max_diff=%e\n", fail_count, max_diff);
        std::exit(EXIT_FAILURE);
    }
    std::printf("softmax_1d: PASS (max_diff=%e)\n", max_diff);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
    cudaCheck(cudaFree(d_max));
    cudaCheck(cudaFree(d_sum));
}

void test_softmax_row() {
    std::printf("Testing softmax_row_kernel...\n");
    
    constexpr int M = 64;
    constexpr int N = 256;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(M * N);
    std::vector<float> h_output_gpu(M * N);
    
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = static_cast<float>((i % N) - N/2) * 0.01f;
    }
    
    cpu_softmax_row(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    softmax_row_kernel<<<M, N>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(h_output_gpu[i], h_output_cpu[i], 1e-4f)) {
            fail_count++;
        }
    }
    
    if (fail_count > 0) {
        std::printf("softmax_row_kernel FAILED: %d mismatches, max_diff=%e\n", fail_count, max_diff);
        std::exit(EXIT_FAILURE);
    }
    std::printf("softmax_row_kernel: PASS (max_diff=%e)\n", max_diff);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_softmax_row_shfl_xor() {
    std::printf("Testing softmax_row_kernel_shfl_xor...\n");
    
    constexpr int M = 64;
    constexpr int N = 256;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(M * N);
    std::vector<float> h_output_gpu(M * N);
    
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = static_cast<float>((i % N) - N/2) * 0.01f;
    }
    
    cpu_softmax_row(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    softmax_row_kernel_shfl_xor<<<M, N>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(h_output_gpu[i], h_output_cpu[i], 1e-4f)) {
            fail_count++;
        }
    }
    
    if (fail_count > 0) {
        std::printf("softmax_row_kernel_shfl_xor FAILED: %d mismatches, max_diff=%e\n", fail_count, max_diff);
        std::exit(EXIT_FAILURE);
    }
    std::printf("softmax_row_kernel_shfl_xor: PASS (max_diff=%e)\n", max_diff);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_softmax_col() {
    std::printf("Testing softmax_col_kernel_shfl_xor...\n");
    
    constexpr int M = 256;
    constexpr int N = 64;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(M * N);
    std::vector<float> h_output_gpu(M * N);
    
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = static_cast<float>((i / N) - M/2) * 0.01f;
    }
    
    cpu_softmax_col(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // One block per column, blockDim.x threads per block
    softmax_col_kernel_shfl_xor<<<N, 256>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(h_output_gpu[i], h_output_cpu[i], 1e-4f)) {
            fail_count++;
        }
    }
    
    if (fail_count > 0) {
        std::printf("softmax_col_kernel_shfl_xor FAILED: %d mismatches, max_diff=%e\n", fail_count, max_diff);
        std::exit(EXIT_FAILURE);
    }
    std::printf("softmax_col_kernel_shfl_xor: PASS (max_diff=%e)\n", max_diff);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

// ============================================================================
// Performance Tests
// ============================================================================

void perf_sum_v2() {
    constexpr int N = 32 * 1024 * 1024;  // 32M elements
    constexpr int BLOCK = 256;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int grid = CEIL_DIV(N, BLOCK);

    float *d_in = nullptr, *d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, grid * sizeof(float)));

    // Fill input
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 13);
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        sum_v2<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sum_v2<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));

    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (N * sizeof(float)) / (m.avg_ms * 1e6);  // Read N floats
    m.gflops = (N / 1e9f) / (m.avg_ms / 1000.0f);  // N-1 additions
    m.blocks = grid;
    m.threads_per_block = BLOCK;
    print_metrics("sum_v2 (dynamic shared)", m, N);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
}

void perf_sum_v3() {
    constexpr int N = 32 * 1024 * 1024;
    constexpr int BLOCK = 256;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int grid = CEIL_DIV(N, BLOCK);

    float *d_in = nullptr, *d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, sizeof(float)));

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 13);
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
        sum_v3<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
        sum_v3<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));

    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (N * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (N / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = grid;
    m.threads_per_block = BLOCK;
    print_metrics("sum_v3 (shared + atomic)", m, N);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
}

void perf_sum_v4() {
    constexpr int N = 32 * 1024 * 1024;
    constexpr int BLOCK = 256;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int grid = CEIL_DIV(N, BLOCK);

    float *d_in = nullptr, *d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, grid * sizeof(float)));

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 13);
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        sum_v4<<<grid, BLOCK>>>(d_in, d_out, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sum_v4<<<grid, BLOCK>>>(d_in, d_out, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));

    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (N * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (N / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = grid;
    m.threads_per_block = BLOCK;
    print_metrics("sum_v4 (warp shuffle)", m, N);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
}

void perf_sum_v5() {
    constexpr int N = 32 * 1024 * 1024;
    constexpr int BLOCK = 256;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int grid = CEIL_DIV(N, BLOCK * 4);

    float *d_in = nullptr, *d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, grid * sizeof(float)));

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 13);
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        sum_v5<<<grid, BLOCK>>>(d_in, d_out, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sum_v5<<<grid, BLOCK>>>(d_in, d_out, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));

    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (N * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (N / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = grid;
    m.threads_per_block = BLOCK;
    print_metrics("sum_v5 (float4 + shuffle)", m, N);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
}

// ============================================================================
// Scalability Analysis
// ============================================================================

void perf_scalability() {
    std::printf("\n=== Scalability Analysis (Reduction Sum) ===\n");
    std::vector<int> sizes = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 33554432};
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    
    std::printf("\n--- sum_v4 (warp shuffle) ---\n");
    std::printf("%-12s %-12s %-12s %-12s\n", "Size", "Time (ms)", "BW (GB/s)", "Efficiency (%)");
    std::printf("--------------------------------------------------------\n");
    
    for (int n : sizes) {
        constexpr int BLOCK = 256;
        const int grid = CEIL_DIV(n, BLOCK);
        
        float *d_in, *d_out;
        cudaCheck(cudaMalloc(&d_in, n * sizeof(float)));
        cudaCheck(cudaMalloc(&d_out, grid * sizeof(float)));
        
        // Warmup
        for (int i = 0; i < 3; ++i) {
            sum_v4<<<grid, BLOCK>>>(d_in, d_out, n);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        
        int iters = (n < 1048576) ? 1000 : 100;
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            sum_v4<<<grid, BLOCK>>>(d_in, d_out, n);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / iters;
        float bandwidth_gb = (n * sizeof(float)) / (avg_ms * 1e6);
        float efficiency = (bandwidth_gb / peak_bw) * 100.0f;
        
        std::printf("%-12d %-12.4f %-12.2f %-12.1f\n", n, avg_ms, bandwidth_gb, efficiency);
        
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
        cudaCheck(cudaFree(d_in));
        cudaCheck(cudaFree(d_out));
    }
    
    std::printf("\n--- sum_v5 (float4 + warp shuffle) ---\n");
    std::printf("%-12s %-12s %-12s %-12s\n", "Size", "Time (ms)", "BW (GB/s)", "Efficiency (%)");
    std::printf("--------------------------------------------------------\n");
    
    for (int n : sizes) {
        constexpr int BLOCK = 256;
        const int grid = CEIL_DIV(n, BLOCK * 4);
        
        float *d_in, *d_out;
        cudaCheck(cudaMalloc(&d_in, n * sizeof(float)));
        cudaCheck(cudaMalloc(&d_out, grid * sizeof(float)));
        
        // Warmup
        for (int i = 0; i < 3; ++i) {
            sum_v5<<<grid, BLOCK>>>(d_in, d_out, n);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        
        int iters = (n < 1048576) ? 1000 : 100;
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            sum_v5<<<grid, BLOCK>>>(d_in, d_out, n);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / iters;
        float bandwidth_gb = (n * sizeof(float)) / (avg_ms * 1e6);
        float efficiency = (bandwidth_gb / peak_bw) * 100.0f;
        
        std::printf("%-12d %-12.4f %-12.2f %-12.1f\n", n, avg_ms, bandwidth_gb, efficiency);
        
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
        cudaCheck(cudaFree(d_in));
        cudaCheck(cudaFree(d_out));
    }
}

// ============================================================================
// CPU Baseline Comparison
// ============================================================================

void perf_cpu_baseline() {
    constexpr int N = 1024 * 1024;  // 1M elements for CPU
    
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i % 13);
    }
    
    std::printf("\n=== CPU Baseline (1M elements) ===\n");
    
    // CPU Reduce Sum
    auto start = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;
    for (int iter = 0; iter < 10; ++iter) {
        sum = cpu_reduce_sum(h_in.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    double cpu_bw = (N * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("REDUCE_SUM (CPU)     : %.3f ms/iter, %.2f GB/s, sum=%.0f\n", cpu_ms, cpu_bw, sum);
    
    // CPU Reduce Max
    start = std::chrono::high_resolution_clock::now();
    float max_val = 0.0f;
    for (int iter = 0; iter < 10; ++iter) {
        max_val = cpu_reduce_max(h_in.data(), N);
    }
    end = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    cpu_bw = (N * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("REDUCE_MAX (CPU)     : %.3f ms/iter, %.2f GB/s, max=%.0f\n", cpu_ms, cpu_bw, max_val);
    
    // CPU Softmax
    std::vector<float> h_out(N);
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        cpu_softmax(h_in.data(), h_out.data(), N);
    }
    end = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    cpu_bw = (2.0 * N * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("SOFTMAX (CPU)        : %.3f ms/iter, %.2f GB/s\n", cpu_ms, cpu_bw);
}

void perf_softmax_row() {
    constexpr int M = 4096;
    constexpr int N = 1024;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int total = M * N;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, total * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, total * sizeof(float)));
    
    std::vector<float> h_input(total);
    for (int i = 0; i < total; ++i) h_input[i] = static_cast<float>(i % 100) * 0.01f;
    cudaCheck(cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        softmax_row_kernel<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        softmax_row_kernel<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * total * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (3.0f * total / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = M;
    m.threads_per_block = N;
    
    std::printf("%-25s: %.4f ms/iter, %.2f GB/s, %.2f GFLOPS\n", 
                "softmax_row_kernel", m.avg_ms, m.bandwidth_gb, m.gflops);
    std::printf("%-25s  Matrix: %d x %d = %d elements\n\n", "", M, N, total);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_softmax_row_shfl_xor() {
    constexpr int M = 4096;
    constexpr int N = 1024;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int total = M * N;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, total * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, total * sizeof(float)));
    
    std::vector<float> h_input(total);
    for (int i = 0; i < total; ++i) h_input[i] = static_cast<float>(i % 100) * 0.01f;
    cudaCheck(cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < WARMUP; ++i) {
        softmax_row_kernel_shfl_xor<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        softmax_row_kernel_shfl_xor<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * total * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (3.0f * total / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = M;
    m.threads_per_block = N;
    
    std::printf("%-25s: %.4f ms/iter, %.2f GB/s, %.2f GFLOPS\n", 
                "softmax_row_shfl_xor", m.avg_ms, m.bandwidth_gb, m.gflops);
    std::printf("%-25s  Matrix: %d x %d = %d elements\n\n", "", M, N, total);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_softmax_col() {
    constexpr int M = 4096;
    constexpr int N = 1024;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int total = M * N;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, total * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, total * sizeof(float)));
    
    std::vector<float> h_input(total);
    for (int i = 0; i < total; ++i) h_input[i] = static_cast<float>(i % 100) * 0.01f;
    cudaCheck(cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    // One block per column
    for (int i = 0; i < WARMUP; ++i) {
        softmax_col_kernel_shfl_xor<<<N, 256>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        softmax_col_kernel_shfl_xor<<<N, 256>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * total * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (3.0f * total / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = N;
    m.threads_per_block = 256;
    
    std::printf("%-25s: %.4f ms/iter, %.2f GB/s, %.2f GFLOPS\n", 
                "softmax_col_shfl_xor", m.avg_ms, m.bandwidth_gb, m.gflops);
    std::printf("%-25s  Matrix: %d x %d = %d elements (column-wise)\n\n", "", M, N, total);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

// ============================================================================
// Kernel Comparison Summary
// ============================================================================

void perf_comparison_summary() {
    std::printf("\n=== Kernel Performance Comparison (32M elements) ===\n");
    std::printf("%-25s %-12s %-12s %-12s\n", "Kernel", "Time (ms)", "BW (GB/s)", "Speedup");
    std::printf("------------------------------------------------------------------------\n");
    
    constexpr int N = 32 * 1024 * 1024;
    constexpr int BLOCK = 256;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_in = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 13);
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    float baseline_ms = 0.0f;
    
    // Test each kernel variant
    struct KernelConfig {
        const char* name;
        int grid;
        bool uses_shared;
        bool uses_atomic;
        int elements_per_thread;
    };
    
    std::vector<KernelConfig> configs = {
        {"sum_v2 (shared)", CEIL_DIV(N, BLOCK), true, false, 1},
        {"sum_v3 (shared+atomic)", CEIL_DIV(N, BLOCK), true, true, 1},
        {"sum_v4 (warp shuffle)", CEIL_DIV(N, BLOCK), false, false, 1},
        {"sum_v5 (float4+shuffle)", CEIL_DIV(N, BLOCK * 4), false, false, 4},
    };
    
    for (size_t k = 0; k < configs.size(); ++k) {
        const auto& cfg = configs[k];
        float *d_out;
        cudaCheck(cudaMalloc(&d_out, cfg.grid * sizeof(float)));
        
        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            if (k == 0) sum_v2<<<cfg.grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
            else if (k == 1) {
                cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
                sum_v3<<<cfg.grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
            }
            else if (k == 2) sum_v4<<<cfg.grid, BLOCK>>>(d_in, d_out, N);
            else if (k == 3) sum_v5<<<cfg.grid, BLOCK>>>(d_in, d_out, N);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        // Timing
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            if (k == 0) sum_v2<<<cfg.grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
            else if (k == 1) {
                cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
                sum_v3<<<cfg.grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
            }
            else if (k == 2) sum_v4<<<cfg.grid, BLOCK>>>(d_in, d_out, N);
            else if (k == 3) sum_v5<<<cfg.grid, BLOCK>>>(d_in, d_out, N);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / ITERS;
        float bandwidth_gb = (N * sizeof(float)) / (avg_ms * 1e6);
        
        if (k == 0) baseline_ms = avg_ms;
        float speedup = baseline_ms / avg_ms;
        
        std::printf("%-25s %-12.4f %-12.2f %-12.2fx\n", cfg.name, avg_ms, bandwidth_gb, speedup);
        
        cudaCheck(cudaFree(d_out));
    }

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_in));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_device_info();
    
    std::printf("=== Correctness Tests (Sum/Max) ===\n");
    test_sum_kernels();
    std::printf("\n");
    test_max_kernel();
    
    std::printf("\n=== Correctness Tests (Softmax) ===\n");
    test_softmax_1d();
    test_softmax_row();
    test_softmax_row_shfl_xor();
    test_softmax_col();
    std::printf("\nAll reduction correctness tests passed.\n\n");
    
    std::printf("=== Performance Tests (Sum, 32M elements) ===\n");
    perf_sum_v2();
    perf_sum_v3();
    perf_sum_v4();
    perf_sum_v5();
    
    std::printf("=== Performance Tests (Softmax, 4096x1024 matrix) ===\n");
    perf_softmax_row();
    perf_softmax_row_shfl_xor();
    perf_softmax_col();
    
    perf_comparison_summary();
    perf_scalability();
    perf_cpu_baseline();
    
    std::printf("\n=== Profiling Commands ===\n");
    std::printf("For detailed kernel metrics, run:\n");
    std::printf("  ncu --set full --target-processes all -o reduction_profile ./build/test_reduction\n");
    std::printf("  nsys profile -o reduction_trace ./build/test_reduction\n\n");

    return 0;
}
