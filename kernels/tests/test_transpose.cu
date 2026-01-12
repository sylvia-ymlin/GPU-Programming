#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

#include "../transpose/transpose.cuh"

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
    int blocks;
    int threads_per_block;
};

// ============================================================================
// CPU Baseline
// ============================================================================

void cpu_transpose(const float* input, float* output, int M, int N) {
    // Input: M x N, Output: N x M
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

bool close_enough(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

// ============================================================================
// Helper Functions
// ============================================================================

void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::printf("=== Device Information ===\n");
    std::printf("Device: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    std::printf("SM Count: %d\n", prop.multiProcessorCount);
    std::printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    std::printf("Peak Memory Bandwidth: %.2f GB/s\n\n", 
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

void print_metrics(const char* name, const PerfMetrics& m, int M, int N) {
    std::printf("%-25s: %.4f ms/iter, %.2f GB/s\n", name, m.avg_ms, m.bandwidth_gb);
    std::printf("%-25s  Matrix: %d x %d, Grid: %d blocks\n\n", "", M, N, m.blocks);
}

void randomize_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verify_transpose(const float* result, const float* reference, int size) {
    int fail_count = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(result[i] - reference[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(result[i], reference[i])) {
            if (fail_count < 5) {
                std::printf("  Mismatch at %d: got %f expected %f\n", i, result[i], reference[i]);
            }
            fail_count++;
        }
    }
    return fail_count == 0;
}

// ============================================================================
// Correctness Tests
// ============================================================================

void test_transpose_v0() {
    std::printf("Testing transpose_v0 (naive)...\n");
    
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int BLOCK = 16;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(N * M);
    std::vector<float> h_output_gpu(N * M);
    
    randomize_matrix(h_input.data(), M * N);
    cpu_transpose(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK, BLOCK);
    dim3 grid(CEIL_DIV(M, BLOCK), CEIL_DIV(N, BLOCK));
    transpose_v0<<<grid, block>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (!verify_transpose(h_output_gpu.data(), h_output_cpu.data(), N * M)) {
        std::printf("transpose_v0 FAILED\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf("transpose_v0: PASS\n");
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_transpose_v1() {
    std::printf("Testing transpose_v1 (coalesced write)...\n");
    
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int BLOCK = 16;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(N * M);
    std::vector<float> h_output_gpu(N * M);
    
    randomize_matrix(h_input.data(), M * N);
    cpu_transpose(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK, BLOCK);
    dim3 grid(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));  // Note: swapped for coalesced write
    transpose_v1<<<grid, block>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (!verify_transpose(h_output_gpu.data(), h_output_cpu.data(), N * M)) {
        std::printf("transpose_v1 FAILED\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf("transpose_v1: PASS\n");
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_transpose_v2() {
    std::printf("Testing transpose_v2 (coalesced + __ldg)...\n");
    
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int BLOCK = 16;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(N * M);
    std::vector<float> h_output_gpu(N * M);
    
    randomize_matrix(h_input.data(), M * N);
    cpu_transpose(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK, BLOCK);
    dim3 grid(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));
    transpose_v2<<<grid, block>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (!verify_transpose(h_output_gpu.data(), h_output_cpu.data(), N * M)) {
        std::printf("transpose_v2 FAILED\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf("transpose_v2: PASS\n");
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_transpose_v3() {
    std::printf("Testing transpose_v3 (shared memory, bank conflicts)...\n");
    
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int TILE = 32;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(N * M);
    std::vector<float> h_output_gpu(N * M);
    
    randomize_matrix(h_input.data(), M * N);
    cpu_transpose(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    transpose_v3<TILE><<<grid, block>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (!verify_transpose(h_output_gpu.data(), h_output_cpu.data(), N * M)) {
        std::printf("transpose_v3 FAILED\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf("transpose_v3: PASS\n");
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_transpose_v4() {
    std::printf("Testing transpose_v4 (shared + padding)...\n");
    
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int TILE = 32;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(N * M);
    std::vector<float> h_output_gpu(N * M);
    
    randomize_matrix(h_input.data(), M * N);
    cpu_transpose(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    transpose_v4<TILE><<<grid, block>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (!verify_transpose(h_output_gpu.data(), h_output_cpu.data(), N * M)) {
        std::printf("transpose_v4 FAILED\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf("transpose_v4: PASS\n");
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_transpose_v5() {
    std::printf("Testing transpose_v5 (shared + swizzling)...\n");
    
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int TILE = 32;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(N * M);
    std::vector<float> h_output_gpu(N * M);
    
    randomize_matrix(h_input.data(), M * N);
    cpu_transpose(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    transpose_v5<TILE><<<grid, block>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (!verify_transpose(h_output_gpu.data(), h_output_cpu.data(), N * M)) {
        std::printf("transpose_v5 FAILED\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf("transpose_v5: PASS\n");
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

// ============================================================================
// Performance Tests
// ============================================================================

void perf_transpose_v0() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE = 32;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    
    std::vector<float> h_input(M * N);
    randomize_matrix(h_input.data(), M * N);
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(M, TILE), CEIL_DIV(N, TILE));
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        transpose_v0<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        transpose_v0<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * M * N * sizeof(float)) / (m.avg_ms * 1e6);
    m.blocks = grid.x * grid.y;
    m.threads_per_block = block.x * block.y;
    print_metrics("transpose_v0 (naive)", m, M, N);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_transpose_v1() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE = 32;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    
    std::vector<float> h_input(M * N);
    randomize_matrix(h_input.data(), M * N);
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    
    for (int i = 0; i < WARMUP; ++i) {
        transpose_v1<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        transpose_v1<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * M * N * sizeof(float)) / (m.avg_ms * 1e6);
    m.blocks = grid.x * grid.y;
    m.threads_per_block = block.x * block.y;
    print_metrics("transpose_v1 (coalesced)", m, M, N);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_transpose_v2() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE = 32;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    
    std::vector<float> h_input(M * N);
    randomize_matrix(h_input.data(), M * N);
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    
    for (int i = 0; i < WARMUP; ++i) {
        transpose_v2<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        transpose_v2<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * M * N * sizeof(float)) / (m.avg_ms * 1e6);
    m.blocks = grid.x * grid.y;
    m.threads_per_block = block.x * block.y;
    print_metrics("transpose_v2 (coalesced+ldg)", m, M, N);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_transpose_v3() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE = 32;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    
    std::vector<float> h_input(M * N);
    randomize_matrix(h_input.data(), M * N);
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    
    for (int i = 0; i < WARMUP; ++i) {
        transpose_v3<TILE><<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        transpose_v3<TILE><<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * M * N * sizeof(float)) / (m.avg_ms * 1e6);
    m.blocks = grid.x * grid.y;
    m.threads_per_block = block.x * block.y;
    print_metrics("transpose_v3 (shared)", m, M, N);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_transpose_v4() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE = 32;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    
    std::vector<float> h_input(M * N);
    randomize_matrix(h_input.data(), M * N);
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    
    for (int i = 0; i < WARMUP; ++i) {
        transpose_v4<TILE><<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        transpose_v4<TILE><<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * M * N * sizeof(float)) / (m.avg_ms * 1e6);
    m.blocks = grid.x * grid.y;
    m.threads_per_block = block.x * block.y;
    print_metrics("transpose_v4 (shared+pad)", m, M, N);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_transpose_v5() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE = 32;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    
    std::vector<float> h_input(M * N);
    randomize_matrix(h_input.data(), M * N);
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    
    for (int i = 0; i < WARMUP; ++i) {
        transpose_v5<TILE><<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        transpose_v5<TILE><<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * M * N * sizeof(float)) / (m.avg_ms * 1e6);
    m.blocks = grid.x * grid.y;
    m.threads_per_block = block.x * block.y;
    print_metrics("transpose_v5 (shared+swizzle)", m, M, N);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

// ============================================================================
// Kernel Comparison Summary
// ============================================================================

void perf_comparison_summary() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE = 32;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    std::printf("\n=== Kernel Performance Comparison (%dx%d matrix) ===\n", M, N);
    std::printf("%-30s %-12s %-12s %-12s\n", "Kernel", "Time (ms)", "BW (GB/s)", "Speedup");
    std::printf("------------------------------------------------------------------------\n");
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
    
    std::vector<float> h_input(M * N);
    randomize_matrix(h_input.data(), M * N);
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    float baseline_ms = 0.0f;
    
    struct KernelInfo { const char* name; int version; };
    KernelInfo kernels[] = {
        {"transpose_v0 (naive)", 0},
        {"transpose_v1 (coalesced)", 1},
        {"transpose_v2 (coalesced+ldg)", 2},
        {"transpose_v3 (shared, bank conflicts)", 3},
        {"transpose_v4 (shared+padding)", 4},
        {"transpose_v5 (shared+swizzle)", 5},
    };
    
    for (const auto& k : kernels) {
        dim3 block(TILE, TILE);
        dim3 grid = (k.version == 0) ? dim3(CEIL_DIV(M, TILE), CEIL_DIV(N, TILE))
                                     : dim3(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
        
        for (int i = 0; i < WARMUP; ++i) {
            switch (k.version) {
                case 0: transpose_v0<<<grid, block>>>(d_input, d_output, M, N); break;
                case 1: transpose_v1<<<grid, block>>>(d_input, d_output, M, N); break;
                case 2: transpose_v2<<<grid, block>>>(d_input, d_output, M, N); break;
                case 3: transpose_v3<TILE><<<grid, block>>>(d_input, d_output, M, N); break;
                case 4: transpose_v4<TILE><<<grid, block>>>(d_input, d_output, M, N); break;
                case 5: transpose_v5<TILE><<<grid, block>>>(d_input, d_output, M, N); break;
            }
        }
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            switch (k.version) {
                case 0: transpose_v0<<<grid, block>>>(d_input, d_output, M, N); break;
                case 1: transpose_v1<<<grid, block>>>(d_input, d_output, M, N); break;
                case 2: transpose_v2<<<grid, block>>>(d_input, d_output, M, N); break;
                case 3: transpose_v3<TILE><<<grid, block>>>(d_input, d_output, M, N); break;
                case 4: transpose_v4<TILE><<<grid, block>>>(d_input, d_output, M, N); break;
                case 5: transpose_v5<TILE><<<grid, block>>>(d_input, d_output, M, N); break;
            }
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / ITERS;
        float bandwidth_gb = (2.0f * M * N * sizeof(float)) / (avg_ms * 1e6);
        
        if (k.version == 0) baseline_ms = avg_ms;
        float speedup = baseline_ms / avg_ms;
        
        std::printf("%-30s %-12.4f %-12.2f %-12.2fx\n", k.name, avg_ms, bandwidth_gb, speedup);
    }
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

// ============================================================================
// Scalability Analysis
// ============================================================================

void perf_scalability() {
    std::printf("\n=== Scalability Analysis (transpose_v4 vs transpose_v5) ===\n");
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192};
    constexpr int TILE = 32;
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    
    std::printf("%-10s %-12s %-12s %-12s %-12s\n", "Size", "v4 (ms)", "v4 BW", "v5 (ms)", "v5 BW");
    std::printf("------------------------------------------------------------------------\n");
    
    for (int size : sizes) {
        int M = size, N = size;
        
        float *d_input, *d_output;
        cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
        cudaCheck(cudaMalloc(&d_output, N * M * sizeof(float)));
        
        dim3 block(TILE, TILE);
        dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
        
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        
        int iters = (size < 2048) ? 500 : 100;
        
        // Warmup & time v4
        for (int i = 0; i < 3; ++i) transpose_v4<TILE><<<grid, block>>>(d_input, d_output, M, N);
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            transpose_v4<TILE><<<grid, block>>>(d_input, d_output, M, N);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms_v4;
        cudaCheck(cudaEventElapsedTime(&ms_v4, start, stop));
        float avg_v4 = ms_v4 / iters;
        float bw_v4 = (2.0f * M * N * sizeof(float)) / (avg_v4 * 1e6);
        
        // Warmup & time v5
        for (int i = 0; i < 3; ++i) transpose_v5<TILE><<<grid, block>>>(d_input, d_output, M, N);
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            transpose_v5<TILE><<<grid, block>>>(d_input, d_output, M, N);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms_v5;
        cudaCheck(cudaEventElapsedTime(&ms_v5, start, stop));
        float avg_v5 = ms_v5 / iters;
        float bw_v5 = (2.0f * M * N * sizeof(float)) / (avg_v5 * 1e6);
        
        std::printf("%-10d %-12.4f %-12.2f %-12.4f %-12.2f\n", size, avg_v4, bw_v4, avg_v5, bw_v5);
        
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
        cudaCheck(cudaFree(d_input));
        cudaCheck(cudaFree(d_output));
    }
    std::printf("Peak Memory Bandwidth: %.2f GB/s\n", peak_bw);
}

// ============================================================================
// CPU Baseline Comparison
// ============================================================================

void perf_cpu_baseline() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output(N * M);
    randomize_matrix(h_input.data(), M * N);
    
    std::printf("\n=== CPU Baseline (%d x %d matrix) ===\n", M, N);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        cpu_transpose(h_input.data(), h_output.data(), M, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    double cpu_bw = (2.0 * M * N * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("TRANSPOSE (CPU)      : %.3f ms/iter, %.2f GB/s\n", cpu_ms, cpu_bw);
}

void perf_edge_cases() {
    std::printf("\n=== Edge Cases (Non-Square Matrices) ===\n");
    
    // Non-square, non-power-of-two: 4096 x 1023
    constexpr int M1 = 4096, N1 = 1023;
    // Square power-of-two reference: 4096 x 1024  
    constexpr int M2 = 4096, N2 = 1024;
    constexpr int TILE = 32;
    constexpr int ITERS = 20;
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    // Test 4096 x 1023 (non-power-of-two columns)
    {
        float *d_in, *d_out;
        cudaCheck(cudaMalloc(&d_in, M1 * N1 * sizeof(float)));
        cudaCheck(cudaMalloc(&d_out, N1 * M1 * sizeof(float)));
        
        dim3 block(TILE, TILE);
        dim3 grid(CEIL_DIV(N1, TILE), CEIL_DIV(M1, TILE));
        
        // Warmup
        for (int i = 0; i < 3; ++i) transpose_v4<TILE><<<grid, block>>>(d_in, d_out, M1, N1);
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            transpose_v4<TILE><<<grid, block>>>(d_in, d_out, M1, N1);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / ITERS;
        float bw = (2.0f * M1 * N1 * sizeof(float)) / (avg_ms * 1e6);
        
        std::printf("transpose_v4 (%d x %d): %.4f ms, %.2f GB/s\n", M1, N1, avg_ms, bw);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    // Test 4096 x 1024 (reference)
    {
        float *d_in, *d_out;
        cudaCheck(cudaMalloc(&d_in, M2 * N2 * sizeof(float)));
        cudaCheck(cudaMalloc(&d_out, N2 * M2 * sizeof(float)));
        
        dim3 block(TILE, TILE);
        dim3 grid(CEIL_DIV(N2, TILE), CEIL_DIV(M2, TILE));
        
        // Warmup
        for (int i = 0; i < 3; ++i) transpose_v4<TILE><<<grid, block>>>(d_in, d_out, M2, N2);
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            transpose_v4<TILE><<<grid, block>>>(d_in, d_out, M2, N2);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / ITERS;
        float bw = (2.0f * M2 * N2 * sizeof(float)) / (avg_ms * 1e6);
        
        std::printf("transpose_v4 (%d x %d): %.4f ms, %.2f GB/s (reference)\n", M2, N2, avg_ms, bw);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    std::printf("Note: Non-power-of-two causes partial tiles at boundaries\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    srand(42);  // Fixed seed for reproducibility
    
    print_device_info();
    
    std::printf("=== Correctness Tests ===\n");
    test_transpose_v0();
    test_transpose_v1();
    test_transpose_v2();
    test_transpose_v3();
    test_transpose_v4();
    test_transpose_v5();
    std::printf("\nAll transpose correctness tests passed.\n\n");
    
    std::printf("=== Performance Tests (4096x4096 matrix) ===\n");
    perf_transpose_v0();
    perf_transpose_v1();
    perf_transpose_v2();
    perf_transpose_v3();
    perf_transpose_v4();
    perf_transpose_v5();
    
    perf_comparison_summary();
    perf_scalability();
    perf_cpu_baseline();
    perf_edge_cases();
    
    std::printf("\n=== Profiling Commands ===\n");
    std::printf("For detailed kernel metrics, run:\n");
    std::printf("  ncu --set full --target-processes all -o transpose_profile ./build/test_transpose\n");
    std::printf("  nsys profile -o transpose_trace ./build/test_transpose\n\n");

    return 0;
}

