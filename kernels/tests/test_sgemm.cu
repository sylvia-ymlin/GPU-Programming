#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

#include "../sgemm/sgemm.cuh"

#define cudaCheck(err) _cudaCheck((err), __FILE__, __LINE__)
inline void _cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::printf("[CUDA ERROR] at %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Performance timing constants
constexpr int WARMUP = 5;
constexpr int ITERS = 20;

struct PerfMetrics {
    float avg_ms;
    float gflops;
    int blocks_x;
    int blocks_y;
    int threads_per_block;
};

// ============================================================================
// CPU Baseline
// ============================================================================

void cpu_sgemm(int M, int N, int K, float alpha, const float* A, 
               const float* B, float beta, float* C) {
    // C = alpha * A @ B + beta * C
    // A: MxK, B: KxN, C: MxN
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

bool close_enough(float a, float b, float tol = 1e-3f) {
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
    std::printf("Peak Memory Bandwidth: %.2f GB/s\n", 
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    
    // Calculate theoretical peak GFLOPS (FMA = 2 ops)
    float peak_gflops = prop.multiProcessorCount * 
                        (prop.major >= 7 ? 64 : 128) *  // CUDA cores per SM (approx)
                        2.0f *  // FMA = 2 ops
                        prop.clockRate / 1e6f;  // Convert kHz to GHz
    std::printf("Theoretical Peak GFLOPS: ~%.0f (estimate)\n\n", peak_gflops);
}

void print_metrics(const char* name, const PerfMetrics& m, int M, int N, int K) {
    std::printf("%-20s: %.4f ms/iter, %.2f GFLOPS\n", name, m.avg_ms, m.gflops);
    std::printf("%-20s  Matrix: %dx%d x %dx%d, Grid: %dx%d blocks\n\n", 
                "", M, K, K, N, m.blocks_x, m.blocks_y);
}

void randomize_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }
}

bool verify_sgemm(const float* result, const float* reference, int size, const char* name) {
    int fail_count = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(result[i] - reference[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(result[i], reference[i])) {
            if (fail_count < 5) {
                std::printf("  %s Mismatch at %d: got %f expected %f (diff=%f)\n", 
                           name, i, result[i], reference[i], diff);
            }
            fail_count++;
        }
    }
    if (fail_count > 0) {
        std::printf("  %s: %d mismatches (max_diff=%.6f)\n", name, fail_count, max_diff);
    }
    return fail_count == 0;
}

float compute_gflops(int M, int N, int K, float ms) {
    // GEMM: 2*M*N*K FLOPs (multiply + add)
    float flops = 2.0f * M * N * K;
    return (flops / 1e9f) / (ms / 1e3f);  // GFLOPS
}

// ============================================================================
// Correctness Tests
// ============================================================================

void test_sgemm_naive() {
    std::printf("Testing sgemm_naive...\n");
    
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 64;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    // Host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    // CPU reference
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
    
    // Device memory
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    // Launch kernel
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaCheck(cudaDeviceSynchronize());
    
    // Copy back and verify
    cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_sgemm(h_C.data(), h_C_ref.data(), M * N, "sgemm_naive")) {
        std::printf("  sgemm_naive PASSED\n\n");
    } else {
        std::printf("  sgemm_naive FAILED\n\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_sgemm_v2() {
    std::printf("Testing sgemm_v2 (shared memory)...\n");
    
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 64;
    constexpr int TILE = 16;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    sgemm_v2<TILE><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_sgemm(h_C.data(), h_C_ref.data(), M * N, "sgemm_v2")) {
        std::printf("  sgemm_v2 PASSED\n\n");
    } else {
        std::printf("  sgemm_v2 FAILED\n\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_sgemm_v3() {
    std::printf("Testing sgemm_v3 (1D thread tiling)...\n");
    
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 64;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    constexpr int thread_num = (BM * BN) / TM;
    dim3 block(thread_num);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_v3<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_sgemm(h_C.data(), h_C_ref.data(), M * N, "sgemm_v3")) {
        std::printf("  sgemm_v3 PASSED\n\n");
    } else {
        std::printf("  sgemm_v3 FAILED\n\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_sgemm_v4() {
    std::printf("Testing sgemm_v4 (2D thread tiling)...\n");
    
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 64;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_v4<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_sgemm(h_C.data(), h_C_ref.data(), M * N, "sgemm_v4")) {
        std::printf("  sgemm_v4 PASSED\n\n");
    } else {
        std::printf("  sgemm_v4 FAILED\n\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_sgemm_v5() {
    std::printf("Testing sgemm_v5 (register caching)...\n");
    
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 64;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_v5<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_sgemm(h_C.data(), h_C_ref.data(), M * N, "sgemm_v5")) {
        std::printf("  sgemm_v5 PASSED\n\n");
    } else {
        std::printf("  sgemm_v5 FAILED\n\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Note: v6 and v7 require specific alignment for float4 vectorized access
// Test with sizes that are multiples of 128
void test_sgemm_v6() {
    std::printf("Testing sgemm_v6 (vectorized)...\n");
    
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 128;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_v6<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_sgemm(h_C.data(), h_C_ref.data(), M * N, "sgemm_v6")) {
        std::printf("  sgemm_v6 PASSED\n\n");
    } else {
        std::printf("  sgemm_v6 FAILED\n\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_sgemm_v7() {
    std::printf("Testing sgemm_v7 (double buffering)...\n");
    
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 128;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_v7<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_sgemm(h_C.data(), h_C_ref.data(), M * N, "sgemm_v7")) {
        std::printf("  sgemm_v7 PASSED\n\n");
    } else {
        std::printf("  sgemm_v7 FAILED\n\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Performance Tests
// ============================================================================

void perf_sgemm_naive() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = grid.x;
    m.blocks_y = grid.y;
    m.threads_per_block = block.x * block.y;
    
    print_metrics("sgemm_naive", m, M, N, K);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void perf_sgemm_v2() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int TILE = 32;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_v2<TILE><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_v2<TILE><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = grid.x;
    m.blocks_y = grid.y;
    m.threads_per_block = block.x * block.y;
    
    print_metrics("sgemm_v2", m, M, N, K);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void perf_sgemm_v3() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    constexpr int thread_num = (BM * BN) / TM;
    dim3 block(thread_num);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_v3<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_v3<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = grid.x;
    m.blocks_y = grid.y;
    m.threads_per_block = thread_num;
    
    print_metrics("sgemm_v3", m, M, N, K);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void perf_sgemm_v4() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_v4<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_v4<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = grid.x;
    m.blocks_y = grid.y;
    m.threads_per_block = block_threads;
    
    print_metrics("sgemm_v4", m, M, N, K);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void perf_sgemm_v5() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_v5<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_v5<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = grid.x;
    m.blocks_y = grid.y;
    m.threads_per_block = block_threads;
    
    print_metrics("sgemm_v5", m, M, N, K);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void perf_sgemm_v6() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_v6<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_v6<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = grid.x;
    m.blocks_y = grid.y;
    m.threads_per_block = block_threads;
    
    print_metrics("sgemm_v6", m, M, N, K);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void perf_sgemm_v7() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    constexpr int block_threads = (BN / TN) * (BM / TM);
    dim3 block(block_threads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_v7<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_v7<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = grid.x;
    m.blocks_y = grid.y;
    m.threads_per_block = block_threads;
    
    print_metrics("sgemm_v7", m, M, N, K);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// cuBLAS Reference
// ============================================================================

void perf_cublas() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        // cuBLAS uses column-major, so we compute B^T * A^T = (AB)^T
        // to get row-major result
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K,
                   &alpha, d_B, N, d_A, K,
                   &beta, d_C, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K,
                   &alpha, d_B, N, d_A, K,
                   &beta, d_C, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float total_ms;
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    PerfMetrics m;
    m.avg_ms = avg_ms;
    m.gflops = compute_gflops(M, N, K, avg_ms);
    m.blocks_x = 0;
    m.blocks_y = 0;
    m.threads_per_block = 0;
    
    std::printf("%-20s: %.4f ms/iter, %.2f GFLOPS (reference)\n", "cuBLAS", m.avg_ms, m.gflops);
    std::printf("%-20s  Matrix: %dx%d x %dx%d\n\n", "", M, K, K, N);
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Comparison Summary
// ============================================================================

void perf_comparison_summary() {
    std::printf("=== Performance Comparison Summary ===\n");
    std::printf("All tests run on 1024x1024 @ 1024 matrix multiply\n");
    std::printf("GFLOPS = 2 * M * N * K / (time_ms * 1e6)\n\n");
}

void perf_scalability() {
    std::printf("=== Scalability Analysis ===\n");
    
    // Test with different sizes
    int sizes[] = {256, 512, 1024, 2048, 4096};
    
    std::printf("Matrix Size | v2 GFLOPS | v5 GFLOPS | v7 GFLOPS | cuBLAS GFLOPS\n");
    std::printf("------------|-----------|-----------|-----------|---------------\n");
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        float alpha = 1.0f, beta = 0.0f;
        
        float *d_A, *d_B, *d_C;
        cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
        cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
        cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
        
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        float total_ms;
        
        // v2 (shared memory)
        constexpr int TILE = 32;
        dim3 block_v2(TILE, TILE);
        dim3 grid_v2(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
        
        for (int i = 0; i < WARMUP; ++i) {
            sgemm_v2<TILE><<<grid_v2, block_v2>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            sgemm_v2<TILE><<<grid_v2, block_v2>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
        float gflops_v2 = compute_gflops(M, N, K, total_ms / ITERS);
        
        // v5 (register caching)
        constexpr int BM5 = 64, BN5 = 64, BK5 = 8, TM5 = 8, TN5 = 8;
        constexpr int block_threads_v5 = (BN5 / TN5) * (BM5 / TM5);
        dim3 block_v5(block_threads_v5);
        dim3 grid_v5(CEIL_DIV(N, BN5), CEIL_DIV(M, BM5));
        
        for (int i = 0; i < WARMUP; ++i) {
            sgemm_v5<BM5, BN5, BK5, TM5, TN5><<<grid_v5, block_v5>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            sgemm_v5<BM5, BN5, BK5, TM5, TN5><<<grid_v5, block_v5>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
        float gflops_v5 = compute_gflops(M, N, K, total_ms / ITERS);
        
        // v7 (double buffering) - only for sizes >= 512 (needs alignment)
        float gflops_v7 = 0.0f;
        if (size >= 512) {
            constexpr int BM7 = 128, BN7 = 128, BK7 = 8, TM7 = 8, TN7 = 8;
            constexpr int block_threads_v7 = (BN7 / TN7) * (BM7 / TM7);
            dim3 block_v7(block_threads_v7);
            dim3 grid_v7(CEIL_DIV(N, BN7), CEIL_DIV(M, BM7));
            
            for (int i = 0; i < WARMUP; ++i) {
                sgemm_v7<BM7, BN7, BK7, TM7, TN7><<<grid_v7, block_v7>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
            }
            cudaCheck(cudaDeviceSynchronize());
            
            cudaCheck(cudaEventRecord(start));
            for (int i = 0; i < ITERS; ++i) {
                sgemm_v7<BM7, BN7, BK7, TM7, TN7><<<grid_v7, block_v7>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
            }
            cudaCheck(cudaEventRecord(stop));
            cudaCheck(cudaEventSynchronize(stop));
            cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
            gflops_v7 = compute_gflops(M, N, K, total_ms / ITERS);
        }
        
        // cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        for (int i = 0; i < WARMUP; ++i) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
        float gflops_cublas = compute_gflops(M, N, K, total_ms / ITERS);
        
        cublasDestroy(handle);
        
        if (size >= 512) {
            std::printf("%4d x %4d | %9.2f | %9.2f | %9.2f | %13.2f\n", 
                       size, size, gflops_v2, gflops_v5, gflops_v7, gflops_cublas);
        } else {
            std::printf("%4d x %4d | %9.2f | %9.2f | %9s | %13.2f\n", 
                       size, size, gflops_v2, gflops_v5, "N/A", gflops_cublas);
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    std::printf("\n");
}

void perf_edge_cases() {
    std::printf("=== Edge Cases (Non-Power-of-Two Dimensions) ===\n");
    
    // Test odd dimensions: 1000x1000x1000
    int M = 1000, N = 1000, K = 1000;
    float alpha = 1.0f, beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float total_ms;
    
    std::printf("Testing %dx%dx%d (non-power-of-two):\n", M, N, K);
    
    // v5
    constexpr int BM5 = 64, BN5 = 64, BK5 = 8, TM5 = 8, TN5 = 8;
    constexpr int block_threads_v5 = (BN5 / TN5) * (BM5 / TM5);
    dim3 block_v5(block_threads_v5);
    dim3 grid_v5(CEIL_DIV(N, BN5), CEIL_DIV(M, BM5));
    
    for (int i = 0; i < WARMUP; ++i) {
        sgemm_v5<BM5, BN5, BK5, TM5, TN5><<<grid_v5, block_v5>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_v5<BM5, BN5, BK5, TM5, TN5><<<grid_v5, block_v5>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float gflops_v5 = compute_gflops(M, N, K, total_ms / ITERS);
    std::printf("  v5: %.2f GFLOPS (%.4f ms)\n", gflops_v5, total_ms / ITERS);
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    for (int i = 0; i < WARMUP; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
    float gflops_cublas = compute_gflops(M, N, K, total_ms / ITERS);
    std::printf("  cuBLAS: %.2f GFLOPS (%.4f ms)\n", gflops_cublas, total_ms / ITERS);
    
    // Compare with 1024x1024x1024
    std::printf("  Comparison: 1000^3 vs 1024^3 efficiency = %.1f%% (due to partial tiles)\n",
               100.0f * gflops_v5 / 1719.0f);  // 1719 is v5@1024 from your results
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    std::printf("\n");
}

void perf_cpu_baseline() {
    std::printf("=== CPU Baseline ===\n");
    
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);
    
    auto start = std::chrono::high_resolution_clock::now();
    cpu_sgemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C.data());
    auto end = std::chrono::high_resolution_clock::now();
    
    float cpu_ms = std::chrono::duration<float, std::milli>(end - start).count();
    float cpu_gflops = compute_gflops(M, N, K, cpu_ms);
    
    std::printf("CPU SGEMM (%dx%d): %.2f ms, %.4f GFLOPS\n", M, N, cpu_ms, cpu_gflops);
    std::printf("Note: CPU baseline is single-threaded naive implementation\n\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    srand(42);  // Fixed seed for reproducibility
    
    print_device_info();
    
    std::printf("=== Correctness Tests ===\n");
    test_sgemm_naive();
    test_sgemm_v2();
    test_sgemm_v3();
    test_sgemm_v4();
    test_sgemm_v5();
    test_sgemm_v6();
    test_sgemm_v7();
    std::printf("All SGEMM correctness tests completed.\n\n");
    
    std::printf("=== Performance Tests (1024 x 1024 x 1024) ===\n");
    perf_sgemm_naive();
    perf_sgemm_v2();
    perf_sgemm_v3();
    perf_sgemm_v4();
    perf_sgemm_v5();
    perf_sgemm_v6();
    perf_sgemm_v7();
    perf_cublas();
    
    perf_comparison_summary();
    perf_scalability();
    perf_edge_cases();
    perf_cpu_baseline();
    
    std::printf("\n=== Profiling Commands ===\n");
    std::printf("For detailed kernel metrics, run:\n");
    std::printf("  ncu --set full --target-processes all -o sgemm_profile ./build/test_sgemm\n");
    std::printf("  nsys profile -o sgemm_trace ./build/test_sgemm\n\n");
    
    return 0;
}

