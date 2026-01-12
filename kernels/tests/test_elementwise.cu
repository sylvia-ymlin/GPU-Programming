#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#include "../elementwise/elementwise.cuh"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define cudaCheck(err) _cudaCheck((err), __FILE__, __LINE__)
inline void _cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::printf("[CUDA ERROR] at %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

bool close_enough(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

// CPU baseline implementations
void cpu_elementwise_add(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

void cpu_sigmoid(const float* X, float* Y, int n) {
    for (int i = 0; i < n; ++i) {
        Y[i] = 1.0f / (1.0f + std::exp(-X[i]));
    }
}

void cpu_relu(const float* X, float* Y, int n) {
    for (int i = 0; i < n; ++i) {
        Y[i] = X[i] > 0.f ? X[i] : 0.f;
    }
}

struct PerfMetrics {
    float avg_ms;
    float bandwidth_gb;
    float gflops;
    int blocks;
    int threads_per_block;
};

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
    std::printf("%-20s: %.3f ms/iter, %.2f GB/s", name, m.avg_ms, m.bandwidth_gb);
    if (m.gflops > 0) std::printf(", %.2f GFLOPS", m.gflops);
    std::printf("\n");
    std::printf("%-20s  Grid: %d blocks x %d threads = %d threads\n", 
                "", m.blocks, m.threads_per_block, m.blocks * m.threads_per_block);
    std::printf("%-20s  Elements: %d, Occupancy estimate: %.1f%%\n\n", 
                "", n, (m.blocks * m.threads_per_block * 100.0f) / (n / 4));
}

void test_elementwise_add() {
    constexpr int n = 10;          // covers vectorized path (8) + tail (2)
    constexpr int n4 = (n / 4) * 4;

    float hA[n], hB[n], hC[n];
    for (int i = 0; i < n; ++i) { hA[i] = float(i); hB[i] = 2.0f * float(i); }

    float *dA, *dB, *dC;
    cudaCheck(cudaMalloc(&dA, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dB, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, n * sizeof(float)));
    cudaCheck(cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    if (n4 > 0) {
        dim3 grid_vec(CEIL_DIV(n4 / 4, block.x));
        elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        cudaCheck(cudaGetLastError());
    }
    if (n > n4) {
        dim3 grid_tail(CEIL_DIV(n - n4, block.x));
        elementwise_add_scalar<<<grid_tail, block>>>(dA, dB, dC, n, n4);
        cudaCheck(cudaGetLastError());
    }

    cudaCheck(cudaMemcpy(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dA)); cudaCheck(cudaFree(dB)); cudaCheck(cudaFree(dC));

    for (int i = 0; i < n; ++i) {
        float expected = hA[i] + hB[i];
        if (!close_enough(hC[i], expected)) {
            std::printf("elementwise_add failed at %d: got %f expected %f\n", i, hC[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("elementwise_add: PASS\n");
}

void test_sigmoid() {
    constexpr int n4 = 8; // must be multiple of 4 for float4 path
    float hX[n4] = { -4.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f };
    float hY[n4];

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));
    cudaCheck(cudaMemcpy(dX, hX, n4 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));
    sigmoid_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                    reinterpret_cast<float4*>(dY),
                                    n4);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(hY, dY, n4 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));

    for (int i = 0; i < n4; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-hX[i]));
        if (!close_enough(hY[i], expected, 1e-4f)) {
            std::printf("sigmoid failed at %d: got %f expected %f\n", i, hY[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("sigmoid: PASS\n");
}

void test_relu() {
    constexpr int n4 = 8;
    float hX[n4] = { -3.f, -1.f, 0.f, 1.f, 2.f, -2.f, 5.f, -5.f };
    float hY[n4];

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));
    cudaCheck(cudaMemcpy(dX, hX, n4 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));
    relu_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                 reinterpret_cast<float4*>(dY),
                                 n4);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(hY, dY, n4 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));

    for (int i = 0; i < n4; ++i) {
        float expected = hX[i] > 0.f ? hX[i] : 0.f;
        if (!close_enough(hY[i], expected)) {
            std::printf("relu failed at %d: got %f expected %f\n", i, hY[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("relu: PASS\n");
}

void perf_elementwise_add() {
    constexpr int n = 32 * 1024 * 1024;  // 32M elements
    constexpr int n4 = (n / 4) * 4;
    constexpr int warmup = 3;
    constexpr int iters = 100;

    float *dA, *dB, *dC;
    cudaCheck(cudaMalloc(&dA, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dB, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, n * sizeof(float)));

    dim3 block(256);
    dim3 grid_vec(CEIL_DIV(n4 / 4, block.x));
    dim3 grid_tail(CEIL_DIV(n - n4, block.x));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        if (n4 > 0) elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        if (n > n4) elementwise_add_scalar<<<grid_tail, block>>>(dA, dB, dC, n, n4);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        if (n4 > 0) elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        if (n > n4) elementwise_add_scalar<<<grid_tail, block>>>(dA, dB, dC, n, n4);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics metrics;
    metrics.avg_ms = ms / iters;
    metrics.bandwidth_gb = (3.0f * n * sizeof(float)) / (metrics.avg_ms * 1e6);
    metrics.gflops = (n / 1e9) / (metrics.avg_ms / 1000.0f); // 1 FLOP per element
    metrics.blocks = grid_vec.x;
    metrics.threads_per_block = block.x;
    
    print_metrics("ADD (GPU float4)", metrics, n);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(dA)); cudaCheck(cudaFree(dB)); cudaCheck(cudaFree(dC));
}

void perf_sigmoid() {
    constexpr int n4 = 32 * 1024 * 1024;  // 32M elements
    constexpr int warmup = 3;
    constexpr int iters = 100;

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        sigmoid_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                        reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        sigmoid_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                        reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics metrics;
    metrics.avg_ms = ms / iters;
    metrics.bandwidth_gb = (2.0f * n4 * sizeof(float)) / (metrics.avg_ms * 1e6);
    metrics.gflops = (n4 * 4.0f / 1e9) / (metrics.avg_ms / 1000.0f); // ~4 FLOPs per element (exp, add, div, neg)
    metrics.blocks = grid.x;
    metrics.threads_per_block = block.x;
    
    print_metrics("SIGMOID (GPU float4)", metrics, n4);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));
}

void perf_relu() {
    constexpr int n4 = 32 * 1024 * 1024;  // 32M elements
    constexpr int warmup = 3;
    constexpr int iters = 100;

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        relu_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                     reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        relu_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                     reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics metrics;
    metrics.avg_ms = ms / iters;
    metrics.bandwidth_gb = (2.0f * n4 * sizeof(float)) / (metrics.avg_ms * 1e6);
    metrics.gflops = (n4 / 1e9) / (metrics.avg_ms / 1000.0f); // 1 comparison per element
    metrics.blocks = grid.x;
    metrics.threads_per_block = block.x;
    
    print_metrics("RELU (GPU float4)", metrics, n4);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));
}

// Scalability tests across different sizes
void perf_scalability() {
    std::printf("\n=== Scalability Analysis (ADD) ===\n");
    std::vector<int> sizes = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 33554432};
    
    std::printf("%-12s %-12s %-12s %-12s\n", "Size", "Time (ms)", "BW (GB/s)", "Efficiency (%)");
    std::printf("--------------------------------------------------------\n");
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    
    for (int n : sizes) {
        int n4 = (n / 4) * 4;
        float *dA, *dB, *dC;
        cudaCheck(cudaMalloc(&dA, n * sizeof(float)));
        cudaCheck(cudaMalloc(&dB, n * sizeof(float)));
        cudaCheck(cudaMalloc(&dC, n * sizeof(float)));
        
        dim3 block(256);
        dim3 grid_vec(CEIL_DIV(n4 / 4, block.x));
        
        // Warmup
        for (int i = 0; i < 3; ++i) {
            if (n4 > 0) elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        
        int iters = (n < 1048576) ? 1000 : 100;
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            if (n4 > 0) elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / iters;
        float bandwidth_gb = (3.0f * n * sizeof(float)) / (avg_ms * 1e6);
        float efficiency = (bandwidth_gb / peak_bw) * 100.0f;
        
        std::printf("%-12d %-12.4f %-12.2f %-12.1f\n", n, avg_ms, bandwidth_gb, efficiency);
        
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
        cudaCheck(cudaFree(dA)); cudaCheck(cudaFree(dB)); cudaCheck(cudaFree(dC));
    }
}

// CPU baseline comparison
void perf_cpu_baseline() {
    constexpr int n = 1024 * 1024;  // 1M elements for CPU
    
    std::vector<float> hA(n), hB(n), hC(n);
    for (int i = 0; i < n; ++i) {
        hA[i] = float(i);
        hB[i] = 2.0f * float(i);
    }
    
    std::printf("\n=== CPU Baseline (1M elements) ===\n");
    
    // ADD
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        cpu_elementwise_add(hA.data(), hB.data(), hC.data(), n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    double cpu_bw = (3.0 * n * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("ADD (CPU)            : %.3f ms/iter, %.2f GB/s\n", cpu_ms, cpu_bw);
    
    // SIGMOID
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        cpu_sigmoid(hA.data(), hC.data(), n);
    }
    end = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    cpu_bw = (2.0 * n * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("SIGMOID (CPU)        : %.3f ms/iter, %.2f GB/s\n", cpu_ms, cpu_bw);
    
    // RELU
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        cpu_relu(hA.data(), hC.data(), n);
    }
    end = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    cpu_bw = (2.0 * n * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("RELU (CPU)           : %.3f ms/iter, %.2f GB/s\n", cpu_ms, cpu_bw);
}

int main() {
    print_device_info();
    
    std::printf("=== Correctness Tests ===\n");
    test_elementwise_add();
    test_sigmoid();
    test_relu();
    std::printf("All elementwise tests passed.\n\n");

    std::printf("=== Performance Tests (32M elements) ===\n");
    perf_elementwise_add();
    perf_sigmoid();
    perf_relu();
    
    perf_scalability();
    perf_cpu_baseline();
    
    std::printf("\n=== Profiling Commands ===\n");
    std::printf("For detailed kernel metrics, run:\n");
    std::printf("  ncu --set full --target-processes all -o elementwise_profile ./build/test_elementwise\n");
    std::printf("  nsys profile -o elementwise_trace ./build/test_elementwise\n\n");

    return 0;
}