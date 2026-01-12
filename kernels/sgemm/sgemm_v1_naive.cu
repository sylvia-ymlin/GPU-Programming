// SGEMM v1: Naive implementation
// Matrix sizes: MxK * KxN = MxN
// C = α*(A@B)+β*C

#include <cuda_runtime.h>

// inputs: dims of matrices, alpha, A, B, beta, C
__global__ __launch_bounds__(1024)
void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                 const float *B, float beta, float *C) {
    
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // bounds check to avoid out-of-bounds access
    if (row < M && col < N) {
        float sum = 0.0f;
        // take the row of A and column of B to compute the dot product
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        // the result is written to C at (row, col)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/*
 * Analysis:
 * - Low compute-to-memory ratio: each iteration needs 1 FMA and 2 global memory reads
 * - High memory latency: direct global memory access
 * - Each element read multiple times (2*K*M*N total reads)
 */
