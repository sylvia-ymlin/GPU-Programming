// SGEMM v2: Shared memory tiling
// Use shared memory to reduce global memory access
// Tile size matches block size

#include <cuda_runtime.h>

template<const int TILE_SIZE>
__global__ void sgemm_v2(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C) {

    // block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // allocate shared memory for the tile
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float temp = 0.0f;
    
    // loop over tiles
    for (int k = 0; k < K; k += TILE_SIZE) {
        // load data to shared memory with bounds checking
        if (row < M && (k + tx) < K) {
            s_A[ty][tx] = A[row * K + k + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        if ((k + ty) < K && col < N) {
            s_B[ty][tx] = B[(k + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }
        __syncthreads();

        // calculate the dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            temp += s_A[ty][i] * s_B[i][tx];
        }
        __syncthreads();
    }

    // write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
    }
}

// Explicit template instantiations
template __global__ void sgemm_v2<16>(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C);

template __global__ void sgemm_v2<32>(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C);

/*
 * Analysis:
 * - Global memory reads reduced by factor of TILE_SIZE
 * - Each element in shared memory used TILE_SIZE times
 * - Block size limited by shared memory availability
 */
