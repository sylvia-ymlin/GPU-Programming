// SGEMM v3: 1D Thread Tiling
// Each thread computes TM elements (a column of the output tile)
// Uses registers to cache intermediate results

#include <cuda_runtime.h>

template<const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_v3(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C) {
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread responsible for TM elements in a column
    int tx = threadIdx.x % BN;           // column within block
    int ty = (threadIdx.x / BN) * TM;    // starting row within block

    // Shared memory for tiles
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    // Thread info for loading A
    int thread_num = (BM * BN) / TM;
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    // Thread info for loading B
    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    // Register array for output (each thread computes TM elements)
    float tmp[TM] = {0.0f};

    // Loop over K dimension in tiles
    for (int k = 0; k < K; k += BK) {
        // Load A tile to shared memory
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int a_row = a_tile_row + i;
            if (by * BM + a_row < M && k + a_tile_col < K) {
                s_A[a_row][a_tile_col] = A[(by * BM + a_row) * K + k + a_tile_col];
            } else {
                s_A[a_row][a_tile_col] = 0.0f;
            }
        }
        
        // Load B tile to shared memory
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            int b_row = b_tile_row + i;
            if (k + b_row < K && bx * BN + b_tile_col < N) {
                s_B[b_row][b_tile_col] = B[(k + b_row) * N + bx * BN + b_tile_col];
            } else {
                s_B[b_row][b_tile_col] = 0.0f;
            }
        }
        __syncthreads();

        // Compute partial results
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            float b_val = s_B[i][tx];
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                tmp[j] += s_A[ty + j][i] * b_val;
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        int global_row = by * BM + ty + j;
        int global_col = bx * BN + tx;
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = alpha * tmp[j] + beta * C[global_row * N + global_col];
        }
    }
}

// Explicit template instantiation
template __global__ void sgemm_v3<64, 64, 8, 8>(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C);

template __global__ void sgemm_v3<128, 128, 8, 8>(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C);
