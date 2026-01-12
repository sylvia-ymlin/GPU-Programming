// SGEMM v4: 2D Thread Tiling
// Each thread computes a TM x TN tile of the output
// Better register utilization

#include <cuda_runtime.h>

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v4(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C) {
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate thread position within the block
    int block_row_threads = BN / TN;
    int block_col_threads = BM / TM;
    int thread_num = block_row_threads * block_col_threads;

    // Thread tile position
    int tx = (threadIdx.x % block_row_threads) * TN;  // column offset
    int ty = (threadIdx.x / block_row_threads) * TM;  // row offset

    // Shared memory for tiles
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    // Thread info for loading A
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    // Thread info for loading B  
    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    // Register array for output (TM x TN per thread)
    float tmp[TM][TN] = {0.0f};

    // Loop over K dimension in tiles
    for (int k = 0; k < K; k += BK) {
        // Load A tile to shared memory
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int a_row = a_tile_row + i;
            if (a_row < BM) {
                int global_row = by * BM + a_row;
                int global_col = k + a_tile_col;
                if (global_row < M && global_col < K) {
                    s_A[a_row][a_tile_col] = A[global_row * K + global_col];
                } else {
                    s_A[a_row][a_tile_col] = 0.0f;
                }
            }
        }
        
        // Load B tile to shared memory
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            int b_row = b_tile_row + i;
            if (b_row < BK) {
                int global_row = k + b_row;
                int global_col = bx * BN + b_tile_col;
                if (global_row < K && global_col < N) {
                    s_B[b_row][b_tile_col] = B[global_row * N + global_col];
                } else {
                    s_B[b_row][b_tile_col] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute partial results for TM x TN tile
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    tmp[m][n] += s_A[ty + m][i] * s_B[i][tx + n];
                }
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int global_row = by * BM + ty + m;
            int global_col = bx * BN + tx + n;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = alpha * tmp[m][n] + beta * C[global_row * N + global_col];
            }
        }
    }
}

// Explicit template instantiation
template __global__ void sgemm_v4<64, 64, 8, 8, 8>(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C);

template __global__ void sgemm_v4<128, 128, 8, 8, 8>(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C);
