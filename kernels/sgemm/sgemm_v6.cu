// SGEMM v6: Vectorized Memory Access
// Uses float4 for coalesced 128-bit loads/stores
// Transpose A in shared memory for better access patterns

#include <cuda_runtime.h>

// Helper macros for vectorized access
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v6(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C) {

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread position within block tile
    constexpr int block_row_threads = BN / TN;
    constexpr int block_col_threads = BM / TM;

    int tx = (threadIdx.x % block_row_threads) * TN;
    int ty = (threadIdx.x / block_row_threads) * TM;
    
    // Shared memory - A is transposed for coalesced access
    __shared__ float s_A[BK][BM];  // Transposed
    __shared__ float s_B[BK][BN];

    // Vectorized loading parameters
    // Each thread loads 4 floats at a time
    constexpr int ldg_a_num = BK * BM / ((BN / TN) * (BM / TM)) / 4;
    constexpr int ldg_b_num = BK * BN / ((BN / TN) * (BM / TM)) / 4;

    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = (threadIdx.x % (BK / 4)) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
    int b_tile_stride = BK / ldg_b_num;

    // Accumulators in registers
    float accum[TM][TN] = {0.0f};
    
    // Register caches
    float ldg_a_reg[4 * ldg_a_num];
    float a_frag[TM];
    float b_frag[TN];

    // Move pointers to starting position
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // Loop over K dimension
    for (int k = 0; k < K; k += BK) {
        // Load A with transpose into shared memory
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;
            // Load 4 floats from global A (use const version for reads)
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = 
                FETCH_FLOAT4_CONST(A[OFFSET(a_tile_row + i, a_tile_col + k, K)]);
            // Store transposed to shared memory
            s_A[a_tile_col][i + a_tile_row] = ldg_a_reg[ldg_index];
            s_A[a_tile_col + 1][i + a_tile_row] = ldg_a_reg[ldg_index + 1];
            s_A[a_tile_col + 2][i + a_tile_row] = ldg_a_reg[ldg_index + 2];
            s_A[a_tile_col + 3][i + a_tile_row] = ldg_a_reg[ldg_index + 3];
        }
    
        // Load B directly (no transpose needed)
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(s_B[b_tile_row + i][b_tile_col]) = 
                FETCH_FLOAT4_CONST(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
        }
        __syncthreads();

        // Compute with register caching
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            // Load fragments to registers
            #pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(s_A[i][ty + m]);
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(s_B[i][tx + n]);
            }
            
            // Outer product
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }

    // Write results back with vectorized stores
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

// Explicit template instantiation
template __global__ void sgemm_v6<128, 128, 8, 8, 8>(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C);
