// SGEMM v7: Double Buffering
// Prefetch next tile while computing current tile
// Overlaps memory access with computation

#include <cuda_runtime.h>

// Helper macros
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v7(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C) {

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread position
    constexpr int block_row_threads = BN / TN;
    constexpr int block_col_threads = BM / TM;
    
    int tx = (threadIdx.x % block_row_threads) * TN;
    int ty = (threadIdx.x / block_row_threads) * TM;

    // Double-buffered shared memory
    __shared__ float s_A[2][BK][BM];  // Transposed
    __shared__ float s_B[2][BK][BN];
    
    // Vectorized loading parameters
    constexpr int ldg_a_num = BK * BM / ((BN / TN) * (BM / TM)) / 4;
    constexpr int ldg_b_num = BK * BN / ((BN / TN) * (BM / TM)) / 4;

    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = (threadIdx.x % (BK / 4)) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
    int b_tile_stride = BK / ldg_b_num;

    // Registers
    float accum[TM][TN] = {0.0f};
    float ldg_a_reg[4 * ldg_a_num];
    float ldg_b_reg[4 * ldg_b_num];
    float a_frag[2][TM];
    float b_frag[2][TN];

    // Move pointers
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // Load first tile to buffer 0
    #pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
            FETCH_FLOAT4_CONST(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        s_A[0][a_tile_col][i + a_tile_row] = ldg_a_reg[ldg_index];
        s_A[0][a_tile_col + 1][i + a_tile_row] = ldg_a_reg[ldg_index + 1];
        s_A[0][a_tile_col + 2][i + a_tile_row] = ldg_a_reg[ldg_index + 2];
        s_A[0][a_tile_col + 3][i + a_tile_row] = ldg_a_reg[ldg_index + 3];
    }
    #pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(s_B[0][b_tile_row + i][b_tile_col]) =
            FETCH_FLOAT4_CONST(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }
    __syncthreads();

    int write_stage = 1;
    int load_stage = 0;
    
    // Main loop
    for (int k = BK; k <= K; k += BK) {
        // Prefetch next tile (if not last iteration)
        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                    FETCH_FLOAT4_CONST(A[OFFSET(a_tile_row + i, a_tile_col + k, K)]);
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                    FETCH_FLOAT4_CONST(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            }
        }

        // Load first fragment from shared memory
        #pragma unroll
        for (int m = 0; m < TM; m += 4) {
            FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(s_A[load_stage][0][ty + m]);
        }
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(s_B[load_stage][0][tx + n]);
        }

        // Compute with double-buffered registers
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {
            // Prefetch next fragment
            #pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = 
                    FETCH_FLOAT4(s_A[load_stage][bk + 1][ty + m]);
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = 
                    FETCH_FLOAT4(s_B[load_stage][bk + 1][tx + n]);
            }
            
            // Compute current fragment
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }

        // Compute last fragment of current tile
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }

        // Store prefetched data to next buffer
        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;
                s_A[write_stage][a_tile_col][i + a_tile_row] = ldg_a_reg[ldg_index];
                s_A[write_stage][a_tile_col + 1][i + a_tile_row] = ldg_a_reg[ldg_index + 1];
                s_A[write_stage][a_tile_col + 2][i + a_tile_row] = ldg_a_reg[ldg_index + 2];
                s_A[write_stage][a_tile_col + 3][i + a_tile_row] = ldg_a_reg[ldg_index + 3];
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(s_B[write_stage][b_tile_row + i][b_tile_col]) =
                    FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            
            // Swap buffers
            write_stage ^= 1;
            load_stage ^= 1;
        }
        __syncthreads();
    }

    // Write results with vectorized stores
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
template __global__ void sgemm_v7<128, 128, 8, 8, 8>(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C);
