/**
 * Matrix softmax kernels
 * - Row-wise softmax: compute softmax for each row of M x N matrix
 * - Column-wise softmax: compute softmax for each column
 * 
 * Uses warp-level primitives for efficient reduction within rows/columns.
 */

#include <cuda_runtime.h>
#include <cfloat>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Row-wise softmax using shared memory for warp reduction results
// One block per row, uses __shfl_down_sync
__global__ void softmax_row_kernel(float* input, float* output, int M, int N) {
    __shared__ float s_sum;
    __shared__ float s_max;

    int laneId = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    int num_warps = blockDim.x / warpSize;
    
    int row = blockIdx.x;
    if (row >= M) return;

    // Each thread handles multiple elements
    int iterations = CEIL_DIV(N, blockDim.x);
    
    // Step 1: Find max value in this row
    float max_val = -FLT_MAX;
    for (int i = 0; i < iterations; i++) {
        int col = i * blockDim.x + threadIdx.x;
        if (col < N) {
            max_val = fmaxf(max_val, input[row * N + col]);
        }
    }
    
    // Warp-level reduction for max
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    // First thread of each warp writes to shared memory
    __shared__ float warp_max[32];
    if (laneId == 0) warp_max[warp_id] = max_val;
    __syncthreads();
    
    // First warp reduces across all warps
    if (warp_id == 0) {
        max_val = (laneId < num_warps) ? warp_max[laneId] : -FLT_MAX;
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (laneId == 0) s_max = max_val;
    }
    __syncthreads();
    max_val = s_max;

    // Step 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = 0; i < iterations; i++) {
        int col = i * blockDim.x + threadIdx.x;
        if (col < N) {
            sum += expf(input[row * N + col] - max_val);
        }
    }
    
    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    __shared__ float warp_sum[32];
    if (laneId == 0) warp_sum[warp_id] = sum;
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (laneId < num_warps) ? warp_sum[laneId] : 0.0f;
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (laneId == 0) s_sum = sum;
    }
    __syncthreads();
    sum = s_sum;

    // Step 3: Normalize
    for (int i = 0; i < iterations; i++) {
        int col = i * blockDim.x + threadIdx.x;
        if (col < N) {
            output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
        }
    }
}

// Row-wise softmax using __shfl_xor_sync (all threads get the reduced value)
// More efficient as it avoids shared memory for reduction broadcast
__global__ void softmax_row_kernel_shfl_xor(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    int num_warps = blockDim.x / warpSize;
    
    int row = blockIdx.x;
    if (row >= M) return;

    int iterations = CEIL_DIV(N, blockDim.x);
    
    // Step 1: Find max
    float max_val = -FLT_MAX;
    for (int i = 0; i < iterations; i++) {
        int col = i * blockDim.x + threadIdx.x;
        if (col < N) {
            max_val = fmaxf(max_val, input[row * N + col]);
        }
    }
    
    // Warp reduction with xor - all threads get the result
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    
    // Cross-warp reduction via shared memory
    __shared__ float warp_max[32];
    if (laneId == 0) warp_max[warp_id] = max_val;
    __syncthreads();
    
    // All threads read and reduce
    if (num_warps > 1) {
        max_val = -FLT_MAX;
        for (int i = 0; i < num_warps; i++) {
            max_val = fmaxf(max_val, warp_max[i]);
        }
    }

    // Step 2: Compute sum
    float sum = 0.0f;
    for (int i = 0; i < iterations; i++) {
        int col = i * blockDim.x + threadIdx.x;
        if (col < N) {
            sum += expf(input[row * N + col] - max_val);
        }
    }
    
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    
    __shared__ float warp_sum[32];
    if (laneId == 0) warp_sum[warp_id] = sum;
    __syncthreads();
    
    if (num_warps > 1) {
        sum = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            sum += warp_sum[i];
        }
    }

    // Step 3: Normalize
    for (int i = 0; i < iterations; i++) {
        int col = i * blockDim.x + threadIdx.x;
        if (col < N) {
            output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
        }
    }
}

// Column-wise softmax using __shfl_xor_sync
// One block per column
__global__ void softmax_col_kernel_shfl_xor(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    int num_warps = blockDim.x / warpSize;
    
    int col = blockIdx.x;
    if (col >= N) return;

    int iterations = CEIL_DIV(M, blockDim.x);
    
    // Step 1: Find max in this column
    float max_val = -FLT_MAX;
    for (int i = 0; i < iterations; i++) {
        int row = i * blockDim.x + threadIdx.x;
        if (row < M) {
            max_val = fmaxf(max_val, input[row * N + col]);
        }
    }
    
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    
    __shared__ float warp_max[32];
    if (laneId == 0) warp_max[warp_id] = max_val;
    __syncthreads();
    
    if (num_warps > 1) {
        max_val = -FLT_MAX;
        for (int i = 0; i < num_warps; i++) {
            max_val = fmaxf(max_val, warp_max[i]);
        }
    }

    // Step 2: Compute sum
    float sum = 0.0f;
    for (int i = 0; i < iterations; i++) {
        int row = i * blockDim.x + threadIdx.x;
        if (row < M) {
            sum += expf(input[row * N + col] - max_val);
        }
    }
    
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    
    __shared__ float warp_sum[32];
    if (laneId == 0) warp_sum[warp_id] = sum;
    __syncthreads();
    
    if (num_warps > 1) {
        sum = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            sum += warp_sum[i];
        }
    }

    // Step 3: Normalize
    for (int i = 0; i < iterations; i++) {
        int row = i * blockDim.x + threadIdx.x;
        if (row < M) {
            output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
        }
    }
}
