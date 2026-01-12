// Softmax kernels for 1D vector
// Uses a 3-pass approach: 1) find max, 2) compute sum of exp(x-max), 3) normalize

#include <cuda_runtime.h>
#include <cfloat>

// Helper kernel to initialize max value
__global__ void setToNegativeMax(float* d_value) {
    *d_value = -FLT_MAX;
}

// Sum kernel for softmax denominator: computes sum of exp(input[i] - max_val)
// Note: This kernel expects input to already have exp(x - max) applied, OR
// you can modify to compute exp internally
__global__ void sum_kernel(float* input, float* sum, float* max_val, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // Load and compute exp(input - max) for numerical stability
    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    if (laneId == 0) s_mem[warp_id] = val;
    __syncthreads();

    // Block-level reduction in first warp
    if (warp_id == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (laneId == 0) atomicAdd(sum, val);
    }
}

// Softmax normalization kernel: output[i] = exp(input[i] - max_val) / sum
__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        output[idx] = expf(input[idx] - *max_val) / (*sum);
    }
}
