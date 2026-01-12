#include <cuda_runtime.h>
#include <math.h>

#include "elementwise_add.cuh"

// Vectorized float4 addition; assumes A/B/C are 16-byte aligned.
__global__ void elementwise_add_float4(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int n4) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4;
    if (idx >= n4) return;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    float4 a = A4[tid];
    float4 b = B4[tid];

    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;

    C4[tid] = c;
}

// Scalar tail for elements that are not a multiple of 4.
__global__ void elementwise_add_scalar(const float* A,
                                       const float* B,
                                       float* C,
                                       int n,
                                       int offset) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
