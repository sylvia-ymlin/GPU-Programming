#include <cuda_runtime.h>
#include <math.h>

#include "activations.cuh"

// Sigmoid over float4, n4 must be multiple of 4.
__global__ void sigmoid_float4(const float4* X, float4* Y, int n4) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4;
    if (idx >= n4) return;

    const float4* X4 = reinterpret_cast<const float4*>(X);
    float4* Y4 = reinterpret_cast<float4*>(Y);

    float4 x = X4[tid];
    float4 y;
    y.x = 1.0f / (1.0f + expf(-x.x));
    y.y = 1.0f / (1.0f + expf(-x.y));
    y.z = 1.0f / (1.0f + expf(-x.z));
    y.w = 1.0f / (1.0f + expf(-x.w));
    Y4[tid] = y;
}

// ReLU over float4, n4 must be multiple of 4.
__global__ void relu_float4(const float4* X, float4* Y, int n4) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4;
    if (idx >= n4) return;

    const float4* X4 = reinterpret_cast<const float4*>(X);
    float4* Y4 = reinterpret_cast<float4*>(Y);

    float4 x = X4[tid];
    float4 y;
    y.x = max(0.0f, x.x);
    y.y = max(0.0f, x.y);
    y.z = max(0.0f, x.z);
    y.w = max(0.0f, x.w);
    Y4[tid] = y;
}
