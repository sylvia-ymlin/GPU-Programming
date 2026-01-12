#pragma once
#include <cuda_runtime.h>

__global__ void sigmoid_float4(const float4* X, float4* Y, int n4);
__global__ void relu_float4(const float4* X, float4* Y, int n4);
