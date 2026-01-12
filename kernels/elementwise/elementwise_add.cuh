#pragma once
#include <cuda_runtime.h>

__global__ void elementwise_add_float4(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int n4);

__global__ void elementwise_add_scalar(const float* A,
                                       const float* B,
                                       float* C,
                                       int n,
                                       int offset);
