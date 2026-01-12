#pragma once
#include <cuda_runtime.h>

// Naive transpose
__global__ void transpose_v0(const float* input, float* output, int M, int N);

// Coalesced write
__global__ void transpose_v1(const float* input, float* output, int M, int N);

// Coalesced write + __ldg()
__global__ void transpose_v2(const float* input, float* output, int M, int N);

// Shared memory tiling (has bank conflicts)
template<int TILE_DIM>
__global__ void transpose_v3(const float* input, float* output, int M, int N);

// Shared memory tiling + padding (no bank conflicts)
template<int TILE_DIM>
__global__ void transpose_v4(const float* input, float* output, int M, int N);

// Shared memory tiling + swizzling (no bank conflicts)
template<int TILE_DIM>
__global__ void transpose_v5(const float* input, float* output, int M, int N);

