#pragma once

#include <cuda_runtime.h>

// SGEMM: Single-precision General Matrix Multiply
// C = α*(A@B)+β*C where A is MxK, B is KxN, C is MxN

// v1: Naive implementation - one thread per output element
__global__ __launch_bounds__(1024)
void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                 const float *B, float beta, float *C);

// v2: Shared memory tiling - reduces global memory access
template<const int TILE_SIZE>
__global__ void sgemm_v2(int M, int N, int K, 
    float alpha, const float *A, const float *B, float beta, float *C);

// v3: 1D thread tiling - each thread computes TM elements
template<const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_v3(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C);

// v4: 2D thread tiling - each thread computes TM x TN elements
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v4(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C);

// v5: 2D tiling with register caching for A and B fragments
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v5(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C);

// v6: Vectorized memory access (float4) with transposed A in shared memory
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v6(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C);

// v7: Double buffering - overlaps memory access with computation
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v7(int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C);

