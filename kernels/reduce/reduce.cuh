#pragma once
#include <cuda_runtime.h>

// Sum reductions (various implementations in reduce_sum.cu)
__global__ void sum_v0(float* X, float* Y);
__global__ void sum_v2(const float* X, float* Y, int n);
__global__ void sum_v3(const float* X, float* Y, int n);
__global__ void sum_v4(const float* X, float* Y, int n);
__global__ void sum_v5(const float* X, float* Y, int n);

// Max reduction (reduce_max.cu)
__global__ void max_kernel(float* input, float* output, int N);

// Softmax (vector version in reduce_softmax.cu)
__global__ void setToNegativeMax(float* d_value);
__global__ void sum_kernel(float* input, float* sum, float* max_val, int N);
__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, int N);

// Note: sum_kernel computes sum of exp(input[i] - max_val) internally

// Softmax (matrix versions in reduce_softmax_matrix.cu)
__global__ void softmax_row_kernel(float* input, float* output, int M, int N);
__global__ void softmax_row_kernel_shfl_xor(float* input, float* output, int M, int N);
__global__ void softmax_col_kernel_shfl_xor(float* input, float* output, int M, int N);
