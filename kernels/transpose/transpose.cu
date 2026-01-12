/**
 * Matrix transpose kernels with various optimizations:
 * - v0: Naive implementation
 * - v1: Coalesced write (swap row/col indexing)
 * - v2: Coalesced write + __ldg() for cached read
 * - v3: Shared memory tiling (has bank conflicts)
 * - v4: Shared memory tiling + padding (no bank conflicts)
 * - v5: Shared memory tiling + swizzling (no bank conflicts, no extra memory)
 */

#include <cuda_runtime.h>

// Naive implementation - non-coalesced writes
__global__ void transpose_v0(const float* input, float* output, int M, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

// Coalesced memory write - swap grid indexing for coalesced output writes
__global__ void transpose_v1(const float* input, float* output, int M, int N) {
    // Swap: threadIdx.x runs along columns for coalesced writes
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

// Coalesced write + __ldg() for cached read
__global__ void transpose_v2(const float* input, float* output, int M, int N) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        output[col * M + row] = __ldg(&input[row * N + col]);
    }
}

// Shared memory tiling - has bank conflicts
template<int TILE_DIM>
__global__ void transpose_v3(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    // Input tile position
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile from input (coalesced read)
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();
    
    // Output tile position (transposed block)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile to output (coalesced write, but bank conflicts on read)
    if (x < M && y < N) {
        // Reading tile[threadIdx.x][threadIdx.y] causes 32-way bank conflicts
        output[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Shared memory tiling + padding to avoid bank conflicts
template<int TILE_DIM>
__global__ void transpose_v4(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding avoids bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < M && y < N) {
        output[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Shared memory tiling + swizzling to avoid bank conflicts
// Works best when TILE_DIM is a power of 2
template<int TILE_DIM>
__global__ void transpose_v5(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // XOR swizzling: different threads access different banks
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y * N + x];
    }
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < M && y < N) {
        output[y * M + x] = tile[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}

// Explicit template instantiations for common tile sizes
template __global__ void transpose_v3<32>(const float*, float*, int, int);
template __global__ void transpose_v4<32>(const float*, float*, int, int);
template __global__ void transpose_v5<32>(const float*, float*, int, int);
