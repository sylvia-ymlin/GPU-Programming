#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(a) *(reinterpret_cast<const float4 *>(&(a)))

// Naive single-block reduction; used for quick sanity tests
__global__ void sum_v0(float *X, float *Y) {
    __shared__ float s_y[256];
    const int tid = threadIdx.x;

    s_y[tid] = X[tid];
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }
    if (tid == 0) Y[blockIdx.x] = s_y[0];
}

// Shared-memory reduction (static shared)
template <int BLOCK>
__global__ void sum_v1(const float *X, float *Y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * BLOCK + tid;
    __shared__ float s_y[BLOCK];

    s_y[tid] = (idx < N) ? X[idx] : 0.0f;
    __syncthreads();

    for (int offset = BLOCK >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }
    if (tid == 0) Y[blockIdx.x] = s_y[0];
}

template <int BLOCK>
void call_sum_v1(const float *d_x, float *d_y, float *h_y, int N, float *sum) {
    const int GRID = CEIL(N, BLOCK);
    sum_v1<BLOCK><<<GRID, BLOCK>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    *sum = 0.0f;
    for (int i = 0; i < GRID; ++i) *sum += h_y[i];
}

// Dynamic shared-memory reduction
__global__ void sum_v2(const float *X, float *Y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    extern __shared__ float s_y[];

    s_y[tid] = (idx < N) ? X[idx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }
    if (tid == 0) Y[blockIdx.x] = s_y[0];
}

template <int BLOCK>
void call_sum_v2(const float *d_x, float *d_y, float *h_y, int N, float *sum) {
    const int GRID = CEIL(N, BLOCK);
    sum_v2<<<GRID, BLOCK, BLOCK * sizeof(float)>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    *sum = 0.0f;
    for (int i = 0; i < GRID; ++i) *sum += h_y[i];
}

// Atomic add per-block result (dynamic shared)
__global__ void sum_v3(const float *X, float *Y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    extern __shared__ float s_y[];

    s_y[tid] = (idx < N) ? X[idx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(Y, s_y[0]);
}

template <int BLOCK>
void call_sum_v3(const float *d_x, float *d_y, float *h_y, int N) {
    const int GRID = CEIL(N, BLOCK);
    cudaMemset(d_y, 0, sizeof(float));
    sum_v3<<<GRID, BLOCK, BLOCK * sizeof(float)>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
}

// Warp-shuffle reduction producing per-block partials
__global__ void sum_v4(const float *X, float *Y, int N) {
    __shared__ float s_y[32];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    float val = (idx < N) ? X[idx] : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) s_y[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        int warp_num = blockDim.x / warpSize;
        val = (lane < warp_num) ? s_y[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) Y[blockIdx.x] = val;
    }
}

template <int BLOCK>
void call_sum_v4(const float *d_x, float *d_y, float *h_y, int N, float *sum) {
    const int GRID = CEIL(N, BLOCK);
    sum_v4<<<GRID, BLOCK>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    *sum = 0.0f;
    for (int i = 0; i < GRID; ++i) *sum += h_y[i];
}

// float4 vectorized load + warp shuffle, per-block partials
__global__ void sum_v5(const float *X, float *Y, int N) {
    __shared__ float s_y[32];
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int warp_id = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    float val = 0.0f;
    if (idx < N) {
        float4 v = FLOAT4(X[idx]);
        val = v.x + v.y + v.z + v.w;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) s_y[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        int warp_num = blockDim.x / warpSize;
        val = (lane < warp_num) ? s_y[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) Y[blockIdx.x] = val;
    }
}

template <int BLOCK>
void call_sum_v5(const float *d_x, float *d_y, float *h_y, int N, float *sum) {
    const int GRID = CEIL(N, BLOCK * 4);
    sum_v5<<<GRID, BLOCK>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    *sum = 0.0f;
    for (int i = 0; i < GRID; ++i) *sum += h_y[i];
}
