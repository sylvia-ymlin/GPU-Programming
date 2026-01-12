/**
 * v8.cu - Pure FP16 Implementation
 * 
 * Key changes from v6:
 * 1. All weights, activations, gradients in FP16 (__half)
 * 2. cublasGemmEx with CUBLAS_COMPUTE_16F for Tensor Core acceleration
 * 3. Half intrinsics for element-wise operations (__hadd, __hmul, etc.)
 * 4. Pre-converted FP16 training data
 * 
 * T4 (Turing SM 7.5) supports:
 * - FP16 Tensor Cores (FP16 accumulation)
 * - Native half precision CUDA operations
 * 
 * Expected: ~2x memory bandwidth reduction, similar or better speed
 * Trade-off: Slight precision loss (acceptable for MNIST)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half, __half2
#include <cublas_v2.h>

typedef struct {
    double h2d_submit;       // Time to call cudaMemcpyAsync (CPU returns immediately)
    double kernel_launch;    // Time to call kernel<<<>>> and cuBLAS (CPU returns immediately)
    double stream_sync;      // Time in cudaStreamSynchronize (actual GPU execution)
    double total_time;
} TimingStats;

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.01f
#define NUM_STREAMS 2

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// ============================================================
// [NEW in v8] All buffers are FP16 (__half)
// v6: float *d_weights1
// v8: __half *d_weights1
// ============================================================
typedef struct {
    // FP16 weights and gradients
    __half *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    __half *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    
    // Double-buffered FP16 activations
    __half *d_fc1_output[NUM_STREAMS];
    __half *d_fc2_output[NUM_STREAMS];
    __half *d_grad_hidden[NUM_STREAMS];
    __half *d_grad_output[NUM_STREAMS];
    __half *d_input_batch[NUM_STREAMS];
    
    // Labels stay int, loss stays float for reduction
    int *d_labels[NUM_STREAMS];
    float *d_loss[NUM_STREAMS];
    
    // FP32 buffers for bias gradient accumulation (atomicAdd compatibility)
    float *d_grad_bias1_fp32, *d_grad_bias2_fp32;

    cudaStream_t streams[NUM_STREAMS];
    cublasHandle_t cublas_handle;
} NeuralNetworkCUDA;

void load_data(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen data"); exit(EXIT_FAILURE); }
    if (fread(data, sizeof(float), size, f) != (size_t)size) {
        perror("fread data"); exit(EXIT_FAILURE);
    }
    fclose(f);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen labels"); exit(EXIT_FAILURE); }
    if (fread(labels, sizeof(int), size, f) != (size_t)size) {
        perror("fread labels"); exit(EXIT_FAILURE);
    }
    fclose(f);
}

// ============================================================
// [NEW in v8] Convert FP32 data to FP16 on host
// ============================================================
void convert_to_fp16(float *src, __half *dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2half(src[i]);
    }
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void initialize_weights_host_fp16(__half *weights, int rows, int cols) {
    float scale = sqrtf(2.0f / rows);
    for (int i = 0; i < rows * cols; i++) {
        float val = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
        weights[i] = __float2half(val);
    }
}

void initialize_bias_host_fp16(__half *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = __float2half(0.0f);
    }
}

// ============================================================
// [NEW in v8] FP16 Fused bias + ReLU kernel
// Note: Use float comparison for portability (half comparison varies by CUDA version)
// ============================================================
__global__ void bias_add_relu_fp16_kernel(__half *x, __half *bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        float val = __half2float(x[idx]) + __half2float(bias[bias_idx]);
        // ReLU: max(0, val)
        x[idx] = __float2half(fmaxf(0.0f, val));
    }
}

// Bias add only (no ReLU) for output layer
__global__ void bias_add_fp16_kernel(__half *x, __half *bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        x[idx] = __hadd(x[idx], bias[bias_idx]);
    }
}

// ============================================================
// [NEW in v8] FP16 ReLU backward
// ============================================================
__global__ void relu_backward_fp16_kernel(__half *grad, __half *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        // grad *= (x > 0) ? 1 : 0
        float x_val = __half2float(x[idx]);
        grad[idx] = (x_val > 0.0f) ? grad[idx] : __float2half(0.0f);
    }
}

// ============================================================
// [NEW in v8] FP16 bias backward - accumulate to float buffer
// Note: atomicAdd for __half has limited support; use float accumulation
// ============================================================
__global__ void bias_backward_fp16_kernel(__half *grad_output, float *grad_bias_fp32, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias_fp32[bias_idx], __half2float(grad_output[idx]));
    }
}

// Convert FP32 bias gradient back to FP16
__global__ void convert_bias_grad_kernel(float *grad_bias_fp32, __half *grad_bias_fp16, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_bias_fp16[idx] = __float2half(grad_bias_fp32[idx]);
    }
}

// ============================================================
// [NEW in v8] FP16 Softmax + Cross-entropy + Backward
// Compute in FP32 for numerical stability, store grad as FP16
// ============================================================
__global__ void softmax_cross_entropy_backward_fp16_kernel(
    __half *logits, int *labels, __half *grad_output, float *loss_per_sample,
    int batch_size, int num_classes
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    float *sample_logits = shared;
    int tid = threadIdx.x;

    // Load FP16 logits to FP32 shared memory for stability
    if (tid < num_classes) {
        sample_logits[tid] = __half2float(logits[tid + b * num_classes]);
    }
    __syncthreads();

    __shared__ float max_logit;
    __shared__ float sum_exp;

    if (tid == 0) {
        max_logit = sample_logits[0];
        for (int i = 1; i < num_classes; i++) {
            if (sample_logits[i] > max_logit) max_logit = sample_logits[i];
        }
    }
    __syncthreads();

    if (tid < num_classes) {
        sample_logits[tid] = expf(sample_logits[tid] - max_logit);
    }
    __syncthreads();

    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += sample_logits[i];
        }
    }
    __syncthreads();

    if (tid < num_classes) {
        float prob = sample_logits[tid] / sum_exp;
        int label = labels[b];
        float grad = prob;
        if (tid == label) {
            grad -= 1.0f;
        }
        grad /= (float)batch_size;
        // Convert gradient back to FP16
        grad_output[tid + b * num_classes] = __float2half(grad);

        if (tid == label) {
            loss_per_sample[b] = -logf(fmaxf(prob, 1e-7f));
        }
    }
}

// ============================================================
// [NEW in v8] Forward pass with cublasGemmEx (FP16)
// ============================================================
void forward_pass_stream(NeuralNetworkCUDA *nn, int stream_idx, int batch_size) {
    cudaStream_t stream = nn->streams[stream_idx];
    CUBLAS_CHECK(cublasSetStream(nn->cublas_handle, stream));
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    // FC1: Y = W1 @ X (all FP16, CUBLAS_COMPUTE_16F for Tensor Cores)
    CUBLAS_CHECK(cublasGemmEx(nn->cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HIDDEN_SIZE, batch_size, INPUT_SIZE,
        &alpha,
        nn->d_weights1, CUDA_R_16F, HIDDEN_SIZE,
        nn->d_input_batch[stream_idx], CUDA_R_16F, INPUT_SIZE,
        &beta,
        nn->d_fc1_output[stream_idx], CUDA_R_16F, HIDDEN_SIZE,
        CUBLAS_COMPUTE_16F,  // FP16 compute (uses Tensor Cores on T4)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Fused bias + ReLU
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255) / 256;
    bias_add_relu_fp16_kernel<<<grid_hidden, 256, 0, stream>>>(
        nn->d_fc1_output[stream_idx], nn->d_bias1, batch_size, HIDDEN_SIZE);

    // FC2: Y = W2 @ H
    CUBLAS_CHECK(cublasGemmEx(nn->cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
        &alpha,
        nn->d_weights2, CUDA_R_16F, OUTPUT_SIZE,
        nn->d_fc1_output[stream_idx], CUDA_R_16F, HIDDEN_SIZE,
        &beta,
        nn->d_fc2_output[stream_idx], CUDA_R_16F, OUTPUT_SIZE,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Bias add (no ReLU)
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_add_fp16_kernel<<<grid_out, 256, 0, stream>>>(
        nn->d_fc2_output[stream_idx], nn->d_bias2, batch_size, OUTPUT_SIZE);
}

// ============================================================
// [NEW in v8] Backward pass with cublasGemmEx (FP16)
// ============================================================
void backward_pass_stream(NeuralNetworkCUDA *nn, int stream_idx, int batch_size) {
    cudaStream_t stream = nn->streams[stream_idx];
    CUBLAS_CHECK(cublasSetStream(nn->cublas_handle, stream));
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    // Zero gradients (FP16 for weights, FP32 for bias accumulation)
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(__half), stream));
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_weights2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__half), stream));
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_bias1_fp32, 0, HIDDEN_SIZE * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_bias2_fp32, 0, OUTPUT_SIZE * sizeof(float), stream));

    // dW2 = grad_output @ hidden.T
    CUBLAS_CHECK(cublasGemmEx(nn->cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
        &alpha,
        nn->d_grad_output[stream_idx], CUDA_R_16F, OUTPUT_SIZE,
        nn->d_fc1_output[stream_idx], CUDA_R_16F, HIDDEN_SIZE,
        &beta,
        nn->d_grad_weights2, CUDA_R_16F, OUTPUT_SIZE,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_backward_fp16_kernel<<<grid_out, 256, 0, stream>>>(
        nn->d_grad_output[stream_idx], nn->d_grad_bias2_fp32, batch_size, OUTPUT_SIZE);

    // grad_hidden = W2.T @ grad_output
    CUBLAS_CHECK(cublasGemmEx(nn->cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
        &alpha,
        nn->d_weights2, CUDA_R_16F, OUTPUT_SIZE,
        nn->d_grad_output[stream_idx], CUDA_R_16F, OUTPUT_SIZE,
        &beta,
        nn->d_grad_hidden[stream_idx], CUDA_R_16F, HIDDEN_SIZE,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // ReLU backward
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255) / 256;
    relu_backward_fp16_kernel<<<grid_hidden, 256, 0, stream>>>(
        nn->d_grad_hidden[stream_idx], nn->d_fc1_output[stream_idx], total_hidden);

    // dW1 = grad_hidden @ input.T
    CUBLAS_CHECK(cublasGemmEx(nn->cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN_SIZE, INPUT_SIZE, batch_size,
        &alpha,
        nn->d_grad_hidden[stream_idx], CUDA_R_16F, HIDDEN_SIZE,
        nn->d_input_batch[stream_idx], CUDA_R_16F, INPUT_SIZE,
        &beta,
        nn->d_grad_weights1, CUDA_R_16F, HIDDEN_SIZE,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    bias_backward_fp16_kernel<<<grid_hidden, 256, 0, stream>>>(
        nn->d_grad_hidden[stream_idx], nn->d_grad_bias1_fp32, batch_size, HIDDEN_SIZE);
    
    // Convert FP32 bias gradients back to FP16
    int grid_b1 = (HIDDEN_SIZE + 255) / 256;
    int grid_b2 = (OUTPUT_SIZE + 255) / 256;
    convert_bias_grad_kernel<<<grid_b1, 256, 0, stream>>>(nn->d_grad_bias1_fp32, nn->d_grad_bias1, HIDDEN_SIZE);
    convert_bias_grad_kernel<<<grid_b2, 256, 0, stream>>>(nn->d_grad_bias2_fp32, nn->d_grad_bias2, OUTPUT_SIZE);
}

// ============================================================
// [NEW in v8] FP16 weight update kernel using half intrinsics
// W = W - lr * dW
// ============================================================
__global__ void sgd_update_fp16_kernel(__half *weights, __half *grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // W -= lr * grad
        __half neg_lr_grad = __float2half(-lr * __half2float(grads[idx]));
        weights[idx] = __hadd(weights[idx], neg_lr_grad);
    }
}

void update_weights_stream(NeuralNetworkCUDA *nn, float lr, int stream_idx) {
    cudaStream_t stream = nn->streams[stream_idx];
    
    int block = 256;
    
    int grid1 = (INPUT_SIZE * HIDDEN_SIZE + block - 1) / block;
    sgd_update_fp16_kernel<<<grid1, block, 0, stream>>>(
        nn->d_weights1, nn->d_grad_weights1, lr, INPUT_SIZE * HIDDEN_SIZE);
    
    int grid2 = (HIDDEN_SIZE * OUTPUT_SIZE + block - 1) / block;
    sgd_update_fp16_kernel<<<grid2, block, 0, stream>>>(
        nn->d_weights2, nn->d_grad_weights2, lr, HIDDEN_SIZE * OUTPUT_SIZE);
    
    int grid_b1 = (HIDDEN_SIZE + block - 1) / block;
    sgd_update_fp16_kernel<<<grid_b1, block, 0, stream>>>(
        nn->d_bias1, nn->d_grad_bias1, lr, HIDDEN_SIZE);
    
    int grid_b2 = (OUTPUT_SIZE + block - 1) / block;
    sgd_update_fp16_kernel<<<grid_b2, block, 0, stream>>>(
        nn->d_bias2, nn->d_grad_bias2, lr, OUTPUT_SIZE);
}

float compute_loss_on_gpu_stream(NeuralNetworkCUDA *nn, int stream_idx, int batch_size, float *h_loss) {
    cudaStream_t stream = nn->streams[stream_idx];
    int shared_mem = OUTPUT_SIZE * sizeof(float);
    
    softmax_cross_entropy_backward_fp16_kernel<<<batch_size, 32, shared_mem, stream>>>(
        nn->d_fc2_output[stream_idx], nn->d_labels[stream_idx], 
        nn->d_grad_output[stream_idx], nn->d_loss[stream_idx],
        batch_size, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpyAsync(h_loss, nn->d_loss[stream_idx], 
                               batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    
    return 0.0f;
}

void initialize_random_weights_cuda(NeuralNetworkCUDA *nn) {
    __half *h_weights1 = (__half *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(__half));
    initialize_weights_host_fp16(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_weights1);

    __half *h_weights2 = (__half *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__half));
    initialize_weights_host_fp16(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_weights2);

    __half *h_bias1 = (__half *)malloc(HIDDEN_SIZE * sizeof(__half));
    initialize_bias_host_fp16(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_bias1);

    __half *h_bias2 = (__half *)malloc(OUTPUT_SIZE * sizeof(__half));
    initialize_bias_host_fp16(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, OUTPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_bias2);
}

void initialize_nn_cuda(NeuralNetworkCUDA *nn) {
    // FP16 weights and gradients
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(__half)));
    // FP32 buffers for bias gradient accumulation
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1_fp32, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2_fp32, OUTPUT_SIZE * sizeof(float)));

    // Double-buffered FP16 activations
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMalloc(&nn->d_fc1_output[i], BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&nn->d_fc2_output[i], BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden[i], BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&nn->d_grad_output[i], BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&nn->d_input_batch[i], BATCH_SIZE * INPUT_SIZE * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&nn->d_labels[i], BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&nn->d_loss[i], BATCH_SIZE * sizeof(float)));
        
        CUDA_CHECK(cudaStreamCreate(&nn->streams[i]));
    }

    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    
    // Enable Tensor Core math
    CUBLAS_CHECK(cublasSetMathMode(nn->cublas_handle, CUBLAS_TENSOR_OP_MATH));
    
    initialize_random_weights_cuda(nn);
}

void free_nn_cuda(NeuralNetworkCUDA *nn) {
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1_fp32));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2_fp32));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaFree(nn->d_fc1_output[i]));
        CUDA_CHECK(cudaFree(nn->d_fc2_output[i]));
        CUDA_CHECK(cudaFree(nn->d_grad_hidden[i]));
        CUDA_CHECK(cudaFree(nn->d_grad_output[i]));
        CUDA_CHECK(cudaFree(nn->d_input_batch[i]));
        CUDA_CHECK(cudaFree(nn->d_labels[i]));
        CUDA_CHECK(cudaFree(nn->d_loss[i]));
        CUDA_CHECK(cudaStreamDestroy(nn->streams[i]));
    }

    CUBLAS_CHECK(cublasDestroy(nn->cublas_handle));
}

// Evaluate model accuracy (convert weights to FP32 for CPU evaluation)
void evaluate(NeuralNetworkCUDA *nn, float *X_test, int *y_test) {
    int correct = 0;
    int num_batches = TEST_SIZE / BATCH_SIZE;
    
    float *hidden = (float *)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Copy FP16 weights to host and convert to FP32
    __half *h_W1_fp16 = (__half *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(__half));
    __half *h_W2_fp16 = (__half *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__half));
    __half *h_b1_fp16 = (__half *)malloc(HIDDEN_SIZE * sizeof(__half));
    __half *h_b2_fp16 = (__half *)malloc(OUTPUT_SIZE * sizeof(__half));
    
    float *h_W1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_W2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_b1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_b2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_W1_fp16, nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_W2_fp16, nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1_fp16, nn->d_bias1, HIDDEN_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2_fp16, nn->d_bias2, OUTPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
    
    // Convert FP16 to FP32
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) h_W1[i] = __half2float(h_W1_fp16[i]);
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) h_W2[i] = __half2float(h_W2_fp16[i]);
    for (int i = 0; i < HIDDEN_SIZE; i++) h_b1[i] = __half2float(h_b1_fp16[i]);
    for (int i = 0; i < OUTPUT_SIZE; i++) h_b2[i] = __half2float(h_b2_fp16[i]);
    
    free(h_W1_fp16); free(h_W2_fp16); free(h_b1_fp16); free(h_b2_fp16);
    
    for (int batch = 0; batch < num_batches; batch++) {
        float *batch_x = X_test + batch * BATCH_SIZE * INPUT_SIZE;
        int *batch_y = y_test + batch * BATCH_SIZE;
        
        // Forward: hidden = relu(W1 @ X + b1)
        for (int b = 0; b < BATCH_SIZE; b++) {
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                float sum = h_b1[h];
                for (int i = 0; i < INPUT_SIZE; i++) {
                    sum += batch_x[b * INPUT_SIZE + i] * h_W1[h + i * HIDDEN_SIZE];
                }
                hidden[b * HIDDEN_SIZE + h] = fmaxf(0.0f, sum);
            }
        }
        
        // Forward: output = W2 @ hidden + b2
        for (int b = 0; b < BATCH_SIZE; b++) {
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                float sum = h_b2[o];
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    sum += hidden[b * HIDDEN_SIZE + h] * h_W2[o + h * OUTPUT_SIZE];
                }
                output[b * OUTPUT_SIZE + o] = sum;
            }
        }
        
        // Count correct predictions
        for (int b = 0; b < BATCH_SIZE; b++) {
            int pred = 0;
            float max_val = output[b * OUTPUT_SIZE];
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[b * OUTPUT_SIZE + j] > max_val) {
                    max_val = output[b * OUTPUT_SIZE + j];
                    pred = j;
                }
            }
            if (pred == batch_y[b]) correct++;
        }
    }
    
    free(hidden); free(output);
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);
    
    float accuracy = 100.0f * correct / (num_batches * BATCH_SIZE);
    printf("Test Accuracy: %.2f%%\n", accuracy);
}

int main() {
    srand(12345);

    // Load FP32 data first
    float *train_data_fp32 = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    load_data("./data/X_train.bin", train_data_fp32, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(train_data_fp32, TRAIN_SIZE * INPUT_SIZE);
    
    // ============================================================
    // [NEW in v8] Pre-convert to FP16 pinned memory
    // v6: float *train_data (FP32)
    // v8: __half *train_data (FP16) - 2x less H2D transfer
    // ============================================================
    __half *train_data;
    CUDA_CHECK(cudaMallocHost(&train_data, TRAIN_SIZE * INPUT_SIZE * sizeof(__half)));
    convert_to_fp16(train_data_fp32, train_data, TRAIN_SIZE * INPUT_SIZE);
    free(train_data_fp32);
    
    int *train_labels;
    CUDA_CHECK(cudaMallocHost(&train_labels, TRAIN_SIZE * sizeof(int)));
    load_labels("./data/y_train.bin", train_labels, TRAIN_SIZE);
    
    // Test data (FP32 for CPU evaluation)
    float *test_data = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *test_labels = (int *)malloc(TEST_SIZE * sizeof(int));
    load_data("./data/X_test.bin", test_data, TEST_SIZE * INPUT_SIZE);
    normalize_data(test_data, TEST_SIZE * INPUT_SIZE);
    load_labels("./data/y_test.bin", test_labels, TEST_SIZE);

    float *h_loss[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMallocHost(&h_loss[i], BATCH_SIZE * sizeof(float)));
    }

    NeuralNetworkCUDA nn;
    initialize_nn_cuda(&nn);

    int num_batches = TRAIN_SIZE / BATCH_SIZE;
    
    TimingStats stats = {0};
    struct timespec start, end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int stream_idx = batch % NUM_STREAMS;
            __half *batch_input = train_data + batch * BATCH_SIZE * INPUT_SIZE;
            int *batch_labels_ptr = train_labels + batch * BATCH_SIZE;

            if (batch >= NUM_STREAMS) {
                clock_gettime(CLOCK_MONOTONIC, &step_start);
                CUDA_CHECK(cudaStreamSynchronize(nn.streams[stream_idx]));
                clock_gettime(CLOCK_MONOTONIC, &step_end);
                stats.stream_sync += get_time_diff(step_start, step_end);
                
                for (int i = 0; i < BATCH_SIZE; i++) {
                    total_loss += h_loss[stream_idx][i];
                }
            }

            // H2D transfer (FP16 - half the data!)
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpyAsync(nn.d_input_batch[stream_idx], batch_input, 
                                       BATCH_SIZE * INPUT_SIZE * sizeof(__half),  // FP16!
                                       cudaMemcpyHostToDevice, nn.streams[stream_idx]));
            CUDA_CHECK(cudaMemcpyAsync(nn.d_labels[stream_idx], batch_labels_ptr, 
                                       BATCH_SIZE * sizeof(int), 
                                       cudaMemcpyHostToDevice, nn.streams[stream_idx]));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.h2d_submit += get_time_diff(step_start, step_end);

            // GPU Computation
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            
            forward_pass_stream(&nn, stream_idx, BATCH_SIZE);
            compute_loss_on_gpu_stream(&nn, stream_idx, BATCH_SIZE, h_loss[stream_idx]);
            backward_pass_stream(&nn, stream_idx, BATCH_SIZE);
            update_weights_stream(&nn, LEARNING_RATE, stream_idx);
            
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.kernel_launch += get_time_diff(step_start, step_end);
        }

        // Collect remaining losses
        for (int s = 0; s < NUM_STREAMS; s++) {
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaStreamSynchronize(nn.streams[s]));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.stream_sync += get_time_diff(step_start, step_end);
            
            int remaining_batch = num_batches - NUM_STREAMS + s;
            if (remaining_batch >= 0 && remaining_batch < num_batches) {
                for (int i = 0; i < BATCH_SIZE; i++) {
                    total_loss += h_loss[s][i];
                }
            }
        }

        printf("Epoch %d loss: %.4f\n", epoch, total_loss / (num_batches * BATCH_SIZE));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    stats.total_time = get_time_diff(start, end);
    
    printf("\n=== PURE FP16 IMPLEMENTATION (TENSOR CORES) ===\n");
    printf("Total training time: %.3f seconds\n\n", stats.total_time);

    printf("Timing Breakdown:\n");
    printf("  H2D submit:     %6.3fs (%5.1f%%)  <- FP16 = half the data\n", 
           stats.h2d_submit, 100.0 * stats.h2d_submit / stats.total_time);
    printf("  Kernel launch:  %6.3fs (%5.1f%%)  <- Tensor Core accelerated\n", 
           stats.kernel_launch, 100.0 * stats.kernel_launch / stats.total_time);
    printf("  Stream sync:    %6.3fs (%5.1f%%)  <- actual GPU execution time\n", 
           stats.stream_sync, 100.0 * stats.stream_sync / stats.total_time);
    
    double accounted = stats.h2d_submit + stats.kernel_launch + stats.stream_sync;
    printf("  Other:          %6.3fs (%5.1f%%)\n",
           stats.total_time - accounted, 100.0 * (stats.total_time - accounted) / stats.total_time);

    evaluate(&nn, test_data, test_labels);

    free_nn_cuda(&nn);
    
    CUDA_CHECK(cudaFreeHost(train_data));
    CUDA_CHECK(cudaFreeHost(train_labels));
    free(test_data);
    free(test_labels);
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaFreeHost(h_loss[i]));
    }

    return 0;
}

