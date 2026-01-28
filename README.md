# CUDA Kernels & High-Performance Computing

## Why This Exists

This started as coursework for a High-Performance Computing class, but became a deep dive into GPU programming fundamentals. I wanted to understand what happens under the hood of libraries like cuBLAS and PyTorch — how do you actually get 80%+ of theoretical peak performance? The lectures covered the theory, but I needed to implement everything from scratch to really get it.

What began as "write some CUDA kernels" turned into a systematic optimization journey: naive implementations → shared memory tiling → register blocking → vectorized loads → double buffering. Each step taught me something new about the GPU memory hierarchy and where performance actually comes from.

## What I Built

Four interconnected projects that build from basic kernels to production-level implementations:

- **[kernels/](kernels/)** - Core CUDA kernels (SGEMM, reductions, elementwise) with systematic optimization
- **[mnist-cuda/](mnist-cuda/)** - End-to-end MLP training: PyTorch → C → CUDA → FP16/Tensor Cores  
- **[pytorch-extension/](pytorch-extension/)** - Custom PyTorch CUDA operator integration
- **[triton/](triton/)** - Modern kernel development with Triton (blocked softmax, vector add)

## Key Results

- **SGEMM**: 83.4% of cuBLAS (15,309 vs 18,356 GFLOPS)
- **Memory Bandwidth**: 90.4% of peak (846 GB/s theoretical)
- **MNIST Training**: 1012x speedup vs C baseline (FP16/Tensor Cores)
- **Optimization**: 5.8x improvement from naive to optimized

**Tested on:** NVIDIA GeForce RTX 3090, CUDA 12.1

## What I Learned (The Hard Parts)

### Memory layout killed me silently

**Problem**: My custom GEMM kernel (v7) was producing exploding loss — training started at 2.3, but by epoch 9 reached 14.3. Test accuracy was stuck at 8% (random guessing level for 10 classes). No crashes, no obvious errors, just completely wrong results.

**Root cause**: cuBLAS uses column-major storage (Fortran convention), but my kernel assumed row-major (C convention). This meant my shared memory tiling was loading the wrong matrix elements:

```c
// WRONG: assumed B is row-major
Bs[ty][tx] = B[b_row + col * K];  

// CORRECT: cuBLAS stores weights column-major
Ws[ty][tx] = W[row + w_col * M];  // W[hidden + input * HIDDEN_SIZE]
```

**Lesson**: Silent correctness bugs are the worst kind. When loss explodes but code runs, it's usually a data layout issue, not hyperparameters. Always validate intermediate outputs against a working reference (v6 cuBLAS in my case). Memory layout mismatches create mathematically valid but semantically wrong computations.

### Vectorized loads gave 2.7x breakthrough

**Problem**: Register caching alone (v5) barely helped — only 1.05x speedup over v4. I expected more from reducing shared memory pressure.

**Breakthrough**: Adding float4 vectorized loads (v6) suddenly jumped performance 2.74x (1,477 → 4,052 GFLOPS). The combination of register tiling + 128-bit coalesced memory access transformed the kernel from memory-bound to compute-bound.

**Lesson**: Individual optimizations can be underwhelming, but the right combination creates phase transitions. Register blocking needs vectorized loads to shine. Always measure arithmetic intensity — when you cross from memory-bound to compute-bound, performance characteristics completely change.

### Profiling revealed the real bottlenecks

**Problem**: I assumed kernel execution time was the main bottleneck, but Nsight Systems showed my optimized sgemm_v7 only consumed 8.1% of GPU execution time, while the baseline sgemm_v2 took 55.5%.

**Surprise**: 74.1% of total time was spent in `cudaEventSynchronize` — API overhead, not computation. My "fast" kernels were so efficient they made synchronization the limiting factor.

**Lesson**: Profile everything, assume nothing. The bottleneck moves as you optimize. What looks like a kernel problem might be a host-device synchronization issue. Tools like Nsight Systems show you where time actually goes, not where you think it goes.

## Repository Structure

```
├── kernels/           # Core CUDA kernels (SGEMM, reductions, elementwise)
├── mnist-cuda/        # End-to-end MLP: PyTorch → C → CUDA → FP16
├── pytorch-extension/ # Custom PyTorch CUDA operators
├── triton/           # Modern kernel development with Triton
└── docs/             # Performance analysis and benchmarks
```

## Quick Start

```bash
# CUDA Kernels
cd kernels && mkdir build && cd build && cmake .. && make
./test_sgemm        # 83.4% cuBLAS performance
./test_elementwise  # 90.4% memory bandwidth

# MNIST Training  
cd mnist-cuda && python3 src/downloader.py
mkdir build && cd build && cmake .. && make
./bin/v8           # 1012x speedup with FP16/Tensor Cores
```

## Performance Analysis

Detailed benchmarks, profiling data, and optimization breakdowns available in [docs/](docs/):
- [Complete Performance Analysis](docs/complete_performance_analysis.md)
- [Interview Summary](docs/interview_summary.md) 
- [Nsight Systems Results](docs/nsight_analysis_results.txt)

![SGEMM Optimization Progress](./docs/sgemm_optimization_progress.png)
![Memory Bandwidth Utilization](./docs/memory_bandwidth_utilization.png)

## References

- [CUDA Matrix Multiplication Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- Kirk & Hwu, *Programming Massively Parallel Processors* (Morgan Kaufmann, 2016)

