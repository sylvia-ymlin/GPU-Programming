# GPU Computing

High-performance CUDA kernel optimization from basic implementations to production-level performance.

**Key Results:**
- 83.4% of cuBLAS performance for SGEMM
- 90.4% of peak memory bandwidth for elementwise operations
- 1012x speedup for MNIST training (FP16 vs C baseline)
- Complete Nsight Systems profiling analysis

**Tested on:** NVIDIA GeForce RTX 3090, CUDA 12.1

## Repository Structure

```
cuda-kernels-from-scratch/
├── kernels/                 # Core CUDA kernels
├── mnist-cuda/              # End-to-end MLP training
├── pytorch-extension/       # Custom PyTorch CUDA extension
├── triton/                  # Triton kernel examples
├── docs/                    # Performance analysis and documentation
└── lectures/                # GPU programming lecture notes
```

## Projects

| Project | Description | Highlights |
|---------|-------------|------------|
| [kernels/](kernels/) | CUDA kernels | 90.4% peak BW, 83.4% cuBLAS |
| [mnist-cuda/](mnist-cuda/) | MLP training | PyTorch → C → CUDA → FP16, 1012× speedup |
| [pytorch-extension/](pytorch-extension/) | PyTorch extension | Polynomial activation kernel |
| [triton/](triton/) | Triton kernels | vec_add, softmax |
| [docs/](docs/) | Performance analysis | Complete Nsight Systems analysis |
| [lectures/](lectures/) | Lecture notes | GPU programming fundamentals |

## Build & Run

### CUDA Kernels
```bash
cd kernels && mkdir build && cd build
cmake .. && make
./test_sgemm        # Matrix multiplication benchmarks
./test_elementwise  # Memory bandwidth tests
./test_reduction    # Parallel reduction tests
./test_transpose    # Memory access pattern tests
```

### MNIST Training
```bash
cd mnist-cuda
python3 src/downloader.py   # Download MNIST data
mkdir build && cd build && cmake .. && make
./bin/v8                    # FP16 + Tensor Cores
```

### PyTorch Extension
```bash
cd pytorch-extension
pip install -e .
python polynomial_activation.py
```

### Triton Kernels
```bash
cd triton
pip install -r requirements.txt
python vec_add.py
python softmax.py
```

Performance visualization:
![SGEMM Optimization](./docs/sgemm_optimization_progress.png)
![Memory Bandwidth](./docs/memory_bandwidth_utilization.png)

## References

- [CUDA Matrix Multiplication Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- Kirk & Hwu, *Programming Massively Parallel Processors* (Morgan Kaufmann, 2016)

