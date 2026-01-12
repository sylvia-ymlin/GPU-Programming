# PyTorch CUDA Extension: Polynomial Activation

A custom CUDA kernel implementing a polynomial activation function `f(x) = x² + x + 1`, wrapped as a PyTorch extension.

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- CUDA Toolkit (matching your PyTorch CUDA version)
- A C++ compiler (g++ on Linux, MSVC on Windows)

## Project Structure

```
pytorch_extension/
├── polynomial_cuda.cu          # CUDA kernel implementation
├── polynomial_activation.py    # Python wrapper and benchmark
├── setup.py                    # Build configuration
└── README.md
```

## Installation

### Option 1: Install as a package

```bash
cd pytorch_extension
pip install .
```

### Option 2: Development mode (recommended for testing)

```bash
cd pytorch_extension
pip install -e .
```

### Option 3: JIT compilation (no install needed)

You can also use PyTorch's JIT compilation by modifying the Python file to use `torch.utils.cpp_extension.load()` instead.

## Usage

After installation, run the benchmark:

```bash
python polynomial_activation.py
```

### Expected Output

```
tensor([...], device='cuda:0')
PyTorch built-in: X.XXXX ms
CUDA extension: X.XXXX ms
```

## How It Works

### CUDA Kernel (`polynomial_cuda.cu`)

The kernel computes `x² + x + 1` for each element in parallel:

```cuda
template <typename scalar_t>
__global__ void polynomial_activation_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    size_t size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = x[idx];
        output[idx] = val * val + val + 1;
    }
}
```

### Python Integration

The extension is exposed to Python via pybind11:

```python
import polynomial_cuda

# Direct usage
output = polynomial_cuda.polynomial_activation(input_tensor)

# Or via the nn.Module wrapper
activation = PolynomialActivation(implementation='cuda')
output = activation(input_tensor)
```

## Troubleshooting

### CUDA not found
Make sure `nvcc` is in your PATH:
```bash
which nvcc
nvcc --version
```

### Version mismatch
Ensure your CUDA Toolkit version matches PyTorch's CUDA version:
```python
import torch
print(torch.version.cuda)
```

### Rebuild after changes
If you modify the CUDA code, reinstall:
```bash
pip install -e . --force-reinstall --no-deps
```

