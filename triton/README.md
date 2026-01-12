# Triton

Triton is an abstraction on top of CUDA.

- CUDA -> scalar program + blocked threads
- Triton -> blocked program + scalar threads

cuda is a scalar program with blocked threads because we write a kernel to operate at the level of threads (scalars), whereas triton is abstracted up to thread blocks (compiler takes care of thread level operations for us).

Besides, cuda has blocked threads in the context of "worrying" about inter-thread at the level of blocks, whereas triton has scalar threads in the context of "not worrying" about inter-thread at the level of threads (compiler also takes care of this).

Why does this actually mean on an intuitive level?

- higher level of abstract for deep learning operations (activations functions, convolutions, matmul, etc)
- the compiler will take care of boilerplate complexities of load and store instructions, tiling, SRAM caching, etc
- python programmers can write kernels comparable to cuBLAS, cuDNN (which is difficult for most CUDA/GPU programmers)

---

## Installation & Dependencies

```bash
pip install -r requirements.txt
```
Requires NVIDIA GPU with CUDA drivers. Recommended: PyTorch 2.0+ and Triton 2.1+.

---

## Project Structure

- `softmax.py`: High-performance softmax kernel implemented in Triton.
- `vec_add.py`: Vector addition kernel implemented in Triton.
- `requirements.txt`: Dependency list.

---

## Usage Example

```python
import torch
from softmax import triton_softmax

x = torch.randn(128, 256, device='cuda')
y = triton_softmax(x)
print(y)
```

---

## Triton vs CUDA

- Triton enables Python users to write high-performance GPU kernels, abstracting away thread/block management and memory access details.
- CUDA requires manual management of threads, blocks, and synchronization, resulting in higher code complexity.

---

## Performance & Visualization

Triton kernels can replace PyTorch/CUDA kernels, achieving performance close to cuBLAS/cuDNN. Use `matplotlib` for result visualization.

---

