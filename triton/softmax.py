import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    
    # offset for row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    row_start_output_ptr = output_ptr + row_idx * output_row_stride

    row = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    # consider the numerical stability
    softmax = numerator / (denominator + 1e-10)

    tl.store(row_start_output_ptr + tl.arange(0, BLOCK_SIZE), softmax, mask=tl.arange(0, BLOCK_SIZE) < n_cols)


def triton_softmax(input: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = input.shape
    output = torch.empty_like(input)
    
    # determine the grid size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    grid = (n_rows, )
    softmax_kernel[grid](
        output, input, 
        input.stride(0), output.stride(0), 
        n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

torch.manual_seed(0)
input = torch.randn(256, 1024, device="cuda", dtype=torch.float32)
torch_result = torch.softmax(input, dim=1)

triton_result = triton_softmax(input)

# Compare results
max_diff = torch.max(torch.abs(torch_result - triton_result))
print(f"Maximum difference between PyTorch and Triton results: {max_diff:.2e}")

# Check if results are close
is_close = torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)
print(f"Results are close: {is_close}")