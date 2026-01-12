import torch
import triton
import triton.language as tl

@triton.jit
def vec_add_kernel(
    x_ptr, # pointer to the first input vector
    y_ptr, # pointer to the second input vector
    output_ptr, # pointer to the output vector
    n, # number of elements in the input vectors
    BLOCK_SIZE: tl.constexpr # block size
    ):
    # identify the block index
    block_idx = tl.program_id(0)

    # identify the start and end of the block
    block_start = block_idx * BLOCK_SIZE 
    offset = block_start + tl.arange(0, BLOCK_SIZE) # 0, 1, 2, ..., BLOCK_SIZE - 1, a list of indices

    # then we need to create a mosk to avoid out of bounds access
    mask = offset < n

    # load x and y from DRAM, and mask out the elements that are out of bounds
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    # then we can add the elements
    output = x + y

    # write the results back: position, value, mask
    tl.store(output_ptr + offset, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # preallocate output tensor
    output = torch.empty_like(x)

    # make sure the tensors are on the same device
    assert x.is_cuda and y.is_cuda and output.is_cuda

    # get the number of elements
    n = x.numel()

    # define the grid, meta is a dictionary that contains the block size and other metadata
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    # launch the kernel
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    vec_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)

    return output

# use the triton test utility to run the kernel
@triton.testing.perf_report(
    triton.testing.Benchmark( # define the configuration for the benchmark
        x_names=["size"], # use for the x-axis of the plot
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider", # use for the legend
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        ylabel="GB/s",
        plot_name="vec_add_performance",
        args={},
    ))

def benchmark(size, provider):
    # generate some random input tensors
    torch.manual_seed(0) # fix the random seed to make the results reproducible
    size = 2**25
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    elif provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(min_ms), gbps(max_ms)

benchmark.run(print_data=True, show_plots=True, save_path="vec_add")