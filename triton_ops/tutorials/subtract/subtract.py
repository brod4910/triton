import torch

import triton
import triton.language as tl


@triton.jit
def subtract_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Seems rather weird to silently mask out-of-bounds accesses
    # but that's what the example does
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x - y
    tl.store(out_ptr + offsets, output, mask=mask)


def subtract(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    subtract_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


torch.manual_seed(0)
size = 2**20
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")

output_torch = x - y
output_trition = subtract(x, y)
print(output_torch)
print(output_trition)
print(f"Max diff: {(output_torch - output_trition).abs().max()}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-subtract-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x - y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: subtract(x, y), quantiles=quantiles
        )

    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(min_ms), gbps(max_ms)


benchmark.run(print_data=True, save_path="")
