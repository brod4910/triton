import torch

import triton
import triton.language as tl

"""
Follow along with the tutorial.

Important notes, Triton compiler only supports blocks that are a power of 2. 
Padding and memory guarding is required to satisfy this constraint.
"""


@torch.jit.script
def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]


@triton.jit
def softmax_kernel(
    out_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # program id is cuda launch params. 1-D, 2-D, or 3-D
    # concerned with only row so access 1st dimension
    row_idx = tl.program_id(0)
    # get the starting address of the row: i.e  0x0000000 + 3 * 128 = 0x0000180
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # creates a tensor of size BLOCK_SIZE (0, 1, 2,  ..., BlOCK_SIZE - 1)
    # used to access the columns of the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # computes the addresses of the columns within the row
    input_ptrs = row_start_ptr + col_offsets
    # create a mask to guard against out of bounds memory accesses
    # required since Triton compiler only supports blocks that are a power of 2
    mask = col_offsets < n_cols
    # load the columns of the row into SRAM
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    # numeric stability for softmax since softmax is shift in-variant
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # again get starting address of the output row
    output_row_start_ptr = out_ptr + row_idx * output_row_stride
    # compute the addresses of the columns within the output row
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    # pad block size to next power of 2 with respect to the number of columns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE)
    return y

torch.manual_seed(0)
x = torch.randn(1823, 781, device="cuda")
y_triton = softmax(x)
y_torch = torch.softmax(x, dim=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, save_path=".")