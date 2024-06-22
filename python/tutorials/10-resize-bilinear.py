"""
Resize Bilinear
====================
In this tutorial, you will write a performant interpolation
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

 * Implementing an image transformation

"""

from functools import partial

import torch

import triton
import triton.language as tl


# %%
# Motivations
# -----------
#
# Triton can also be used to optimize other stages of an ML pipeline.
# In this tutorial, we will be implementing single channel bilinear interpolation 
# 
# Exercise:
#   After this tutorial, try extending this to 3-channel interpolation for RGB images.
# 

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=5,
                      num_warps=2),
    ],
    key=['out_h', 'out_w'],
)
@triton.jit
def _resize_bilinear(in_ptr, in_h, in_w,
                     stride_in_h, stride_in_w,
                     out_ptr, out_h, out_w,
                     stride_out_h, stride_out_w,
                     scale_x, scale_y,
                     BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr):
    # Grab the Micro-tile index. This is equivalent to CUDA:
    #  
    # >>> int x_id = blockIdx.x
    # >>> int y_id = BlockIdx.y
    x_id = tl.program_id(0)
    y_id = tl.program_id(1)

    # We grab a block of x,y offsets that we will then calculate the addresses
    # that we want to store the output of interpolation. This is equivalent to CUDA:
    #
    # >>> int offsets_x[BLOCK_SIZE_M];
    # >>> for (int i = 0; i < BLOCK_SIZE_M; ++i) {
    #       offsets_x[i] = (blockIdx.x * blockDim.x + threadId.x) % out_w
    #   }
    #
    # The modulo operator is important, so we don't go out of bounds when accessing the output array.
    # i.e,
    #   >>> out_w = 512
    #   >>> offsets = [0, 1, 2, 3, 4,..., 127]
    #   >>> offsets = offsets % out_w
    #
    # The modulo operator keeps the offsets within the bounds of the output with variable
    # since 512 % 512 = 0, 513 % 512 = 1, etc.
    #
    # A more full proof operation would be to mask the offsets to prevent out-of-bounds loads.
    #
    # (BM,)
    out_x_offsets = (x_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % out_w
    # (BN,)
    out_y_offsets = (y_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % out_h

    # We vectorize the calculated x,y offsets and create an BM x BN matrix of
    # memory addresses (pointers) these addresses are then used to store the
    # result of interpolation.
    #
    # Results in a (BM, 1) matrix
    # >>> stride_out_w = 1
    # >>> out_x_offsets = out_x_offsets[:, None] * stride_out_w
    #
    # Results in a (1, BN) matrix
    # >>> stride_out_h = 512
    # >>> out_y_offsets = out_y_offsets[None, :] * stride_out_h
    #
    # The goal here is to compute the vectorized offsets of the output for later use
    #
    # (BM, BN)
    out_ptrs = out_ptr + (out_x_offsets[:, None] * stride_out_w + out_y_offsets[None, :] * stride_out_h)

    # We multiply by the scale to get the input offsets from the output
    in_x_offsets = out_x_offsets * scale_x # (BM,)
    in_y_offsets = out_y_offsets * scale_y # (BN,)

    # We calculate the x,y points used for bilinear interpolation
    x1 = tl.math.floor(in_x_offsets)
    y1 = tl.math.floor(in_y_offsets)
    x2 = tl.math.ceil(in_x_offsets)
    y2 = tl.math.ceil(in_y_offsets)

    # Calculate x,y weights and expand them to (BM, 1) and (1, BN) matrices
    dx = (in_x_offsets - x1)[:, None]
    dy = (in_y_offsets - y1)[None, :]

    # Convert the dtypes of the x,y coordinates to int32 from float32.
    # We use these points to index and load the addresses required for bilinear interpolation
    # Expand the dimensions to (BM, 1), (1, BN), (BM, 1), and (1, BN) respectively.
    #
    # We also modulo the last x2 and y2 points so we don't device-side assert a segmentation fault.
    # Again, the modulo operator keeps the offsets within the range of the dimensions of the image in this case.
    x1 = x1.to(tl.int32)[:, None]
    y1 = y1.to(tl.int32)[None, :]
    x2 = x2.to(tl.int32)[:, None] % in_w
    y2 = y2.to(tl.int32)[None, :] % in_h

    # Vectorize the calculated x,y offsets and create an BM x BN matrix of
    # memory addresses (pointers) these addresses are then used to load the
    # values required for producing the interpolation result.
    a = tl.load(in_ptr + (x1 * stride_in_w + y1 * stride_in_h))
    b = tl.load(in_ptr + (x2 * stride_in_w + y1 * stride_in_h))
    c = tl.load(in_ptr + (x1 * stride_in_w + y2 * stride_in_h))
    d = tl.load(in_ptr + (x2 * stride_in_w + y2 * stride_in_h))

    # Calculate the block of pixel values for the matrix of output addresses.
    P = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * dy * (1 - dx) + d * dx * dy

    # store the result in P.
    tl.store(out_ptrs, P)


def resize_bilinear(X: torch.Tensor, Y: torch.Tensor):
    in_h, in_w = X.size(-2), X.size(-1)
    out_h, out_w = Y.size(-2), Y.size(-1)

    scale_x = (in_w - 1) / (out_w - 1)
    scale_y = (in_h - 1) / (out_h - 1)

    grid = lambda META: (triton.cdiv(out_w, META["BLOCK_SIZE_M"]), triton.cdiv(out_h, META["BLOCK_SIZE_N"]))

    _resize_bilinear[grid](X, in_h, in_w,
                           X.stride(0), X.stride(1),
                           Y, out_h, out_w,
                           Y.stride(0), Y.stride(1),
                           scale_x, scale_y)


A = torch.arange(0, 224, 1, dtype=torch.float32).repeat(224, 1).cuda()
B = torch.zeros(512, 512, dtype=torch.float32).cuda()

resize_bilinear(A, B)

# Torch interpolate requires that the tensor is 4-D for bilinear interpolation
torch_ref = torch.nn.functional.interpolate(A.unsqueeze(0).unsqueeze(0), (512,512), mode="bilinear", align_corners=True).squeeze(0).squeeze(0)

assert torch.allclose(torch_ref, B)

torch_interp = partial(torch.nn.functional.interpolate, mode="bilinear", align_corners=True)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['pytorch', 'triton'],
        # Label name for the lines
        line_names=["Pytorch", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="GB/s",  # Label name for the y-axis
        plot_name="resize-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, provider):
    a = torch.randn((224, 224), device='cuda', dtype=torch.float32)
    b = torch.zeros((N, M), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_interp(a.unsqueeze(0).unsqueeze(0), (N, M)), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: resize_bilinear(a, b), quantiles=quantiles)
    gbps = lambda ms: 2 * a.numel() * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, save_path=".")
