import torch

import triton
import triton.language as tl


# For ref: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UpSample.cuh
@triton.jit
def cubic_convolution2(x, A):
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A

@triton.jit
def cubic_convolution1(x, A):
    return ((A + 2) * x - (A + 3)) * x * x + 1

@triton.jit
def get_cubic_upsampling_coeffcients(x, t):
    A = -0.75
    # place holder (replace with correct matrix) for the 4 coefficients of the cubic interpolation
    coeffs = tl.zeros_like(x)
    coeffs[0] = cubic_convolution2(x + 1.0, A)
    coeffs[1] = cubic_convolution1(x, A)
    
    x2 = 1.0 - x
    coeffs[2] = cubic_convolution1(x2, A)
    coeffs[3] = cubic_convolution2(x2 + 1.0, A)

    return coeffs

@triton.jit
def cubic_interp1d(x, t):
    coeffs = get_cubic_upsampling_coeffcients(x, t)
    return coeffs[0] * x[0] + coeffs[1] * x[1] + coeffs[2] * x[2] + coeffs[3] * x[3]


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
def _resize_bicubic(in_ptr, in_h, in_w,
                     stride_in_h, stride_in_w,
                     out_ptr, out_h, out_w,
                     stride_out_h, stride_out_w,
                     scale_x, scale_y,
                     BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr):
    x_id = tl.program_id(0)
    y_id = tl.program_id(1)

    # (BM,)
    out_x_offsets = (x_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % out_w
    # (BN,)
    out_y_offsets = (y_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % out_h

    # (BM, BN)
    out_ptrs = out_ptr + (out_x_offsets[:, None] * stride_out_w + out_y_offsets[None, :] * stride_out_h)
    
    in_x_offsets = out_x_offsets * scale_x # (BM,)
    in_y_offsets = out_y_offsets * scale_y # (BN,)

    x_t = in_x_offsets - tl.math.floor(in_x_offsets)
    y_t = in_y_offsets - tl.math.floor(in_y_offsets)




def resize_bicubic(X: torch.Tensor, Y: torch.Tensor):
    in_h, in_w = X.size(-2), X.size(-1)
    out_h, out_w = Y.size(-2), Y.size(-1)

    scale_x = (in_w - 1) / (out_w - 1)
    scale_y = (in_h - 1) / (out_h - 1)

    grid = lambda META: (triton.cdiv(out_w, META["BLOCK_SIZE_M"]), triton.cdiv(out_h, META["BLOCK_SIZE_N"]))

    _resize_bicubic[grid](X, in_h, in_w,
                           X.stride(0), X.stride(1),
                           Y, out_h, out_w,
                           Y.stride(0), Y.stride(1),
                           scale_x, scale_y)