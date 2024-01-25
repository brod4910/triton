import torch

import triton
import triton.language as tl


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
    pass


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