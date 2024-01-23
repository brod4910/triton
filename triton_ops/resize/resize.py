import torch

import triton
import triton.language as tl


def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@triton.jit#(interpret=True)
def _resize_bilinear(in_ptr, in_h, in_w,
                     stride_in_h, stride_in_w,
                     out_ptr, out_h, out_w,
                     stride_out_h, stride_out_w,
                     scale_x, scale_y,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    # Grab the Micro-tile index. For the initiated, this is equivalent in CUDA:
    #  
    #
    #
    x_id = tl.program_id(0)
    y_id = tl.program_id(1)

    # BM
    out_x_offsets = (x_id * BLOCK_M + tl.arange(0, BLOCK_M)) % out_w
    # BN
    out_y_offsets = (y_id * BLOCK_N + tl.arange(0, BLOCK_N)) % out_h

    # BM x BN
    out_ptrs = out_ptr + (out_x_offsets[:, None] * stride_out_w + out_y_offsets[None, :] * stride_out_h)

    in_x_offsets = out_x_offsets * scale_x # BM
    in_y_offsets = out_y_offsets * scale_y # BN

    x1 = tl.math.floor(in_x_offsets)
    y1 = tl.math.floor(in_y_offsets)
    x2 = tl.math.ceil(in_x_offsets)
    y2 = tl.math.ceil(in_y_offsets)

    # x1 = in_x_offsets // 1
    # y1 = in_y_offsets // 1
    # x2 = in_x_offsets // 1 + 1
    # y2 = in_y_offsets // 1 + 1

    # x,y weights
    dx = (in_x_offsets - x1)[:, None]
    dy = (in_y_offsets - y1)[None, :]

    x1 = x1.to(tl.int32)[:, None]
    y1 = y1.to(tl.int32)[None, :]
    x2 = x2.to(tl.int32)[:, None] % in_w
    y2 = y2.to(tl.int32)[None, :] % in_h

    # x1 = x1.to(torch.int32)
    # y1 = y1.to(torch.int32)
    # x2 = x2.to(torch.int32) % in_w
    # y2 = y2.to(torch.int32) % in_h

    a = tl.load(in_ptr + (x1 * stride_in_w + y1 * stride_in_h))
    b = tl.load(in_ptr + (x2 * stride_in_w + y1 * stride_in_h))
    c = tl.load(in_ptr + (x1 * stride_in_w + y2 * stride_in_h))
    d = tl.load(in_ptr + (x2 * stride_in_w + y2 * stride_in_h))

    P = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * dy * (1 - dx) + d * dx * dy
    tl.store(out_ptrs, P)


def resize_bilinear(X: torch.Tensor, Y: torch.Tensor):
    in_h, in_w = X.size(-2), X.size(-1)
    out_h, out_w = Y.size(-2), Y.size(-1)

    scale_x = (in_w - 1) / (out_w - 1)
    scale_y = (in_h - 1) / (out_h - 1)
    BLOCK_SIZE = 128
    # replace with tl cdiv
    grid = (cdiv(out_h, BLOCK_SIZE), cdiv(out_w, BLOCK_SIZE))

    _resize_bilinear[grid](X, in_h, in_w,
                           X.stride(0), X.stride(1),
                           Y, out_h, out_w,
                           Y.stride(0), Y.stride(1),
                           scale_x, scale_y,
                           BLOCK_M=BLOCK_SIZE,
                           BLOCK_N=BLOCK_SIZE)


A = torch.arange(0, 224, 1, dtype=torch.float32).repeat(224, 1).cuda()
B = torch.zeros(512, 512, dtype=torch.float32).cuda()

resize_bilinear(A, B)

torch_ref = torch.nn.functional.interpolate(A.unsqueeze(0).unsqueeze(0), (512,512), mode="bilinear", align_corners=True).squeeze(0).squeeze(0)

print(torch.allclose(torch_ref, B))