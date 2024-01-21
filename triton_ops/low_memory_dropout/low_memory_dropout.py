import torch

import triton
import triton.language as tl
import tabulate


@triton.jit
def _dropout(
        x_ptr,
        x_keep_ptr,
        out_ptr,
        num_elements,
        p,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(out_ptr + offsets, output, mask=mask)

def dropout(
        x,
        x_keep,
        p
):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# Input tensor
x = torch.randn(size=(10, )).cuda()
# Dropout mask
p = 0.5
x_keep = (torch.rand(size=(10, )) > p).to(torch.int32).cuda()
#
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))

@triton.jit
def _seeded_dropout(
        x_ptr,
        out_ptr,
        num_elements,
        p,
        seed,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(out_ptr + offsets, output, mask=mask)

def seeded_dropout(
        x,
        p,
        seed
):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10,), device='cuda')
output = seeded_dropout(x, 0.5, 123)
output2 = seeded_dropout(x, 0.5, 123)
output3 = seeded_dropout(x, 0.5, 512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))

# Challenge: implement matrix dropout
@triton.jit
def _seeded_matrix_dropout(x_ptr,
                           out_ptr,
                           M, N,
                           num_elements,
                           p,
                           seed,
                           BLOCK_SIZE: tl.constexpr):
    pass