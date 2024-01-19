import torch

import triton
import triton.language as tl


"""
From the paper: The GELU nonlinearity is represented by the following equation:
    GELU(x) = x * P(X <= x) = x * 0.5 * (1.0 + erf(x / sqrt(2.0)))

An approximation of the GELU nonlinearity is:
0.5 * x * (1.0 + tanh([sqrt(2 / pi) * (x + 0.044715 * x^3)])) or
x * sigmoid(1.702 * x)
"""

@triton.jit
def approx_gelu_kernel(x):
    return x * tl.sigmoid(1.702 * x)

@triton.jit
def gelu_kernel(x):
    return x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))

@torch.jit.script
def torch_jit_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(2.0)))

def torch_gelu(x):
    return torch.nn.functional.gelu(x)


torch.manual_seed(0)
x = torch.randn(1024, 1024, device="cuda")

triton_approx_y = approx_gelu_kernel(x)
triton_y = gelu_kernel(x)
torch_jit_y = torch_jit_gelu(x)
torch_y = torch_gelu(x)

assert torch.allclose(torch_jit_y, torch_y, atol=1e-5)
assert torch.allclose(triton_approx_y, torch_y, atol=1e-5)
assert torch.allclose(triton_y, torch_y, atol=1e-5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'triton-approx',
            'torch-native',
            'torch-jit',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Triton (approx)",
            "Torch (native)",
            "Torch (jit)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '--'), ('yellow', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="gelu-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gelu_kernel(x), quantiles=quantiles)
    if provider == 'triton-approx':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: approx_gelu_kernel(x), quantiles=quantiles)
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_gelu(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_jit_gelu(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, save_path=".")