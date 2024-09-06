import triton
import torch
from .matmul import matmul as matmul_naive
from .matmul_grouped import matmul as matmul_grouped

# allow tf32 on torch for fair comparison
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def matmul_torch(a, b):
    return torch.matmul(a, b)


configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[
            128 * i for i in range(2, 33)
        ],
        line_arg="provider",
        line_vals=[
            "triton_grouped",
            "triton_naive",
            "torch",
        ],
        line_names=["Triton Grouped", "Triton Naive", "Torch"],
        ylabel="TFLOPS",
        plot_name="matmul-performance-float16",
        args=dict(dtype=torch.float16),
    ),
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["triton_grouped", "triton_naive", "torch"],
        line_names=["Triton Grouped", "Triton Naive", "Torch"],
        ylabel="TFLOPS",
        plot_name="matmul-performance-float32",
        args=dict(dtype=torch.float32),
    ),
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["triton_grouped", "triton_naive", "torch"],
        line_names=["Triton Grouped", "Triton Naive", "Torch"],
        ylabel="TFLOPS",
        plot_name="matmul-performance-bfloat16",
        args=dict(dtype=torch.bfloat16),
    ),
]


provider_to_fn = {
    "triton_grouped": matmul_grouped,
    "triton_naive": matmul_naive,
    "torch": matmul_torch,
}


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, dtype):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    fn = provider_to_fn[provider]
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
