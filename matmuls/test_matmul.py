import pytest
import torch
from matmuls.matmul import matmul as matmul_naive
from matmuls.matmul_grouped import matmul as matmul_grouped

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def matmul_torch(a, b):
    return torch.matmul(a, b)

@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256), (512, 512, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_matmul_implementations(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    result_torch = matmul_torch(a, b)
    result_naive = matmul_naive(a, b)
    result_grouped = matmul_grouped(a, b)
    assert torch.allclose(
        result_torch, result_naive, atol=1e-2
    ), f"Results of naive matmul differ significantly from Torch (M={M}, N={N}, K={K})"
    assert torch.allclose(
        result_torch, result_grouped, atol=1e-2
    ), f"Results of grouped matmul differ significantly from Torch (M={M}, N={N}, K={K})"


if __name__ == "__main__":
    pytest.main()
