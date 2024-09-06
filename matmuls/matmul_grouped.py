import triton
import triton.language as tl
import torch
import math


@triton.jit
def group_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    a_stride_m,
    a_stride_k,
    b_stride_k,
    b_stride_n,
    c_stride_m,
    c_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pids_n = tl.cdiv(n, BLOCK_SIZE_N)
    n_pids_per_group = GROUP_SIZE_M * n_pids_n

    group_id = pid // n_pids_per_group
    pid_inside_group = pid % n_pids_per_group
    pid_m = group_id * GROUP_SIZE_M + pid_inside_group // n_pids_n
    pid_n = pid_inside_group % n_pids_n

    a_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # across rows
    b_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # across columns
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptr = a_ptr + a_offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k
    b_ptr = b_ptr + offs_k[:, None] * b_stride_k + b_offs_n[None, :] * b_stride_n
    a_mask_m = a_offs_m[:, None] < m
    b_mask_n = b_offs_n[None, :] < n

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, k, BLOCK_SIZE_K):
        mask_k = offs_k + kk < k
        a_frag = tl.load(a_ptr, mask=a_mask_m & mask_k[None, :], other=0)
        b_frag = tl.load(b_ptr, mask=mask_k[:, None] & b_mask_n, other=0)
        acc = tl.dot(a_frag, b_frag, acc)

        a_ptr += BLOCK_SIZE_K * a_stride_k
        b_ptr += BLOCK_SIZE_K * b_stride_k

    c_offs_m, c_offs_n = pid_m * BLOCK_SIZE_M + tl.arange(
        0, BLOCK_SIZE_M
    ), pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = c_offs_m[:, None] < m
    c_mask &= c_offs_n[None, :] < n
    tl.store(
        c_ptr + c_offs_m[:, None] * c_stride_m + c_offs_n[None, :] * c_stride_n,
        acc,
        mask=c_mask,
    )


def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == 2
    assert b.dim() == 2
    assert a.size(1) == b.size(0)

    m, k = a.shape
    k, n = b.shape

    c = torch.zeros((m, n), dtype=torch.float32, device="cuda")

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32
    GROUP_SIZE_M = 2
    grid = ((math.ceil(m / BLOCK_SIZE_M) + 1) * (math.ceil(n / BLOCK_SIZE_N) + 1),)
    group_matmul_kernel[grid](
        a.contiguous(),
        b.contiguous(),
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        GROUP_SIZE_M,
    )
    return c


a = torch.rand((133, 779), dtype=torch.float32, device="cuda") * 10
b = torch.rand((779, 133), dtype=torch.float32, device="cuda") * 10
ctorch = torch.matmul(a, b)

ctrion = matmul(a, b)
torch.testing.assert_close(
    ctrion, ctorch, rtol=1e-3, atol=1e-3
)  # higher rtol because it uses tf32
