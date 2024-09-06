import triton
import triton.language as tl
import torch

def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["m", "n", "k"],
)
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
    n_pids_per_group = GROUP_SIZE_M * tl.cdiv(n, BLOCK_SIZE_N)

    group_id = pid // n_pids_per_group
    pid_inside_group = pid % n_pids_per_group
    group_size_m = min(tl.cdiv(m, BLOCK_SIZE_M) - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + pid_inside_group % group_size_m
    pid_n = pid_inside_group // group_size_m

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
    assert a.dtype == b.dtype

    m, k = a.shape
    k, n = b.shape

    c = torch.empty((m, n), dtype=a.dtype, device="cuda")

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
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
    )
    return c
