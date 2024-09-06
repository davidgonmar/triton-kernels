import numpy as np
import matplotlib.pyplot as plt


def simulate_memory_access_by_pid(
    M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, pids_to_check
):
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    memory_access_patterns = {}

    for pid in pids_to_check:
        grid = np.full((num_pid_m, num_pid_n), -1)

        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        if pid_m < num_pid_m and pid_n < num_pid_n:
            grid[pid_m, pid_n] = pid

        memory_access_patterns[pid] = grid

    return memory_access_patterns


M = 128
N = 128
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
GROUP_SIZE_M = 4

pids_to_visualize = list(range(11))

memory_access_patterns_by_pid = simulate_memory_access_by_pid(
    M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, pids_to_visualize
)

fig, axs = plt.subplots(3, 4, figsize=(16, 12))
axs = axs.flatten()

for i, pid in enumerate(pids_to_visualize):
    axs[i].imshow(memory_access_patterns_by_pid[pid], cmap="tab20", origin="upper")
    axs[i].set_title(f"Memory Access Pattern for PID {pid}")
    axs[i].set_xlabel("N-axis (Columns)")
    axs[i].set_ylabel("M-axis (Rows)")
    axs[i].grid(False)

for j in range(len(pids_to_visualize), len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()
