import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

formatter = EngFormatter(unit="")

# -------------------------
# Non-power-of-two Scan Results
# -------------------------
scan_sizes = np.array([61, 253, 1021, 4093, 16381, 65533,
                       262141, 1048573, 4194301, 16777213, 67108861])
cpu_scan     = [0.000044, 0.000063, 0.000215, 0.000812, 0.003308,
                0.012721, 0.196031, 1.340560, 3.897540, 7.871300, 29.286100]
naive_scan   = [0.017664, 0.019936, 0.024704, 0.027968, 0.032448,
                0.079456, 0.265952, 1.336480, 5.601060, 22.084500, 96.962200]
eff_scan     = [0.029760, 0.036512, 0.044544, 0.074656, 0.060960,
                0.078720, 0.205312, 0.283712, 0.649696, 2.671620, 14.517200]
thrust_scan  = [0.016928, 0.012896, 0.016256, 0.012768, 0.016160,
                0.017376, 0.067072, 0.122976, 0.132608, 0.321632, 1.468770]

plt.figure(figsize=(8,6))
plt.plot(scan_sizes, cpu_scan, marker='o', label="CPU Scan")
plt.plot(scan_sizes, naive_scan, marker='o', label="Naive GPU Scan")
plt.plot(scan_sizes, eff_scan, marker='o', label="Work-Efficient GPU Scan")
plt.plot(scan_sizes, thrust_scan, marker='o', label="Thrust Scan")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Array Size (N)")
plt.ylabel("Time (ms)")
plt.title("Scan Performance (Non-Power-of-Two)")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.gca().xaxis.set_major_formatter(formatter)
plt.ticklabel_format(style='plain', axis='x')

plt.tight_layout()
plt.savefig("scan_non_power2.png", dpi=200)

# -------------------------
# Block Size Sweep (N = 262,144)
# -------------------------
block_sizes = np.array([8, 16, 32, 128, 256, 512, 1024])

cpu_block   = [0.318709, 0.339354, 0.329103, 0.342237,
               0.331082, 0.321641, 0.335945]

naive_block = [1.225980, 0.790464, 0.580896, 2.145920,
               0.479200, 2.856510, 0.504672]

eff_block   = [0.281600, 0.153600, 0.139808, 0.260320,
               0.246752, 0.239296, 0.265760]

thrust_block = [2.091140, 2.059780, 1.200060, 2.066340,
                2.134780, 2.046940, 2.137790]


plt.figure(figsize=(8,6))
plt.plot(block_sizes, cpu_block, marker='o', label="CPU Scan")
plt.plot(block_sizes, naive_block, marker='o', label="Naive GPU Scan")
plt.plot(block_sizes, eff_block, marker='o', label="Work-Efficient GPU Scan")
plt.plot(block_sizes, thrust_block, marker='o', label="Thrust Scan")

plt.xlabel("Block Size")
plt.ylabel("Time (ms)")
plt.title("Elapsed Time to Block Size (N=262,144, less is better)")
plt.legend()
plt.grid(True, ls="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("scan_blocksize.png", dpi=200)

plt.show()
