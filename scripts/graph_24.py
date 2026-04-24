import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

formatter = EngFormatter(unit="")  # no extra unit

# -------------------------
# Data from your tables
# -------------------------

# Power-of-two scan results
scan_sizes = np.array([64, 256, 1024, 4096, 16384, 65536,
                       262144, 1048576, 4194304, 16777216, 67108864])
cpu_scan     = [0.000146, 0.000081, 0.000215, 0.000862, 0.003271,
                0.013142, 0.132031, 1.756160, 3.965870, 10.041000, 35.037100]
naive_scan   = [0.134752, 0.020032, 0.023872, 0.029248, 0.033248,
                0.216288, 0.481088, 1.441090, 7.939580, 21.344200, 99.206900]
eff_scan     = [0.102656, 0.035904, 0.044480, 0.052896, 0.063680,
                0.078528, 0.203040, 0.383552, 0.653216, 2.688540, 11.000500]
thrust_scan  = [0.048384, 0.015776, 0.017056, 0.013088, 0.017504,
                0.022432, 0.080640, 0.160512, 0.138400, 0.321024, 1.465890]

# Power-of-two compaction results
comp_sizes = np.array([64, 256, 1024, 4096, 16384, 65536,
                       262144, 1048576, 4194304, 16777216, 67108864])
cpu_no_scan   = [0.000205, 0.000573, 0.001745, 0.006417, 0.024499,
                 0.103375, 0.412454, 2.178310, 7.599850, 28.617200, 113.149000]
cpu_with_scan = [0.000428, 0.000715, 0.003360, 0.012418, 0.052908,
                 0.264867, 0.881458, 5.862320, 22.914500, 85.529100, 427.137000]
gpu_compact   = [0.079904, 0.073536, 0.084352, 0.097120, 0.183584,
                 0.616352, 1.587170, 8.874690, 21.235400, 65.237300, 253.671000]

# -------------------------
# Plot Scan Results
# -------------------------
plt.figure(figsize=(8,6))
plt.plot(scan_sizes, cpu_scan, marker='o', label="CPU Scan")
plt.plot(scan_sizes, naive_scan, marker='o', label="Naive GPU Scan")
plt.plot(scan_sizes, eff_scan, marker='o', label="Work-Efficient GPU Scan")
plt.plot(scan_sizes, thrust_scan, marker='o', label="Thrust Scan")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Array Size (N)")
plt.ylabel("Time (ms)")
plt.title("Scan Performance")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

# For Scan plot
plt.gca().xaxis.set_major_formatter(formatter)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.ticklabel_format(style='plain', axis='x')
# plt.ticklabel_format(style='plain', axis='y')
plt.savefig("scan_performance.png", dpi=200)

# -------------------------
# Plot Compaction Results
# -------------------------
plt.figure(figsize=(8,6))
plt.plot(comp_sizes, cpu_no_scan, marker='o', label="CPU Compact (No Scan)")
plt.plot(comp_sizes, cpu_with_scan, marker='o', label="CPU Compact (With Scan)")
plt.plot(comp_sizes, gpu_compact, marker='o', label="Work-Efficient GPU Compact")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Array Size (N)")
plt.ylabel("Time (ms)")
plt.title("Compaction Performance")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

# For Compaction plot
plt.gca().xaxis.set_major_formatter(formatter)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.ticklabel_format(style='plain', axis='x')
# plt.ticklabel_format(style='plain', axis='y')
plt.savefig("compaction_performance.png", dpi=200)

plt.show()
