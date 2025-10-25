import numpy as np
import matplotlib.pyplot as plt

# Select pixel to analyze
r, c = 50, 50

# Load Ground Truth & Gapped Input
y_true = np.load("../../data/original.npy")[:, r, c]
y_gap = np.load("../../data/gapped.npy")[:, r, c]

# Load reconstructed outputs
Y = {
    "DTS": np.load("../../results/DTS/recon1.npy")[:, r, c],
    "Cubic": np.load("../../results/Cubic/recon.npy")[:, r, c],
    "LOWESS": np.load("../../results/LOWESS/recon.npy")[:, r, c],
    "SG": np.load("../../results/SG/recon.npy")[:, r, c],
    "Whittaker": np.load("../../results/Whittaker/recon.npy")[:, r, c]
}

# Ensure equal timeline by trimming to common length
min_T = y_true.shape[0]
for y in Y.values():
    min_T = min(min_T, len(y))

y_true = y_true[:min_T]
y_gap = y_gap[:min_T]
for m in Y:
    Y[m] = Y[m][:min_T]

# Plotting
plt.figure(figsize=(14,6))
plt.plot(y_true, "k-", linewidth=2, label="Original")
plt.plot(y_gap, "ko", markersize=4, label="Gapped")

for m, y in Y.items():
    plt.plot(y, label=m)

plt.title("Temporal Reconstruction Comparison — Pixel (50,50)")
plt.xlabel("Time Index")
plt.ylabel("NDVI-like Value")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc="lower right")
plt.tight_layout()

save_path = "../../results/Comparison/temporal_compare.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Combined Temporal Plot saved → {save_path}")
