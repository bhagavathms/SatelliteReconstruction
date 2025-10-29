import numpy as np
import matplotlib.pyplot as plt

# Select pixel (choose a gap-heavy one)
r, c = 50, 50  

# Load Ground Truth & Gapped Input
y_true = np.load("../../data/original.npy")[:, r, c]
y_gap = np.load("../../data/gapped.npy")[:, r, c]

# Load reconstructed outputs (all methods)
Y = {
    "DTS": np.load("../../results/DTS/recon1.npy")[:, r, c],
    "Cubic": np.load("../../results/Cubic/recon.npy")[:, r, c],
    "LOWESS": np.load("../../results/LOWESS/recon.npy")[:, r, c],
    "SG": np.load("../../results/SG/recon.npy")[:, r, c],
    "Whittaker": np.load("../../results/Whittaker/recon.npy")[:, r, c],
    "GAN": np.load("../../results/GAN/recon.npy")[:, r, c],
    "DTS-GAN": np.load("../../results/DTS_GAN/recon.npy")[:, r, c]
}

# Ensure equal timeline
min_T = y_true.shape[0]
for y in Y.values():
    min_T = min(min_T, len(y))

y_true = y_true[:min_T]
y_gap = y_gap[:min_T]
for m in Y:
    Y[m] = Y[m][:min_T]

# Colors for better visibility
colors = {
    "Original": "black",
    "Gapped": "gray",
    "Cubic": "blue",
    "LOWESS": "green",
    "SG": "purple",
    "DTS": "red",
    "GAN": "olive",
    "Whittaker": "orange",
    "DTS-GAN": "#AA00FF"  # bright violet
}

# Plotting
plt.figure(figsize=(15,7))
plt.plot(y_true, color=colors["Original"], linewidth=3, label="Original")
plt.scatter(range(min_T), y_gap, color="gray", s=25, label="Gapped Input")

for m, y in Y.items():
    plt.plot(y, color=colors[m], linewidth=1.8, label=m)

plt.title("Temporal Reconstruction Comparison – Pixel (50, 50)", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Reflectance / NDVI-like Value", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()

save_path = "../../results/Comparison/temporal_compare_all_methods.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Combined Temporal Plot Saved → {save_path}")
