import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../../results/Comparison", exist_ok=True)

# Load original truth
y_true = np.load("../../data/original.npy")

# Load reconstructions
Y = {
    "DTS": np.load("../../results/DTS/recon1.npy"),
    "Cubic": np.load("../../results/Cubic/recon.npy"),
    "LOWESS": np.load("../../results/LOWESS/recon.npy"),
    "SG": np.load("../../results/SG/recon.npy"),
    "Whittaker": np.load("../../results/Whittaker/recon.npy"),
    "GAN": np.load("../../results/GAN/recon.npy"),
    "DTS-GAN": np.load("../../results/DTS_GAN/recon.npy")
}

# Ensure time dimension match
min_T = min(y_true.shape[0], *[y.shape[0] for y in Y.values()])
y_true = y_true[:min_T]

rmse_scores = {}
colors = {
    "DTS": "red",
    "Cubic": "blue",
    "LOWESS": "green",
    "SG": "purple",
    "Whittaker": "orange",
    "GAN": "olive",
    "DTS-GAN": "#AA00FF"
}

# Compute RMSE
for method, y in Y.items():
    y = y[:min_T]
    # Pixel-wise RMSE
    rmse = np.sqrt(np.nanmean((y - y_true) ** 2))
    rmse_scores[method] = rmse

# Sort methods by RMSE
sorted_methods = sorted(rmse_scores, key=rmse_scores.get)
sorted_values = [rmse_scores[m] for m in sorted_methods]
sorted_colors = [colors[m] for m in sorted_methods]

print("\n✅ RMSE values:")
for m in sorted_methods:
    print(f"{m:10s}: {rmse_scores[m]:.6f}")

# Plot RMSE Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(sorted_methods, sorted_values, color=sorted_colors)

# Highlight best model with annotation
best = sorted_methods[0]
best_rmse = rmse_scores[best]
plt.text(sorted_methods.index(best), best_rmse,
         " ⭐ Best", ha='center', va='bottom',
         fontsize=12, color="black", fontweight='bold')

# Add value labels on bars
for i, v in enumerate(sorted_values):
    plt.text(i, v + 0.0008, f"{v:.3f}", ha='center', fontsize=9)

plt.ylabel("RMSE (Lower = Better)", fontsize=12)
plt.title("RMSE Comparison - All Reconstruction Methods", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.35)
plt.tight_layout()

save_path = "../../results/Comparison/rmse_methods_bar_improved.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"\n✅ RMSE plot saved at: {save_path}")
