import numpy as np
import matplotlib.pyplot as plt

# File mapping for each method
files = {
    "DTS": "../../results/DTS/recon1.npy",
    "Cubic": "../../results/Cubic/recon.npy",
    "LOWESS": "../../results/LOWESS/recon.npy",
    "SG": "../../results/SG/recon.npy",
    "Whittaker": "../../results/Whittaker/recon.npy"
}

colors = {
    "DTS": "r",
    "Cubic": "b",
    "LOWESS": "g",
    "SG": "m",
    "Whittaker": "orange"
}

# Load ground truth
y_true = np.load("../../data/original.npy")

# ✅ Find common time length
min_T = y_true.shape[0]
for method, fpath in files.items():
    y = np.load(fpath)
    min_T = min(min_T, y.shape[0])

print("✅ Common Time Length:", min_T)

# ✅ Flatten truth
y_true_flat = y_true[:min_T].flatten()

plt.figure(figsize=(8, 8))

# 1:1 line
min_val, max_val = y_true_flat.min(), y_true_flat.max()
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 Line")

# ✅ Plot each method
for method, fpath in files.items():
    y = np.load(fpath)[:min_T]
    y_flat = y.flatten()
    plt.scatter(y_true_flat, y_flat,
                s=4, alpha=0.25,
                color=colors[method],
                label=method)

plt.title("User vs Reconstructed Values (All Methods)")
plt.xlabel("Original (User)")
plt.ylabel("Reconstructed")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

save_path = "../../results/Comparison/user_vs_recon_scatter.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Scatter Plot Saved → {save_path}")
