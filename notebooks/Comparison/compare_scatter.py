import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# File mapping for each method
files = {
    "DTS": "../../results/DTS/recon1.npy",
    "Cubic": "../../results/Cubic/recon.npy",
    "LOWESS": "../../results/LOWESS/recon.npy",
    "SG": "../../results/SG/recon.npy",
    "Whittaker": "../../results/Whittaker/recon.npy",
    "GAN": "../../results/GAN/recon.npy",
    "DTS-GAN": "../../results/DTS_GAN/recon.npy"
}

colors = {
    "DTS": "red",
    "Cubic": "blue",
    "LOWESS": "green",
    "SG": "purple",
    "Whittaker": "orange",
    "GAN": "olive",
    "DTS-GAN": "#AA00FF"  # standout purple-pink
}

os.makedirs("../../results/Comparison", exist_ok=True)

# Load ground truth
y_true = np.load("../../data/original.npy")
min_T = y_true.shape[0]

# ✅ Determine common temporal length
for method, fpath in files.items():
    y = np.load(fpath)
    min_T = min(min_T, y.shape[0])

print("✅ Common Time Length:", min_T)

# ✅ Flatten ground truth for comparison
y_true_flat = y_true[:min_T].flatten()

# ✅ Random sampling to avoid clutter
N = min(10000, len(y_true_flat))
idx = np.random.choice(len(y_true_flat), N, replace=False)
y_true_flat = y_true_flat[idx]

# ✅ Start plotting
plt.figure(figsize=(8, 8))

# ✅ 1:1 line
min_val, max_val = y_true_flat.min(), y_true_flat.max()
plt.plot([min_val, max_val], [min_val, max_val],
         'k--', lw=2, label="1:1 Line")

metrics_text = ["Method | RMSE | MAE | R²", "-----------------------------------"]

# ✅ Plot each method
for method, fpath in files.items():
    y = np.load(fpath)[:min_T]
    y_flat = y.flatten()[idx]

    # ✅ Metrics
    mse = mean_squared_error(y_true_flat, y_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_flat)
    r2 = r2_score(y_true_flat, y_flat)

    metrics_text.append(f"{method:9s} | {rmse:.4f} | {mae:.4f} | {r2:.4f}")

    plt.scatter(
        y_true_flat, y_flat,
        s=6, alpha=0.20,
        color=colors[method],
        label=f"{method}"
    )

# ✅ Final plot settings
plt.title("Original vs Reconstructed Values (All Methods)", fontsize=14)
plt.xlabel("Original Pixel Reflectance", fontsize=12)
plt.ylabel("Reconstructed Pixel Reflectance", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(markerscale=3)
plt.tight_layout()

save_path = "../../results/Comparison/user_vs_recon_scatter_improved.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"\n✅ Scatter Plot Saved → {save_path}\n")
print("\n📌 Metrics Report:")
for line in metrics_text:
    print(line)
