import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
y_original = np.load("../../data/original.npy")
y_gap = np.load("../../data/gapped.npy")
y_lowess = np.load("../../results/LOWESS/recon.npy")

# Choose pixel (center)
r, c = 50, 50

# Adjust plot to only show original time length
T = y_original.shape[0]

plt.figure(figsize=(10,5))
plt.plot(y_original[:, r, c], "g-", label="Original (True)")
plt.plot(y_gap[:, r, c], "bo", label="Gapped Input")
plt.plot(y_lowess[:T, r, c], "r-", label="LOWESS")

plt.title(f"LOWESS Reconstruction Validation — Pixel ({r},{c})")
plt.xlabel("Time Index")
plt.ylabel("EVI2")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save Figure
save_path = f"../../results/LOWESS/lowess.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Plot saved at: {save_path}")
