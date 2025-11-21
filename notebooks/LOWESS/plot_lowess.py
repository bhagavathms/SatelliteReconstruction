import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
y_original = np.load("../../data/original_h.npy")
y_gap = np.load("../../data/gapped_h.npy")
y_lowess = np.load("../../results/LOWESS/recon_h.npy")

# Choose pixel (center)
r, c = 25, 30

# Adjust plot to only show original time length
T = y_original.shape[0]

plt.figure(figsize=(10,5))
plt.plot(y_original[:, r, c], "g-", label="Original")
plt.plot(y_gap[:, r, c], "bo", label="Gaps")
plt.plot(y_lowess[:T, r, c], "r-", label="LOWESS")

plt.title(f"LOWESS Reconstruction")
plt.xlabel("Time")
plt.ylabel("EVI2")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save Figure
save_path = f"../../results/LOWESS/lowess_h.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Plot saved at: {save_path}")
