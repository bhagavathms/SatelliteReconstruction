import numpy as np
import matplotlib.pyplot as plt
import os

# Load required data
y_original = np.load("../../data/original.npy")
y_gap = np.load("../../data/gapped.npy")
y_whitt = np.load("../../results/Whittaker/recon.npy")

# Pick a pixel for validation
r, c = 50, 50
T = y_original.shape[0]  # Trim in case sizes differ

plt.figure(figsize=(10, 5))

plt.plot(y_original[:, r, c], "g-", label="Original")
plt.plot(y_gap[:, r, c], "bo", markersize=4, label="Gaps")
plt.plot(y_whitt[:T, r, c], "m-", label="Whittaker")

plt.title("Whittaker Reconstruction")
plt.xlabel("Time")
plt.ylabel("EVI2")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_path = "../../results/Whittaker/whitt.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Plot saved at: {save_path}")
