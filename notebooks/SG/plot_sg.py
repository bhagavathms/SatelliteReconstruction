import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
y_original = np.load("../../data/original_h.npy")
y_gap = np.load("../../data/gapped_h.npy")
y_sg = np.load("../../results/SG/recon_h.npy")

# Choose pixel
r, c = 25, 30
T = y_original.shape[0]

plt.figure(figsize=(10,5))
plt.plot(y_original[:, r, c], "g-", label="Original")
plt.plot(y_gap[:, r, c], "bo", label="Gaps")
plt.plot(y_sg[:T, r, c], "r-", label="SG")

plt.title("Savitzky–Golay Reconstruction")
plt.xlabel("Time")
plt.ylabel("EVI2")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_path = "../../results/SG/sg_h.png"
plt.savefig(save_path, dpi=300)
plt.show()

print("✅ Plot saved at:", save_path)
