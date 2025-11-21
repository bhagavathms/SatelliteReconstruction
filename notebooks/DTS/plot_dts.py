import numpy as np
import matplotlib.pyplot as plt

y_original = np.load("../../data/original_h.npy")
y_gap = np.load("../../data/gapped_h.npy")
y_dts = np.load("../../results/DTS/recon_h.npy")

r, c = 25, 30
T = y_original.shape[0]

plt.figure(figsize=(10,5))
plt.plot(y_original[:, r, c], "g-", label="Original")
plt.plot(y_gap[:, r, c], "bo", label="Gapped")
plt.plot(y_dts[:T, r, c], "r-", label="DTS")

plt.title("DTS Reconstruction — Pixel (50,50)")
plt.xlabel("Time Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_path = "../../results/DTS/dts_h.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Plot saved at: {save_path}")
