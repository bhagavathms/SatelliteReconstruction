import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
orig = np.load("../../data/multi_original.npy")
gap  = np.load("../../data/multi_gapped.npy")
rec  = np.load("../../results/DTS_multi/recon.npy")

band_names = ["Blue", "Green", "Red", "NIR"]
r, c = 40, 40  # pixel location to visualize

plt.figure(figsize=(12, 8))

for b in range(4):
    plt.subplot(2, 2, b + 1)

    # Scatter original missing-data version
    plt.plot(gap[:, r, c, b], "o-", alpha=0.4, label="Gapped")

    # DTS output
    plt.plot(rec[:, r, c, b], "-", lw=2, label="DTS Reconstructed")

    plt.title(f"{band_names[b]} Band")
    plt.xlabel("Time Index")
    plt.ylabel("Reflectance Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.suptitle("DTS Multi-band Reconstruction: Before vs After", fontsize=14)
plt.tight_layout()

save_path = "../../results/DTS_multi/multiband_pixel_plot.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved → {save_path}")

plt.show()
