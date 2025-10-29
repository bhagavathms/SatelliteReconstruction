import numpy as np
import os

os.makedirs("../../results/DTS_GAN", exist_ok=True)

# Load inputs
y_true = np.load("../../data/original.npy")
y_gap  = np.load("../../data/gapped.npy")
y_dts  = np.load("../../results/DTS/recon1.npy")
y_gan  = np.load("../../results/GAN/recon.npy")

# Match time dimension
min_T = min(y_true.shape[0], y_gap.shape[0], y_dts.shape[0], y_gan.shape[0])
y_gap = y_gap[:min_T]
y_dts = y_dts[:min_T]
y_gan = y_gan[:min_T]

# Identify missing values (gaps)
mask = np.isnan(y_gap) | (y_gap == 0)

# Fuse = DTS baseline + GAN replaced only where missing
y_dts_gan = y_dts.copy()
y_dts_gan[mask] = y_gan[mask]

# Save final hybrid output
np.save("../../results/DTS_GAN/recon.npy", y_dts_gan)
print("✅ DTS+GAN Fusion Completed!")
print("✅ Shape:", y_dts_gan.shape)
