import numpy as np
from scipy.interpolate import CubicSpline
import os

os.makedirs('../../results/Cubic', exist_ok=True)

orig = np.load('../../data/original.npy')
gap = np.load('../../data/gapped.npy')

T, H, W = gap.shape
rec = np.zeros_like(orig)

print("⏳ Running Cubic Spline...")

for i in range(H):
    for j in range(W):
        y = gap[:, i, j]
        x = np.arange(T)

        valid = ~np.isnan(y)
        if valid.sum() < 4:
            rec[:, i, j] = np.nan  # not enough points
            continue

        spline = CubicSpline(x[valid], y[valid])
        rec[:, i, j] = spline(x)

np.save('../../results/Cubic/recon.npy', rec)
print("✅ Cubic Spline Reconstruction saved!")
print("Output shape:", rec.shape)
