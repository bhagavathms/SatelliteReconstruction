# '''
import numpy as np
from scipy.interpolate import CubicSpline
import os

os.makedirs('../../results/Cubic', exist_ok=True)

orig = np.load('../../data/original_h.npy')
gap = np.load('../../data/gapped_h.npy')

T, H, W = gap.shape
rec = np.full_like(orig, np.nan)  # start with NaN

print("⏳ Running Improved Cubic Spline...")

x = np.arange(T)

for i in range(H):
    for j in range(W):
        y = gap[:, i, j]
        valid = ~np.isnan(y)

        # Must have at least 4 valid points
        if valid.sum() < 4:
            continue

        xv = x[valid]
        yv = y[valid]

        try:
            # ✅ Cubic only inside data range
            cubic = CubicSpline(xv, yv, extrapolate=False)
            yc = cubic(x)

            # ✅ Linear fallback for extrapolation edges
            if np.isnan(yc).any():
                lin = np.interp(x, xv, yv)
                yc = np.where(np.isnan(yc), lin, yc)

            # ✅ Clip physically valid reflectance
            rec[:, i, j] = np.clip(yc, 0, 1)

        except:
            continue  # Skip if curve fails

np.save('../../results/Cubic/recon_h.npy', rec)
print("✅ Improved Cubic Reconstruction saved!")
print("Output shape:", rec.shape)
# '''



