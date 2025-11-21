# import numpy as np
# from statsmodels.nonparametric.smoothers_lowess import lowess
# from tqdm import tqdm
# import os

# os.makedirs("../../results/LOWESS", exist_ok=True)

# # Load data
# y_gap = np.load("../../data/gapped_h.npy")

# time_steps, rows, cols = y_gap.shape
# x = np.arange(time_steps)

# y_lowess = np.zeros_like(y_gap)

# for i in tqdm(range(rows)):
#     for j in range(cols):
#         y = y_gap[:, i, j]
#         mask = np.isfinite(y)
#         if mask.sum() > 5:
#             vals = lowess(y[mask], x[mask], frac=0.2, return_sorted=False)
#             # Interpolate back to original size
#             y_lowess[:, i, j] = np.interp(x, x[mask], vals)

#         else:
#             y_lowess[:, i, j] = y

# np.save("../../results/LOWESS/recon_h.npy", y_lowess)
# print("✅ LOWESS Reconstruction Saved → ../../results/LOWESS/recon.npy")




import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
import os

os.makedirs("../../results/LOWESS", exist_ok=True)

# Load data
y_gap = np.load("../../data/gapped_h.npy")

T, H, W = y_gap.shape
x = np.arange(T)

# Output
y_lowess = np.zeros_like(y_gap)

print("⏳ Running LOWESS smoothing...")

for i in tqdm(range(H)):
    for j in range(W):
        y = y_gap[:, i, j]
        mask = np.isfinite(y)

        if mask.sum() >= 6:
            # dynamic smoothing strength
            frac = min(0.3, 20 / mask.sum())

            # lowess on only valid points
            smoothed = lowess(y[mask], x[mask], frac=frac, return_sorted=False)

            # interpolate to full timeline
            y_interp = np.interp(x, x[mask], smoothed)

            # clip NDVI range
            y_lowess[:, i, j] = np.clip(y_interp, 0, 1)

        else:
            # too few points → fallback to original
            y_lowess[:, i, j] = y

# Save
np.save("../../results/LOWESS/recon_h.npy", y_lowess)

print("✅ LOWESS Reconstruction Saved → ../../results/LOWESS/recon_h.npy")
