import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
import os

os.makedirs("../../results/LOWESS", exist_ok=True)

# Load data
y_gap = np.load("../../data/gapped.npy")

time_steps, rows, cols = y_gap.shape
x = np.arange(time_steps)

y_lowess = np.zeros_like(y_gap)

for i in tqdm(range(rows)):
    for j in range(cols):
        y = y_gap[:, i, j]
        mask = np.isfinite(y)
        if mask.sum() > 5:
            vals = lowess(y[mask], x[mask], frac=0.2, return_sorted=False)
            # Interpolate back to original size
            y_lowess[:, i, j] = np.interp(x, x[mask], vals)

        else:
            y_lowess[:, i, j] = y

np.save("../../results/LOWESS/recon.npy", y_lowess)
print("✅ LOWESS Reconstruction Saved → ../../results/LOWESS/recon.npy")
