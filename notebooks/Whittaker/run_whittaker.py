# import numpy as np
# from pybaselines.whittaker import asls
# from tqdm import tqdm
# import os

# os.makedirs("../../results/Whittaker", exist_ok=True)

# # Load input data
# y_gap = np.load("../../data/gapped_h.npy")
# time_steps, rows, cols = y_gap.shape

# y_whitt = np.zeros_like(y_gap)

# for i in tqdm(range(rows), desc="Whittaker"):
#     for j in range(cols):
#         y = np.nan_to_num(y_gap[:, i, j])
#         baseline, _ = asls(y, lam=50, p=0.01)
#         y_whitt[:, i, j] = baseline

# np.save("../../results/Whittaker/recon_h.npy", y_whitt)
# print("✅ Whittaker Reconstruction Saved → ../../results/Whittaker/recon_h.npy")



import numpy as np
from pybaselines.whittaker import asls
from tqdm import tqdm
import os

os.makedirs("../../results/Whittaker", exist_ok=True)

# Load data
y_gap = np.load("../../data/gapped_h.npy")

T, H, W = y_gap.shape
x = np.arange(T)

y_whitt = np.zeros_like(y_gap)

print("⏳ Running Whittaker-Eilers smoothing...")

for i in tqdm(range(H)):
    for j in range(W):
        y = y_gap[:, i, j]
        mask = np.isfinite(y)

        if mask.sum() < 6:
            # too few points → fallback
            y_whitt[:, i, j] = np.nan_to_num(y, nan=0.0)
            continue

        # apply Whittaker only on valid points
        baseline, _ = asls(y[mask], lam=200, p=0.05)

        # interpolate back to full timeline
        smooth_full = np.interp(x, x[mask], baseline)

        # NDVI range constraint
        smooth_full = np.clip(smooth_full, 0, 1)

        y_whitt[:, i, j] = smooth_full

# save output
np.save("../../results/Whittaker/recon_h.npy", y_whitt)
print("✅ Whittaker Reconstruction Saved → ../../results/Whittaker/recon_h.npy")
