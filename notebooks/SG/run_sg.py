# import numpy as np
# from scipy.signal import savgol_filter
# from tqdm import tqdm
# import os

# os.makedirs("../../results/SG", exist_ok=True)

# # Load input data
# y_gap = np.load("../../data/gapped.npy")

# time_steps, rows, cols = y_gap.shape

# # Output array
# y_sg = np.zeros_like(y_gap)

# for i in tqdm(range(rows), desc="Savitzky-Golay"):
#     for j in range(cols):
#         y = np.nan_to_num(y_gap[:, i, j])

#         # Ensure window length < number of time points & odd
#         window = min(21, time_steps - 1)
#         if window % 2 == 0:
#             window -= 1  # make odd

#         # Apply SG filter
#         y_sg[:, i, j] = savgol_filter(y, window, polyorder=3)

# np.save("../../results/SG/recon.npy", y_sg)
# print("✅ SG Reconstruction Saved → ../../results/SG/recon.npy")



import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm
import os

os.makedirs("../../results/SG", exist_ok=True)

# Load the gapped NDVI data
y_gap = np.load("../../data/gapped_h.npy")

T, H, W = y_gap.shape
x = np.arange(T)

y_sg = np.zeros_like(y_gap)

print("⏳ Running Savitzky–Golay smoothing...")

for i in tqdm(range(H)):
    for j in range(W):

        y = y_gap[:, i, j]
        mask = np.isfinite(y)

        if mask.sum() < 7:      # SG requires enough valid points
            y_sg[:, i, j] = np.nan_to_num(y, nan=0.0)
            continue

        # window length must be odd and < number of valid points
        win = min(21, mask.sum() - 1)
        if win % 2 == 0:
            win -= 1

        # apply SG filter on valid points only
        smooth_valid = savgol_filter(y[mask], window_length=win, polyorder=3)

        # interpolate back to full timeline
        smooth_full = np.interp(x, x[mask], smooth_valid)

        # NDVI valid range
        smooth_full = np.clip(smooth_full, 0, 1)

        y_sg[:, i, j] = smooth_full

np.save("../../results/SG/recon_h.npy", y_sg)
print("✅ SG Reconstruction Saved → ../../results/SG/recon_h.npy")
