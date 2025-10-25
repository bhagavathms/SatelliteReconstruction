import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm
import os

os.makedirs("../../results/SG", exist_ok=True)

# Load input data
y_gap = np.load("../../data/gapped.npy")

time_steps, rows, cols = y_gap.shape

# Output array
y_sg = np.zeros_like(y_gap)

for i in tqdm(range(rows), desc="Savitzky-Golay"):
    for j in range(cols):
        y = np.nan_to_num(y_gap[:, i, j])

        # Ensure window length < number of time points & odd
        window = min(21, time_steps - 1)
        if window % 2 == 0:
            window -= 1  # make odd

        # Apply SG filter
        y_sg[:, i, j] = savgol_filter(y, window, polyorder=3)

np.save("../../results/SG/recon.npy", y_sg)
print("✅ SG Reconstruction Saved → ../../results/SG/recon.npy")
