import numpy as np
from pybaselines.whittaker import asls
from tqdm import tqdm
import os

os.makedirs("../../results/Whittaker", exist_ok=True)

# Load input data
y_gap = np.load("../../data/gapped.npy")
time_steps, rows, cols = y_gap.shape

y_whitt = np.zeros_like(y_gap)

for i in tqdm(range(rows), desc="Whittaker"):
    for j in range(cols):
        y = np.nan_to_num(y_gap[:, i, j])
        baseline, _ = asls(y, lam=50, p=0.01)
        y_whitt[:, i, j] = baseline

np.save("../../results/Whittaker/recon.npy", y_whitt)
print("✅ Whittaker Reconstruction Saved → ../../results/Whittaker/recon.npy")
