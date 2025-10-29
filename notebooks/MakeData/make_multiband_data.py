import numpy as np
import datetime, os

os.makedirs("../../data", exist_ok=True)

# Parameters
T, H, W, B = 100, 80, 80, 4  # 4 bands: Blue, Green, Red, NIR

np.random.seed(42)
t = np.arange(T)

# Base vegetation seasonal signal
season = 0.4 + 0.3 * np.sin(2 * np.pi * t / T)

# Each band has its own spectral scaling factor
band_scales = [0.5, 0.7, 0.9, 1.1]  # Blue < ... < NIR

Y = np.zeros((T, H, W, B), dtype=np.float32)
for b in range(B):
    noise = 0.05 * np.random.randn(T, H, W)
    Y[:, :, :, b] = band_scales[b] * season[:, None, None] + noise

# Introduce 40% missing values
mask = np.random.rand(T, H, W, B) < 0.4
Y_gap = Y.copy()
Y_gap[mask] = np.nan

# Dates array
dates = np.array([
    datetime.datetime(2018,1,1) + datetime.timedelta(days=16*k)
    for k in range(T)
], dtype=object)

# Save
np.save("../../data/multi_original.npy", Y)
np.save("../../data/multi_gapped.npy", Y_gap)
np.save("../../data/multi_dates.npy", dates)

print("✅ Multi-band dataset saved: (T,H,W,B) =", Y.shape)
