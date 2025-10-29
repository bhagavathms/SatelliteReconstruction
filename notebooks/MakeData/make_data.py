import numpy as np, datetime, os
os.makedirs('../../data', exist_ok=True)

# Dataset size
time_steps = 100      # 100 time points
rows, cols = 100, 100 # 100x100 pixel region

np.random.seed(42)
t = np.arange(time_steps)

# Seasonal vegetation cycle (EVI2-like)
base = 0.5 + 0.3 * np.sin(2 * np.pi * t / time_steps)

# Build dataset with slight noise variation per pixel
y = np.zeros((time_steps, rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        y[:, i, j] = base + 0.05 * np.random.randn(time_steps)

# Introduce 40% random missing values (gaps)
gap_ratio = 0.40
mask = np.random.rand(*y.shape) < gap_ratio
y_gap = y.copy()
y_gap[mask] = np.nan

# Landsat-like dates (100 images spaced by ~16 days)
dates = np.array([
    datetime.datetime(2017,1,1) + datetime.timedelta(days=16*k)
    for k in range(time_steps)
], dtype=object)

# Save to data folder
np.save('../../data/original.npy', y)
np.save('../../data/gapped.npy', y_gap)
np.save('../../data/dates.npy', dates)

print("✅ Synthetic dataset created: 100x100 pixels, 100 time steps, 40% gaps")
