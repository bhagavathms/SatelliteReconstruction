import numpy as np, datetime, os
from scipy.ndimage import gaussian_filter

os.makedirs("../../data", exist_ok=True)

# Dimensions
time_steps = 60  # ~ 3 years seasonal cycles
rows, cols = 100, 100

# Seasonal NDVI base waveform (real-like vegetation)
t = np.linspace(0, 2*np.pi, time_steps)
season = 0.6 + 0.25*np.sin(t)  # mean ~0.6, amplitude ~0.25

# Spatial structure: smoothed noise
np.random.seed(42)
noise_map = gaussian_filter(np.random.rand(rows, cols), sigma=3)
noise_map = 0.1 * (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())

# Construct NDVI cube (T,H,W)
y = np.zeros((time_steps, rows, cols), dtype=np.float32)
for i in range(time_steps):
    y[i] = season[i] + noise_map + np.random.normal(0, 0.02, (rows, cols))

y = np.clip(y, 0.05, 0.95)  # realistic NDVI bounds

# Add realistic cloud gaps: 30% time steps → 20x20 patches erased
mask = np.zeros_like(y, dtype=bool)
for i in range(time_steps):
    if np.random.rand() < 0.3:
        x = np.random.randint(0, rows-20)
        yx = np.random.randint(0, cols-20)
        mask[i, x:x+20, yx+20] = True

y_gap = y.copy()
y_gap[mask] = np.nan

# Generate Landsat-like dates every 16 days
start_date = datetime.datetime(2017,1,1)
dates = np.array([start_date + datetime.timedelta(days=16*i)
                 for i in range(time_steps)], dtype=object)

np.save("../../data/original.npy", y)
np.save("../../data/gapped.npy", y_gap)
np.save("../../data/dates.npy", dates)

print("✅ Realistic Synthetic NDVI Created!")
print("Shape:", y.shape)
print("Gap Ratio:", np.isnan(y_gap).sum() / y_gap.size)
