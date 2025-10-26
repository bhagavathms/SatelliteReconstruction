import numpy as np, os
from scipy.ndimage import uniform_filter
from tqdm import tqdm

ORIG = "../../data/original.npy"   # fine true (T,H,W)
GAP  = "../../data/gapped.npy"     # sparse fine for comparison
OUT  = "../../results/STARFM"; os.makedirs(OUT, exist_ok=True)

y_true = np.load(ORIG).astype("float32")
y_gap  = np.load(GAP).astype("float32")
T,H,W  = y_true.shape

# 1️⃣ Generate "MODIS-like" coarse daily by smoothing spatially
# simulate coarse 5x5 pixel blending
def to_coarse(img, k=5):
    return uniform_filter(img, size=(0,k,k), mode='nearest')

y_modis_daily = to_coarse(y_true, k=5)  # (T,H,W)

# 2️⃣ Sparse "Landsat" reference dates: same as gaps ~ every 10th sample
landsat_idx = np.arange(0, T, 10)

L_ref = y_true[landsat_idx]        # fine at sparse dates
M_ref = y_modis_daily[landsat_idx] # coarse same days

# 3️⃣ STARFM core: Apply reference delta to all coarse dates
recon = np.zeros_like(y_true)
for t in tqdm(range(T), desc="STARFM Fusion"):
    j = landsat_idx[np.argmin(np.abs(landsat_idx - t))]
    delta = L_ref[np.where(landsat_idx==j)[0][0]] - M_ref[np.where(landsat_idx==j)[0][0]]
    recon[t] = y_modis_daily[t] + delta

recon = np.clip(recon, 0.0, 1.0)
np.save(f"{OUT}/recon.npy", recon)
print("✅ STARFM reconstruction saved →", f"{OUT}/recon.npy")
