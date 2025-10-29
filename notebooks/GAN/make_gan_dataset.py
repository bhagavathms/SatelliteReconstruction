# make_gan_dataset.py (v2)
import numpy as np, os, json
from sklearn.model_selection import train_test_split

# Inputs (from Stage-1 synthetic)
Y_FULL = "../../data/original.npy"  # (T,H,W)
Y_GAP  = "../../data/gapped.npy"    # (T,H,W)
OUT    = "../../data/gan_dataset"
os.makedirs(OUT, exist_ok=True)

# ---- Load
y_full = np.load(Y_FULL).astype(np.float32)  # (T,H,W)
y_gap  = np.load(Y_GAP ).astype(np.float32)
T,H,W  = y_full.shape

# ---- Flatten to per-pixel series: (Npix, T)
X = y_gap.transpose(1,2,0).reshape(H*W, T)    # gapped inputs
Y = y_full.transpose(1,2,0).reshape(H*W, T)   # full targets

# ---- Valid mask & keep mask for potential GAN conditioning
valid_counts = np.isfinite(X).sum(axis=1)
keep = valid_counts > int(0.5 * T)            # a bit stricter than 0.4
X_mask = np.isfinite(X[keep]).astype(np.float32)

# ---- Fill NaNs with 0 after saving mask
X = np.nan_to_num(X[keep], nan=0.0)
Y = np.nan_to_num(Y[keep], nan=0.0)

# ---- Optional normalization (commented for synthetic)
# # Per-series min-max (robust for varied scales)
# eps = 1e-6
# x_min = X.min(axis=1, keepdims=True)
# x_max = X.max(axis=1, keepdims=True)
# X = (X - x_min) / (x_max - x_min + eps)
# y_min = Y.min(axis=1, keepdims=True)
# y_max = Y.max(axis=1, keepdims=True)
# Y = (Y - y_min) / (y_max - y_min + eps)

# ---- Subsample to keep training light
N = min(12000, X.shape[0])
rng = np.random.default_rng(42)
idx = rng.choice(X.shape[0], size=N, replace=False)
X, Y, X_mask = X[idx], Y[idx], X_mask[idx]

# ---- Train/val split
Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=0.15, random_state=42)
Mtr, Mva = train_test_split(X_mask, test_size=0.15, random_state=42)

# ---- Save arrays
np.save(f"{OUT}/X_train.npy", Xtr)
np.save(f"{OUT}/Y_train.npy", Ytr)
np.save(f"{OUT}/X_val.npy",   Xva)
np.save(f"{OUT}/Y_val.npy",   Yva)
np.save(f"{OUT}/M_train.npy", Mtr)   # masks (optional input to GAN)
np.save(f"{OUT}/M_val.npy",   Mva)

# ---- Save metadata: shapes & the pixel indices used
meta = {
    "T": int(T), "H": int(H), "W": int(W),
    "N_after_filter": int(X.shape[0] + Xva.shape[0]),  # before split
    "train_size": int(Xtr.shape[0]),
    "val_size": int(Xva.shape[0]),
    "note": "Values are float32; NaNs in X replaced with 0. Mask arrays mark valid(1)/gap(0)."
}
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ GAN dataset saved in", OUT,
      "| shapes:",
      Xtr.shape, Ytr.shape, Xva.shape, Yva.shape,
      "| masks:", Mtr.shape, Mva.shape)
