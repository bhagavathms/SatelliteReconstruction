import numpy as np, os
from sklearn.model_selection import train_test_split

# Inputs (from Stage-1 synthetic)
Y_FULL = "../../data/original.npy"  # (T,H,W)
Y_GAP  = "../../data/gapped.npy"    # (T,H,W)
OUT    = "../../data/gan_dataset"
os.makedirs(OUT, exist_ok=True)

y_full = np.load(Y_FULL)  # float32
y_gap  = np.load(Y_GAP)
T,H,W  = y_full.shape

# Flatten pixels → (Npix, T)
X = y_gap.transpose(1,2,0).reshape(H*W, T)
Y = y_full.transpose(1,2,0).reshape(H*W, T)

# Keep only pixels with some valid data
valid = np.isfinite(X).sum(axis=1) > int(0.4*T)
X = np.nan_to_num(X[valid], nan=0.0)
Y = np.nan_to_num(Y[valid], nan=0.0)

# Subsample to keep training lightweight
N = min(12000, X.shape[0])
rng = np.random.default_rng(42)
idx = rng.choice(X.shape[0], size=N, replace=False)
X, Y = X[idx], Y[idx]

# Train/val split
Xtr,Xva,Ytr,Yva = train_test_split(X, Y, test_size=0.15, random_state=42)

np.save(f"{OUT}/X_train.npy", Xtr)
np.save(f"{OUT}/Y_train.npy", Ytr)
np.save(f"{OUT}/X_val.npy",   Xva)
np.save(f"{OUT}/Y_val.npy",   Yva)
print("✅ GAN dataset saved in", OUT, "| shapes:", Xtr.shape, Ytr.shape, Xva.shape, Yva.shape)
