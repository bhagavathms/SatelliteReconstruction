import numpy as np
from pathlib import Path

IN_STACK = Path("../../data/L7_salta_npy/L7_salta_stack.npy")
IN_DATES = Path("../../data/L7_salta_npy/L7_salta_dates.npy")
OUT_DIR  = Path("../../data/L7_salta_evi2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

Y = np.load(IN_STACK)   # (T,H,W,6)
dates = np.load(IN_DATES, allow_pickle=True)

# Landsat 7 SR band order in our stack: [B1,B2,B3,B4,B5,B7]
B2 = Y[..., 1]  # Green
B4 = Y[..., 3]  # NIR

# EVI2 formula
num = 2.5 * (B4 - B2)
den = (B4 + 2.4 * B2 + 1.0)

EVI2 = np.divide(
    num, den,
    out=np.full_like(num, np.nan, dtype=np.float32),
    where=np.isfinite(den)
).astype(np.float32)

np.save(OUT_DIR / "evi2_original.npy", EVI2)
np.save(OUT_DIR / "evi2_dates.npy", dates)

print("✅ Saved EVI2:", EVI2.shape, "| Dates:", dates.shape)
