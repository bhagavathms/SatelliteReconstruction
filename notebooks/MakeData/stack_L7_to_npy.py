import os, re, glob
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from datetime import datetime

ROOT = Path("../../data/L7_salta/tifs")
OUT  = Path("../../data/L7_salta_npy")
OUT.mkdir(parents=True, exist_ok=True)

# === CONFIG ===
SCALE = 4  # 4x coarser in each dimension → ~16x less RAM/IO
BANDS_WANTED = ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"]

# Landsat C2 L2 SR scale/offset + fill
SR_SCALE  = 0.0000275
SR_OFFSET = -0.2
FILL_VALS = {-9999, -9998}

# Find all TIFs
all_tifs = sorted(glob.glob(str(ROOT / "*.TIF")))
if not all_tifs:
    raise SystemExit(f"No TIFs found in {ROOT}")

# Group by date key from filename (LE07_L2SP_231077_YYYYMMDD_...)
pat = re.compile(r"LE07_L2\w+_\d{6}_(\d{8})_")
by_date = {}
for fp in all_tifs:
    if os.path.getsize(fp) == 0:
        continue
    m = pat.search(os.path.basename(fp))
    if not m:
        continue
    ymd = m.group(1)  # '20190626'
    by_date.setdefault(ymd, []).append(fp)

# Choose a reference grid (first valid SR_B4 we see, else any)
ref_path = None
for ymd, files in sorted(by_date.items()):
    for f in files:
        if "SR_B4" in f and os.path.getsize(f) > 0:
            ref_path = f
            break
    if ref_path:
        break
if ref_path is None:
    for f in all_tifs:
        if os.path.getsize(f) > 0:
            ref_path = f
            break
if ref_path is None:
    raise SystemExit("No non-empty TIFs found to use as reference.")

with rasterio.open(ref_path) as ref:
    src_transform = ref.transform
    src_crs = ref.crs
    src_height, src_width = ref.height, ref.width
    src_xres, src_yres = ref.res  # pixel sizes

# Target grid = same CRS, but downsampled transform/size
dst_crs = src_crs
dst_transform = Affine(
    src_transform.a * SCALE, src_transform.b, src_transform.c,
    src_transform.d, src_transform.e * SCALE, src_transform.f
)
dst_height = src_height // SCALE
dst_width  = src_width  // SCALE

print(f"Reference size: {src_height}x{src_width}, pixel {src_xres}m")
print(f"Target size   : {dst_height}x{dst_width}, pixel {src_xres*SCALE}m")

dates_list = []
stack_list = []  # holds (H,W,6) per date

def read_to_target_grid(src_path):
    """Read band and reproject/resample to target grid."""
    with rasterio.open(src_path) as src:
        # Allocate destination array (float32)
        dst = np.zeros((dst_height, dst_width), dtype=np.float32)

        # Reproject/Resample no matter what (ensures uniform grid)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        # Scale/offset + fill to NaN
        mask = np.isin(dst, list(FILL_VALS))
        dst = dst * SR_SCALE + SR_OFFSET
        dst[mask] = np.nan
        return dst

# Walk dates in order; build per-date 6-band cube
for ymd, files in sorted(by_date.items()):
    # Find wanted bands for this date
    per_date = {}
    for bname in BANDS_WANTED:
        cands = [f for f in files if bname in f and os.path.getsize(f) > 0]
        if cands:
            # if duplicates exist (e.g., " (1).TIF"), pick the largest
            per_date[bname] = max(cands, key=lambda f: os.path.getsize(f))

    # Only keep dates with all 6 bands
    if len(per_date) != len(BANDS_WANTED):
        continue

    try:
        cube = []
        for b in BANDS_WANTED:
            arr = read_to_target_grid(per_date[b])
            cube.append(arr)
        cube = np.stack(cube, axis=-1)  # (H,W,6)
    except Exception as e:
        print(f"Skip {ymd} due to read/reproject error: {e}")
        continue

    dates_list.append(datetime.strptime(ymd, "%Y%m%d").date())
    stack_list.append(cube)

if not stack_list:
    raise SystemExit("No valid (all-6-bands) dates assembled. Check files and names.")

Y = np.stack(stack_list, axis=0)  # (T,H,W,6)
dates = np.array(dates_list, dtype=object)

np.save(OUT / "L7_salta_stack.npy", Y)
np.save(OUT / "L7_salta_dates.npy", dates)
print("✅ Saved:",
      OUT / "L7_salta_stack.npy", Y.shape,
      "|", OUT / "L7_salta_dates.npy", dates.shape)
