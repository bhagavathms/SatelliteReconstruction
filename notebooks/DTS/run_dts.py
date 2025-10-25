import numpy as np
import satsmooth as sm
from satsmooth.utils import prepare_x, nd_to_columns, columns_to_nd
import os

os.makedirs("../../results/DTS", exist_ok=True)

# 1) Load data (keep NaNs as NaNs)
y_gap = np.load("../../data/gapped.npy")                 # (T, H, W)
dates = np.load("../../data/dates.npy", allow_pickle=True).tolist()

T, H, W = y_gap.shape

# 2) Prepare DTS time info
# Use daily smoothing but we'll sample back to the ORIGINAL observation days.
xinfo = prepare_x(dates, str(dates[0].date()), str(dates[-1].date()),
                  rule='D',  # daily smooth grid
                  skip='N',  # no ~weekly skip
                  write_skip=1)

# 3) Flatten to (pixels, time)
Y = nd_to_columns(y_gap, T, H, W).astype(np.float64)

# 4) Build interpolator and get DAILY output (full grid), not indexed
interp = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)

Y_daily = interp.interpolate_smooth(
    np.ascontiguousarray(Y),
    fill_no_data=True,
    no_data_value=np.nan,      # <<< keep NaNs as NoData
    remove_outliers=True,
    max_outlier_days1=120,
    max_outlier_days2=120,
    min_outlier_values=7,
    outlier_iters=1,
    dev_thresh1=0.2,
    dev_thresh2=0.2,
    return_indexed=False,      # <<< get the full daily series
    max_window=61,
    min_window=21,
    mid_g=0.5, r_g=-10.0,
    mid_k=0.5, r_k=-10.0,
    mid_t=0.5, r_t=15.0,
    sigma_color=0.1,
    n_iters=2,
    n_jobs=4
)
# Y_daily shape: (pixels, Td), where Td = len(xinfo.xd_smooth)

# 5) Sample the DAILY curve back to the ORIGINAL observation times
# Find positions of original xd inside xd_smooth
idx = np.searchsorted(xinfo.xd_smooth, xinfo.xd)  # length == T

Y_at_obs = Y_daily[:, idx]         # (pixels, T) aligned with original cadence

# 6) Back to (T, H, W)
y_dts = columns_to_nd(Y_at_obs, T, H, W)

np.save("../../results/DTS/recon1.npy", y_dts)
print("✅ DTS (aligned to original timestamps) → ../../results/DTS/recon1.npy")
