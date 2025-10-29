import numpy as np
import satsmooth as sm
from satsmooth.utils import prepare_x, nd_to_columns, columns_to_nd
import datetime, os

os.makedirs("../../results/DTS_multi", exist_ok=True)

Y_gap = np.load("../../data/multi_gapped.npy")  # (T,H,W,B)
dates = np.load("../../data/multi_dates.npy", allow_pickle=True)


T, H, W, B = Y_gap.shape

# Prepare time axis
xinfo = prepare_x(
    dates,
    str(dates[0].date()),
    str(dates[-1].date()),
    rule='D2',
    skip='N',
    write_skip=10
)

# Flatten: (samples, time)
Yf = Y_gap.copy()
Yf = Yf.transpose(3,0,1,2).reshape(B, T, H*W)  # (B,T,N)

out_list = []

for b in range(B):
    print(f"✔ Processing band {b+1}/{B}")
    y = Yf[b].T  # (samples,time)
    interpolator = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)
    y_smooth = interpolator.interpolate_smooth(
        np.ascontiguousarray(y, dtype='float64'),
        fill_no_data=True,
        remove_outliers=True,
        return_indexed=True,
        indices = np.ascontiguousarray(xinfo.skip_idx + xinfo.start_idx, dtype="uint64")
    )
    out_list.append(y_smooth.T)


# Convert list → array (B, T_new, N)
Yout = np.stack(out_list, axis=0)
print("Intermediate stacked shape:", Yout.shape)

# Restore spatial layout: (B, T_new, H, W) → (T_new, H, W, B)
T_new = Yout.shape[1]
# Rearrange → (T_new, N, B)
Yout = Yout.transpose(1, 2, 0)

# Restore spatial shape
Yout = Yout.reshape(Yout.shape[0], H, W, B)


# Save result
np.save("../../results/DTS_multi/recon.npy", Yout)
print("✅ DTS Multi-band reconstruction saved! Shape:", Yout.shape)

