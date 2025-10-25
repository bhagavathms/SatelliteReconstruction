import numpy as np
import satsmooth as sm
from satsmooth.utils import prepare_x, nd_to_columns, columns_to_nd

y = np.load('../../data/gapped.npy')
dates = np.load('../../data/dates.npy', allow_pickle=True)

dims, nrows, ncols = y.shape

print("⏳ DTS running on Synthetic NDVI...")
start = dates.min().strftime('%Y-%m-%d')
end   = dates.max().strftime('%Y-%m-%d')
xinfo = prepare_x(dates, start, end, rule='D2', skip='N', write_skip=10)


y_cols = nd_to_columns(y, dims, nrows, ncols)
interpolator = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)
indices = np.ascontiguousarray(xinfo.skip_idx + xinfo.start_idx, dtype='uint64')

rec_cols = interpolator.interpolate_smooth(
    np.ascontiguousarray(y_cols, dtype='float64'),
    fill_no_data=True,
    remove_outliers=True,
    return_indexed=True,
    indices=indices,
    n_jobs=4
)

rec = columns_to_nd(rec_cols, rec_cols.shape[1], nrows, ncols)
np.save('../../results/DTS/recon.npy', rec)

print("✅ DTS Done!")
print("Output Shape:", rec.shape)
