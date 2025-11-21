import numpy as np
import pandas as pd

# Generate 365 daily timestamps
dates_365 = pd.date_range("2017-01-01", periods=365, freq="D").to_list()

np.save("../../data/dates_h.npy", np.array(dates_365, dtype=object))

print("Saved 365-day dates to ../../data/dates_h.npy")
