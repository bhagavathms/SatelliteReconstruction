import numpy as np

arr = np.load("../../data/gapped_h.npy")
print("gapped_h shape:", arr.shape)

arr2 = np.load("../../data/original_h.npy")
print("original_h shape:", arr2.shape)

dates = np.load("../../data/dates_h.npy", allow_pickle=True)
print("dates shape:", dates.shape)


# import numpy as np
# import pandas as pd

# # load your existing 60 dates
# dates60 = np.load("../../data/dates.npy", allow_pickle=True).tolist()

# start = dates60[0]
# end   = dates60[-1]

# dates365 = pd.date_range(start, end, freq='D').to_list()

# np.save("../../data/dates_h.npy", np.array(dates365, dtype=object))

# print("Saved 365-day dates to ../../data/dates_h.npy")


# import numpy as np

# dates60 = np.load("../../data/dates_h.npy", allow_pickle=True)
# print("First date:", dates60[0])
# print("Last date:", dates60[-1])
# print("Total span days:", (dates60[-1] - dates60[0]).days)