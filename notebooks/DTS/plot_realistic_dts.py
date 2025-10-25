import numpy as np
import matplotlib.pyplot as plt

# Load dataset
orig = np.load('../../data/original.npy')
gap = np.load('../../data/gapped.npy')
rec = np.load('../../results/DTS/recon.npy')

# DTS returned more time steps — match lengths
rec = rec[:orig.shape[0]]

# Select any pixel — center pixel recommended
r, c = 50, 50

plt.figure(figsize=(10, 5))
plt.plot(orig[:, r, c], 'g-', label='Original NDVI')
plt.plot(gap[:, r, c], 'bo', label='Gapped (Input)')
plt.plot(rec[:, r, c], 'r-', label='DTS Reconstruction')

plt.title("DTS Reconstruction of EVI2 (Pixel 50,50)")
plt.xlabel("Time Index")
plt.ylabel("EVI2 Value")
plt.grid(True)
plt.legend(title="Time Series")

# Save plot
out = '../../results/DTS/DTS.png'
plt.savefig(out, dpi=300)

print("✅ Plot Created and Saved :", out)
plt.show()
