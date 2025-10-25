import numpy as np
import matplotlib.pyplot as plt

# Load data
orig = np.load('../../data/original.npy')
gap = np.load('../../data/gapped.npy')
rec = np.load('../../results/Cubic/recon.npy')

# Choose a pixel to validate
r, c = 50, 50  # same pixel as DTS for consistent comparison

plt.figure(figsize=(10,5))
plt.plot(orig[:,r,c], 'g-', label='Original EVI2')
plt.plot(gap[:,r,c], 'bo', label='Gapped Input')
plt.plot(rec[:,r,c], 'r-', label='Cubic Spline Reconstruction')

plt.title("Cubic Spline Reconstruction - Pixel (50,50)")
plt.xlabel("Time Index")
plt.ylabel("EVI2 Value")
plt.grid(True)
plt.legend()

output_path = '../../results/Cubic/Cubic.png'
plt.savefig(output_path, dpi=300)
plt.show()
print("✅ Cubic Plot saved:", output_path)
