import numpy as np
import matplotlib.pyplot as plt

# Load data
orig = np.load('../../data/original_h.npy')
gap = np.load('../../data/gapped_h.npy')
rec = np.load('../../results/Cubic/recon_h.npy')

# Choose a pixel to validate
r, c = 25, 30

plt.figure(figsize=(10,5))
plt.plot(orig[:,r,c], 'y-', label='Original EVI2')
plt.plot(gap[:,r,c], 'bo', label='Gaps')
plt.plot(rec[:,r,c], 'g-', label='Cubic Spline Reconstruction')

plt.title("Cubic Spline Reconstruction")
plt.xlabel("Time")
plt.ylabel("EVI2 Value")
plt.grid(True)
plt.legend()

output_path = '../../results/Cubic/Cubic_h.png'
plt.savefig(output_path, dpi=300)
plt.show()
print("✅ Cubic Plot saved:", output_path)
