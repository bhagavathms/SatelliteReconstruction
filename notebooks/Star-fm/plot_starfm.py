import numpy as np
import matplotlib.pyplot as plt

orig = np.load("../../data/original.npy")
star = np.load("../../results/STARFM/recon.npy")
gap  = np.load("../../data/gapped.npy")

r,c = 40,40  # sample pixel

plt.figure(figsize=(10,5))
plt.plot(orig[:,r,c], label="Original", lw=2.5, color='green')
plt.plot(gap[:,r,c], 'o-', alpha=0.4, label="Gapped Input", color='gray')
plt.plot(star[:,r,c], '-', lw=2, label="STARFM", color='blue')

plt.title("STARFM Reconstruction vs Original vs Gapped")
plt.xlabel("Time Index")
plt.ylabel("EVI2 Value")
plt.grid(True, alpha=0.3)
plt.legend()

path = "../../results/STARFM/starfm_comparison_pixel.png"
plt.savefig(path, dpi=300)
print(f"✅ Saved: {path}")
plt.show()
