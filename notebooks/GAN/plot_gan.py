import numpy as np
import matplotlib.pyplot as plt

# Load true + gapped + GAN reconstruction
orig   = np.load("../../data/original.npy")
gapped = np.load("../../data/gapped.npy")
gan    = np.load("../../results/GAN/recon.npy")

# Choose pixel
r, c = 40, 40  

y_orig = orig[:, r, c]
y_gap  = gapped[:, r, c]
y_gan  = gan[:, r, c]

plt.figure(figsize=(10,5))

# True ground truth
plt.plot(y_orig, label="Original True", color='green', lw=2.5)

# Gapped input
plt.plot(y_gap, 'o-', alpha=0.4, label="Gapped Input", color='gray')

# GAN reconstruction
plt.plot(y_gan, '-', lw=2.5, label="GAN Reconstruction", color='red')

plt.title("GAN Reconstruction vs Original vs Gapped")
plt.xlabel("Time Index")
plt.ylabel("EVI2 Value")
plt.grid(True, alpha=0.3)
plt.legend()

save_path = "../../results/GAN/gan_reconstruction_comparison.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Saved: {save_path}")

plt.show()
