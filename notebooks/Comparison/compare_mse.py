import numpy as np
import matplotlib.pyplot as plt

# Load original reference truth
y_true = np.load("../../data/original.npy")

# Load reconstructions
Y = {
    "DTS": np.load("../../results/DTS/recon1.npy"),
    "Cubic": np.load("../../results/Cubic/recon.npy"),
    "LOWESS": np.load("../../results/LOWESS/recon.npy"),
    "SG": np.load("../../results/SG/recon.npy"),
    "Whittaker": np.load("../../results/Whittaker/recon.npy"),
}

mse_scores = {}

for method, y in Y.items():
    mse_scores[method] = np.nanmean((y - y_true)**2)

print("✅ MSE values:")
for m, v in mse_scores.items():
    print(m, ":", v)

# Plot MSE Bar Chart
plt.figure(figsize=(7,5))
methods = list(mse_scores.keys())
values = list(mse_scores.values())

plt.bar(methods, values)
plt.ylabel("Mean Squared Error")
plt.title("MSE Comparison - All Methods")
plt.grid(True, linestyle='--', alpha=0.4)

save_path = "../../results/Comparison/mse_methods_bar.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ MSE plot saved at: {save_path}")
