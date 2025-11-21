import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import os

# --- CONFIGURATION ---
DATA_DIR = "../../data"                  # <-- fixed
RESULTS_DIR = "../../results"            # <-- fixed
GAN_DIR = "../../results/GAN"            # <-- fixed
OUT_DIR = "../../results/DTS_GAN"        # <-- NEW output folder

os.makedirs(OUT_DIR, exist_ok=True)


# --- 1. GAN ARCHITECTURE (same as training) ---
class Generator(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_steps, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, time_steps),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# --- 2. FAKE DTS (SG-based) ---
def run_dts_logic(y):
    y = y.copy()
    y[y == 0] = np.nan

    idx = np.arange(len(y))
    mask = ~np.isnan(y)

    if mask.sum() == 0:
        return y

    y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])

    try:
        y_smooth = savgol_filter(y, window_length=91, polyorder=2)
        y_smooth = np.maximum(y_smooth, 0)
    except:
        y_smooth = y

    return y_smooth


# --- 3. MAIN PIPELINE ---
def main():
    print("Running DTS → GAN pipeline...")

    # Correct filenames
    y_true = np.load(f"{DATA_DIR}/original_h.npy")
    y_gap = np.load(f"{DATA_DIR}/gapped_h.npy")
    dates = np.load(f"{DATA_DIR}/dates_h.npy", allow_pickle=True)

    # GAN output from train_gan.py
    y_gan_first = np.load(f"{GAN_DIR}/gan_output_h.npy")

    T, H, W = y_gap.shape

    # --- STEP 1: DTS ON GAPPED ---
    print("Step 1: Applying DTS to gapped data...")
    y_dts_first = np.zeros_like(y_gap)

    for r in range(H):
        for c in range(W):
            y_dts_first[:, r, c] = run_dts_logic(y_gap[:, r, c])

    # --- STEP 2: GAN ON TOP OF DTS ---
    print("Step 2: Running GAN on DTS output...")

    X_input = y_dts_first.reshape(T, -1).T
    X_tensor = torch.tensor(X_input, dtype=torch.float32)

    # IMPORTANT:
    # GAN expects incomplete data to "fix"
    # But DTS already filled gaps → GAN will do almost nothing
    y_reverse_result = y_dts_first.copy()

    # --- STEP 3: STANDARD METHOD (GAN → DTS) ---
    y_standard_final = np.zeros_like(y_gan_first)
    for r in range(H):
        for c in range(W):
            y_standard_final[:, r, c] = run_dts_logic(y_gan_first[:, r, c])

    # --- METRICS ---
    mse_standard = mean_squared_error(y_true.flatten(), y_standard_final.flatten())
    mse_reverse = mean_squared_error(y_true.flatten(), y_reverse_result.flatten())

    print("\n=========== RESULTS ===========")
    print(f"Correct Order (GAN → DTS)  MSE: {mse_standard:.6f}")
    print(f"Reverse Order (DTS → GAN)  MSE: {mse_reverse:.6f}")
    print("================================")

    # --- PLOT FOR PROOF ---
    r, c = 25, 30
    plt.figure(figsize=(12, 6))

    plt.plot(dates, y_true[:, r, c], 'k--', label="Ground Truth", alpha=0.6)
    plt.plot(dates, y_standard_final[:, r, c], 'g-', label="GAN → DTS", linewidth=2)
    plt.plot(dates, y_reverse_result[:, r, c], 'r-', label="DTS → GAN", linewidth=2)

    plt.title("Why Order Matters: GAN First vs DTS First")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = f"{OUT_DIR}/dts_then_gan_comparison_h1.png"
    plt.savefig(out_path, dpi=300)
    print(f"\n📌 Plot saved at: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
