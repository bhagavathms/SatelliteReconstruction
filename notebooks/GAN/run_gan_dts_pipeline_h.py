import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import os

# --- CONFIGURATION ---
DATA_DIR = "../../data"                # <-- UPDATED
RESULTS_DIR = "../../results/GAN"      # <-- UPDATED
OUTPUT_DIR = "../../results/GAN_DTS"   # <-- NEW

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 1. FAKE DTS (SG FILTER) FOR DEMO ---
def run_dts(data_3d):
    """
    Simulated DTS using Savitzky-Golay smoothing.
    Input: (T, H, W)
    Output: (T, H, W) smoothed
    """
    T, H, W = data_3d.shape
    smoothed_data = np.zeros_like(data_3d)

    print("Running smoothing using DTS ...")

    for r in range(H):
        for c in range(W):
            ts = data_3d[:, r, c]

            try:
                sm_series = savgol_filter(ts, window_length=45, polyorder=2)
                # upper envelope logic
                sm_series = np.maximum(sm_series, ts)
            except:
                sm_series = ts

            smoothed_data[:, r, c] = sm_series

    return smoothed_data


# --- 2. MAIN PIPELINE ---
def main():
    print("Loading data...")

    # LOAD your dataset (correct names)
    y_true = np.load(f"{DATA_DIR}/original_h.npy")     # ground truth
    y_gap  = np.load(f"{DATA_DIR}/gapped_h.npy")       # gapped input

    # LOAD GAN output (correct folder)
    y_gan  = np.load(f"{RESULTS_DIR}/gan_output_h.npy")  # GAN reconstruction

    # LOAD dates (correct file)
    dates  = np.load(f"{DATA_DIR}/dates_h.npy", allow_pickle=True)

    # ---- METHOD 1: DTS over gapped ----
    print("\n--- Method 1: DTS Only ---")
    y_dts_only = run_dts(y_gap)

    # ---- METHOD 2: GAN → DTS ----
    print("\n--- Method 2: GAN + DTS ---")
    y_hybrid = run_dts(y_gan)

    # ---- EVALUATION ----
    mse_dts = mean_squared_error(y_true.flatten(), y_dts_only.flatten())
    mse_hybrid = mean_squared_error(y_true.flatten(), y_hybrid.flatten())

    print("\n" + "="*30)
    print("RESULTS SUMMARY")
    print("="*30)
    print(f"1. DTS Only MSE:      {mse_dts:.6f}")
    print(f"2. GAN + DTS MSE:     {mse_hybrid:.6f}")
    print("-" * 30)
    improvement = ((mse_dts - mse_hybrid) / mse_dts) * 100
    print(f"Improvement:          {improvement:.2f}%")
    print("="*30)

    # ---- VISUALIZATION ----
    r, c = 25, 30     # pixel to visualize

    plt.figure(figsize=(14, 6))

    plt.plot(dates, y_true[:, r, c], 'k--', label="Ground Truth", linewidth=1.5, alpha=0.7)

    valid = y_gap[:, r, c] > 0.01
    plt.plot(dates[valid], y_gap[valid, r, c], 'ro', label="Gapped Input", markersize=4)

    plt.plot(dates, y_dts_only[:, r, c], 'b-', label="DTS Only", linewidth=2)
    plt.plot(dates, y_hybrid[:, r, c], 'g-', label="GAN + DTS", linewidth=2.5)

    plt.title(f"Pixel ({r},{c}) Comparison: GAN + DTS vs DTS")
    plt.ylabel("EVI2")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = f"{OUTPUT_DIR}/comparison_plot_h.png"
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"\n📌 Plot saved: {out_path}")


if __name__ == "__main__":
    main()
