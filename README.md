# Satellite Time-Series Reconstruction using DTS & Classical Smoothers

This project implements **Dynamic Temporal Smoothing (DTS)** – a gap-filling and denoising algorithm used in satellite time-series reconstruction.  
We compare DTS with four commonly used smoothing techniques:

✅ DTS (Dynamic Temporal Smoother)  
✅ Cubic Smoothing Splines  
✅ LOWESS  
✅ Savitzky–Golay  
✅ Whittaker Smoothing

---

## 🎯 Objective

Reconstruct missing satellite observations (ex: due to clouds) and analyze:

- Spatial reconstruction accuracy (error maps)
- Temporal smoothness vs. real variability
- Runtime performance
- Method-wise strengths & weaknesses

This contributes to the evaluation of techniques used for **vegetation dynamics monitoring** such as NDVI/EVI2 over croplands.

---

SatelliteReconstruction
│
├── notebooks
│ ├── Data/ → Synthetic dataset creation
│ ├── DTS/ → DTS reconstruction + plots
│ ├── Cubic/
│ ├── LOWESS/
│ ├── SG/
│ ├── Whittaker/
│ └── Comparison/ → Scatter, temporal & error metrics
│
├── requirements.txt
└── .gitignore



⚠️ `data/` and `results/` are not stored in GitHub (auto-generated)

---

## 🚀 How to Run

### 1️⃣ Create synthetic dataset
```bash
cd notebooks/Data
python make_synthetic_realistic.py

This generates:

data/original.npy
data/gapped.npy
data/dates.npy


Run all reconstruction methods
cd ../DTS        && python run_dts.py
cd ../Cubic      && python run_cubic.py
cd ../LOWESS     && python run_lowess.py
cd ../SG         && python run_sg.py
cd ../Whittaker  && python run_whittaker.py


Run visual comparisons
cd ../Comparison
python compare_scatter.py
python compare_temporal.py
python compare_mse.py
## 📂 Project Structure

