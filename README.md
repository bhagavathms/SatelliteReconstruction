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
⚠️ `data/` and `results/` are not stored in GitHub (auto-generated)

---

## 🚀 How to Run

```bash
cd notebooks/Data
python make_synthetic_realistic.py
```
This generates:
data/original.npy
data/gapped.npy
data/dates.npy


Run all reconstruction methods
```bash
cd ../DTS        && python run_dts.py
cd ../Cubic      && python run_cubic.py
cd ../LOWESS     && python run_lowess.py
cd ../SG         && python run_sg.py
cd ../Whittaker  && python run_whittaker.py
```

Run visual comparisons
```bash
cd ../Comparison
python compare_scatter.py
python compare_temporal.py
python compare_mse.py
```

If cloning
```bss
git clone https://github.com/bhagavathms/SatelliteReconstruction.git
cd SatelliteReconstruction
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
