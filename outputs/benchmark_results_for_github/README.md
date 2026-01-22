# Benchmark Results

## 📊 Contents

This directory contains benchmark results for ECG denoising models on three noise types:
- **BW (Baseline Wander)**
- **EM (Electrode Motion)**
- **MA (Muscle Artifact)**

### Models Evaluated
1. **Proposed** (Our model)
2. **DeepFilter**
3. **FCN+DAE**
4. **DRNN**

### Directory Structure
```
benchmark_results_for_github/
├── bw/
│   ├── csv/results_summary.csv
│   └── plots/*_snr0dB.png
├── em/
│   ├── csv/results_summary.csv
│   └── plots/*_snr0dB.png
└── ma/
    ├── csv/results_summary.csv
    └── plots/*_snr0dB.png
```

### Files Included
- **CSV:** Summary statistics (SNR, RMSE, PRD, SSIM) for all SNR levels (0, 5, 10, 15 dB)
- **Plots:** Visual comparison at 0dB SNR for all test records

### Metrics
- **SNR (dB):** Signal-to-Noise Ratio (higher is better)
- **RMSE:** Root Mean Square Error (lower is better)
- **PRD (%):** Percent Root-mean-square Difference (lower is better)
- **SSIM:** Structural Similarity Index (0-1, higher is better)

---

**Dataset:** MIT-BIH Arrhythmia Database  
**Test Records:** 113, 201, 203, 205, 219, 222, 230
