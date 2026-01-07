# UNet (General-Purpose Model) - Archived

## 📦 Reason for Archiving
UNet is a **general-purpose denoising model** (not specialized for ECG baseline removal). 
It was initially tested as a comparison baseline but deemed unsuitable for direct comparison with domain-specific baseline removal algorithms.

## 📂 Contents
- `model_UNet/` - UNet implementation (1D variant)
- `train_unet.py` - Training script
- `UNet/` - Training outputs (best_model.pth, logs, config)

## 📊 Benchmark Results (for reference)
- **0dB SNR**: RMSE 0.328 (failed, identity mapping)
- **15dB SNR**: RMSE 0.106 (excellent performance)

## 🔄 Replacement
Consider using **DeepFilter** (https://github.com/fperdigon/DeepFilter) instead - a deep learning model specifically designed for ECG baseline wander removal.

---
*Archived on: 2026-01-07*
*Reason: Switching to baseline-removal-specific deep learning model for fair comparison*
