
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append("/home/subi/VSCodeProjects/ECG")

from scripts.run_benchmark import _load_dae_model, _dae_denoise_fullsignal
from scripts.run_benchmark import _load_unet_model, _unet_denoise_fullsignal

def test_dae_identity():
    print("Testing DAE Identity...")
    ckpt_path = Path("/home/subi/VSCodeProjects/ECG/outputs/Improved_DAE/best_model.pth")
    if not ckpt_path.exists():
        print(f"Skipping DAE: {ckpt_path} not found")
        return

    model = _load_dae_model(ckpt_path, device="cpu")
    
    # Create random noisy signal
    x = np.random.randn(1000).astype(np.float32)
    
    # Run inference
    # Note: This runs WITHOUT wavelet denoising, matching current run_benchmark.py
    y = _dae_denoise_fullsignal(x, model, torch.device("cpu"), window_len=101, stride=1, batch_size=32)
    
    diff = np.abs(x - y).mean()
    print(f"DAE Input/Output Mean Diff: {diff:.6f}")
    if diff < 1e-5:
        print("!! DAE Output is Identical to Input !!")
    else:
        print("DAE is working (not identity).")

def test_unet_identity():
    print("\nTesting UNet Identity...")
    ckpt_path = Path("/home/subi/VSCodeProjects/ECG/outputs/UNet/best_model.pth")
    if not ckpt_path.exists():
        print(f"Skipping UNet: {ckpt_path} not found")
        return

    model = _load_unet_model(ckpt_path, device="cpu")
    
    x = np.random.randn(2000).astype(np.float32)
    
    # Run inference
    y = _unet_denoise_fullsignal(x, model, torch.device("cpu"), win_len=512, hop_len=512, batch_size=16)
    
    # Check length
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    
    diff = np.abs(x - y).mean()
    print(f"UNet Input/Output Mean Diff: {diff:.6f}")
    if diff < 1e-5:
        print("!! UNet Output is Identical to Input !!")
    else:
        print("UNet is working (not identity).")

if __name__ == "__main__":
    test_dae_identity()
    test_unet_identity()
