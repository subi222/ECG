# -*- coding: utf-8 -*-
"""
train_DAE.py

Train Improved DAE baseline (Xiong et al., 2016) with Layer-wise Pretraining:
- Wavelet: Daubechies6 (db6), 8-level, soft thresholding (Eq.1)
- DAE Structure: 101 -> 50 -> 50 -> 101
- Activation: Sigmoid (for all layers)
- Loss: BCELoss (Bernoulli cross-entropy)
- Pretraining: Greedy Layer-wise (AE1: 101-50-101 -> AE2: 50-50-50)
- Fine-tuning: End-to-end backpropagation

Refactored for Antigravity environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import wfdb
import pywt

from model_DAE import ImprovedDAE, SingleLayerAE

# ===========================
# Project Paths (Relative)
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Antigravity/ECG
MITDB_DIR_DEFAULT = PROJECT_ROOT / "MITDB_data"
NSTDB_DIR_DEFAULT = PROJECT_ROOT / "noise_data"
OUTPUT_DIR_DEFAULT = Path("outputs")

# ===========================
# Defaults
# ===========================
record_ids_default = [100, 101, 103, 105, 106, 107, 108, 111, 112, 113]
START_SAMPLE_DEFAULT = 0
DURATION_SEC_DEFAULT = 10
FS_DEFAULT = 360
NSTDB_RECORD_DEFAULT = "bw"
SNR_LEVELS_DEFAULT = [0, 5, 10, 15]


# ===========================
# Reproducibility
# ===========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================
# Helpers
# ===========================
def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def calculate_snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    # Use removed-DC version for calculation
    clean0 = remove_dc(clean)
    est0 = remove_dc(est)
    s = clean0
    e = est0 - clean0
    ps = np.mean(s ** 2)
    pe = np.mean(e ** 2)
    if pe < 1e-12:
        return float("inf")
    return 10.0 * np.log10(ps / pe)


def calculate_rmse(clean: np.ndarray, processed: np.ndarray) -> float:
    clean0 = remove_dc(clean)
    proc0 = remove_dc(processed)
    return float(np.sqrt(np.mean((clean0 - proc0) ** 2)))


# ===========================
# Data Loading & Mixing
# ===========================
def load_mitdb_csv(mitdb_dir: Path, record: int, start_sample: int, duration_sec: int, fs: int) -> Tuple[
    np.ndarray, int]:
    csv_path = mitdb_dir / f"{record}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MITDB file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip("'").strip('"') for c in df.columns]

    if "MLII" in df.columns:
        ecg = df["MLII"].values
    elif "V5" in df.columns:
        ecg = df["V5"].values
    else:
        # Fallback to first valid column that isn't Sample #
        valid_cols = [c for c in df.columns if "Sample" not in c]
        if valid_cols:
            ecg = df[valid_cols[0]].values
        else:
            raise ValueError(f"No ECG channel found in {csv_path}")

    start = start_sample
    end = start_sample + int(fs * duration_sec)
    # Handle bounds
    if end > len(ecg):
        end = len(ecg)

    return ecg[start:end].astype(np.float64), fs


def load_nstdb_noise(nstdb_dir: Path, record: str, start_sample: int, duration_sec: int, fs: int) -> Tuple[
    np.ndarray, int]:
    # wfdb reads without extension
    rec_path = nstdb_dir / record
    if not (nstdb_dir / (record + ".hea")).exists():
        raise FileNotFoundError(f"NSTDB header not found: {rec_path}.hea")

    sig, _ = wfdb.rdsamp(str(rec_path))
    noise = sig[:, 0]
    end = start_sample + int(fs * duration_sec)
    if end > len(noise):
        noise = np.pad(noise, (0, end - len(noise)), mode='wrap')

    return noise[start_sample:end].astype(np.float64), fs


def add_baseline_wander_snr(clean_ecg: np.ndarray, bw: np.ndarray, target_snr_db: float) -> Tuple[
    np.ndarray, np.ndarray, float]:
    N = min(len(clean_ecg), len(bw))
    ref = np.asarray(clean_ecg[:N], dtype=np.float64)
    bw_cut = np.asarray(bw[:N], dtype=np.float64)

    bw0 = remove_dc(bw_cut)
    ref0 = remove_dc(ref)

    ps = np.mean(ref0 ** 2)
    pn = np.mean(bw0 ** 2)

    target_noise_power = ps / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / (pn + 1e-12))

    noisy = ref + bw0 * scale
    actual_snr = calculate_snr_db(ref, noisy)
    return noisy, ref, float(actual_snr)


# ===========================
# Configuration
# ===========================
@dataclass
class DAEConfig:
    paper_id: str = "Xiong2016"
    model_name: str = "ImprovedDAE_Reproduced"
    window_len: int = 101
    hidden1: int = 50
    hidden2: int = 50
    wavelet: str = "db6"
    level: int = 8

    # Training Params
    fs: int = FS_DEFAULT
    epochs_pretrain: int = 20
    epochs_finetune: int = 50
    batch_size: int = 128
    lr_pre: float = 0.001
    lr_fine: float = 0.0001

    training_records: List[int] = None
    snr_levels: List[float] = None
    seed: int = 42
    timestamp: str = ""

    def to_json(self) -> Dict:
        d = asdict(self)
        if d["training_records"] is None: d["training_records"] = record_ids_default
        if d["snr_levels"] is None: d["snr_levels"] = SNR_LEVELS_DEFAULT
        return d


# ===========================
# Preprocessing (WT + Windowing)
# ===========================
def _soft_threshold(d: np.ndarray, T: float) -> np.ndarray:
    return np.sign(d) * np.maximum(np.abs(d) - T, 0.0)


def wavelet_denoise_db6_level8_soft(x: np.ndarray, level: int = 8) -> np.ndarray:
    # Scale-adaptive soft thresholding as per paper Eq.1
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    wavelet = "db6"
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    use_level = int(min(level, max_level))

    coeffs = pywt.wavedec(x, wavelet, level=use_level)
    cA = coeffs[0]
    cDs = coeffs[1:]

    new_cDs = []
    for idx, d in enumerate(cDs, start=1):
        j = use_level - idx + 1
        # Sigma estimation using MAD
        sigma_j = np.median(np.abs(d)) / 0.6745 if d.size > 0 else 0.0
        # Threshold formula
        Tj = sigma_j * math.sqrt(2.0 * math.log(n + 1e-12)) / math.exp(max(j - 1, 0))
        new_cDs.append(_soft_threshold(d, Tj))

    return pywt.waverec([cA] + new_cDs, wavelet)[:n]


def extract_windows(sig: np.ndarray, radius: int = 50) -> np.ndarray:
    # Sliding window with reflect padding
    sig = np.asarray(sig, dtype=np.float64)
    wlen = 2 * radius + 1
    pad = radius
    padded = np.pad(sig, (pad, pad), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, wlen)
    return windows.copy()


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x_windows: np.ndarray, t_windows: np.ndarray):
        super().__init__()
        # Normalization: Map each input window to [0,1]
        # And apply SAME transformation to target window (paper implies mapping V to [0,1])

        self.x_wins = x_windows
        self.t_wins = t_windows

        # Calculate min/max per window for normalization
        self.min_vals = self.x_wins.min(axis=1, keepdims=True)
        self.max_vals = self.x_wins.max(axis=1, keepdims=True)
        self.denom = (self.max_vals - self.min_vals) + 1e-8

        self.x_norm = (self.x_wins - self.min_vals) / self.denom
        self.t_norm = (self.t_wins - self.min_vals) / self.denom

        # Clip to [0,1] for BCELoss stability
        self.x_norm = np.clip(self.x_norm, 0.0, 1.0)
        self.t_norm = np.clip(self.t_norm, 0.0, 1.0)

    def __len__(self):
        return len(self.x_norm)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x_norm[idx]).float(),
                torch.from_numpy(self.t_norm[idx]).float())


# ===========================
# Training Routines
# ===========================
def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0
    cnt = 0
    for x, t in loader:
        x, t = x.to(device), t.to(device)
        opt.zero_grad()
        y = model(x)
        loss = crit(y, t)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        cnt += x.size(0)
    return total_loss / max(cnt, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_pre", type=int, default=10, help="Pretraining epochs per layer")
    parser.add_argument("--epochs_fine", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    # 1. Config & Paths
    cfg = DAEConfig(epochs_pretrain=args.epochs_pre, epochs_finetune=args.epochs_fine)
    OUTPUT_DIR_DEFAULT.mkdir(exist_ok=True, parents=True)

    # 2. Data Preparation
    print(">>> Loading Data...")
    all_x, all_t = [], []

    records = cfg.training_records if cfg.training_records else record_ids_default
    snrs = cfg.snr_levels if cfg.snr_levels else SNR_LEVELS_DEFAULT

    mitdb_dir = MITDB_DIR_DEFAULT
    nstdb_dir = NSTDB_DIR_DEFAULT

    if not mitdb_dir.exists():
        print(f"ERROR: MITDB directory not found at {mitdb_dir}")
        return

    for rec in records:
        try:
            clean, fs = load_mitdb_csv(mitdb_dir, rec, START_SAMPLE_DEFAULT, DURATION_SEC_DEFAULT, FS_DEFAULT)
            noise, _ = load_nstdb_noise(nstdb_dir, NSTDB_RECORD_DEFAULT, START_SAMPLE_DEFAULT, DURATION_SEC_DEFAULT,
                                        FS_DEFAULT)
        except Exception as e:
            print(f"Skipping Record {rec}: {e}")
            continue

        for snr in snrs:
            noisy, ref, _ = add_baseline_wander_snr(clean, noise, snr)

            # WT Logic
            wt_denoised = wavelet_denoise_db6_level8_soft(noisy, level=cfg.level)

            # Windowing (Radius 50 -> Len 101)
            x_w = extract_windows(wt_denoised, radius=50)  # Input: WT Output
            t_w = extract_windows(ref, radius=50)  # Target: Clean Reference

            all_x.append(x_w)
            all_t.append(t_w)

    if not all_x:
        print("No training data loaded!")
        return

    X = np.concatenate(all_x, axis=0)
    T = np.concatenate(all_t, axis=0)

    dataset = WindowDataset(X, T)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    print(f">>> Data Loaded. {len(dataset)} windows.")

    # 3. Layer-wise Pretraining
    print("\n=== Stage 1: Pretraining Layer 1 (101 -> 50 -> 101) ===")
    ae1 = SingleLayerAE(101, 50).to(device)
    crit = nn.BCELoss()  # Using BCELoss as data is [0,1]
    opt1 = optim.Adam(ae1.parameters(), lr=cfg.lr_pre)

    for ep in range(cfg.epochs_pretrain):
        # For AE1, input is x, target is x (self-reconstruction) ?
        # Or denoising? The paper says "Greedy layer-wise... Denoising Autoencoder".
        # Usually DAE implies input=noisy, target=clean.
        # But for pretraining deep networks, often it's Unsupervised (x->x).
        # HOWEVER, we have pairs (x, t). We can do Supervised DAE Training layer-wise?
        # Actually standard Stacked DAE is Unsupervised.
        # Let's assume Standard Stacked DAE on the Input distribution.
        # Input: dataset.x (WT output). Target: dataset.x (Reconstruct itself)

        train_loss = 0
        cnt = 0
        ae1.train()
        for x, _ in loader:  # Unsupervised
            x = x.to(device)
            opt1.zero_grad()
            recon, _ = ae1(x)
            loss = crit(recon, x)  # Reconstruction of Input
            loss.backward()
            opt1.step()
            train_loss += loss.item() * x.size(0)
            cnt += x.size(0)
        print(f"AE1 Epoch {ep + 1}/{cfg.epochs_pretrain} Loss: {train_loss / cnt:.6f}")

    # Generate Hidden Features for Layer 2
    print("\nTargeting Hidden Features for Layer 2...")
    hidden_features = []
    ae1.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, h = ae1(x)
            hidden_features.append(h.cpu())

    H1 = torch.cat(hidden_features, dim=0)
    # H1 is already in [0,1] due to sigmoid
    ds2 = torch.utils.data.TensorDataset(H1, H1)  # Reconstruct hidden
    loader2 = torch.utils.data.DataLoader(ds2, batch_size=cfg.batch_size, shuffle=True)

    print("\n=== Stage 2: Pretraining Layer 2 (50 -> 50 -> 50) ===")
    # Note: Structure is 101-50-50-101.
    # Center layer is 50-50.
    ae2 = SingleLayerAE(50, 50).to(device)
    opt2 = optim.Adam(ae2.parameters(), lr=cfg.lr_pre)

    for ep in range(cfg.epochs_pretrain):
        train_loss = 0
        cnt = 0
        ae2.train()
        for h, _ in loader2:
            h = h.to(device)
            opt2.zero_grad()
            recon, _ = ae2(h)
            loss = crit(recon, h)
            loss.backward()
            opt2.step()
            train_loss += loss.item() * h.size(0)
            cnt += h.size(0)
        print(f"AE2 Epoch {ep + 1}/{cfg.epochs_pretrain} Loss: {train_loss / cnt:.6f}")

    # 4. Fine-tuning (End-to-End)
    print("\n=== Stage 3: Fine-tuning Full Network (101 -> 50 -> 50 -> 101) ===")
    final_model = ImprovedDAE(window_len=101, hidden1=50, hidden2=50).to(device)

    # Initialize weights from Pretrained AEs
    # Layer 1 (Encoder)
    final_model.net[0].weight.data = ae1.encoder.weight.data.clone()
    final_model.net[0].bias.data = ae1.encoder.bias.data.clone()

    # Layer 2 (Hidden) - Matrix 50x50
    final_model.net[2].weight.data = ae2.encoder.weight.data.clone()
    final_model.net[2].bias.data = ae2.encoder.bias.data.clone()

    # Layer 3 (Decoder) - Matrix 101x50
    # Use AE1 decoder weights? Or AE2 decoder?
    # Logic: 101 ->(W1)-> 50 ->(W2)-> 50 ->(W3)-> 101
    # AE1 trained 101<->50. W1=Encoder, W1_dec=Decoder(50->101).
    # AE2 trained 50<->50. W2=Encoder, W2_dec=Decoder(50->50).
    # So Layer 3 corresponds to W1_dec? No, the signal flows forward.
    # The structure 101-50-50-101 implies "Encoding-Encoding-Decoding" ?
    # Or "Encoding-Mapping-Decoding"?
    # If using stacked AE:
    # 1. 101->50 (W1)
    # 2. 50->50 (W2)
    # 3. 50->101 (W3)
    # Typically, W3 is initialized from AE1's decoder (transposed or direct).
    final_model.net[4].weight.data = ae1.decoder.weight.data.clone()
    final_model.net[4].bias.data = ae1.decoder.bias.data.clone()

    opt_fine = optim.Adam(final_model.parameters(), lr=cfg.lr_fine)

    for ep in range(cfg.epochs_finetune):
        train_loss = 0
        cnt = 0
        final_model.train()
        for x, t in loader:  # Supervised: x(Noisy) -> t(Clean)
            x, t = x.to(device), t.to(device)
            opt_fine.zero_grad()
            y = final_model(x)
            loss = crit(y, t)
            loss.backward()
            opt_fine.step()
            train_loss += loss.item() * x.size(0)
            cnt += x.size(0)
        print(f"FineTune Epoch {ep + 1}/{cfg.epochs_finetune} Loss: {train_loss / cnt:.6f}")

    # 5. Save
    torch.save(final_model.state_dict(), OUTPUT_DIR_DEFAULT / "dae_model.pth")
    with open(OUTPUT_DIR_DEFAULT / "dae_config.json", "w") as f:
        json.dump(cfg.to_json(), f, indent=4)

    print(f"\nCompleted. Model saved to {OUTPUT_DIR_DEFAULT / 'dae_model.pth'}")


if __name__ == "__main__":
    main()