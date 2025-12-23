# -*- coding: utf-8 -*-
"""
train_DAE.py

Train Improved DAE baseline (WT + DAE) following paper description:
- Wavelet: Daubechies6 (db6), 8-level, soft thresholding
- Threshold formula (paper Eq.1): Tj = sigma_j * sqrt(2 ln(n)) / exp(j-1)
  where sigma_j = median(|d_j|) / 0.6745
- Neighborhood radius δ = 50 → window_len = 101
- DAE: 101-50-50-101 with sigmoid, optimized with Bernoulli cross-entropy (BCELoss)

Training data construction is aligned with run_synthetic_test.py:
- clean reference: MITDB raw (CSV: MLII preferred, else V5)
- baseline noise: NSTDB bw (wfdb), with DC removed and scaled to target SNR
- noisy = ref + scaled_bw  (reference kept raw; DC removal used in scaling and metric calc)
- For DAE input: apply WT denoise to noisy to get "initial denoised" signal a
- Build window samples from a (input), target from clean reference (but normalized using input-window min/max)

Outputs:
- outputs/dae_model.pth
- outputs/dae_config.json
- outputs/dae_train_log.csv  (epoch loss)

Sanity checks:
- (SC1) data sanity: length match, NaN/Inf, input SNR matches target within tol
- (SC2) model I/O sanity: shapes, output range ~[0,1]
- (SC3) training sanity: overall loss trend decreases
- (SC4) quick mini test: WT-only vs WT+DAE on 1 record/1 SNR/10s, compare SNR/RMSE
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

from model_DAE import ImprovedDAE


# ===========================
# Defaults aligned to run_synthetic_test.py
# ===========================
record_ids_default = [100, 101, 103, 105, 106, 107, 108, 111, 112, 113]
START_SAMPLE_DEFAULT = 0
DURATION_SEC_DEFAULT = 10
FS_DEFAULT = 360
NSTDB_RECORD_DEFAULT = "bw"
SNR_LEVELS_DEFAULT = [0, 5, 10, 15]

# These paths are taken from run_synthetic_test.py; change if needed.
MITDB_DIR_DEFAULT = Path("/home/subi/PycharmProjects/ECG/MITDB_data")
NSTDB_DIR_DEFAULT = Path("/home/subi/PycharmProjects/ECG/noise_data")

OUTPUT_DIR_DEFAULT = Path("./outputs")


# ===========================
# Reproducibility
# ===========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================
# Metrics (match run_synthetic_test.py)
# ===========================
def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def calculate_snr_db(clean: np.ndarray, est: np.ndarray, remove_mean: bool = True) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    if remove_mean:
        clean0 = clean - clean.mean()
        est0 = est - est.mean()
    else:
        clean0 = clean
        est0 = est
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
# Data loading (match run_synthetic_test.py)
# ===========================
def load_mitdb_csv(mitdb_dir: Path, record: int, start_sample: int, duration_sec: int, fs: int) -> Tuple[np.ndarray, int]:
    csv_path = mitdb_dir / f"{record}.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip("'").strip('"') for c in df.columns]
    if "MLII" in df.columns:
        ecg = df["MLII"].values
    elif "V5" in df.columns:
        ecg = df["V5"].values
    else:
        raise ValueError(f"No ECG channel found in {csv_path}. Available columns: {df.columns.tolist()}")
    start = start_sample
    end = start_sample + int(fs * duration_sec)
    return ecg[start:end].astype(np.float64), fs


def load_nstdb_noise(nstdb_dir: Path, record: str, start_sample: int, duration_sec: int, fs: int) -> Tuple[np.ndarray, int]:
    sig, _ = wfdb.rdsamp(str(nstdb_dir / record))
    noise = sig[:, 0]
    end = start_sample + int(fs * duration_sec)
    return noise[start_sample:end].astype(np.float64), fs


def add_baseline_wander_snr(clean_ecg: np.ndarray, bw: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Same logic as run_synthetic_test.py:
    - reference = MITDB raw (no DC removal)
    - bw0 = DC removed baseline wander
    - scale bw0 to meet target SNR based on DC-removed ref power
    - noisy = ref + scaled_bw0
    - actual input SNR computed with calculate_snr_db(remove_mean=True)
    """
    N = int(min(len(clean_ecg), len(bw)))
    ref = np.asarray(clean_ecg[:N], dtype=np.float64)
    bw0 = remove_dc(np.asarray(bw[:N], dtype=np.float64))

    ref0 = remove_dc(ref)
    ps = np.mean(ref0 ** 2)
    pn = np.mean(bw0 ** 2)
    target_noise_power = ps / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / (pn + 1e-12))
    noisy = ref + bw0 * scale
    actual_snr = calculate_snr_db(ref, noisy, remove_mean=True)
    return noisy, ref, float(actual_snr)


# ===========================
# Wavelet denoise (paper Eq.1 & Eq.2)
# ===========================
def _soft_threshold(d: np.ndarray, T: float) -> np.ndarray:
    return np.sign(d) * np.maximum(np.abs(d) - T, 0.0)


def wavelet_denoise_db6_level8_soft(x: np.ndarray, level: int = 8) -> np.ndarray:
    """
    Apply db6, 8-level DWT, scale-adaptive soft threshold:
      T_j = sigma_j * sqrt(2 ln(n)) / exp(j-1)
      sigma_j = median(|d_j|) / 0.6745
    Reconstruct from thresholded coefficients.

    Note: DWT level is capped by signal length; we use pywt.dwt_max_level.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    wavelet = "db6"

    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    use_level = int(min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=use_level)

    # coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    cA = coeffs[0]
    cDs = coeffs[1:]

    new_cDs = []
    for idx, d in enumerate(cDs, start=1):
        # Map to paper's layer index j in {1..L}. Here we treat:
        # d at cD_L is "j=L", ..., cD_1 is "j=1".
        # We can compute j = use_level - idx + 1.
        j = use_level - idx + 1

        sigma_j = np.median(np.abs(d)) / 0.6745 if d.size > 0 else 0.0
        Tj = sigma_j * math.sqrt(2.0 * math.log(n + 1e-12)) / math.exp(max(j - 1, 0))
        new_cDs.append(_soft_threshold(d, Tj))

    new_coeffs = [cA] + new_cDs
    y = pywt.waverec(new_coeffs, wavelet)
    # waverec can return slightly different length
    return y[:n]


# ===========================
# Windowing utilities (δ=50)
# ===========================
def extract_windows(sig: np.ndarray, radius: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping windows centered at each sample i with neighborhood radius δ.
    Returns:
      windows: (N, 2δ+1)
      centers: indices (N,)
    Boundary handling: reflect padding.
    """
    sig = np.asarray(sig, dtype=np.float64)
    wlen = 2 * radius + 1
    N = sig.size
    pad = radius
    padded = np.pad(sig, (pad, pad), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, wlen)
    centers = np.arange(N, dtype=np.int64)
    return windows.copy(), centers


def overlap_mean_fusion(windows_out: np.ndarray, centers: np.ndarray, N: int, radius: int = 50) -> np.ndarray:
    """
    windows_out: (N, wlen) predicted windows (original scale)
    Fuse by overlap-add with mean in overlaps.
    """
    wlen = 2 * radius + 1
    out = np.zeros(N + 2 * radius, dtype=np.float64)
    cnt = np.zeros(N + 2 * radius, dtype=np.float64)

    for i, c in enumerate(centers):
        start = c
        out[start:start + wlen] += windows_out[i]
        cnt[start:start + wlen] += 1.0

    # remove padding and average
    out = out / np.maximum(cnt, 1e-12)
    return out[radius:radius + N]


# ===========================
# Dataset
# ===========================
class WindowDataset(torch.utils.data.Dataset):
    """
    Input: WT-denoised noisy signal a
    Target: clean reference signal (raw MITDB segment)
    Both are cut into windows with δ=50.
    Normalization: per-window min-max based on INPUT window a_w.
      x_norm = (a_w - min_w) / (max_w - min_w + eps)
      t_norm = (clean_w - min_w) / (max_w - min_w + eps)  (same min/max for inference feasibility)
    """
    def __init__(self, a_sig: np.ndarray, clean_sig: np.ndarray, radius: int = 50, eps: float = 1e-8):
        assert len(a_sig) == len(clean_sig)
        self.radius = int(radius)
        self.eps = float(eps)

        self.a_win, self.centers = extract_windows(a_sig, radius=self.radius)
        self.c_win, _ = extract_windows(clean_sig, radius=self.radius)

        # window 중심값 + scale
        self.w_min = self.a_win.min(axis=1, keepdims=True)
        self.w_max = self.a_win.max(axis=1, keepdims=True)
        denom = (self.w_max - self.w_min) + self.eps

        # 입력/타깃 모두 "입력 창의 min/max"로 정규화 (중요!)
        self.x = (self.a_win - self.w_min) / denom
        self.t = (self.c_win - self.w_min) / denom

        # BCE 안정성: [0,1]로 clip
        self.x = np.clip(self.x, 0.0, 1.0)
        self.t = np.clip(self.t, 0.0, 1.0)

        print("[DBG] x_norm range:", self.x.min(), self.x.max())
        print("[DBG] t_norm range:", self.t.min(), self.t.max(),
              "clip_rate=", np.mean((self.t < 0) | (self.t > 1)))
        if not hasattr(self, "_dbg_printed"):
            print("[DBG] x_norm range:", self.x.min(), self.x.max())
            print("[DBG] t_norm range:", self.t.min(), self.t.max(),
                  "clip_rate=", np.mean((self.t < 0) | (self.t > 1)))
            self._dbg_printed = True


    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).float()
        t = torch.from_numpy(self.t[idx]).float()
        return x, t


# ===========================
# Config
# ===========================
@dataclass
class DAEConfig:
    paper_id: str = "Xiong2016_ImprovedDAE"
    model_name: str = "improved_dae_reimplementation_end2end"
    window_radius: int = 50
    window_len: int = 101
    wavelet: str = "db6"
    level: int = 8
    thresholding: str = "soft"
    threshold_formula: str = "Eq.(1) scale-adaptive: Tj = sigma_j*sqrt(2ln(n))/exp(j-1), sigma_j=median(|d_j|)/0.6745"
    training_records: List[int] = None
    noise_record: str = "bw"
    snr_levels_train: List[float] = None
    fs: int = FS_DEFAULT
    duration_sec: int = DURATION_SEC_DEFAULT
    start_sample: int = START_SAMPLE_DEFAULT
    optimizer: str = "Adam"
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 256
    seed: int = 42
    code_version: str = "v1"
    timestamp: str = ""

    def to_json(self) -> Dict:
        d = asdict(self)
        # fill defaults
        if d["training_records"] is None:
            d["training_records"] = record_ids_default
        if d["snr_levels_train"] is None:
            d["snr_levels_train"] = SNR_LEVELS_DEFAULT
        return d


# ===========================
# Training loop
# ===========================
def train_one_epoch(model: nn.Module, loader, opt, loss_fn, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, t in loader:
        x = x.to(device)
        t = t.to(device)
        opt.zero_grad(set_to_none=True)

        y = model(x)
        y = torch.clamp(y, 0.0, 1.0)  # ⭐ 추가 (학습용 clamp)

        loss = loss_fn(y, t)
        loss.backward()
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model: nn.Module, loader, loss_fn, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, t in loader:
        x = x.to(device)
        t = t.to(device)
        y = model(x)
        loss = loss_fn(y, t)
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def _print_sc(msg: str) -> None:
    print(f"[SANITY] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitdb_dir", type=str, default=str(MITDB_DIR_DEFAULT))
    parser.add_argument("--nstdb_dir", type=str, default=str(NSTDB_DIR_DEFAULT))
    parser.add_argument("--noise_record", type=str, default=NSTDB_RECORD_DEFAULT, choices=["bw", "ma", "em"])
    parser.add_argument("--snr_levels", type=float, nargs="+", default=SNR_LEVELS_DEFAULT)
    parser.add_argument("--records", type=int, nargs="+", default=record_ids_default)
    parser.add_argument("--start_sample", type=int, default=START_SAMPLE_DEFAULT)
    parser.add_argument("--duration_sec", type=int, default=DURATION_SEC_DEFAULT)
    parser.add_argument("--fs", type=int, default=FS_DEFAULT)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--sanity_tol_db", type=float, default=0.2)
    parser.add_argument("--mini_record", type=int, default=record_ids_default[0])
    parser.add_argument("--mini_snr", type=float, default=SNR_LEVELS_DEFAULT[-1])

    args = parser.parse_args()

    set_seed(args.seed)
    mitdb_dir = Path(args.mitdb_dir)
    nstdb_dir = Path(args.nstdb_dir)
    out_dir = OUTPUT_DIR_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Prepare config
    cfg = DAEConfig(
        training_records=list(args.records),
        noise_record=args.noise_record,
        snr_levels_train=list(args.snr_levels),
        fs=args.fs,
        duration_sec=args.duration_sec,
        start_sample=args.start_sample,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Build dataset by concatenating all windows from (records x snr_levels)
    all_x = []
    all_t = []
    # For sanity (SC1), track actual input SNRs
    snr_checks = []

    for rec in args.records:
        clean, _ = load_mitdb_csv(mitdb_dir, rec, args.start_sample, args.duration_sec, args.fs)
        noise, _ = load_nstdb_noise(nstdb_dir, args.noise_record, args.start_sample, args.duration_sec, args.fs)

        for snr_t in args.snr_levels:
            noisy, ref, snr_in = add_baseline_wander_snr(clean, noise, snr_t)
            snr_checks.append((rec, snr_t, snr_in))

            # WT initial denoise
            a = wavelet_denoise_db6_level8_soft(noisy, level=cfg.level)
            a = a[:len(ref)]

            ds = WindowDataset(a, ref, radius=cfg.window_radius)
            all_x.append(ds.x)
            all_t.append(ds.t)

    X = np.concatenate(all_x, axis=0)
    T = np.concatenate(all_t, axis=0)

    # (SC1) data sanity
    if not np.isfinite(X).all() or not np.isfinite(T).all():
        raise RuntimeError("Found NaN/Inf in training data.")
    _print_sc(f"(SC1) data finite OK. X shape={X.shape}, T shape={T.shape}")

    # check SNR tolerance
    bad = []
    for rec, target, actual in snr_checks:
        if abs(actual - target) > args.sanity_tol_db:
            bad.append((rec, target, actual))
    if bad:
        _print_sc(f"(SC1) WARNING: {len(bad)} cases out of tol ±{args.sanity_tol_db} dB. Example: {bad[0]}")
    else:
        _print_sc(f"(SC1) input SNR matches targets within ±{args.sanity_tol_db} dB")

    # Torch dataset / loader
    tensor_ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(T).float())
    loader = torch.utils.data.DataLoader(tensor_ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=False)

    # Model
    model = ImprovedDAE(window_len=cfg.window_len).to(device)
    #loss_fn = nn.BCELoss()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # (SC2) I/O sanity
    with torch.no_grad():
        xb, tb = next(iter(loader))
        yb = model(xb.to(device))
        _print_sc(f"(SC2) model IO shape: in={tuple(xb.shape)} out={tuple(yb.shape)}")
        y_min = float(yb.min().cpu().item())
        y_max = float(yb.max().cpu().item())
        _print_sc(f"(SC2) output range (should be ~[0,1]): min={y_min:.4f}, max={y_max:.4f}")

    # Train
    log_rows = []
    loss_hist = []
    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, opt, loss_fn, device)
        loss_hist.append(train_loss)
        log_rows.append({"epoch": ep, "train_loss": train_loss})
        print(f"[TRAIN] epoch {ep:03d}/{args.epochs:03d} loss={train_loss:.6f}")

    # (SC3) trend
    if len(loss_hist) >= 2 and loss_hist[-1] < loss_hist[0]:
        _print_sc(f"(SC3) loss decreased overall: {loss_hist[0]:.6f} -> {loss_hist[-1]:.6f}")
    else:
        _print_sc(f"(SC3) WARNING: loss did not decrease overall: {loss_hist[0]:.6f} -> {loss_hist[-1]:.6f}")

    # Save model + config + log
    model_path = out_dir / "dae_model.pth"
    torch.save(model.state_dict(), model_path)

    cfg_path = out_dir / "dae_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_json(), f, indent=2, ensure_ascii=False)

    log_path = out_dir / "dae_train_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False, float_format="%.8f")

    print(f"\n✓ Saved: {model_path}")
    print(f"✓ Saved: {cfg_path}")
    print(f"✓ Saved: {log_path}")

    # (SC4) quick mini test: WT-only vs WT+DAE
    sanity = run_quick_mini_test(
        model=model,
        device=device,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        rec=args.mini_record,
        noise_record=args.noise_record,
        snr_db=args.mini_snr,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs=args.fs,
        radius=cfg.window_radius,
    )
    sanity_path = out_dir / "sanity_check.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2, ensure_ascii=False)
    _print_sc(f"(SC4) mini test saved: {sanity_path}")


@torch.no_grad()
def dae_denoise_signal(model: nn.Module, device: torch.device, noisy: np.ndarray, radius: int = 50) -> np.ndarray:
    """
    Inference pipeline for a 1D signal:
      noisy -> WT -> windowing -> DAE -> overlap mean fusion
    Denormalization uses per-window min/max from WT-window (input).
    """
    noisy = np.asarray(noisy, dtype=np.float64)
    a = wavelet_denoise_db6_level8_soft(noisy, level=8)  # fixed to paper
    a = a[:noisy.size]

    win, centers = extract_windows(a, radius=radius)  # (N,101)
    w_min = win.min(axis=1, keepdims=True)
    w_max = win.max(axis=1, keepdims=True)
    denom = (w_max - w_min) + 1e-8
    x_norm = (win - w_min) / denom
    x_norm = np.clip(x_norm, 0.0, 1.0)

    x_t = torch.from_numpy(x_norm).float().to(device)
    y_norm = model(x_t).cpu().numpy()

    # denorm
    y = y_norm * denom + w_min
    y_sig = overlap_mean_fusion(y, centers, N=noisy.size, radius=radius)
    return y_sig


def run_quick_mini_test(model: nn.Module, device: torch.device, mitdb_dir: Path, nstdb_dir: Path,
                        rec: int, noise_record: str, snr_db: float,
                        start_sample: int, duration_sec: int, fs: int, radius: int = 50) -> Dict:
    clean, _ = load_mitdb_csv(mitdb_dir, rec, start_sample, duration_sec, fs)
    noise, _ = load_nstdb_noise(nstdb_dir, noise_record, start_sample, duration_sec, fs)
    noisy, ref, snr_in = add_baseline_wander_snr(clean, noise, snr_db)

    # WT-only
    wt = wavelet_denoise_db6_level8_soft(noisy, level=8)[:len(ref)]
    snr_wt = calculate_snr_db(ref, wt, remove_mean=True)
    rmse_wt = calculate_rmse(ref, wt)

    # WT+DAE
    dae = dae_denoise_signal(model, device, noisy, radius=radius)[:len(ref)]
    snr_dae = calculate_snr_db(ref, dae, remove_mean=True)
    rmse_dae = calculate_rmse(ref, dae)

    out = {
        "record": rec,
        "target_snr_db": snr_db,
        "input_snr_db": snr_in,
        "wt_only": {"snr_db": snr_wt, "rmse": rmse_wt},
        "wt_plus_dae": {"snr_db": snr_dae, "rmse": rmse_dae},
        "improvement": {"snr_db": snr_dae - snr_wt, "rmse": rmse_wt - rmse_dae},
        "note": "SC4 quick check: expecting WT+DAE >= WT in SNR and/or <= in RMSE (not guaranteed on tiny training).",
    }

    _print_sc(f"(SC4) mini record={rec} target={snr_db}dB input={snr_in:.2f}dB | "
              f"WT SNR={snr_wt:.2f}, RMSE={rmse_wt:.6f} | "
              f"WT+DAE SNR={snr_dae:.2f}, RMSE={rmse_dae:.6f}")
    return out


if __name__ == "__main__":
    main()