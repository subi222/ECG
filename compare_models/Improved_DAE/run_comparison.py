# -*- coding: utf-8 -*-
"""
run_comparison.py

Comparison runner with the same loop/metrics/CSV/plot format as run_synthetic_test.py,
but with pluggable methods.

Methods:
- our_algorithm: calls baseline_array.process_ecg_array(...) exactly like run_synthetic_test.py
- improved_dae: paper-like pipeline:
    noisy -> WT(db6, level=8, soft threshold Eq.1) -> windowing(δ=50) -> DAE inference -> overlap mean fusion

Outputs (same spirit as run_synthetic_test.py):
- outputs/ per-case plot saved
- outputs/synthetic_test_results.csv : raw + summary(mean±std)

The model weights/config default to:
- outputs/dae_model.pth
- outputs/dae_config.json

Config consistency checks:
- window_len, radius, wavelet, level, fs are validated against current settings.
  If mismatched, a warning is printed (but execution continues).

Note:
- This script expects the MITDB CSVs and NSTDB WFDB records to be in the same locations
  used by run_synthetic_test.py, unless you override paths via CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wfdb
import pywt

# try to import shared definitions from run_synthetic_test.py for perfect consistency
try:
    import run_synthetic_test as rst
except Exception:
    rst = None

from model_DAE import ImprovedDAE


# ===========================
# Defaults (fallback if rst is unavailable)
# ===========================
DEFAULT_RECORD_IDS = [100, 101, 103, 105, 106, 107, 108, 111, 112, 113]
DEFAULT_START_SAMPLE = 0
DEFAULT_DURATION_SEC = 10
DEFAULT_FS = 360
DEFAULT_NSTDB_RECORD = "bw"
DEFAULT_SNR_LEVELS = [0, 5, 10, 15]

DEFAULT_MITDB_DIR = Path("/MITDB_data")
DEFAULT_NSTDB_DIR = Path("/noise_data")

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================
# Helpers (match run_synthetic_test.py)
# ===========================
def remove_dc(x):
    return x - np.mean(x)

def calculate_snr_db(clean, est, remove_mean=True):
    clean = np.asarray(clean, dtype=np.float64)
    est   = np.asarray(est, dtype=np.float64)
    if remove_mean:
        clean0 = clean - clean.mean()
        est0   = est - est.mean()
    else:
        clean0 = clean
        est0   = est
    s = clean0
    e = est0 - clean0
    ps = np.mean(s ** 2)
    pe = np.mean(e ** 2)
    if pe < 1e-12:
        return np.inf
    return 10.0 * np.log10(ps / pe)

def calculate_rmse(clean, processed):
    clean0 = remove_dc(clean)
    proc0 = remove_dc(processed)
    return float(np.sqrt(np.mean((clean0 - proc0) ** 2)))

def load_mitdb_csv(mitdb_dir: Path, record, start_sample, duration_sec, fs):
    import pandas as pd
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

def load_nstdb_noise(nstdb_dir: Path, record: str, start_sample, duration_sec, fs):
    sig, _ = wfdb.rdsamp(str(nstdb_dir / record))
    noise = sig[:, 0]
    end = start_sample + int(fs * duration_sec)
    return noise[start_sample:end].astype(np.float64), fs

def add_baseline_wander_snr(clean_ecg, bw, target_snr_db):
    N = min(len(clean_ecg), len(bw))
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

def plot_triplet(clean, noisy, processed, title, fs):
    t = np.arange(len(clean)) / fs
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax[0].plot(t, remove_dc(clean))
    ax[0].set_title("Reference ECG (MITDB raw, DC removed for visualization)")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, remove_dc(noisy))
    ax[1].set_title("Noisy ECG (Baseline Wander, DC removed for visualization)")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t[:len(processed)], processed)
    ax[2].set_title("Processed ECG")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


# ===========================
# Wavelet + windowing + fusion (same as train script)
# ===========================
import math

def _soft_threshold(d: np.ndarray, T: float) -> np.ndarray:
    return np.sign(d) * np.maximum(np.abs(d) - T, 0.0)

def wavelet_denoise_db6_level8_soft(x: np.ndarray, level: int = 8) -> np.ndarray:
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
        sigma_j = np.median(np.abs(d)) / 0.6745 if d.size > 0 else 0.0
        Tj = sigma_j * math.sqrt(2.0 * math.log(n + 1e-12)) / math.exp(max(j - 1, 0))
        new_cDs.append(_soft_threshold(d, Tj))
    new_coeffs = [cA] + new_cDs
    y = pywt.waverec(new_coeffs, wavelet)
    return y[:n]

def extract_windows(sig: np.ndarray, radius: int = 50):
    sig = np.asarray(sig, dtype=np.float64)
    wlen = 2 * radius + 1
    N = sig.size
    padded = np.pad(sig, (radius, radius), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(padded, wlen)
    centers = np.arange(N, dtype=np.int64)
    return win.copy(), centers

def overlap_mean_fusion(windows_out: np.ndarray, centers: np.ndarray, N: int, radius: int = 50) -> np.ndarray:
    wlen = 2 * radius + 1
    out = np.zeros(N + 2 * radius, dtype=np.float64)
    cnt = np.zeros(N + 2 * radius, dtype=np.float64)
    for i, c in enumerate(centers):
        start = c
        out[start:start + wlen] += windows_out[i]
        cnt[start:start + wlen] += 1.0
    out = out / np.maximum(cnt, 1e-12)
    return out[radius:radius + N]


# ===========================
# Method implementations
# ===========================
def run_our_algorithm(noisy: np.ndarray, fs: float, **kwargs) -> np.ndarray:
    # Use the same call as run_synthetic_test.py
    from baseline_array import process_ecg_array
    y = process_ecg_array(ecg_raw=noisy, fs_raw=fs, fs_target=None, return_time=False)
    return np.asarray(y, dtype=np.float64)

@torch.no_grad()
def run_improved_dae(noisy: np.ndarray, fs: float, model: ImprovedDAE, device: torch.device,
                     radius: int = 50, **kwargs) -> np.ndarray:
    noisy = np.asarray(noisy, dtype=np.float64)
    # pipeline: WT -> windows -> DAE -> overlap mean
    a = wavelet_denoise_db6_level8_soft(noisy, level=8)[:noisy.size]
    win, centers = extract_windows(a, radius=radius)  # (N,101)

    w_min = win.min(axis=1, keepdims=True)
    w_max = win.max(axis=1, keepdims=True)
    denom = (w_max - w_min) + 1e-8
    x_norm = (win - w_min) / denom
    x_norm = np.clip(x_norm, 0.0, 1.0)

    x_t = torch.from_numpy(x_norm).float().to(device)
    y_norm = model(x_t).cpu().numpy()

    y_win = y_norm * denom + w_min
    y_sig = overlap_mean_fusion(y_win, centers, N=noisy.size, radius=radius)
    return y_sig


def load_dae(model_path: Path, config_path: Path, device: torch.device) -> Tuple[ImprovedDAE, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = ImprovedDAE(window_len=int(cfg.get("window_len", 101)))
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, cfg


def check_config(cfg: dict, fs: int, radius: int):
    # validate key fields, warn on mismatch
    def warn(msg):
        print(f"[CONFIG WARNING] {msg}")

    if int(cfg.get("fs", fs)) != fs:
        warn(f"fs mismatch: cfg={cfg.get('fs')} vs current={fs}")
    if int(cfg.get("window_radius", radius)) != radius:
        warn(f"window_radius mismatch: cfg={cfg.get('window_radius')} vs current={radius}")
    if int(cfg.get("window_len", 2*radius+1)) != (2 * radius + 1):
        warn(f"window_len mismatch: cfg={cfg.get('window_len')} vs expected={2*radius+1}")
    if cfg.get("wavelet", "db6") != "db6":
        warn(f"wavelet mismatch: cfg={cfg.get('wavelet')} vs db6")
    if int(cfg.get("level", 8)) != 8:
        warn(f"level mismatch: cfg={cfg.get('level')} vs 8")


# ===========================
# Comparison engine
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitdb_dir", type=str, default=str(getattr(rst, "MITDB_DIR", DEFAULT_MITDB_DIR)))
    parser.add_argument("--nstdb_dir", type=str, default=str(getattr(rst, "NSTDB_DIR", DEFAULT_NSTDB_DIR)))
    parser.add_argument("--noise_record", type=str, default=str(getattr(rst, "NSTDB_RECORD", DEFAULT_NSTDB_RECORD)))
    parser.add_argument("--records", type=int, nargs="+", default=list(getattr(rst, "record_ids", DEFAULT_RECORD_IDS)))
    parser.add_argument("--snr_levels", type=float, nargs="+", default=list(getattr(rst, "SNR_LEVELS", DEFAULT_SNR_LEVELS)))
    parser.add_argument("--start_sample", type=int, default=int(getattr(rst, "START_SAMPLE", DEFAULT_START_SAMPLE)))
    parser.add_argument("--duration_sec", type=int, default=int(getattr(rst, "DURATION_SEC", DEFAULT_DURATION_SEC)))
    parser.add_argument("--fs", type=int, default=int(getattr(rst, "FS", DEFAULT_FS)))

    parser.add_argument("--method", type=str, default="improved_dae", choices=["our_algorithm", "improved_dae"])
    parser.add_argument("--dae_model", type=str, default=str(OUTPUT_DIR / "dae_model.pth"))
    parser.add_argument("--dae_config", type=str, default=str(OUTPUT_DIR / "dae_config.json"))
    parser.add_argument("--radius", type=int, default=50)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    mitdb_dir = Path(args.mitdb_dir)
    nstdb_dir = Path(args.nstdb_dir)
    device = torch.device(args.device)

    # method registry
    model = None
    cfg = None
    if args.method == "improved_dae":
        model, cfg = load_dae(Path(args.dae_model), Path(args.dae_config), device)
        check_config(cfg, fs=args.fs, radius=args.radius)

    methods: Dict[str, Callable] = {
        "our_algorithm": lambda noisy, fs, **kw: run_our_algorithm(noisy, fs, **kw),
        "improved_dae": lambda noisy, fs, **kw: run_improved_dae(noisy, fs, model=model, device=device, radius=args.radius, **kw),
    }

    results = []
    case_idx = 0

    for rec in args.records:
        case_idx += 1
        clean, fs = load_mitdb_csv(mitdb_dir, rec, args.start_sample, args.duration_sec, args.fs)
        noise, _ = load_nstdb_noise(nstdb_dir, args.noise_record, args.start_sample, args.duration_sec, args.fs)

        for snr_t in args.snr_levels:
            case_name = f"{args.method}_Case{rec}_SNR{snr_t}dB"
            print(f"\n[{case_name}]")

            noisy, ref, snr_in = add_baseline_wander_snr(clean, noise, snr_t)
            processed = methods[args.method](noisy, fs)

            N = min(len(ref), len(processed))
            ref = ref[:N]
            noisy = noisy[:N]
            processed = processed[:N]

            snr_out = calculate_snr_db(ref, processed, remove_mean=True)
            snr_imp = snr_out - snr_in
            rmse = calculate_rmse(ref, processed)

            print(f"  Input SNR : {snr_in:.2f} dB")
            print(f"  Output SNR: {snr_out:.2f} dB")
            print(f"  SNR_imp  : {snr_imp:.2f} dB")
            print(f"  RMSE     : {rmse:.6f}")

            results.append({
                "Case": case_idx,
                "MITDB": rec,
                "Target_SNR_dB": snr_t,
                "Input_SNR_dB": snr_in,
                "Output_SNR_dB": snr_out,
                "SNR_Improvement_dB": snr_imp,
                "RMSE": rmse,
                "Method": args.method,
            })

            fig = plot_triplet(ref, noisy, processed, title=case_name, fs=fs)
            fig.savefig(OUTPUT_DIR / f"{case_name}.png", dpi=150)
            plt.close(fig)

    df = pd.DataFrame(results)

    # summary (same columns style as run_synthetic_test.py)
    summary = (
        df.groupby("Target_SNR_dB")
          .agg(
              Output_SNR_mean=("Output_SNR_dB", "mean"),
              Output_SNR_std=("Output_SNR_dB", "std"),
              RMSE_mean=("RMSE", "mean"),
              RMSE_std=("RMSE", "std"),
          )
          .reset_index()
    )
    summary["Output_SNR_mean±std"] = (
        summary["Output_SNR_mean"].round(2).astype(str)
        + " ± "
        + summary["Output_SNR_std"].round(2).astype(str)
    )
    summary["RMSE_mean±std"] = (
        summary["RMSE_mean"].round(6).astype(str)
        + " ± "
        + summary["RMSE_std"].round(6).astype(str)
    )

    csv_path = OUTPUT_DIR / "synthetic_test_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("# Raw results (per case)\n")
        df.to_csv(f, index=False, float_format="%.6f")
        f.write("\n\n")
        f.write("# Summary by input SNR (mean ± std)\n")
        summary.to_csv(f, index=False)

    print("\n✓ CSV saved:", csv_path)


if __name__ == "__main__":
    main()