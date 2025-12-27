# -*- coding: utf-8 -*-
"""
run_comparison.py

Comparison Runner: Our Algorithm vs Improved DAE (Xiong 2016) vs UNet1D
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wfdb
import pywt

# ===========================
# Path Setup
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Now we can import project modules
try:
    from baseline_array import process_ecg_array
    from metrics import SNR, RMSE
except ImportError as e:
    print(f"Error importing modules from project root: {e}")
    sys.exit(1)

from compare_models.methods.Improved_DAE.model_DAE import ImprovedDAE
from compare_models.methods.UNet_1D.UNet.unet1d import UNet


# ===========================
# Defaults
# ===========================
MITDB_DIR_DEFAULT = PROJECT_ROOT / "MITDB_data"
NSTDB_DIR_DEFAULT = PROJECT_ROOT / "noise_data"

# ===========================
# Output directories
# ===========================

# (1) Improved DAE 학습 산출물만 저장 (가중치, config)
DAE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "compare_models"
    / "methods"
    / "Improved_DAE"
    / "outputs"
)
DAE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (2) 비교 실험 결과 저장 (csv, png)
RESULTS_DIR = PROJECT_ROOT / "compare_models" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_RECORD_IDS = [100, 101, 103, 105, 106, 107, 108, 111, 112, 113]
DEFAULT_START_SAMPLE = 0
DEFAULT_DURATION_SEC = 10
DEFAULT_FS = 360
DEFAULT_NSTDB_RECORD = "bw"
DEFAULT_SNR_LEVELS = [0, 5, 10, 15]

# UNet checkpoint (train_unet.py saves outputs/unet_model.pth)
UNET_CKPT_DEFAULT = (
    PROJECT_ROOT
    / "compare_models"
    / "methods"
    / "UNet_1D"
    / "outputs"
    / "unet_model.pth"
)


# ===========================
# Helpers
# ===========================
def remove_dc(x):
    return x - np.mean(x)


def calculate_snr_db(clean: np.ndarray, est: np.ndarray) -> float:
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


def load_mitdb_csv(mitdb_dir: Path, record, start_sample, duration_sec, fs):
    csv_path = mitdb_dir / f"{record}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MITDB file {csv_path} not found.")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip("'").strip('"') for c in df.columns]

    valid_cols = [c for c in df.columns if "Sample" not in c]
    if "MLII" in df.columns:
        ecg = df["MLII"].values
    elif "V5" in df.columns:
        ecg = df["V5"].values
    elif valid_cols:
        ecg = df[valid_cols[0]].values
    else:
        raise ValueError("No ECG data found")

    start = start_sample
    end = start_sample + int(fs * duration_sec)
    return ecg[start:end].astype(np.float64), fs


def load_nstdb_noise(nstdb_dir: Path, record: str, start_sample, duration_sec, fs):
    rec_path = nstdb_dir / record
    if not (nstdb_dir / (record + ".hea")).exists():
        raise FileNotFoundError(f"NSTDB header not found: {rec_path}.hea")
    sig, _ = wfdb.rdsamp(str(rec_path))
    noise = sig[:, 0]
    end = start_sample + int(fs * duration_sec)
    if end > len(noise):
        noise = np.pad(noise, (0, end - len(noise)), mode="wrap")
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
    actual_snr = 10 * np.log10(ps / np.mean((noisy - ref) ** 2))
    return noisy, ref, float(actual_snr)


# ===========================
# Improved DAE Pipeline
# ===========================
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

    return pywt.waverec([cA] + new_cDs, wavelet)[:n]


def extract_windows(sig: np.ndarray, radius: int = 50):
    sig = np.asarray(sig, dtype=np.float64)
    wlen = 2 * radius + 1
    padded = np.pad(sig, (radius, radius), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(padded, wlen)
    return win.copy()


def overlap_mean_fusion(windows_out: np.ndarray, N: int, radius: int = 50) -> np.ndarray:
    wlen = 2 * radius + 1
    out = np.zeros(N + 2 * radius, dtype=np.float64)
    cnt = np.zeros(N + 2 * radius, dtype=np.float64)

    for i in range(N):
        out[i:i + wlen] += windows_out[i]
        cnt[i:i + wlen] += 1.0

    out = out / np.maximum(cnt, 1e-12)
    return out[radius:radius + N]


@torch.no_grad()
def run_improved_dae(noisy: np.ndarray, fs: float, model: ImprovedDAE, device: torch.device,
                     radius: int = 50) -> np.ndarray:
    noisy = np.asarray(noisy, dtype=np.float64)

    wt_out = wavelet_denoise_db6_level8_soft(noisy, level=8)[:noisy.size]
    windows = extract_windows(wt_out, radius=radius)

    w_min = windows.min(axis=1, keepdims=True)
    w_max = windows.max(axis=1, keepdims=True)
    denom = (w_max - w_min) + 1e-8

    x_norm = (windows - w_min) / denom
    x_norm = np.clip(x_norm, 0.0, 1.0)

    x_t = torch.from_numpy(x_norm).float().to(device)
    y_norm = model(x_t).cpu().numpy()

    y_win = y_norm * denom + w_min
    y_final = overlap_mean_fusion(y_win, N=noisy.size, radius=radius)
    return y_final


def load_dae(model_path: Path, config_path: Path, device: torch.device) -> Tuple[ImprovedDAE, dict]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model = ImprovedDAE(window_len=101, hidden1=50, hidden2=50)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()
    return model, cfg


# ===========================
# UNet1D Pipeline
# ===========================
def load_unet(model_path: Path, device: torch.device) -> UNet:
    # must match train_unet.py
    model = UNet(in_channels=1, out_classes=1, dimensions=1, padding=True)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_unet1d(noisy: np.ndarray, model: UNet, device: torch.device, input_len: int = 512) -> np.ndarray:
    """
    Match train_unet.py normalization:
    - chunk length 512
    - per-chunk min-max normalization based on input (noisy) chunk
    - denormalize using same min/max
    """
    x = np.asarray(noisy, dtype=np.float64)
    N = x.size

    # pad to multiple of 512
    pad = (input_len - (N % input_len)) % input_len
    if pad > 0:
        x_pad = np.pad(x, (0, pad), mode="reflect")
    else:
        x_pad = x

    chunks = x_pad.reshape(-1, input_len)
    x_min = chunks.min(axis=1, keepdims=True)
    x_max = chunks.max(axis=1, keepdims=True)
    denom = (x_max - x_min) + 1e-8

    x_norm = (chunks - x_min) / denom
    x_norm = np.clip(x_norm, 0.0, 1.0).astype(np.float32)

    x_t = torch.from_numpy(x_norm[:, None, :]).to(device)  # (B,1,L)
    y_norm = model(x_t).detach().cpu().numpy()[:, 0, :]     # (B,L)

    y = y_norm * denom + x_min
    y = y.reshape(-1)
    return y[:N]


# ===========================
# Plotting
# ===========================
def plot_results(ref, noisy, processed, title, save_path):
    plt.figure(figsize=(12, 8))
    t = np.arange(len(ref))

    plt.subplot(3, 1, 1)
    plt.plot(t, ref, 'k', label='Ref')
    plt.title("Clean Reference")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, noisy, 'b', alpha=0.7, label='Noisy')
    plt.title("Noisy Input")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, ref, 'k', alpha=0.3, label='Ref')
    plt.plot(t, processed, 'r', label='Processed')
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===========================
# Main Runner
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=[
            "our_algorithm",
            "improved_dae",
            "unet1d",
            # "wt_only",
            "all",
        ],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    mitdb_dir = MITDB_DIR_DEFAULT
    nstdb_dir = NSTDB_DIR_DEFAULT

    # Load DAE if needed
    dae_model = None
    if not (DAE_OUTPUT_DIR / "dae_model.pth").exists():
        print("DAE Model not found! Please run train_DAE.py first.")
        return
    dae_model, _ = load_dae(
        DAE_OUTPUT_DIR / "dae_model.pth",
        DAE_OUTPUT_DIR / "dae_config.json",
        device
    )

    # Load UNet if needed
    unet_model = None
    if args.method in ["unet1d", "all"]:
        if not UNET_CKPT_DEFAULT.exists():
            print(f"UNet model not found: {UNET_CKPT_DEFAULT} (run train_unet.py first)")
            return
        unet_model = load_unet(UNET_CKPT_DEFAULT, device)

    results = []

    for rec in DEFAULT_RECORD_IDS:
        try:
            clean, fs = load_mitdb_csv(mitdb_dir, rec, DEFAULT_START_SAMPLE, DEFAULT_DURATION_SEC, DEFAULT_FS)
            noise, _ = load_nstdb_noise(nstdb_dir, DEFAULT_NSTDB_RECORD, DEFAULT_START_SAMPLE, DEFAULT_DURATION_SEC,
                                        DEFAULT_FS)
        except Exception as e:
            print(f"Skipping {rec}: {e}")
            continue

        for snr in DEFAULT_SNR_LEVELS:
            print(f"Processing Case: Rec={rec}, SNR={snr}dB")
            noisy, ref, snr_in = add_baseline_wander_snr(clean, noise, snr)

            if args.method == "all":
                methods_to_run = [
                    "improved_dae",
                    "unet1d",
                    "our_algorithm",
                ]
            else:
                methods_to_run = [args.method]

            for m in methods_to_run:
                if m == "our_algorithm":
                    processed = process_ecg_array(noisy, fs_raw=fs)

                elif m == "improved_dae":
                    processed = run_improved_dae(noisy, fs, dae_model, device)

                elif m == "unet1d":
                    processed = run_unet1d(noisy, unet_model, device, input_len=512)

                # elif m == "wt_only":
                #     processed = wavelet_denoise_db6_level8_soft(noisy, level=8)[:len(noisy)]

                else:
                    raise ValueError(f"Unknown method: {m}")

                L = min(len(ref), len(processed))
                ref_c = ref[:L]
                proc_c = processed[:L]

                snr_out = calculate_snr_db(ref_c, proc_c)
                rmse = calculate_rmse(ref_c, proc_c)

                results.append({
                    "Record": rec,
                    "Target_SNR": snr,
                    "Method": m,
                    "Input_SNR": snr_in,
                    "Output_SNR": snr_out,
                    "SNR_Imp": snr_out - snr_in,
                    "RMSE": rmse
                })

                if rec == 100:
                    plot_results(
                        ref_c,
                        noisy[:L],
                        proc_c,
                        f"{m} (SNR={snr}dB)",
                        RESULTS_DIR / f"{m}_rec{rec}_snr{snr}.png"
                    )

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "comparison_results.csv", index=False)

    print("\n=== Summary (Mean SNR Improvement) ===")
    summary = df.groupby(["Method", "Target_SNR"])["SNR_Imp"].mean().unstack()
    print(summary)

    summary.to_csv(RESULTS_DIR / "comparison_summary.csv")
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
