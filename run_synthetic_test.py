# -*- coding: utf-8 -*-
"""
Synthetic ECG Test Script
- MITDB clean ECG (CSV) + NSTDB baseline wander (로컬 WFDB) 조합
- SNR levels: 0, 5, 10, 20 dB
- Output performance SNR / RMSE 계산
- 파형 3장 세트 저장
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from pathlib import Path
from baseline_array import process_ecg_array

# ===========================
# 설정
# ===========================
OUTPUT_DIR = Path("./synthetic_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MITDB_DIR = Path("/home/subi/PycharmProjects/ECG/MITDB_data")
NSTDB_DIR = Path("/home/subi/PycharmProjects/ECG/noise_data")

TEST_CASES = [
    {"mitdb": "100", "nstdb": "bw", "start": 0,    "duration": 10, "fs": 360},
    {"mitdb": "101", "nstdb": "bw", "start": 1000, "duration": 10, "fs": 360},
    {"mitdb": "103", "nstdb": "bw", "start": 2000, "duration": 10, "fs": 360},
]

SNR_LEVELS = [0, 5, 10, 20]


# ===========================
# Helper Functions
# ===========================
def remove_dc(x):
    return x - np.mean(x)


def calculate_snr_db_output(clean, processed, remove_mean_clean=True):
    """
    Output performance SNR
    signal = clean ECG
    noise  = processed - clean
    """
    clean = np.asarray(clean, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    if remove_mean_clean:
        s = clean - clean.mean()
    else:
        s = clean

    e = processed - clean  # noise (오차)

    ps = np.mean(s ** 2)
    pe = np.mean(e ** 2)

    if pe < 1e-12:
        return np.inf
    return 10.0 * np.log10(ps / pe)


def calculate_rmse(clean, processed):
    clean0 = remove_dc(clean)
    proc0 = remove_dc(processed)
    return np.sqrt(np.mean((clean0 - proc0) ** 2))

def load_mitdb_csv(record, start_sample, duration_sec, fs):
    csv_path = MITDB_DIR / f"{record}.csv"
    df = pd.read_csv(csv_path)

    # ✅ 컬럼명 정리: 앞뒤 공백 제거 + 따옴표(' ") 제거
    df.columns = [c.strip().strip("'").strip('"') for c in df.columns]

    # ✅ ECG 채널 선택
    if "MLII" in df.columns:
        ecg = df["MLII"].values
    elif "V5" in df.columns:
        ecg = df["V5"].values
    else:
        raise ValueError(
            f"No ECG channel found in {csv_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    start = start_sample
    end = start_sample + int(fs * duration_sec)

    return ecg[start:end], fs


def load_nstdb_bw(record, start_sample, duration_sec, fs):
    sig, _ = wfdb.rdsamp(str(NSTDB_DIR / record))
    bw = sig[:, 0]

    end = start_sample + int(fs * duration_sec)
    return bw[start_sample:end], fs


def add_baseline_wander_snr(clean_ecg, bw, target_snr_db):
    """
    clean + scaled baseline wander (target input SNR)
    """
    N = min(len(clean_ecg), len(bw))
    clean = remove_dc(clean_ecg[:N])
    bw = remove_dc(bw[:N])

    ps = np.mean(clean ** 2)
    pn = np.mean(bw ** 2)

    target_noise_power = ps / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / (pn + 1e-12))

    noisy = clean + bw * scale

    actual_snr = calculate_snr_db_output(clean, noisy)
    return noisy, clean, actual_snr


def plot_triplet(clean, noisy, processed, title, fs):
    t = np.arange(len(clean)) / fs
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax[0].plot(t, clean)
    ax[0].set_title("Clean ECG")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, noisy)
    ax[1].set_title("Noisy ECG (Baseline Wander)")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t[:len(processed)], processed)
    ax[2].set_title("Processed ECG")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


# ===========================
# Main
# ===========================
def run_synthetic_test():
    results = []

    for i, case in enumerate(TEST_CASES, 1):
        clean_ecg, fs = load_mitdb_csv(
            case["mitdb"], case["start"], case["duration"], case["fs"]
        )
        bw, _ = load_nstdb_bw(
            case["nstdb"], case["start"], case["duration"], case["fs"]
        )

        for snr in SNR_LEVELS:
            case_name = f"Case{i}_SNR{snr}dB"
            print(f"\n[{case_name}]")

            noisy, clean_ref, snr_in = add_baseline_wander_snr(
                clean_ecg, bw, snr
            )

            processed = process_ecg_array(
                ecg_raw=noisy,
                fs_raw=fs,
                fs_target=None,
                return_time=False
            )

            N = min(len(clean_ref), len(processed))
            clean_ref = clean_ref[:N]
            noisy = noisy[:N]
            processed = processed[:N]

            snr_out = calculate_snr_db_output(clean_ref, processed)
            snr_imp = snr_out - snr_in
            rmse = calculate_rmse(clean_ref, processed)

            print(f"  Input SNR : {snr_in:.2f} dB")
            print(f"  Output SNR: {snr_out:.2f} dB")
            print(f"  SNR_imp  : {snr_imp:.2f} dB")
            print(f"  RMSE     : {rmse:.6f}")

            results.append({
                "Case": i,
                "MITDB": case["mitdb"],
                "Target_SNR_dB": snr,
                "Input_SNR_dB": snr_in,
                "Output_SNR_dB": snr_out,
                "SNR_Improvement_dB": snr_imp,
                "RMSE": rmse
            })

            fig = plot_triplet(
                clean_ref, noisy, processed,
                title=case_name, fs=fs
            )
            fig.savefig(OUTPUT_DIR / f"{case_name}.png", dpi=150)
            plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "synthetic_test_results.csv",
              index=False, float_format="%.6f")

    print("\n✓ CSV saved:", OUTPUT_DIR / "synthetic_test_results.csv")


if __name__ == "__main__":
    run_synthetic_test()
