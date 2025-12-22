# -*- coding: utf-8 -*-
"""
Synthetic ECG Test Script
- MITDB clean ECG (CSV) + NSTDB baseline wander (로컬 WFDB) 조합
- SNR levels: 0, 5, 10, 15 dB
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


record_ids = [100, 101, 103, 105, 106, 107, 108, 111, 112, 113]

START_SAMPLE = 0       # 모두 동일 조건
DURATION_SEC = 10
FS = 360
NSTDB_RECORD = "bw"


SNR_LEVELS = [0, 5, 10, 15]


# ===========================
# Helper Functions
# ===========================
def remove_dc(x):
    return x - np.mean(x)

def calculate_snr_db(clean, est, remove_mean_clean=True):
    """
    General SNR definition
    signal = clean
    noise  = est - clean
    """
    clean = np.asarray(clean, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    if remove_mean_clean:
        s = clean - clean.mean()
    else:
        s = clean

    e = est - clean  # noise (오차)

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

    actual_snr = calculate_snr_db(clean, noisy)
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

    case_idx = 0

    for record in record_ids:
        case_idx += 1

        clean_ecg, fs = load_mitdb_csv(
            record,
            START_SAMPLE,
            DURATION_SEC,
            FS
        )

        bw, _ = load_nstdb_bw(
            NSTDB_RECORD,
            START_SAMPLE,
            DURATION_SEC,
            FS
        )
        #plt.figure(figsize=(12, 3))
        #t = np.arange(len(bw)) / fs
        #plt.plot(t, bw)
        #plt.title("Baseline Wander (NSTDB raw)")
        #plt.xlabel("Time (s)")
        #plt.ylabel("Amplitude")
        #plt.grid(True, alpha=0.3)
        # =========================
        # 🔍 디버그용: 파일로 저장
        # =========================
        #debug_path = OUTPUT_DIR / f"debug_raw_bw_Record{record}.png"
        #plt.savefig(debug_path, dpi=150, bbox_inches="tight")
        #plt.close()

        for snr in SNR_LEVELS:
            case_name = f"Case{record}_SNR{snr}dB"
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

            snr_out = calculate_snr_db(clean_ref, processed)
            snr_imp = snr_out - snr_in
            rmse = calculate_rmse(clean_ref, processed)

            print(f"  Input SNR : {snr_in:.2f} dB")
            print(f"  Output SNR: {snr_out:.2f} dB")
            print(f"  SNR_imp  : {snr_imp:.2f} dB")
            print(f"  RMSE     : {rmse:.6f}")

            results.append({
                "Case": case_idx,
                "MITDB": record,
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

    # ===========================
    # Input SNR별 통계 (mean ± std)
    # ===========================
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

    # 보기 좋게 문자열 컬럼도 추가 (논문/보고용)
    summary["Output_SNR_mean±std"] = (
        summary["Output_SNR_mean"].round(2).astype(str)
        + " ± "
        + summary["Output_SNR_std"].round(2).astype(str)
    )
    summary["RMSE_mean±std"] = (
        summary["RMSE_mean"].round(2).astype(str)
        + " ± "
        + summary["RMSE_std"].round(2).astype(str)
    )

    # ===========================
    # raw + summary를 하나의 CSV로 저장
    # ===========================
    csv_path = OUTPUT_DIR / "synthetic_test_results.csv"

    with open(csv_path, "w") as f:
        f.write("# Raw results (per case)\n")
        df.to_csv(f, index=False, float_format="%.6f")
        f.write("\n\n")
        f.write("# Summary by input SNR (mean ± std)\n")
        summary.to_csv(f, index=False)


    print("\n✓ CSV saved:", OUTPUT_DIR / "synthetic_test_results.csv")


if __name__ == "__main__":
    run_synthetic_test()
