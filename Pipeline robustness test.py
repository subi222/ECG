import csv
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
import os
from baseline import process_ecg_from_array


# =========================================
# 평가 함수: SNR, NRMSE
# =========================================
def compute_snr_db(ref, est, remove_mean=True):
    """
    ref: 기준 신호
    est: 비교 대상
    remove_mean=True 이면 DC 제거 후 SNR 계산 (믹싱 로직과 맞춤)
    """
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    if remove_mean:
        ref0 = ref - ref.mean()
        est0 = est - est.mean()
    else:
        ref0 = ref
        est0 = est

    noise = ref0 - est0
    ps = np.mean(ref0 ** 2)
    pn = np.mean(noise ** 2)

    if pn == 0:
        return np.inf
    return 10.0 * np.log10(ps / pn)


def compute_nrmse(ref, est, mode="std"):
    """
    mode="std" : RMSE / std(ref)
    mode="range": RMSE / (max(ref) - min(ref))
    """
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    mse = np.mean((ref - est) ** 2)
    rmse = np.sqrt(mse)

    if mode == "std":
        denom = np.std(ref)
    elif mode == "range":
        denom = float(ref.max() - ref.min())
    else:
        denom = 1.0

    if denom == 0:
        return np.nan
    return rmse / denom


# --- 1) 섞기 ---
def mix_with_snr(clean, noise, snr_db):
    """
    clean, noise: 1D numpy array
    snr_db: 섞고 싶은 SNR (dB). 20, 10, 5, 0 등
    return: mixed_signal, scaled_noise
    """
    # 길이 맞추기 (더 짧은 쪽에 맞춤)
    N = min(len(clean), len(noise))
    clean = clean[:N].astype(np.float64)
    noise = noise[:N].astype(np.float64)

    # DC 성분 제거 후 파워 계산
    s = clean - clean.mean()
    n = noise - noise.mean()

    sig_power = np.mean(s ** 2)
    noise_power = np.mean(n ** 2) + 1e-12  # 0 나눗셈 방지

    # 목표 noise 파워 = sig_power / (10^(SNR/10))
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    n_scaled = n * scale

    mixed = clean + n_scaled
    return mixed, n_scaled


# Pre-cleaned GT 생성 함수
def create_pre_cleaned_gt(raw_sig, fs=360.0, cutoff=0.5):
    """
    원본 신호(raw_sig)에서 0.5Hz 미만의 기저선 변동을
    'Zero-phase' 필터로 완벽히 제거하여 실험용 정답(GT)을 생성함.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # 2차 Butterworth HPF 설계
    b, a = butter(N=2, Wn=normal_cutoff, btype='high', analog=False)

    # filtfilt: 위상 왜곡(밀림 현상) 없이 필터링
    clean_gt = filtfilt(b, a, raw_sig)
    return clean_gt

# --- 2) 실제 실험 ---
def main():
    base_dir = "/home/subi/PycharmProjects/ECG/MITDB_data"
    noise_dir = "/home/subi/PycharmProjects/ECG/noise_data"

    # ✅ 1) MITDB에서 테스트할 레코드 10개
    records_100 = [100, 101, 103, 105, 109]
    records_200 = [200, 201, 203, 207, 208]
    record_ids = records_100 + records_200

    # ✅ 2) 테스트할 SNR 값들
    snr_db_list = [20, 10, 5, 0]

    # ✅ 3) bw 노이즈 로드
    rec_name = f"{noise_dir}/bw"  # bw.dat / bw.hea
    sig, fields = wfdb.rdsamp(rec_name)
    noise = sig[:, 0]

    # ✅ 4) CSV 파일 열기 (스크립트 있는 폴더에 저장)
    out_csv = "noise_experiment_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # 헤더 한 줄 쓰기
        writer.writerow([
            "record_id",
            "clean_std",
            "snr_target_db",
            "snr_input_db",
            "snr_output_db",
            "nrmse_std"
        ])


        # ✅ 5) 각 레코드에 대해 반복
        for rec_id in record_ids:
            # (1) 원본 로드 (Raw Data)
            raw_data = np.loadtxt(f"{base_dir}/{rec_id}.csv",
                                  delimiter=",", skiprows=1)[:, 1]

            # 🔴 [수정 1] 진짜 정답(GT) 생성: 원본을 0.5Hz 필터로 깨끗하게 만듦
            gt_clean = create_pre_cleaned_gt(raw_data, fs=360.0)

            # (참고용) GT의 표준편차 계산
            gt_std = gt_clean.std()

            print("\n" + "=" * 80)
            print(f"[GT CREATED] record {rec_id}, gt std={gt_std:.3f}")

            # (3) SNR별로 노이즈 섞어서 실험
            for snr_db in snr_db_list:
                # 🔴 [수정 2] 노이즈 섞기: 원본이 아니라 'GT'에 노이즈를 섞음 (Input 생성)
                mixed_input, noise_used = mix_with_snr(gt_clean, noise, snr_db)

                # 3-2) 실제 입력 SNR 확인 (GT vs Input)
                snr_in = compute_snr_db(gt_clean, mixed_input, remove_mean=True)

                # 🔴 [수정 3] 모델 돌리기 (내 모델 파이프라인 통과)
                # 모델은 'mixed_input'만 보고 기저선을 지워야 함
                # (함수 리턴값은 사용자 정의에 따라 다를 수 있으나, 최종 정제 신호를 받아야 함)
                # 예시에서는 첫 번째 리턴값이 최종 신호라고 가정 (y_mixed_dbg)
                y_out_final, _, _, _ = process_ecg_from_array(
                    mixed_input, fs_raw=360.0, return_debug=True
                )

                # 🔴 [수정 4] 성능 평가: 'GT' vs '모델 출력' 비교
                # 기존: compute_snr_db(y_clean_dbg, y_mixed_dbg) -> 잘못된 비교
                # 변경: compute_snr_db(gt_clean, y_out_final) -> 공정한 비교
                snr_out = compute_snr_db(gt_clean, y_out_final, remove_mean=True)
                nrmse_out = compute_nrmse(gt_clean, y_out_final, mode="std")

                print(f"\n[SNR target={snr_db} dB] record {rec_id}")
                print(f"  - input  SNR (GT vs mixed): {snr_in:.2f} dB")
                print(f"  - output SNR (GT vs Output): {snr_out:.2f} dB")
                print(f"  - NRMSE (vs GT): {nrmse_out:.4f}")

                writer.writerow([
                    rec_id,
                    f"{gt_std:.6f}",  # clean_std 대신 gt_std 저장
                    snr_db,
                    f"{snr_in:.6f}",
                    f"{snr_out:.6f}",
                    f"{nrmse_out:.6f}",
                ])

    print(f"\nCSV 결과가 '{out_csv}' 파일로 저장되었습니다.")

    # ==================================================
    # 6) SNR 조건별 평균 ± 표준편차 계산 (논문용 요약)
    # ==================================================
    print("\n" + "=" * 80)
    print("[SUMMARY] Mean ± Std by Target SNR")

    # CSV 다시 읽기
    data = np.genfromtxt(
        out_csv,
        delimiter=",",
        skip_header=1,
        dtype=None,
        encoding="utf-8"
    )

    # 컬럼 인덱스 (CSV 헤더 순서 기준)
    IDX_SNR_TARGET = 2
    IDX_SNR_OUT = 4
    IDX_NRMSE = 5

    snr_targets = sorted(set(row[IDX_SNR_TARGET] for row in data))

    print(f"{'SNR(dB)':>8} | {'Output SNR (mean±std)':>25} | {'NRMSE (mean±std)':>25}")
    print("-" * 70)

    for snr_t in snr_targets:
        rows = [row for row in data if row[IDX_SNR_TARGET] == snr_t]

        snr_out_vals = np.array([float(r[IDX_SNR_OUT]) for r in rows])
        nrmse_vals = np.array([float(r[IDX_NRMSE]) for r in rows])

        snr_mean = snr_out_vals.mean()
        snr_std = snr_out_vals.std()

        nrmse_mean = nrmse_vals.mean()
        nrmse_std = nrmse_vals.std()

        print(
            f"{snr_t:8.1f} | "
            f"{snr_mean:6.2f} ± {snr_std:5.2f} dB | "
            f"{nrmse_mean:7.4f} ± {nrmse_std:7.4f}"
        )

    # ==================================================
    # 7) 요약 결과 CSV로 저장
    # ==================================================
    summary_csv = out_csv.replace(".csv", "_summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr_target_db",
            "output_snr_mean",
            "output_snr_std",
            "nrmse_mean",
            "nrmse_std"
        ])

        for snr_t in snr_targets:
            rows = [row for row in data if row[IDX_SNR_TARGET] == snr_t]

            snr_out_vals = np.array([float(r[IDX_SNR_OUT]) for r in rows])
            nrmse_vals = np.array([float(r[IDX_NRMSE]) for r in rows])

            writer.writerow([
                snr_t,
                snr_out_vals.mean(),
                snr_out_vals.std(),
                nrmse_vals.mean(),
                nrmse_vals.std()
            ])

    print(f"\n요약 결과가 '{summary_csv}' 파일로 저장되었습니다.")
    return summary_csv


if __name__ == "__main__":
    main()
