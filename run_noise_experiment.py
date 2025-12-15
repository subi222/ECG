import csv

import numpy as np
import wfdb

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
            # (1) 클린 ECG 로드
            clean = np.loadtxt(f"{base_dir}/{rec_id}.csv",
                               delimiter=",", skiprows=1)[:, 1]

            # (2) 클린에 엔진 돌린 결과 → 기준 신호
            y_clean = process_ecg_from_array(clean, fs_raw=360.0)
            clean_std = y_clean.std()

            print("\n" + "=" * 80)
            print(f"[CLEAN] record {rec_id}, out std={clean_std:.3f}")

            # (3) SNR별로 노이즈 섞어서 실험
            for snr_db in snr_db_list:
                # 3-1) 목표 SNR에 맞게 노이즈 섞기
                mixed, noise_used = mix_with_snr(clean, noise, snr_db)

                # 3-2) 실제 입력 SNR 확인
                snr_in = compute_snr_db(clean, mixed, remove_mean=True)

                # 3-3) 엔진(디버그 모드) 돌리기
                y_clean_dbg, clean_dec, y_corr_clean, r_clean = process_ecg_from_array(
                    clean, fs_raw=360.0, return_debug=True
                )
                y_mixed_dbg, mixed_dec, y_corr_mixed, r_mixed = process_ecg_from_array(
                    mixed, fs_raw=360.0, return_debug=True
                )

                # 3-4) 출력 SNR / NRMSE 계산
                snr_out = compute_snr_db(y_clean_dbg, y_mixed_dbg, remove_mean=True)
                nrmse_out = compute_nrmse(y_clean_dbg, y_mixed_dbg, mode="std")

                print(f"\n[SNR target={snr_db} dB] record {rec_id}")
                print(f"  - input  SNR (clean vs mixed): {snr_in:.2f} dB")
                print(f"  - output SNR (y_clean vs y_mixed): {snr_out:.2f} dB")
                print(f"  - NRMSE (vs y_clean, /std): {nrmse_out:.4f}")

                writer.writerow([
                    rec_id,
                    f"{clean_std:.6f}",
                    snr_db,
                    f"{snr_in:.6f}",
                    f"{snr_out:.6f}",
                    f"{nrmse_out:.6f}",
                ])

    print(f"\nCSV 결과가 '{out_csv}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
