import csv
from pathlib import Path

import numpy as np
import wfdb
import matplotlib.pyplot as plt

from baseline_1st import _resample_to_target
from v37_standalone import v37_baseline_correction
from mixing_utils import (
    add_baseline_wander_snr,
    calculate_snr_db, calculate_rmse,
    calculate_nrmse, calculate_prd
)

# -------------------------
# 설정
# -------------------------
MITDB_DIR = "/home/subi/PycharmProjects/ECG/MITDB_data"
NOISE_DIR = "/home/subi/PycharmProjects/ECG/noise_data"

RECORDS = None          # ✅ 모든 MITDB csv 자동 탐색
NOISE_REC = "bw"
SNR_LEVELS = [0, 5, 10, 15]

FS_TARGET = 250.0
START_SEC = 0
DURATION_SEC = 30

# 결과 폴더
OUT_DIR = Path("results")
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"

DETAIL_CSV = CSV_DIR / "v37_mitdb_nstdb_results_detail.csv"
SUMMARY_CSV = CSV_DIR / "v37_mitdb_nstdb_results_summary.csv"
# -------------------------
# IO Helpers
# -------------------------
def _read_csv_1d(csv_path: Path, lead: str = "MLII") -> np.ndarray:
    """
    MITDB CSV 로드 (네 파일 포맷에 강건):
    - 1행이 "['sample #', 'MLII', 'V5']" 같은 리스트-문자열 헤더일 수 있음
    - 이후 행은 sample, MLII, V5 숫자 3컬럼 (예: 0,995,1011)
    - 기본은 MLII(2번째 컬럼)를 ECG로 사용
    """
    lead = lead.upper().strip()
    col_map = {"MLII": 1, "V5": 2}  # [sample, MLII, V5]
    if lead not in col_map:
        raise ValueError(f"lead must be one of {list(col_map.keys())}, got {lead}")

    target_col = col_map[lead]
    values = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue

            # 첫 줄 헤더/리스트 문자열 스킵
            if i == 0 and (s.startswith("[") or "MLII" in s or "sample" in s.lower()):
                continue

            parts = [p.strip() for p in s.split(",")]
            if len(parts) <= target_col:
                continue

            try:
                v = float(parts[target_col])
                if np.isfinite(v):
                    values.append(v)
            except ValueError:
                # 숫자 변환 실패한 줄은 스킵
                continue

    if len(values) == 0:
        raise ValueError(f"CSV parse failed (no numeric samples): {csv_path}")

    return np.asarray(values, dtype=np.float32)


def _infer_records_from_dir(mitdb_dir: str):
    """
    MITDB 디렉토리에서 레코드 번호 자동 탐색.
    - 기존 CSV 기반(*.csv) + WFDB 기반(*.hea) 둘 다 지원
    - 102-0.atr 같은 보조 파일은 제외 (stem이 숫자가 아님)
    """
    d = Path(mitdb_dir)
    recs = set()

    # 1) CSV 기반 (혹시 남아있으면 같이 지원)
    for fp in d.glob("*.csv"):
        if fp.stem.isdigit():
            recs.add(int(fp.stem))

    # 2) WFDB 기반: 보통 record.hea 가 존재하면 rdrecord(record) 가능
    for fp in d.glob("*.hea"):
        if fp.stem.isdigit():          # "100", "101" 같은 것만
            recs.add(int(fp.stem))

    return sorted(recs)


def load_mitdb_segment(rec_id: int, start_sec: int, duration_sec: int):
    """
    MITDB 신호 로드 (WFDB 우선):
    반환: (x_segment, fs_raw)
    """
    record_path = str(Path(MITDB_DIR) / str(rec_id))
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    x = record.p_signal[:, 0].astype(np.float32)

    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    x_seg = x[s:e]
    if x_seg.size == 0:
        raise ValueError(f"Empty segment for rec={rec_id} (check START_SEC/DURATION_SEC)")
    return x_seg, fs


def load_mitdb_rpeaks(rec_id: int, start_sec: int, duration_sec: int):
    """
    MITDB annotation(atr)에서 R-peak(= 'N' 등 beat 위치)를 읽어
    현재 실험 구간(start~start+duration)에 해당하는 샘플 인덱스 배열을 반환.

    반환:
      r_idx_seg_raw: segment 기준(0부터 시작) 인덱스 (fs_raw 기준)
      fs_raw: MITDB 원본 fs (보통 360Hz)
    """
    record_path = str(Path(MITDB_DIR) / str(rec_id))

    # MIT-BIH annotation 파일: 보통 'atr'
    ann = wfdb.rdann(record_path, "atr")  # ann.sample: 전체 레코드 기준 샘플 인덱스

    # beat 타입 필터링(너무 엄격히 하면 빠질 수 있어서, 일단 전체 beat를 허용)
    # 필요하면: symbols = np.array(ann.symbol); mask = np.isin(symbols, ["N","L","R","A","V",...])
    r_all = np.asarray(ann.sample, dtype=np.int64)

    # 구간 자르기 (전체 레코드 기준 -> segment 기준)
    # 주의: load_mitdb_segment에서 fs를 record.fs로 가져오므로 동일한 fs 사용
    fs_raw = wfdb.rdrecord(record_path).fs
    s = int(start_sec * fs_raw)
    e = s + int(duration_sec * fs_raw)

    r_seg = r_all[(r_all >= s) & (r_all < e)] - s
    return r_seg.astype(np.int64), float(fs_raw)


def load_noise_segment(noise_rec: str, start_sec: int, duration_sec: int):
    """
    NSTDB noise (bw/ma/em) 로드: noise_data/{noise_rec}.hea/.dat
    반환: (n_segment, fs_raw)
    """
    sig, fields = wfdb.rdsamp(str(Path(NOISE_DIR) / noise_rec))
    fs = float(fields["fs"])
    n = sig[:, 0].astype(np.float32)
    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    n_seg = n[s:e]
    if n_seg.size == 0:
        raise ValueError(f"Empty noise segment for {noise_rec} (check START_SEC/DURATION_SEC)")
    return n_seg, fs

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# CSV Writers
# -------------------------
DETAIL_HEADER = [
    "rec_id", "noise_rec", "snr_target_db",
    "snr_in_db", "snr_out_db", "snr_improve_db",
    "nrmse_std", "prd_percent"
]

SUMMARY_HEADER = [
    "noise_rec", "snr_target_db", "count",
    "snr_in_mean", "snr_in_std",
    "snr_out_mean", "snr_out_std",
    "snr_improve_mean", "snr_improve_std",
    "nrmse_mean", "nrmse_std",
    "prd_mean", "prd_std"
]


def _ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_detail_rows(rows):
    _ensure_out_dir()
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        w.writerows(rows)


def _write_summary_rows(rows):
    _ensure_out_dir()
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        w.writerows(rows)

# -------------------------
# 시각화 저장
# -------------------------
# -------------------------
# 시각화 저장 (3단 패널: Ref / Noisy / Corrected)
# -------------------------
def _compute_zref_zero_baseline(x: np.ndarray, method: str = "median"):
    """
    Z-Ref: 임상적으로 의미 있는 0 기준선(isoelectric line)에 맞춘 reference.
    여기서는 간단/강건하게 median(권장) 또는 mean으로 DC/오프셋을 제거해 0 기준선을 만듭니다.

    반환:
      z_ref: x - offset  (0 기준선으로 이동된 신호)
      offset: 제거한 오프셋 값
    """
    x = np.asarray(x, dtype=np.float64)
    if method == "mean":
        offset = float(np.mean(x))
    else:
        # median이 QRS/잡음에 덜 민감해서 baseline 0 기준선으로 더 안정적인 편
        offset = float(np.median(x))
    return (x - offset).astype(np.float32), offset


def save_tripanel_plot(rec, ref, noisy_in, corrected, fs, snr_db, zref_method: str = "median"):
    """
    첨부한 예시처럼 3개 subplot으로 저장:
      (1) Ref (원본) vs Z-Ref(0 기준선)
      (2) Noisy Input
      (3) Corrected vs Z-Ref
    """
    N = int(min(len(ref), len(noisy_in), len(corrected)))
    ref = ref[:N]
    noisy_in = noisy_in[:N]
    corrected = corrected[:N]

    # Z-Ref (0 baseline) 만들기 + 같은 오프셋을 noisy/corrected에도 적용
    z_ref, offset = _compute_zref_zero_baseline(ref, method=zref_method)
    z_noisy = (noisy_in - offset).astype(np.float32)
    z_corr = corrected.astype(np.float32)

    t = np.arange(N) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    # (1) Ref panel
    axes[0].plot(t, ref, label="Original MIT-BIH", linewidth=1.2, alpha=0.9)
    axes[0].plot(t, z_ref, label="Z-Ref (0 baseline)", linewidth=1.2, alpha=0.9, color="0.4")
    axes[0].set_title(f"Patient {rec} - Ref")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.2)

    # (2) Noisy panel
    axes[1].plot(t, z_noisy, label=f"Noisy ({NOISE_REC}, {snr_db}dB)", linewidth=1.0, alpha=0.9)
    axes[1].set_title("Noisy Input")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.2)

    # (3) Corrected panel
    axes[2].plot(t, z_ref, label="Z-Ref", linewidth=1.0, alpha=0.8, color="0.4")
    axes[2].plot(t, z_corr, label="Corrected", linewidth=1.2, alpha=0.9)
    axes[2].set_title("Corrected")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.2)

    # x축을 10~15초로 제한
    for ax in axes:
        ax.set_xlim(10, 15)

    fig.tight_layout()
    out_path = PLOT_DIR / f"rec_{rec}_snr{snr_db}dB.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    ensure_dirs()
    records = _infer_records_from_dir(MITDB_DIR)

    print(f"[Config] MITDB_DIR={MITDB_DIR} | NOISE_DIR={NOISE_DIR}")
    print(f"[Config] records={records}")
    print(f"[Config] noise={NOISE_REC}, FS_TARGET={FS_TARGET}, window={DURATION_SEC}s from {START_SEC}s")
    print(f"[Output] {DETAIL_CSV} / {SUMMARY_CSV}")

    detail_rows = []

    # noise는 1번만 로드 + 250Hz 리샘플
    n_raw, fs_n = load_noise_segment(NOISE_REC, START_SEC, DURATION_SEC)
    n_250_full = _resample_to_target(n_raw, fs_raw=fs_n, fs_target=FS_TARGET)

    for rec in records:
        # MITDB 레코드 로드 + 250Hz 리샘플
        x_ref_raw, fs_m = load_mitdb_segment(rec, START_SEC, DURATION_SEC)
        x_ref_250_full = _resample_to_target(x_ref_raw, fs_raw=fs_m, fs_target=FS_TARGET)

        # --- (추가) MITDB annotation에서 r_idx 추출 + 250Hz로 변환 ---
        r_idx_raw, fs_ann = load_mitdb_rpeaks(rec, START_SEC, DURATION_SEC)

        # 안전 체크: fs_m(신호)와 fs_ann(ann)이 다르면 이상하니 fs_m을 기준으로 사용
        fs_used = float(fs_m)

        # raw(fs_used) -> target(FS_TARGET)로 인덱스 스케일링
        r_idx_250 = np.round(r_idx_raw * (FS_TARGET / fs_used)).astype(np.int64)

        # 범위 밖 제거(리샘플/라운딩으로 끝 인덱스가 넘어갈 수 있음)
        r_idx_250 = r_idx_250[(r_idx_250 >= 0) & (r_idx_250 < len(x_ref_250_full))]

        # 대표 SNR 하나만 시각화
        snr_vis = SNR_LEVELS[0]
        x_in_vis, ref_vis, _ = add_baseline_wander_snr(
            x_ref_250_full, n_250_full, snr_vis
        )

        y_vis, _ = v37_baseline_correction(
            x_in_vis, fs=FS_TARGET, r_idx=r_idx_250,
            adaptive_denoise=True
        )
        save_tripanel_plot(rec, ref_vis, x_in_vis, y_vis, FS_TARGET, snr_vis)

        print(f"\n[rec={rec}] MITDB len={len(x_ref_raw)} fs={fs_m} | NOISE len={len(n_raw)} fs={fs_n}")

        for snr_tgt in SNR_LEVELS:
            # SNR 맞춰 섞기
            x_in_250, ref_250, snr_in = add_baseline_wander_snr(
                x_ref_250_full, n_250_full, snr_tgt
            )

            # 처리
            y_out_250, _ = v37_baseline_correction(x_in_250, fs=FS_TARGET, r_idx=r_idx_250, adaptive_denoise=True)


            # 길이 맞추기
            N = int(min(ref_250.size, y_out_250.size))
            ref_eval = ref_250[:N]
            out_eval = y_out_250[:N]

            # 평가
            snr_out = calculate_snr_db(ref_eval, out_eval, remove_mean=True)
            rmse = float(calculate_rmse(ref_eval, out_eval))
            nrmse = float(calculate_nrmse(ref_eval, out_eval, mode="std"))
            prd = float(calculate_prd(ref_eval, out_eval, remove_mean=True))
            imp = float(snr_out - snr_in)

            detail_rows.append([
                rec, NOISE_REC, snr_tgt,
                float(snr_in), float(snr_out), imp,
                nrmse, prd
            ])

            print(
                f"  [target={snr_tgt:>2}dB] "
                f"in={snr_in:6.2f} dB | out={snr_out:6.2f} dB | imp={imp:6.2f} dB | "
                f"rmse={rmse:.6f} | nrmse(std)={nrmse:.6f} | prd={prd:.2f}%"
            )

    # (A) 상세 결과 저장
    _write_detail_rows(detail_rows)

    # (B) 요약(평균/표준편차) 저장
    summary_rows = []
    if detail_rows:
        idx_noise = DETAIL_HEADER.index("noise_rec")
        idx_snr_tgt = DETAIL_HEADER.index("snr_target_db")
        idx_snr_in = DETAIL_HEADER.index("snr_in_db")
        idx_snr_out = DETAIL_HEADER.index("snr_out_db")
        idx_imp = DETAIL_HEADER.index("snr_improve_db")
        idx_nrmse = DETAIL_HEADER.index("nrmse_std")
        idx_prd = DETAIL_HEADER.index("prd_percent")

        groups = sorted(set((row[idx_noise], row[idx_snr_tgt]) for row in detail_rows))

        def _mean_std(vals):
            vals = np.asarray(vals, dtype=np.float64)
            return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0

        for noise_rec, snr_tgt in groups:
            rows_g = [row for row in detail_rows if row[idx_noise] == noise_rec and row[idx_snr_tgt] == snr_tgt]
            count = len(rows_g)

            snr_in_mean, snr_in_std = _mean_std([r[idx_snr_in] for r in rows_g])
            snr_out_mean, snr_out_std = _mean_std([r[idx_snr_out] for r in rows_g])
            imp_mean, imp_std = _mean_std([r[idx_imp] for r in rows_g])
            nrmse_mean, nrmse_std = _mean_std([r[idx_nrmse] for r in rows_g])
            prd_mean, prd_std = _mean_std([r[idx_prd] for r in rows_g])

            summary_rows.append([
                noise_rec, snr_tgt,
                count,
                snr_in_mean, snr_in_std,
                snr_out_mean, snr_out_std,
                imp_mean, imp_std,
                nrmse_mean, nrmse_std,
                prd_mean, prd_std
            ])

    _write_summary_rows(summary_rows)

    print("\n[Done] Saved:")
    print(f" - Detail : {DETAIL_CSV}")
    print(f" - Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()