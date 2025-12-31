# compare_models/run_test_benchmark.py
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# common: 동일 조건 생성용 모듈들 (가능한 건 다 import)
from common import config as cfg
from common import dataset_split
from common import io_wfdb
from common import metrics
from common import noise as noise_mod
from common import repro
from common import utils

# 우리 알고리즘(비딥러닝)
from models.model_proposed.v37_standalone import v37_baseline_correction

# (선택) 딥러닝 모델들: repo에 맞춰 adapter를 연결해
# 예: from compare_models.methods.UNet_1D.infer import unet1d_infer
# 예: from compare_models.methods.Improved_DAE.infer import dae_infer


# -------------------------
# Output schema
# -------------------------
DETAIL_HEADER = [
    "method", "rec_id", "noise_rec", "snr_target_db",
    "snr_in_db", "snr_out_db", "snr_improve_db",
    "rmse", "nrmse_std", "prd_percent",
    "N"
]

SUMMARY_HEADER = [
    "method", "noise_rec", "snr_target_db", "count",
    "snr_in_mean", "snr_in_std",
    "snr_out_mean", "snr_out_std",
    "snr_improve_mean", "snr_improve_std",
    "rmse_mean", "rmse_std",
    "nrmse_mean", "nrmse_std",
    "prd_mean", "prd_std"
]


# -------------------------
# Helpers
# -------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def _compute_zref(x: np.ndarray, method: str = "median") -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    if method == "mean":
        off = float(np.mean(x))
    else:
        off = float(np.median(x))
    return (x - off).astype(np.float32), off


def save_tripanel_plot(
    out_png: Path,
    rec_id: str,
    method: str,
    ref: np.ndarray,
    noisy_in: np.ndarray,
    corrected: np.ndarray,
    fs: float,
    snr_db: float,
    zref_method: str = "median",
    xlim: Tuple[float, float] = (10, 15),
):
    N = int(min(len(ref), len(noisy_in), len(corrected)))
    ref = ref[:N]
    noisy_in = noisy_in[:N]
    corrected = corrected[:N]

    z_ref, off = _compute_zref(ref, method=zref_method)
    z_noisy = (noisy_in - off).astype(np.float32)

    # corrected는 "baseline 제거 결과"라 가정 (이미 0 기준선이면 그대로)
    z_corr = corrected.astype(np.float32)

    t = np.arange(N) / fs
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t, ref, linewidth=1.1, alpha=0.9, label="Ref (raw)")
    axes[0].plot(t, z_ref, linewidth=1.1, alpha=0.9, color="0.4", label="Z-Ref (0 baseline)")
    axes[0].set_title(f"rec={rec_id} | method={method} | Ref")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, z_noisy, linewidth=1.0, alpha=0.9, label=f"Noisy in ({snr_db} dB)")
    axes[1].set_title("Noisy Input (Z-Ref shifted)")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, z_ref, linewidth=1.0, alpha=0.8, color="0.4", label="Z-Ref")
    axes[2].plot(t, z_corr, linewidth=1.2, alpha=0.9, label="Corrected")
    axes[2].set_title("Corrected")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.set_xlim(*xlim)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# -------------------------
# Model adapters (중요)
# -------------------------
@dataclass
class RunContext:
    fs: float
    r_idx: Optional[np.ndarray] = None
    # 필요하면 여기에 model_dir, device, etc 확장


def run_method_proposed(x_in: np.ndarray, ctx: RunContext) -> np.ndarray:
    y, _dbg = v37_baseline_correction(
        x_in,
        fs=ctx.fs,
        r_idx=ctx.r_idx,
        adaptive_denoise=True
    )
    return y


# TODO: 네 repo의 딥러닝 모델들 inference 함수를 여기 어댑터로 붙이면 됨.
# 아래는 “형식”만 맞춘 placeholder.
def run_method_unet1d(x_in: np.ndarray, ctx: RunContext) -> np.ndarray:
    raise NotImplementedError("UNet1D adapter를 연결해줘야 함 (infer 함수 import해서 여기서 호출).")


def run_method_dae(x_in: np.ndarray, ctx: RunContext) -> np.ndarray:
    raise NotImplementedError("Improved DAE adapter를 연결해줘야 함.")


def build_method_registry() -> Dict[str, Callable[[np.ndarray, RunContext], np.ndarray]]:
    return {
        "proposed": run_method_proposed,  # ✅ 우리 알고리즘
        "unet1d": run_method_unet1d,       # ✅ 여기에 연결
        "dae": run_method_dae,             # ✅ 여기에 연결
        # "hpf": ...
        # "asls": ...
        # "wavelet_median": ...
    }


# -------------------------
# Data loader (test split)
# -------------------------
def load_test_cases_from_split(
    split_path: Path,
    split_name: str = "test",
) -> List[Dict[str, Any]]:
    """
    splits.json 같은 파일에서 test 케이스 리스트를 가져온다고 가정.
    구조는 네 dataset_split 구현에 맞게 바꿔도 됨.
    """
    # ✅ dataset_split 모듈이 이미 읽는 함수가 있으면 그걸 쓰고,
    # 없으면 json 직접 파싱해서 쓰면 됨.
    # 여기서는 "json 직접 파싱 fallback" 포함.
    if hasattr(dataset_split, "load_splits"):
        splits = dataset_split.load_splits(split_path)
    else:
        splits = json.loads(split_path.read_text(encoding="utf-8"))

    if split_name not in splits:
        raise KeyError(f"split '{split_name}' not found in {split_path}")

    # 각 item이 {rec_id, ...} 같은 dict라고 가정
    return splits[split_name]


def read_mitdb_segment(rec_id: str, start_sec: int, duration_sec: int) -> Tuple[np.ndarray, float]:
    """
    io_wfdb 쪽에 읽는 함수가 있으면 그걸 쓰고,
    없으면 wfdb 직접 접근하는 방식으로 fallback.
    """
    # 1) common.io_wfdb에 함수가 있으면 사용
    if hasattr(io_wfdb, "read_record_segment"):
        return io_wfdb.read_record_segment(rec_id, start_sec, duration_sec)

    # 2) fallback: wfdb 직접 (repo에 맞게 root는 config에서 가져오길 권장)
    import wfdb  # local import
    mitdb_root = Path(cfg.MITDB_DIR) if hasattr(cfg, "MITDB_DIR") else Path("data/MITDB_data")
    record_path = str(mitdb_root / str(rec_id))
    record = wfdb.rdrecord(record_path)
    fs = float(record.fs)
    x = record.p_signal[:, 0].astype(np.float32)

    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    x_seg = x[s:e]
    if x_seg.size == 0:
        raise ValueError(f"Empty segment rec={rec_id}")
    return x_seg, fs


def read_rpeaks(rec_id: str, start_sec: int, duration_sec: int) -> Tuple[np.ndarray, float]:
    if hasattr(io_wfdb, "read_rpeaks_segment"):
        return io_wfdb.read_rpeaks_segment(rec_id, start_sec, duration_sec)

    import wfdb  # local import
    mitdb_root = Path(cfg.MITDB_DIR) if hasattr(cfg, "MITDB_DIR") else Path("data/MITDB_data")
    record_path = str(mitdb_root / str(rec_id))
    ann = wfdb.rdann(record_path, "atr")
    r_all = np.asarray(ann.sample, dtype=np.int64)

    fs_raw = float(wfdb.rdrecord(record_path).fs)
    s = int(start_sec * fs_raw)
    e = s + int(duration_sec * fs_raw)
    r_seg = r_all[(r_all >= s) & (r_all < e)] - s
    return r_seg.astype(np.int64), fs_raw


def read_noise_segment(noise_rec: str, start_sec: int, duration_sec: int) -> Tuple[np.ndarray, float]:
    if hasattr(io_wfdb, "read_noise_segment"):
        return io_wfdb.read_noise_segment(noise_rec, start_sec, duration_sec)

    import wfdb  # local import
    noise_root = Path(cfg.NOISE_DIR) if hasattr(cfg, "NOISE_DIR") else Path("data/noise_data")
    sig, fields = wfdb.rdsamp(str(noise_root / noise_rec))
    fs = float(fields["fs"])
    n = sig[:, 0].astype(np.float32)

    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    n_seg = n[s:e]
    if n_seg.size == 0:
        raise ValueError(f"Empty noise segment noise_rec={noise_rec}")
    return n_seg, fs


# -------------------------
# Main test runner
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_path", type=str, default=str(Path("common/splits.json")))
    ap.add_argument("--split", type=str, default="test", choices=["train", "valid", "val", "test"])
    ap.add_argument("--methods", type=str, default="proposed", help="comma separated: proposed,unet1d,dae")
    ap.add_argument("--noise_rec", type=str, default="bw", choices=["bw", "ma", "em"])
    ap.add_argument("--snrs", type=str, default="0,5,10,15")
    ap.add_argument("--fs_target", type=float, default=250.0)
    ap.add_argument("--start_sec", type=int, default=0)
    ap.add_argument("--duration_sec", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=str(Path("outputs/benchmark_test")))
    ap.add_argument("--plot_one", action="store_true", help="대표 케이스 1개만 3단 패널로 저장")
    ap.add_argument("--plot_rec", type=str, default="", help="대표 plot 레코드 id (비우면 첫 케이스)")
    args = ap.parse_args()

    # ✅ 재현성
    if hasattr(repro, "seed_all"):
        repro.seed_all(args.seed)
    else:
        np.random.seed(args.seed)

    out_root = Path(args.out_dir)
    csv_dir = out_root / "csv"
    plot_dir = out_root / "plots"
    _ensure_dir(csv_dir)
    _ensure_dir(plot_dir)

    snr_levels = [float(s) for s in args.snrs.split(",") if s.strip() != ""]
    method_names = [m.strip() for m in args.methods.split(",") if m.strip() != ""]

    registry = build_method_registry()

    for m in method_names:
        if m not in registry:
            raise KeyError(f"Unknown method='{m}'. Available={list(registry.keys())}")

    # ✅ test 케이스 로드
    test_cases = load_test_cases_from_split(Path(args.split_path), split_name=args.split)
    if not test_cases:
        raise RuntimeError(f"No cases in split={args.split} from {args.split_path}")

    # ✅ noise 1회 로드 -> target fs로 리샘플
    n_raw, fs_n = read_noise_segment(args.noise_rec, args.start_sec, args.duration_sec)
    n_250 = utils._resample_to_target(n_raw, fs_raw=fs_n, fs_target=args.fs_target)

    detail_rows: List[List[Any]] = []

    # 대표 plot 저장 조건
    plot_saved = {m: False for m in method_names}
    plot_target_rec = args.plot_rec.strip() if args.plot_rec.strip() else str(test_cases[0].get("rec_id", ""))

    # -------------------------
    # loop: case × snr × method
    # -------------------------
    for case in test_cases:
        rec_id = str(case.get("rec_id", case.get("record", "")))
        if rec_id == "":
            raise ValueError(f"Bad case item (no rec_id): {case}")

        # ref 로드 -> target fs
        x_ref_raw, fs_m = read_mitdb_segment(rec_id, args.start_sec, args.duration_sec)
        x_ref = utils._resample_to_target(x_ref_raw, fs_raw=fs_m, fs_target=args.fs_target)

        # r-peaks -> target fs index 변환 (우리 알고리즘에서 필요)
        r_raw, fs_r = read_rpeaks(rec_id, args.start_sec, args.duration_sec)
        r_250 = np.round(r_raw * (args.fs_target / float(fs_r))).astype(np.int64)
        r_250 = r_250[(r_250 >= 0) & (r_250 < len(x_ref))]

        # SNR loop
        for snr_tgt in snr_levels:
            # noisy mix (공통 조건)
            # common.noise 모듈에 add_baseline_wander_snr가 있다고 했으니 그걸 사용
            x_in, ref_used, snr_in = noise_mod.add_baseline_wander_snr(x_ref, n_250, snr_tgt)

            # method loop
            for method in method_names:
                runner = registry[method]
                ctx = RunContext(fs=args.fs_target, r_idx=r_250)

                y_out = runner(x_in, ctx)

                # 길이 맞추기
                N = int(min(ref_used.size, y_out.size))
                ref_eval = ref_used[:N]
                out_eval = y_out[:N]

                # 평가 (모델 출력 이후 공통)
                snr_out = float(metrics.calculate_snr_db(ref_eval, out_eval, remove_mean=True))
                rmse = float(metrics.calculate_rmse(ref_eval, out_eval))
                nrmse = float(metrics.calculate_nrmse(ref_eval, out_eval, mode="std"))
                prd = float(metrics.calculate_prd(ref_eval, out_eval, remove_mean=True))
                imp = float(snr_out - float(snr_in))

                detail_rows.append([
                    method, rec_id, args.noise_rec, float(snr_tgt),
                    float(snr_in), snr_out, imp,
                    rmse, nrmse, prd,
                    int(N)
                ])

                # 대표 plot 1회 저장 (원하면)
                if args.plot_one and (not plot_saved[method]) and (rec_id == plot_target_rec) and (snr_tgt == snr_levels[0]):
                    out_png = plot_dir / f"{method}_rec_{rec_id}_snr{int(snr_tgt)}dB.png"
                    save_tripanel_plot(
                        out_png=out_png,
                        rec_id=rec_id,
                        method=method,
                        ref=ref_used,
                        noisy_in=x_in,
                        corrected=y_out,
                        fs=args.fs_target,
                        snr_db=snr_tgt
                    )
                    plot_saved[method] = True

        print(f"[Done case] rec={rec_id}")

    # -------------------------
    # Save detail CSV
    # -------------------------
    detail_csv = csv_dir / "results_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        w.writerows(detail_rows)

    # -------------------------
    # Save summary CSV (mean/std by method×noise×snr)
    # -------------------------
    idx = {k: DETAIL_HEADER.index(k) for k in DETAIL_HEADER}
    groups = sorted(set((r[idx["method"]], r[idx["noise_rec"]], r[idx["snr_target_db"]]) for r in detail_rows))

    summary_rows: List[List[Any]] = []
    for method, noise_rec, snr_tgt in groups:
        rows_g = [r for r in detail_rows if r[idx["method"]] == method and r[idx["noise_rec"]] == noise_rec and r[idx["snr_target_db"]] == snr_tgt]
        count = len(rows_g)

        snr_in_mean, snr_in_std = _mean_std([float(r[idx["snr_in_db"]]) for r in rows_g])
        snr_out_mean, snr_out_std = _mean_std([float(r[idx["snr_out_db"]]) for r in rows_g])
        imp_mean, imp_std = _mean_std([float(r[idx["snr_improve_db"]]) for r in rows_g])
        rmse_mean, rmse_std = _mean_std([float(r[idx["rmse"]]) for r in rows_g])
        nrmse_mean, nrmse_std = _mean_std([float(r[idx["nrmse_std"]]) for r in rows_g])
        prd_mean, prd_std = _mean_std([float(r[idx["prd_percent"]]) for r in rows_g])

        summary_rows.append([
            method, noise_rec, float(snr_tgt), count,
            snr_in_mean, snr_in_std,
            snr_out_mean, snr_out_std,
            imp_mean, imp_std,
            rmse_mean, rmse_std,
            nrmse_mean, nrmse_std,
            prd_mean, prd_std
        ])

    summary_csv = csv_dir / "results_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        w.writerows(summary_rows)

    print("\n[Saved]")
    print(f"- Detail : {detail_csv}")
    print(f"- Summary: {summary_csv}")
    print(f"- Plots  : {plot_dir} (if --plot_one used)")


if __name__ == "__main__":
    main()