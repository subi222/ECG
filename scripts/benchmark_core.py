# benchmark/benchmark_core.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# 공통: 동일 조건 생성용 모듈들
from common import config as cfg
from common import dataset_split
from common import io_wfdb
from common import metrics
from common import noise as noise_mod
from common import repro
from common import utils


# -------------------------
# Output schema
# -------------------------
DETAIL_HEADER = [
    "method", "rec_id", "noise_rec", "snr_target_db",
    "snr_in", "snr_out", "snr_improve",
    "rmse", "prd", "ssim",
    "samples"
]

SUMMARY_HEADER = [
    "method", "noise_rec", "snr_target_db", "count",
    "snr_out_mean", "snr_out_std",
    "rmse_mean", "rmse_std",
    "prd_mean", "ssim_mean"
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

    # Calculate consistent Y-limits based on reference signal (with 10% margin)
    y_min, y_max = z_ref.min(), z_ref.max()
    y_margin = (y_max - y_min) * 0.1
    ylim = (y_min - y_margin, y_max + y_margin)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


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
    if hasattr(dataset_split, "load_splits"):
        splits = dataset_split.load_splits(split_path)
    else:
        splits = json.loads(split_path.read_text(encoding="utf-8"))

    if split_name not in splits:
        raise KeyError(f"split '{split_name}' not found in {split_path}")

    items = splits[split_name]

    # splits.json이 [100, 101] 같은 int 리스트인 경우를 dict 포맷으로 통일
    if items and isinstance(items[0], (int, str)):
        return [{"rec_id": str(x)} for x in items]

    # 이미 [{"rec_id":...}, ...] 형태면 그대로
    return items


def read_mitdb_segment(rec_id: str, start_sec: int, duration_sec: int) -> Tuple[np.ndarray, float]:
    """
    io_wfdb 쪽에 읽는 함수가 있으면 그걸 쓰고,
    없으면 wfdb 직접 접근하는 방식으로 fallback.
    """
    if hasattr(io_wfdb, "read_record_segment"):
        return io_wfdb.read_record_segment(rec_id, start_sec, duration_sec)

    import wfdb  # local import
    ROOT = Path(__file__).resolve().parents[1]  # .../ECG
    mitdb_root = Path(cfg.MITDB_DIR) if hasattr(cfg, "MITDB_DIR") else (ROOT / "data" / "MITDB_data")
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
    ROOT = Path(__file__).resolve().parents[1]  # .../ECG
    mitdb_root = Path(cfg.MITDB_DIR) if hasattr(cfg, "MITDB_DIR") else (ROOT / "data" / "MITDB_data")
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

    # ✅ 프로젝트 루트 기준으로 noise_data 경로 고정
    ROOT = Path(__file__).resolve().parents[1]   # .../ECG
    noise_root = ROOT / "data" / "noise_data"

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
# Core runner config & context
# -------------------------
@dataclass
class BenchmarkArgs:
    split_path: Path
    split: str
    methods: List[str]
    noise_rec: str
    snr_levels: List[float]
    fs_target: float
    start_sec: int
    duration_sec: int
    seed: int
    out_dir: Path
    plot_one: bool = False
    plot_rec: str = ""


@dataclass
class RunContext:
    fs: float
    r_idx: Optional[np.ndarray] = None
    device: Optional[str] = None  # <-- 추가 (e.g., "cuda", "cpu")

RunnerFn = Callable[[np.ndarray, RunContext], np.ndarray]


def run_benchmark(
    args: BenchmarkArgs,
    registry: Dict[str, RunnerFn],
):
    # 재현성
    if hasattr(repro, "seed_all"):
        repro.seed_all(args.seed)
    else:
        np.random.seed(args.seed)

    out_root = Path(args.out_dir)
    csv_dir = out_root / "csv"
    plot_dir = out_root / "plots"
    _ensure_dir(csv_dir)
    _ensure_dir(plot_dir)

    for m in args.methods:
        if m not in registry:
            raise KeyError(f"Unknown method='{m}'. Available={list(registry.keys())}")

    # test 케이스 로드
    test_cases = load_test_cases_from_split(Path(args.split_path), split_name=args.split)
    if not test_cases:
        raise RuntimeError(f"No cases in split={args.split} from {args.split_path}")

    # noise 1회 로드 -> target fs로 리샘플
    n_raw, fs_n = read_noise_segment(args.noise_rec, args.start_sec, args.duration_sec)
    n_250 = utils._resample_to_target(n_raw, fs_raw=fs_n, fs_target=args.fs_target)

    detail_rows: List[List[Any]] = []

    # 대표 plot 저장 조건
    plot_target_rec = args.plot_rec.strip() if args.plot_rec.strip() else str(test_cases[0].get("rec_id", ""))

    # loop: case × snr × method
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

        for snr_tgt in args.snr_levels:
            # noisy mix (공통 조건)
            x_in, ref_used, snr_in = noise_mod.add_baseline_wander_snr(x_ref, n_250, snr_tgt)

            for method in args.methods:
                runner = registry[method]
                # run_benchmark.py에서 --device 전달된 값을 쓰려면
                # (가장 간단하게) 환경변수/전역으로 받기보다 여기 ctx에 넣는 게 깔끔해
                import os
                device = os.environ.get("ECG_DEVICE", None)
                ctx = RunContext(fs=args.fs_target, r_idx=r_250, device=device)

                y_out = runner(x_in, ctx)

                # 길이 맞추기
                N = int(min(ref_used.size, y_out.size))
                ref_eval = ref_used[:N]
                out_eval = y_out[:N]

                # 평가 (모델 출력 이후 공통)
                snr_out = float(metrics.calculate_snr_db(ref_eval, out_eval, remove_mean=True))
                rmse = float(metrics.calculate_rmse(ref_eval, out_eval))
                prd = float(metrics.calculate_prd(ref_eval, out_eval, remove_mean=True))
                ssim = float(metrics.calculate_ssim(ref_eval, out_eval, remove_mean=True))
                snr_improve = float(snr_out - float(snr_in))

                detail_rows.append([
                    method, rec_id, args.noise_rec, float(snr_tgt),
                    float(snr_in), snr_out, snr_improve,
                    rmse, prd, ssim,
                    int(N)
                ])
                print(
                    f"[{method}] rec={rec_id} snr={snr_tgt}dB "
                    f"in={snr_in:.2f} out={snr_out:.2f} imp={snr_improve:.2f} "
                    f"rmse={rmse:.6f} prd={prd:.2f}% "
                    f"ssim={ssim:.4f}"
                )

                # ✅ 모든 test rec의 0dB plot 저장
                if args.plot_one and (snr_tgt in [0, 5, 10, 15]):
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

        print(f"[Done case] rec={rec_id}")

    # Save detail CSV
    detail_csv = csv_dir / "results_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        w.writerows(detail_rows)

    # Save summary CSV (mean/std by method×noise×snr)
    idx = {k: DETAIL_HEADER.index(k) for k in DETAIL_HEADER}
    groups = sorted(set((r[idx["method"]], r[idx["noise_rec"]], r[idx["snr_target_db"]]) for r in detail_rows))

    summary_rows: List[List[Any]] = []
    for m, nr, s_tgt in groups:
        group_rows = [
            r for r in detail_rows
            if r[idx["method"]] == m
            and r[idx["noise_rec"]] == nr
            and r[idx["snr_target_db"]] == s_tgt
        ]

        snr_v = [r[idx["snr_out"]] for r in group_rows]
        rmse_v = [r[idx["rmse"]] for r in group_rows]
        prd_v = [r[idx["prd"]] for r in group_rows]
        ssim_v = [r[idx["ssim"]] for r in group_rows]

        summary_rows.append([
            m, nr, s_tgt, len(group_rows),
            np.mean(snr_v), np.std(snr_v),
            np.mean(rmse_v), np.std(rmse_v),
            np.mean(prd_v),
            np.mean(ssim_v)
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