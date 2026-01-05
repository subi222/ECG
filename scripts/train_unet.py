"""
train_unet.py (Universal multi-SNR training)

- Uses common/* modules for fair benchmarking
- Uses fixed record-wise split from common/splits.json (train/val)
- Trains ONE universal UNet using mixed SNR windows (0/5/10/15 by default)
- Saves only:
  1) best_model.pth
  2) training_log.csv
  3) config.json

Requirements:
- models/model_UNet.py provides UNet
- common/ contains: config, io_wfdb, noise, repro, utils
- common/splits.json exists
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Common imports (must match other models)
# -------------------------
from common import config
from common import io_wfdb
from common import noise as noise_mixer
from common import repro
from common import utils

# Model
from models.model_UNet import UNet


# -------------------------
# Helpers
# -------------------------
def load_splits(splits_path: Path) -> Tuple[List[int], List[int]]:
    """
    Expect keys: train/val/test.
    (Optional compatibility: if 'valid' exists, treat as val)
    """
    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    if "train" not in splits:
        raise KeyError(f"'train' key not found in splits: {splits_path}")

    # allow either "val" or "valid"
    if "val" in splits:
        val_key = "val"
    elif "valid" in splits:
        val_key = "valid"
    else:
        raise KeyError(f"'val' (or 'valid') key not found in splits: {splits_path}")

    train_records = [int(x) for x in splits["train"]]
    val_records = [int(x) for x in splits[val_key]]
    return train_records, val_records


def load_and_resample_mitdb_segment(
    mitdb_dir: Path,
    record: int,
    start_sample: int,
    duration_sec: int,
    fs_target: int,
) -> np.ndarray:
    ecg_raw, fs_raw = io_wfdb.load_mitdb_wfdb(
        mitdb_dir=mitdb_dir,
        record=record,
        start_sample=start_sample,
        duration_sec=duration_sec,
    )
    if ecg_raw.size == 0:
        return ecg_raw.astype(np.float32)

    ecg = utils._resample_to_target(ecg_raw, fs_raw=float(fs_raw), fs_target=float(fs_target))
    return ecg.astype(np.float32, copy=False)


def load_and_resample_noise_segment(
    nstdb_dir: Path,
    noise_record: str,
    start_sample: int,
    duration_sec: int,
    fs_target: int,
) -> np.ndarray:
    nz_raw, fs_read = io_wfdb.load_nstdb_noise(
        nstdb_dir=nstdb_dir,
        record=noise_record,
        start_sample=start_sample,
        duration_sec=duration_sec,
        fs=fs_target,  # length calculation base
    )
    if nz_raw.size == 0:
        return nz_raw.astype(np.float32)

    nz = utils._resample_to_target(nz_raw, fs_raw=float(fs_read), fs_target=float(fs_target))
    return nz.astype(np.float32, copy=False)


def make_windows(
    clean: np.ndarray,
    noisy: np.ndarray,
    win_len: int,
    hop_len: int,
    normalize: str = "minmax_by_noisy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    clean/noisy -> (X, Y) windows with shape (N, 1, win_len)

    normalize:
      - "minmax_by_noisy": x_win min-max로 x,y 둘 다 같은 denom 사용 (기존 의도)
      - "none": 정규화 없음
    """
    assert clean.ndim == 1 and noisy.ndim == 1
    N = min(len(clean), len(noisy))
    clean = clean[:N]
    noisy = noisy[:N]

    X_list, Y_list = [], []
    for s in range(0, N - win_len + 1, hop_len):
        x_win = noisy[s : s + win_len]
        y_win = clean[s : s + win_len]

        if normalize == "minmax_by_noisy":
            x_min = float(np.min(x_win))
            x_max = float(np.max(x_win))
            denom = (x_max - x_min) + 1e-8
            x_norm = (x_win - x_min) / denom
            y_norm = (y_win - x_min) / denom
        elif normalize == "none":
            x_norm = x_win
            y_norm = y_win
        else:
            raise ValueError("normalize must be 'minmax_by_noisy' or 'none'")

        X_list.append(x_norm.astype(np.float32, copy=False))
        Y_list.append(y_norm.astype(np.float32, copy=False))

    if not X_list:
        return np.zeros((0, 1, win_len), dtype=np.float32), np.zeros((0, 1, win_len), dtype=np.float32)

    X = np.stack(X_list, axis=0)[:, None, :]  # (N,1,L)
    Y = np.stack(Y_list, axis=0)[:, None, :]
    return X, Y


def build_dataset_from_records(
    records: List[int],
    mitdb_dir: Path,
    nstdb_dir: Path,
    noise_record: str,
    start_sample: int,
    duration_sec: int,
    fs_target: int,
    target_snr_list: List[float],   # ✅ Universal: multiple SNRs merged into ONE dataset
    win_len: int,
    hop_len: int,
    normalize: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal training dataset builder:
    For each record, generate windows for ALL SNRs in target_snr_list, then concatenate.

    Returns:
      X_all: (N,1,win_len) noisy windows
      Y_all: (N,1,win_len) clean windows (ref)
    """
    all_X, all_Y = [], []

    for rec in records:
        clean = load_and_resample_mitdb_segment(
            mitdb_dir=mitdb_dir,
            record=rec,
            start_sample=start_sample,
            duration_sec=duration_sec,
            fs_target=fs_target,
        )
        if clean.size == 0:
            print(f"[Skip] record {rec}: empty clean segment")
            continue

        bw = load_and_resample_noise_segment(
            nstdb_dir=nstdb_dir,
            noise_record=noise_record,
            start_sample=start_sample,
            duration_sec=duration_sec,
            fs_target=fs_target,
        )
        if bw.size == 0:
            print(f"[Skip] record {rec}: empty noise segment")
            continue

        total_windows_rec = 0
        for snr_db in target_snr_list:
            noisy, ref, actual_snr = noise_mixer.add_baseline_wander_snr(clean, bw, float(snr_db))
            X, Y = make_windows(ref, noisy, win_len=win_len, hop_len=hop_len, normalize=normalize)

            if X.shape[0] == 0:
                print(f"[Skip] record {rec} snr={snr_db}: no windows (len={len(ref)})")
                continue

            all_X.append(X)
            all_Y.append(Y)
            total_windows_rec += X.shape[0]

            print(f"[Rec {rec}] snr={snr_db}dB windows={X.shape[0]} | input SNR(actual)={actual_snr:.2f} dB")

        if total_windows_rec == 0:
            print(f"[Skip] record {rec}: no windows for all SNRs")

    if not all_X:
        return np.zeros((0, 1, win_len), dtype=np.float32), np.zeros((0, 1, win_len), dtype=np.float32)

    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)
    return X_all, Y_all


def evaluate_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    crit = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(1, n_batches)


# -------------------------
# Train
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # scripts/train_unet.py 기준: 한 단계 위가 프로젝트 루트(ECG)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_SPLITS = PROJECT_ROOT / "common" / "splits.json"

    parser.add_argument("--mitdb_dir", type=str, default=str(config.MITDB_DIR_DEFAULT))
    parser.add_argument("--nstdb_dir", type=str, default=str(config.NSTDB_DIR_DEFAULT))
    parser.add_argument("--splits", type=str, default=str(DEFAULT_SPLITS))

    parser.add_argument("--noise", type=str, default=config.NSTDB_RECORD_DEFAULT)

    # ✅ Universal multi-SNR: merge data from these SNR levels into one dataset/model
    parser.add_argument(
        "--snr_list",
        type=float,
        nargs="+",
        default=[0, 5, 10, 15],
        help="Target SNR list for universal training (dB). Data from all SNRs are merged into ONE model.",
    )

    parser.add_argument("--fs", type=int, default=config.FS_DEFAULT)
    parser.add_argument("--start_sample", type=int, default=config.START_SAMPLE_DEFAULT)
    parser.add_argument("--duration_sec", type=int, default=config.DURATION_SEC_DEFAULT)

    parser.add_argument("--win_len", type=int, default=512)
    parser.add_argument("--hop_len", type=int, default=512)  # non-overlap default
    parser.add_argument("--normalize", type=str, default="minmax_by_noisy", choices=["minmax_by_noisy", "none"])

    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    # ✅ One model output dir (do NOT split by SNR)
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "UNet"))

    args = parser.parse_args()

    # Reproducibility
    repro.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    mitdb_dir = Path(args.mitdb_dir)
    nstdb_dir = Path(args.nstdb_dir)
    splits_path = Path(args.splits)

    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path.resolve()}")

    train_records, val_records = load_splits(splits_path)
    print(f"[Split] train={len(train_records)} recs | val={len(val_records)} recs")
    print(f"        train={train_records}")
    print(f"        val  ={val_records}")
    print(f"[SNR] universal snr_list={args.snr_list}")

    # -------------------------
    # Build datasets in RAM (Universal: merge SNR windows)
    # -------------------------
    print("[Data] Building train windows (universal, merged SNRs)...")
    X_tr, Y_tr = build_dataset_from_records(
        records=train_records,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        noise_record=args.noise,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs,
        target_snr_list=args.snr_list,   # ✅
        win_len=args.win_len,
        hop_len=args.hop_len,
        normalize=args.normalize,
    )
    print(f"[Train] X={X_tr.shape} | Y={Y_tr.shape}")

    print("[Data] Building val windows (universal, merged SNRs)...")
    X_va, Y_va = build_dataset_from_records(
        records=val_records,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        noise_record=args.noise,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs,
        target_snr_list=args.snr_list,   # ✅
        win_len=args.win_len,
        hop_len=args.hop_len,
        normalize=args.normalize,
    )
    print(f"[Val] X={X_va.shape} | Y={Y_va.shape}")

    if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
        raise RuntimeError("No training/validation samples were generated. Check paths and WFDB files.")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # -------------------------
    # Model / optimizer
    # -------------------------
    model = UNet(in_channels=1, out_classes=1, dimensions=1, padding=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -------------------------
    # Outputs (ONLY 3 artifacts)
    # -------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best_model.pth"
    log_path = out_dir / "training_log.csv"
    config_path = out_dir / "config.json"

    # Save config.json
    run_cfg = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "device": str(device),
        "args": vars(args),
        "paths": {
            "mitdb_dir": str(mitdb_dir.resolve()),
            "nstdb_dir": str(nstdb_dir.resolve()),
            "splits": str(splits_path.resolve()),
            "out_dir": str(out_dir.resolve()),
        },
        "split_records": {
            "train": train_records,
            "val": val_records,
        },
        "data": {
            "noise_record": args.noise,
            "snr_levels": args.snr_list,
            "fs_target": args.fs,
            "start_sample": args.start_sample,
            "duration_sec": args.duration_sec,
            "win_len": args.win_len,
            "hop_len": args.hop_len,
            "normalize": args.normalize,
        },
    }
    config_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[Saved] {config_path}")

    # Prepare training log
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse", "is_best"])
    print(f"[Log] {log_path}")

    # -------------------------
    # Train loop (ONE run, ONE model)
    # -------------------------
    best_val = float("inf")
    print("[Train] start")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        val_loss = evaluate_mse(model, val_loader, device=device)

        print(f"[Epoch {epoch:03d}] train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

        is_best = False
        if val_loss < best_val:
            best_val = val_loss
            is_best = True
            torch.save(model.state_dict(), best_path)
            print(f"  -> [BEST] saved: {best_path} (val_mse={best_val:.6f})")

        # always append log (not only best epochs)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", int(is_best)])

    print(f"[Done] Best val_mse={best_val:.6f}")
    print(f"[Saved] best_model: {best_path}")
    print(f"[Saved] training_log: {log_path}")
    print(f"[Saved] config: {config_path}")


if __name__ == "__main__":
    main()