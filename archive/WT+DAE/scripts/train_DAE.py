"""
train_DAE.py

Paper-guided baseline training for Improved DAE (Xiong et al., 2016),
used as a comparative method under a unified benchmark pipeline.

Core idea (paper-guided):
- Input formation: noisy ECG = clean MITDB segment + NSTDB baseline wander mixed at target SNR
- Wavelet preprocessing: db6, up to 8-level, scale-adaptive soft-thresholding (inspired by Eq.(1)-(2))
- Windowing: δ=50 -> window length 101 (sliding windows, reflect padding)
- Network: Fully-connected 101 -> 50 -> 50 -> 101 with sigmoid activations
- Objective: Bernoulli distance / cross-entropy -> BCELoss on [0,1]-normalized windows
- Training: greedy layer-wise pretraining (AE1: 101-50-101, AE2: 50-50-50) + end-to-end fine-tuning

Important implementation notes (baseline comparison focus; not exact reproduction):
- Normalization: per-window min-max based on the INPUT window; the SAME transform is applied to the target window
  to keep paired training stable under BCELoss.
- Sampling rate: signals may be resampled to a project-standard rate (e.g., 250 Hz) for fair comparison across methods.
- Wavelet reconstruction: this script reconstructs using waverec(cA + all thresholded detail bands);
  the paper’s optional detail-band selection set D is not explicitly implemented here.
- Overlap fusion: this script trains on windows; waveform-level overlap-averaging fusion (if used) should be handled
  in the evaluation/inference code.

This script is intended for baseline training within this repository, not for claiming exact replication of the paper's results.
"""


from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pywt
from typing import List
import csv
from datetime import datetime


from models.model_DAE.model_DAE import ImprovedDAE, SingleLayerAE
from common.config import (
    MITDB_DIR_DEFAULT, NSTDB_DIR_DEFAULT, OUTPUT_DIR_DEFAULT,
    START_SAMPLE_DEFAULT, DURATION_SEC_DEFAULT, FS_DEFAULT,
    NSTDB_RECORD_DEFAULT, SNR_LEVELS_DEFAULT,
)
from pathlib import Path
from common.dataset_split import list_mitdb_records
from common.io_wfdb import load_mitdb_wfdb, load_nstdb_noise
from common.noise import add_baseline_wander_snr
from common.repro import set_seed
from common.utils import _resample_to_target


OUTPUT_DIR = OUTPUT_DIR_DEFAULT / "Improved_DAE"

# ===========================
# Configuration
# ===========================
@dataclass
class DAEConfig:
    paper_id: str = "Xiong2016"
    model_name: str = "ImprovedDAE_Baseline"
    window_len: int = 101
    hidden1: int = 50
    hidden2: int = 50
    wavelet: str = "db6"
    level: int = 8

    # Training Params
    fs: int = FS_DEFAULT
    epochs_pretrain: int = 20
    epochs_finetune: int = 50
    batch_size: int = 128
    lr_pre: float = 0.001
    lr_fine: float = 0.0001

    training_records: List[int] = None
    snr_levels: List[float] = None
    seed: int = 42
    timestamp: str = ""

    def to_json(self) -> Dict:
        d = asdict(self)
        if d["training_records"] is None:
            d["training_records"] = []
        return d


# ===========================
# Preprocessing (WT + Windowing)
# ===========================
def _soft_threshold(d: np.ndarray, T: float) -> np.ndarray:
    return np.sign(d) * np.maximum(np.abs(d) - T, 0.0)


def wavelet_denoise_db6_level8_soft(x: np.ndarray, level: int = 8) -> np.ndarray:
    """
        Wavelet denoise used as a paper-guided preprocessing step.

        - Wavelet: db6
        - Levels: up to 8 (limited by signal length)
        - Threshold: scale-adaptive soft-thresholding using MAD-based sigma estimate
          (inspired by Xiong et al., 2016 Eq.(1)-(2)).

        Note:
        - We reconstruct via waverec(cA + all thresholded details).
          The paper also discusses selecting a subset of detail bands (set D);
          that selection is not explicitly implemented in this baseline training script.
        """

    # Scale-adaptive soft thresholding as per paper Eq.1
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
        # Sigma estimation using MAD
        sigma_j = np.median(np.abs(d)) / 0.6745 if d.size > 0 else 0.0
        # Threshold formula
        Tj = sigma_j * math.sqrt(2.0 * math.log(n + 1e-12)) / math.exp(max(j - 1, 0))
        new_cDs.append(_soft_threshold(d, Tj))

    return pywt.waverec([cA] + new_cDs, wavelet)[:n]


def extract_windows(sig: np.ndarray, radius: int = 50) -> np.ndarray:
    # Sliding window with reflect padding
    sig = np.asarray(sig, dtype=np.float64)
    wlen = 2 * radius + 1
    pad = radius
    padded = np.pad(sig, (pad, pad), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, wlen)
    return windows.copy()


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x_windows: np.ndarray, t_windows: np.ndarray):
        super().__init__()
        # Normalization: Map each input window to [0,1]
        # And apply SAME transformation to target window (paper implies mapping V to [0,1])

        self.x_wins = x_windows
        self.t_wins = t_windows

        # Calculate min/max per window for normalization
        self.min_vals = self.x_wins.min(axis=1, keepdims=True)
        self.max_vals = self.x_wins.max(axis=1, keepdims=True)
        self.denom = (self.max_vals - self.min_vals) + 1e-8

        self.x_norm = (self.x_wins - self.min_vals) / self.denom
        self.t_norm = (self.t_wins - self.min_vals) / self.denom

        # Clip to [0,1] for BCELoss stability
        self.x_norm = np.clip(self.x_norm, 0.0, 1.0)
        self.t_norm = np.clip(self.t_norm, 0.0, 1.0)

    def __len__(self):
        return len(self.x_norm)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x_norm[idx]).float(),
                torch.from_numpy(self.t_norm[idx]).float())


# ===========================
# Training Routines
# ===========================
def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0
    cnt = 0
    for x, t in loader:
        x, t = x.to(device), t.to(device)
        opt.zero_grad()
        y = model(x)
        loss = crit(y, t)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        cnt += x.size(0)
    return total_loss / max(cnt, 1)

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    cnt = 0
    for x, t in loader:
        x, t = x.to(device), t.to(device)
        y = model(x)
        loss = crit(y, t)
        total_loss += loss.item() * x.size(0)
        cnt += x.size(0)
    return total_loss / max(cnt, 1)


def main():
    parser = argparse.ArgumentParser()
    # 프로젝트 루트 = scripts/의 상위 폴더
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    parser.add_argument("--epochs_pre", type=int, default=10, help="Pretraining epochs per layer")
    parser.add_argument("--epochs_fine", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split_path", type=str, default="common/splits.json", help="Path to shared record-wise split JSON")

    args = parser.parse_args()

    # ---- split path 처리 (여기가 정답 위치) ----
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    split_path = Path(args.split_path)
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path

    set_seed(42)
    device = torch.device(args.device)

    # 1. Config & Paths
    cfg = DAEConfig(epochs_pretrain=args.epochs_pre, epochs_finetune=args.epochs_fine)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # -------------------------
    # Prepare output artifacts
    # -------------------------
    best_path = OUTPUT_DIR / "best_model.pth"
    log_path = OUTPUT_DIR / "training_log.csv"
    config_path = OUTPUT_DIR / "config.json"

    # training_log.csv 헤더
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",  # pretrain_ae1 / pretrain_ae2 / finetune
            "epoch",  # 1..N
            "train_loss",
            "val_loss",
            "is_best"
        ])
    print(f"[Log] {log_path}")

    # 2. Data Preparation
    print(">>> Loading Data...")

    # [수정 1] 데이터를 담을 리스트를 Train/Val 별도로 초기화
    train_x_list, train_t_list = [], []
    val_x_list, val_t_list = [], []

    records = list_mitdb_records(MITDB_DIR_DEFAULT)
    if not records:
        print(f"ERROR: No records found in {MITDB_DIR_DEFAULT}")
        return

    # --- Load shared split (scripts/val/test) ---
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path.resolve()}")

    with open(split_path, "r") as f:
        splits = json.load(f)

    # train 키가 없고 scripts만 있는 경우도 있으니 호환 처리
    train_key = "train" if "train" in splits else "scripts"

    train_records = set(splits[train_key])
    val_records = set(splits["val"])
    test_records = set(splits["test"])

    # Optional sanity checks
    overlap_tv = train_records & val_records
    overlap_tt = train_records & test_records
    overlap_vt = val_records & test_records
    missing = (train_records | val_records | test_records) - set(records)
    if missing:
        raise ValueError(f"Split contains records not found in MITDB: {sorted(missing)}")

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError("Split sets overlap! Check splits.json")

    print(f"[Split] Using {split_path} | "
          f"Train={len(train_records)} Val={len(val_records)} Test={len(test_records)}")

    snrs = cfg.snr_levels if cfg.snr_levels else SNR_LEVELS_DEFAULT
    mitdb_dir = MITDB_DIR_DEFAULT
    nstdb_dir = NSTDB_DIR_DEFAULT

    # (추천) scripts/val 레코드만 처리해서 시간 절약 + test 레코드 미사용 보장
    records_to_use = sorted(list(train_records | val_records))

    # 전체 레코드를 돌면서 Train/Val 리스트에 분배
    for rec in records_to_use:
        try:
            clean, fs_mit = load_mitdb_wfdb(mitdb_dir, rec, START_SAMPLE_DEFAULT, DURATION_SEC_DEFAULT)
            noise, _ = load_nstdb_noise(nstdb_dir, NSTDB_RECORD_DEFAULT,
                                        START_SAMPLE_DEFAULT, DURATION_SEC_DEFAULT,
                                        FS_DEFAULT)

        except Exception as e:
            print(f"Skipping Record {rec}: {e}")
            continue

        for snr in snrs:
            noisy, ref, _ = add_baseline_wander_snr(clean, noise, snr)

            # 250Hz로 리샘플링 (사용자 의도 반영)
            noisy = _resample_to_target(noisy, fs_raw=fs_mit, fs_target=FS_DEFAULT)
            ref = _resample_to_target(ref, fs_raw=fs_mit, fs_target=FS_DEFAULT)

            # WT Denoising
            wt_denoised = wavelet_denoise_db6_level8_soft(noisy, level=cfg.level)

            # Windowing (Radius 50 -> Len 101)
            x_w = extract_windows(wt_denoised, radius=50)  # Input: WT Output
            t_w = extract_windows(ref, radius=50)  # Target: Clean Reference

            # [수정 2] 레코드 ID에 따라 알맞은 리스트에 추가
            if rec in train_records:
                train_x_list.append(x_w)
                train_t_list.append(t_w)
            elif rec in val_records:
                val_x_list.append(x_w)
                val_t_list.append(t_w)

    if not train_x_list:
        print("Error: No training data loaded!")
        return

    # Numpy Concatenation
    X_train = np.concatenate(train_x_list, axis=0)
    T_train = np.concatenate(train_t_list, axis=0)
    X_val = np.concatenate(val_x_list, axis=0)
    T_val = np.concatenate(val_t_list, axis=0)

    print(f"[Dataset] Train Windows: {X_train.shape} | Val Windows: {X_val.shape}")

    # Dataset & DataLoader 생성
    train_ds = WindowDataset(X_train, T_train)
    val_ds = WindowDataset(X_val, T_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 3. Layer-wise Pretraining
    print("\n=== Stage 1: Pretraining Layer 1 (101 -> 50 -> 101) ===")
    ae1 = SingleLayerAE(101, 50).to(device)
    crit = nn.BCELoss()
    opt1 = optim.Adam(ae1.parameters(), lr=cfg.lr_pre)

    for ep in range(cfg.epochs_pretrain):
        # [수정 3] loader -> train_loader
        train_loss = 0
        cnt = 0
        ae1.train()
        for x, _ in train_loader:
            x = x.to(device)
            opt1.zero_grad()
            recon, _ = ae1(x)
            loss = crit(recon, x)  # Autoencoder: Input 복원 학습
            loss.backward()
            opt1.step()
            train_loss += loss.item() * x.size(0)
            cnt += x.size(0)
        print(f"AE1 Epoch {ep + 1}/{cfg.epochs_pretrain} Loss: {train_loss / cnt:.6f}")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pretrain_ae1", ep + 1, f"{(train_loss / cnt):.6f}", "", 0])

    # Generate Hidden Features for Layer 2
    print("\nTargeting Hidden Features for Layer 2...")
    hidden_features = []
    ae1.eval()
    with torch.no_grad():
        for x, _ in train_loader:  # [수정 4] loader -> train_loader
            x = x.to(device)
            _, h = ae1(x)
            hidden_features.append(h.cpu())

    H1 = torch.cat(hidden_features, dim=0)
    ds2 = torch.utils.data.TensorDataset(H1, H1)
    loader2 = torch.utils.data.DataLoader(ds2, batch_size=cfg.batch_size, shuffle=True)

    print("\n=== Stage 2: Pretraining Layer 2 (50 -> 50 -> 50) ===")
    ae2 = SingleLayerAE(50, 50).to(device)
    opt2 = optim.Adam(ae2.parameters(), lr=cfg.lr_pre)

    for ep in range(cfg.epochs_pretrain):
        train_loss = 0
        cnt = 0
        ae2.train()
        for h, _ in loader2:
            h = h.to(device)
            opt2.zero_grad()
            recon, _ = ae2(h)
            loss = crit(recon, h)
            loss.backward()
            opt2.step()
            train_loss += loss.item() * h.size(0)
            cnt += h.size(0)
        print(f"AE2 Epoch {ep + 1}/{cfg.epochs_pretrain} Loss: {train_loss / cnt:.6f}")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pretrain_ae2", ep + 1, f"{(train_loss / cnt):.6f}", "", 0])

    # 4. Fine-tuning (End-to-End)
    print("\n=== Stage 3: Fine-tuning Full Network (101 -> 50 -> 50 -> 101) ===")
    final_model = ImprovedDAE(window_len=101, hidden1=50, hidden2=50).to(device)

    # 가중치 복사 (Pretrained Weights Transfer)
    final_model.net[0].weight.data = ae1.encoder.weight.data.clone()
    final_model.net[0].bias.data = ae1.encoder.bias.data.clone()

    final_model.net[2].weight.data = ae2.encoder.weight.data.clone()
    final_model.net[2].bias.data = ae2.encoder.bias.data.clone()

    final_model.net[4].weight.data = ae1.decoder.weight.data.clone()
    final_model.net[4].bias.data = ae1.decoder.bias.data.clone()

    opt_fine = optim.Adam(final_model.parameters(), lr=cfg.lr_fine)

    best_val = float("inf")

    for ep in range(cfg.epochs_finetune):
        train_loss = train_epoch(final_model, train_loader, opt_fine, crit, device)
        val_loss = eval_epoch(final_model, val_loader, crit, device)

        print(f"FineTune Epoch {ep + 1}/{cfg.epochs_finetune} "
              f"Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

        is_best = False
        if val_loss < best_val:
            best_val = val_loss
            is_best = True
            torch.save(final_model.state_dict(), best_path)
            print(f"  ★ Saved Best Model: {best_path} (Val Loss: {best_val:.6f})")

        # ✅ 매 epoch 로그 저장 (if 밖!)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["finetune", ep + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", int(is_best)])

    # Save training records info for reproducibility
    cfg.training_records = sorted(list(train_records))
    # -------------------------
    # Save config.json (unified format)
    # -------------------------
    run_cfg = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "device": str(device),
        "split_path": str(split_path.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "data": {
            "mitdb_dir": str(MITDB_DIR_DEFAULT),
            "nstdb_dir": str(NSTDB_DIR_DEFAULT),
            "noise_record": str(NSTDB_RECORD_DEFAULT),
            "snr_levels": snrs,
            "fs_target": FS_DEFAULT,
            "start_sample": START_SAMPLE_DEFAULT,
            "duration_sec": DURATION_SEC_DEFAULT,
        },
        "split_records": {
            "train": sorted(list(train_records)),
            "val": sorted(list(val_records)),
            "test": sorted(list(test_records)),
        },
        "hyperparams": {
            "epochs_pretrain": cfg.epochs_pretrain,
            "epochs_finetune": cfg.epochs_finetune,
            "batch_size": cfg.batch_size,
            "lr_pre": cfg.lr_pre,
            "lr_fine": cfg.lr_fine,
            "window_len": cfg.window_len,
            "hidden1": cfg.hidden1,
            "hidden2": cfg.hidden2,
            "wavelet": cfg.wavelet,
            "level": cfg.level,
            "seed": cfg.seed,
        },
    }

    config_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[Saved] best_model: {best_path}")
    print(f"[Saved] training_log: {log_path}")
    print(f"[Saved] config: {config_path}")
    print(f"[Done] Outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()