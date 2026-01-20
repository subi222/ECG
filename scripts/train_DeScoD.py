"""
DeScoD Training Script (adapted for WFDB data)

Loads MITDB data directly from .dat/.hea files and mixes with BW noise
Based on train_deepfilter.py data loading pattern

References:
- small_DeScoD.py: ConditionalModel (neural network)
- main_DeScoD.py: DDPM (diffusion process)
"""

import sys
from pathlib import Path
import argparse
import json
import yaml
from datetime import datetime
from typing import List, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Common imports
from common import config
from common import io_wfdb
from common import noise as noise_mixer
from common import repro
from common import utils

# DeScoD model imports
from models.model_DeScoD.small_DeScoD import ConditionalModel
from models.model_DeScoD.main_DeScoD import DDPM


# ============================================================================
# Data Loading Helpers (from train_deepfilter.py)
# ============================================================================

def load_splits(splits_path: Path) -> Tuple[List[int], List[int], List[int]]:
    """Load train/val/test splits from JSON"""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    
    train_records = [int(x) for x in splits["train"]]
    val_records = [int(x) for x in splits.get("val", splits.get("valid", []))]
    test_records = [int(x) for x in splits.get("test", [])]
    
    return train_records, val_records, test_records


def load_and_resample_mitdb_segment(
    mitdb_dir: Path,
    record: int,
    start_sample: int,
    duration_sec: int,
    fs_target: int,
) -> np.ndarray:
    """Load and resample MITDB ECG segment"""
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
    """Load and resample NSTDB noise segment"""
    nz_raw, fs_read = io_wfdb.load_nstdb_noise(
        nstdb_dir=nstdb_dir,
        record=noise_record,
        start_sample=start_sample,
        duration_sec=duration_sec,
        fs=fs_target,
    )
    if nz_raw.size == 0:
        return nz_raw.astype(np.float32)
    
    nz = utils._resample_to_target(nz_raw, fs_raw=float(fs_read), fs_target=float(fs_target))
    return nz.astype(np.float32, copy=False)


def build_dataset_from_records(
    records: List[int],
    mitdb_dir: Path,
    nstdb_dir: Path,
    noise_record: str,
    start_sample: int,
    duration_sec: int,
    fs_target: float,
    target_snrs: List[float] = [0, 5, 10, 15],
    signal_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build dataset from WFDB records.
    
    Returns:
        X: noisy ECG segments (N, 1, signal_size) - PyTorch format (channel first)
        y: clean ECG segments (N, 1, signal_size) - PyTorch format (channel first)
    """
    print(f"Building dataset from {len(records)} records with SNRs={target_snrs}...")
    
    all_X = []
    all_y = []
    
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
        
        # Loop over multiple SNRs for Data Augmentation
        for target_snr in target_snrs:
            # Mix using noise.py add_baseline_wander_snr
            noisy, ref, actual_snr = noise_mixer.add_baseline_wander_snr(
                clean, bw, float(target_snr)
            )
            
            N = min(len(noisy), len(ref))
            clean_ref = ref[:N].astype(np.float32)
            noisy_sig = noisy[:N].astype(np.float32)
            
            # Segment into signal_size windows with 50% overlap
            hop_len = signal_size // 2
            
            for start in range(0, N - signal_size + 1, hop_len):
                x_seg = noisy_sig[start:start + signal_size]
                y_seg = clean_ref[start:start + signal_size]
                
                # PyTorch format: (1, signal_size) - channel first
                # Scale to roughly [-1, 1] range (assuming max amplitude ~4mV)
                all_X.append(x_seg[None, :] / 4.0)
                all_y.append(y_seg[None, :] / 4.0)
    
    if not all_X:
        return np.zeros((0, 1, signal_size), dtype=np.float32), \
               np.zeros((0, 1, signal_size), dtype=np.float32)
    
    X_arr = np.array(all_X, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)
    
    # Shuffle
    indices = np.arange(len(X_arr))
    np.random.shuffle(indices)
    
    print(f"  -> {len(X_arr)} segments created")
    return X_arr[indices], y_arr[indices]


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, ema=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        clean, noisy = batch  # y=clean, x=noisy (condition)
        clean = clean.to(device)
        noisy = noisy.to(device)
        
        optimizer.zero_grad()
        
        # DDPM forward: (clean, noisy_condition)
        loss = model(clean, noisy)
        loss.backward()
        
        # Add gradient clipping to prevent divergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if ema is not None:
            ema.update(model)
            
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            clean, noisy = batch
            clean = clean.to(device)
            noisy = noisy.to(device)
            
            loss = model(clean, noisy)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_descod(args):
    """Train DeScoD model using WFDB data"""
    
    print(f"\n{'='*60}")
    print(f"DeScoD Training (DDPM-based ECG Denoising)")
    print(f"{'='*60}\n")
    
    # Reproducibility
    repro.set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Paths
    mitdb_dir = Path(args.mitdb_dir)
    nstdb_dir = Path(args.nstdb_dir)
    splits_path = Path(args.splits)
    
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path.resolve()}")
    
    # Load config
    config_path = ROOT / "models" / "model_DeScoD" / args.config
    if not config_path.exists():
        # Try absolute or relative path
        config_path = Path(args.config)
    
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"[Config] Loaded from: {config_path}")
    else:
        # Default config
        
        cfg = {
            "train": {
                "batch_size": 64,
                "epochs": 100,
                "lr": 1e-4,
                "feats": 64,
            },
            "diffusion": {
                "num_steps": 1000,
                "schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
            }
        }
        print(f"[Config] Using default config")
        
    
    print(f"[Config] {cfg}")
    
    # Load splits
    train_records, val_records, test_records = load_splits(splits_path)
    print(f"\n[Split] train={len(train_records)} | val={len(val_records)} | test={len(test_records)}")

    # ==================
    # Build Datasets
    # ==================
    
    signal_size = 512
    
    print("\n[Data] Building train dataset...")
    X_train, y_train = build_dataset_from_records(
        records=train_records,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        noise_record=args.noise_record,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs,
        target_snrs=[0, 5, 10, 15],
        signal_size=signal_size,
    )
    
    print("[Data] Building val dataset...")
    X_val, y_val = build_dataset_from_records(
        records=val_records,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        noise_record=args.noise_record,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs,
        target_snrs=[0, 5, 10, 15],
        signal_size=signal_size,
    )
    
    print(f"\n[Dataset] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[Dataset] Val:   X={X_val.shape}, y={y_val.shape}")
    
    if X_train.shape[0] == 0:
        raise RuntimeError("No training samples. Check paths and WFDB files.")
    
    # Create DataLoaders
    # TensorDataset: (clean, noisy) - clean is target, noisy is condition
    train_dataset = TensorDataset(
        torch.FloatTensor(y_train),  # clean (target)
        torch.FloatTensor(X_train),  # noisy (condition)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(y_val),
        torch.FloatTensor(X_val),
    )
    
    batch_size = args.batch_size if args.batch_size else cfg["train"].get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ==================
    # Build Model
    # ==================
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    
    feats = cfg["train"].get("feats", 64)
    base_model = ConditionalModel(feats=feats).to(device)
    model = DDPM(base_model, cfg, device).to(device)
    
    print(f"[Model] ConditionalModel(feats={feats})")
    print(f"[Model] DDPM(num_steps={cfg['diffusion']['num_steps']})")
    
    # ==================
    # Training Setup
    # ==================
    
    lr = cfg["train"].get("lr", 1e-4)
    epochs = args.epochs if args.epochs else cfg["train"].get("epochs", 500)
    batch_size = args.batch_size if args.batch_size else cfg["train"].get("batch_size", 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # More relaxed scheduler for 100 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)
    
    ema_helper = None
    # EMA is disabled in final official run
    
    # Output directory
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    best_model_path = out_dir / "DeScoD_best.pth"
    
    print(f"\n[Training] epochs={epochs}, lr={lr}, batch_size={batch_size}")
    print(f"[Output] {out_dir}")
    
    # ==================
    # Training Loop
    # ==================
    
    history = {"train_loss": [], "val_loss": []}
    
    # Early stopping setup
    patience = args.patience if args.patience else cfg["train"].get("patience", 10)
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, ema=ema_helper)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Save best model and handle early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if ema_helper is not None:
                # Apply EMA weights temporarily for saving
                original_weights = {k: v.clone() for k, v in model.state_dict().items()}
                ema_helper.ema(model)
                torch.save(model.state_dict(), out_dir / "DeScoD_best.pth")
                # Restore original weights for continued training
                model.load_state_dict(original_weights)
                print(f"  -> Best model saved (EMA weights)")
            else:
                torch.save(model.state_dict(), out_dir / "DeScoD_best.pth")
                print(f"  -> Best model saved")
            marker = " *"
        else:
            patience_counter += 1
            marker = ""
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | patience={patience_counter}/{patience}{marker}")
            
        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
            break
    
    # Save final model
    final_model_path = out_dir / "DeScoD_final.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Save history
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # Save config alongside trained model
    config_save_path = out_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"  Best model: {best_model_path}")
    print(f"  Final model: {final_model_path}")
    print(f"  History: {history_path}")
    print(f"  Config: {config_save_path}")
    print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train DeScoD for ECG denoising')
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_SPLITS = PROJECT_ROOT / "common" / "splits.json"
    
    # Paths
    parser.add_argument('--mitdb_dir', type=str, default=str(config.MITDB_DIR_DEFAULT))
    parser.add_argument('--nstdb_dir', type=str, default=str(config.NSTDB_DIR_DEFAULT))
    parser.add_argument('--splits', type=str, default=str(DEFAULT_SPLITS))
    parser.add_argument('--noise_record', type=str, default='bw', help='Noise record (bw/em/ma)')
    parser.add_argument('--config', type=str, default='base.yaml', help='Config file')
    
    # Data
    parser.add_argument('--fs', type=int, default=config.FS_DEFAULT)
    parser.add_argument('--start_sample', type=int, default=config.START_SAMPLE_DEFAULT)
    parser.add_argument('--duration_sec', type=int, default=config.DURATION_SEC_DEFAULT)
    
    # Training
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=0, help='Override batch size (0 to use config)')
    parser.add_argument('--patience', type=int, default=0, help='Override early stopping patience (0 to use config)')
    
    # Output
    parser.add_argument('--out_dir', type=str, default='outputs/train_DeScoD',
                        help='Output directory for models and logs')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_descod(args)