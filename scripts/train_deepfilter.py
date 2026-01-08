"""
DeepFilter Training Script (adapted for WFDB data)

Loads MITDB data directly from .dat/.hea files and mixes with BW noise
Based on train_unet.py data loading pattern
"""

import sys
from pathlib import Path
import argparse
import json
import csv
from datetime import datetime
from typing import List, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses

# Common imports
from common import config
from common import io_wfdb
from common import noise as noise_mixer
from common import repro
from common import utils

from models.model_DeepFilter import deepfilter


# ============================================================================
# Custom Loss Functions
# ============================================================================

def ssd_loss(y_true, y_pred):
    """Sum of Squared Distance"""
    return tf.reduce_sum(tf.square(y_pred - y_true), axis=-2)


def combined_ssd_mad_loss(y_true, y_pred):
    """
    Combined SSD + MAD Loss (DeepFilter Paper)
    
    Loss = SSD + 50 * MAD
    """
    return tf.reduce_max(tf.square(y_true - y_pred), axis=-2) * 50 + \
           tf.reduce_sum(tf.square(y_true - y_pred), axis=-2)


def mad_loss(y_true, y_pred):
    """Maximum Absolute Distance"""
    return tf.reduce_max(tf.square(y_pred - y_true), axis=-2)


# ============================================================================
# Data Loading Helpers (from train_unet.py)
# ============================================================================

def load_splits(splits_path: Path) -> Tuple[List[int], List[int]]:
    """Load train/val splits from JSON"""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    
    if "train" not in splits:
        raise KeyError(f"'train' key not found in splits: {splits_path}")
    
    # Allow either "val" or "valid"
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
    target_snrs: List[float] = [0, 5, 10, 15],  # Multiple SNRs
    signal_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    
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
            # Mix using noise.py
            noisy, ref, actual_snr = noise_mixer.add_baseline_wander_snr(clean, bw, float(target_snr))
            
            # DeepFilter: predict Clean ECG directly
            # X: noisy ECG, y: Clean ECG (MITDB)
            # This solves scaling issues naturally as Clean ECG has large amplitude
            N = min(len(noisy), len(ref))
            clean_ref = ref[:N].astype(np.float32)
            
            # Segment into signal_size windows with 50% overlap
            hop_len = signal_size // 2
            n_windows = 0
            
            for start in range(0, N - signal_size + 1, hop_len):
                x_seg = noisy[start:start + signal_size]
                y_seg = clean_ref[start:start + signal_size]
                
                # Keras format: (signal_size, 1)
                # Fixed Scaling roughly to [-1, 1] range (assuming max amplitude ~4mV)
                # This prevents gradient explosions without destroying DC information
                all_X.append(x_seg[:, None] / 4.0)
                all_y.append(y_seg[:, None] / 4.0)
                n_windows += 1
            
            # print(f"[Rec {rec}] snr={target_snr}dB | windows={n_windows}")

    if not all_X:
        return np.zeros((0, signal_size, 1), dtype=np.float32), \
               np.zeros((0, signal_size, 1), dtype=np.float32)
    
    X_arr = np.array(all_X, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)
    
    # Shuffle
    indices = np.arange(len(X_arr))
    np.random.shuffle(indices)
    
    return X_arr[indices], y_arr[indices]

# ============================================================================
# Training Pipeline
# ============================================================================

def train_deepfilter(args):
    """Train DeepFilter model using WFDB data"""
    
    print(f'Training DeepFilter: {args.model_type}')
    print(f'Target SNR: {args.target_snr} dB')
    
    # Reproducibility
    repro.set_seed(args.seed)
    
    # Paths
    mitdb_dir = Path(args.mitdb_dir)
    nstdb_dir = Path(args.nstdb_dir)
    splits_path = Path(args.splits)
    
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path.resolve()}")
    
    # Load splits
    train_records, val_records = load_splits(splits_path)
    print(f"[Split] train={len(train_records)} recs | val={len(val_records)} recs")
    print(f"        train={train_records}")
    print(f"        val  ={val_records}")
    
    # ==================
    # Build Datasets
    # ==================
    
    signal_size = 512  # Fixed window size (DeepFilter paper)
    
    print("[Data] Building train dataset (Augmented SNRs: 0, 5, 10, 15)...")
    X_train, y_train = build_dataset_from_records(
        records=train_records,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        noise_record=args.noise_record,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs,
        target_snrs=[0, 5, 10, 15],  # Augmentation
        signal_size=signal_size,
    )
    print(f"[Train] X={X_train.shape} | y={y_train.shape}")
    
    print("[Data] Building val dataset (Augmented SNRs: 0, 5, 10, 15)...")
    X_val, y_val = build_dataset_from_records(
        records=val_records,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        noise_record=args.noise_record,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs,
        target_snrs=[0, 5, 10, 15],  # Augmentation
        signal_size=signal_size,
    )
    print(f"[Val] X={X_val.shape} | y={y_val.shape}")
    
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise RuntimeError("No training/validation samples. Check paths and WFDB files.")
    
    # ==================
    # Load Model
    # ==================
    
    # Use fixed signal_size=512 (DeepFilter paper)
    print(f"\n[Model] Signal size: {signal_size} samples (fixed)")
    
    if args.model_type == 'LANL':
        model = deepfilter.deep_filter_I_LANL(signal_size=signal_size)
        model_label = 'DeepFilter_LANL'
    elif args.model_type == 'LANLD':
        model = deepfilter.deep_filter_model_I_LANL_dilated(signal_size=signal_size)
        model_label = 'DeepFilter_LANLD'
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f'\nModel: {model_label}\n')
    model.summary()
    
    # ==================
    # Compile Model
    # ==================
    
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        metrics=['mae', 'mse']
    )
    
    # ==================
    # Callbacks
    # ==================
    
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_filepath = str(out_dir / f'{model_label}_best.weights.h5')  # Keras 3.x format
    checkpoint = ModelCheckpoint(
        model_filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        min_delta=0.05,
        mode='min',
        patience=5,
        min_lr=args.min_lr,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        mode='min',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )
    
    tb_log_dir = str(out_dir / 'logs' / model_label)
    tboard = TensorBoard(
        log_dir=tb_log_dir,
        histogram_freq=0,
        write_graph=False
    )
    
    print(f'\nOutput directory: {out_dir}')
    print(f'Best model: {model_filepath}')
    print(f'TensorBoard: tensorboard --logdir={tb_log_dir}')
    
    # ==================
    # Train
    # ==================
    
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=[early_stop, reduce_lr, checkpoint, tboard]
    )
    
    # ==================
    # Save History
    # ==================
    
    history_file = out_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    print(f'\nTraining history saved: {history_file}')
    
    K.clear_session()
    
    print('\n' + '='*60)
    print('Training completed!')
    print(f'Best model: {model_filepath}')
    print('\nTo evaluate on test set:')
    print(f'  python scripts/test_deepfilter.py --model_path {model_filepath} --model_type {args.model_type}')
    print('='*60)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train DeepFilter for ECG baseline removal')
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_SPLITS = PROJECT_ROOT / "common" / "splits.json"
    
    # Paths
    parser.add_argument('--mitdb_dir', type=str, default=str(config.MITDB_DIR_DEFAULT))
    parser.add_argument('--nstdb_dir', type=str, default=str(config.NSTDB_DIR_DEFAULT))
    parser.add_argument('--splits', type=str, default=str(DEFAULT_SPLITS))
    parser.add_argument('--noise_record', type=str, default='bw', help='Noise record (bw/em/ma)')
    
    # Model
    parser.add_argument('--model_type', type=str, default='LANLD',
                        choices=['LANL', 'LANLD'],
                        help='LANL: basic, LANLD: dilated+dropout (recommended)')
    
    # Data
    parser.add_argument('--target_snr', type=float, default=5.0,
                        help='Target SNR for baseline wander (dB)')
    parser.add_argument('--fs', type=int, default=config.FS_DEFAULT)
    parser.add_argument('--start_sample', type=int, default=config.START_SAMPLE_DEFAULT)
    parser.add_argument('--duration_sec', type=int, default=config.DURATION_SEC_DEFAULT)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-10,
                        help='Minimum learning rate')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--out_dir', type=str, default='outputs/DeepFilter',
                        help='Output directory for models and logs')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_deepfilter(args)
