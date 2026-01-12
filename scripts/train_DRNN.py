"""
DRNN Training Script (adapted for WFDB data)

Based on train_deepfilter.py / train_FCN_DAE.py pipeline:
- Loads MITDB (clean) and NSTDB (baseline wander noise)
- Mixes at multiple SNRs (0, 5, 10, 15 dB)
- Target: Clean ECG (not baseline)
- Uses splits.json for train/val separation

Reference: Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
"""

import sys
from pathlib import Path
import argparse
import json
from typing import List, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses

# Common imports
from common import config
from common import io_wfdb
from common import noise as noise_mixer
from common import utils

# ============================================================================
# Custom Loss Functions (for metrics)
# ============================================================================

def ssd_loss(y_true, y_pred):
    """Sum of Squared Distance"""
    return K.sum(K.square(y_pred - y_true), axis=-2)

def mad_loss(y_true, y_pred):
    """Max Absolute Deviation"""
    return K.max(K.square(y_pred - y_true), axis=-2)

# ============================================================================
# Data Loading Helpers (same as train_FCN_DAE.py)
# ============================================================================

def load_splits(splits_path: Path) -> Tuple[List[int], List[int]]:
    """Load train/val splits from JSON"""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    
    if "train" not in splits:
        raise KeyError(f"'train' key not found in splits: {splits_path}")
    
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
    target_snrs: List[float] = [0, 5, 10, 15],
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
            # Mix using noise.py's add_baseline_wander_snr
            noisy, ref, actual_snr = noise_mixer.add_baseline_wander_snr(clean, bw, float(target_snr))
            
            # DRNN: predict Clean ECG directly (same as DeepFilter)
            N = min(len(noisy), len(ref))
            clean_ref = ref[:N].astype(np.float32)
            
            # Segment into signal_size windows with 50% overlap
            hop_len = signal_size // 2
            n_windows = 0
            
            for start in range(0, N - signal_size + 1, hop_len):
                x_seg = noisy[start:start + signal_size]
                y_seg = clean_ref[start:start + signal_size]
                
                # Keras format: (signal_size, 1)
                # Fixed Scaling to [-1, 1] range (assuming max amplitude ~4mV)
                all_X.append(x_seg[:, None] / 4.0)
                all_y.append(y_seg[:, None] / 4.0)
                n_windows += 1

    if not all_X:
        return np.zeros((0, signal_size, 1), dtype=np.float32), \
               np.zeros((0, signal_size, 1), dtype=np.float32)
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    
    print(f"Dataset built: X={X.shape}, y={y.shape}")
    return X, y


# ============================================================================
# DRNN Model Definition
# ============================================================================

def build_DRNN(signal_size=512, lstm_units=64, dense_units=64):
    """
    DRNN Model (Antczak, 2018)
    
    Architecture:
    - LSTM: 64 units, return_sequences=True
    - Dense: 64 units, ReLU
    - Dense: 64 units, ReLU
    - Output: 1 unit, Linear
    """
    model = Sequential()
    
    # LSTM Layer: (Batch, Time, 1) -> (Batch, Time, lstm_units)
    model.add(LSTM(lstm_units, input_shape=(signal_size, 1), return_sequences=True))
    
    # Dense Layers: applied to each time step independently
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(dense_units, activation='relu'))
    
    # Output Layer: (Batch, Time, 1)
    model.add(Dense(1, activation='linear'))
    
    return model


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=str, default=str(ROOT / 'common' / 'splits.json'))
    parser.add_argument('--mitdb_dir', type=str, default=str(ROOT / 'data' / 'MITDB_data'))
    parser.add_argument('--nstdb_dir', type=str, default=str(ROOT / 'data' / 'noise_data'))
    parser.add_argument('--noise_rec', type=str, default='bw')
    parser.add_argument('--start_sample', type=int, default=0)
    parser.add_argument('--duration_sec', type=int, default=1800)
    parser.add_argument('--fs_target', type=int, default=250)
    parser.add_argument('--signal_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--out_dir', type=str, default=str(ROOT / 'outputs' / 'train_DRNN'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)

    # Load splits
    train_records, val_records = load_splits(Path(args.splits))
    print(f"Train records: {train_records}")
    print(f"Val records: {val_records}")

    # Build datasets
    signal_size = args.signal_size
    
    print("\n=== Building Training Dataset ===")
    X_train, y_train = build_dataset_from_records(
        records=train_records,
        mitdb_dir=Path(args.mitdb_dir),
        nstdb_dir=Path(args.nstdb_dir),
        noise_record=args.noise_rec,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs_target,
        target_snrs=[0, 5, 10, 15],
        signal_size=signal_size,
    )
    
    print("\n=== Building Validation Dataset ===")
    X_val, y_val = build_dataset_from_records(
        records=val_records,
        mitdb_dir=Path(args.mitdb_dir),
        nstdb_dir=Path(args.nstdb_dir),
        noise_record=args.noise_rec,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs_target,
        target_snrs=[0, 5, 10, 15],
        signal_size=signal_size,
    )
    
    print(f"\n[Train] X={X_train.shape} | y={y_train.shape}")
    print(f"[Val] X={X_val.shape} | y={y_val.shape}")
    
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise RuntimeError("No training/validation samples. Check paths and WFDB files.")
    
    # Build Model
    print(f"\n[Model] DRNN with signal_size: {signal_size} samples")
    model = build_DRNN(signal_size=signal_size)
    print('\nModel: DRNN\n')
    model.summary()
    
    # Compile Model (Use MSE for stability, same as DeepFilter)
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_filepath = str(out_dir / 'DRNN_best.weights.h5')
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
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )
    
    tb_log_dir = str(out_dir / 'logs' / 'DRNN')
    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0)
    
    # Training
    print(f"\nOutput directory: {out_dir}")
    print(f"Best model: {model_filepath}")
    print(f"TensorBoard: tensorboard --logdir={tb_log_dir}")
    
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=[checkpoint, reduce_lr, early_stop, tboard]
    )
    
    # Save history
    history_path = out_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2, default=float)
    
    print(f"\nTraining history saved: {history_path}")
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best model: {model_filepath}")
    print("="*60)


if __name__ == '__main__':
    main()