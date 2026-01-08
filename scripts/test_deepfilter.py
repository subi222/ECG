"""
DeepFilter Testing/Inference Script (adapted for WFDB data)

Loads MITDB test data directly from .dat/.hea files and mixes with BW noise
Based on train_deepfilter.py data loading pattern
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
import tensorflow as tf
import keras
from keras import backend as K
from keras import losses

# Common imports
from common import config
from common import io_wfdb
from common import noise as noise_mixer
from common import utils

from models.model_DeepFilter import deepfilter


# ============================================================================
# Custom Loss Functions (same as training)
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
# Data Loading Helpers (from train_deepfilter.py)
# ============================================================================

def load_splits(splits_path: Path) -> List[int]:
    """Load test splits from JSON"""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    
    if "test" not in splits:
        raise KeyError(f"'test' key not found in splits: {splits_path}")
    
    test_records = [int(x) for x in splits["test"]]
    return test_records


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


def build_test_dataset(
    records: List[int],
    mitdb_dir: Path,
    nstdb_dir: Path,
    noise_record: str,
    start_sample: int,
    duration_sec: int,
    fs_target: int,
    target_snr: float,
    signal_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build test dataset from MITDB records
    
    Uses 512-sample windows with 50% overlap
    
    Returns:
        X: (N, signal_size, 1) - noisy ECG windows
        y: (N, signal_size, 1) - baseline wander (target)
    """
    all_X, all_y = [], []
    
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
        
        # Mix using noise.py
        noisy, ref, actual_snr = noise_mixer.add_baseline_wander_snr(clean, bw, float(target_snr))
        
        # Extract baseline wander
        N = min(len(noisy), len(ref))
        bw_extracted = (noisy[:N] - ref[:N]).astype(np.float32)
        
        # Segment into 512 windows with 50% overlap
        hop_len = signal_size // 2
        n_windows = 0
        
        for start in range(0, N - signal_size + 1, hop_len):
            x_seg = noisy[start:start + signal_size]
            y_seg = bw_extracted[start:start + signal_size]
            
            # Keras format: (signal_size, 1)
            all_X.append(x_seg[:, None])
            all_y.append(y_seg[:, None])
            n_windows += 1
        
        print(f"[Rec {rec}] snr={target_snr}dB | windows={n_windows} | actual_snr={actual_snr:.2f}dB")
    
    if not all_X:
        return np.zeros((0, signal_size, 1), dtype=np.float32), \
               np.zeros((0, signal_size, 1), dtype=np.float32)
    
    X = np.stack(all_X, axis=0).astype(np.float32)
    y = np.stack(all_y, axis=0).astype(np.float32)
    return X, y


# ============================================================================
# Testing Pipeline
# ============================================================================

def test_deepfilter(args):
    """Test DeepFilter model using WFDB data"""
    
    print(f'Testing DeepFilter: {args.model_type}')
    print(f'Target SNR: {args.target_snr} dB')
    
    # Paths
    mitdb_dir = Path(args.mitdb_dir)
    nstdb_dir = Path(args.nstdb_dir)
    splits_path = Path(args.splits)
    
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path.resolve()}")
    
    # Load test splits
    test_records = load_splits(splits_path)
    print(f"[Split] test={len(test_records)} recs")
    print(f"        test={test_records}")
    
    # ==================
    # Build Test Dataset
    # ==================
    
    signal_size = 512  # Fixed window size
    
    print("[Data] Building test dataset...")
    X_test, y_test = build_test_dataset(
        records=test_records,
        mitdb_dir=mitdb_dir,
        nstdb_dir=nstdb_dir,
        noise_record=args.noise_record,
        start_sample=args.start_sample,
        duration_sec=args.duration_sec,
        fs_target=args.fs,
        target_snr=args.target_snr,
        signal_size=signal_size,
    )
    print(f"[Test] X={X_test.shape} | y={y_test.shape}")
    
    if X_test.shape[0] == 0:
        raise RuntimeError("No test samples. Check paths and WFDB files.")
    
    # ==================
    # Load Model
    # ==================
    
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
        loss=combined_ssd_mad_loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            losses.mean_squared_error,
            losses.mean_absolute_error,
            ssd_loss,
            mad_loss
        ]
    )
    
    # ==================
    # Load Weights
    # ==================
    
    model_filepath = args.model_path
    print(f'Loading weights from: {model_filepath}')
    model.load_weights(model_filepath)
    
    # ==================
    # Predict
    # ==================
    
    print('\nRunning inference...')
    y_pred = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    
    # ==================
    # Evaluate
    # ==================
    
    print('\n' + '='*60)
    print('Evaluating on test set...')
    test_results = model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)
    
    print('\nTest Results:')
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f'  {metric_name}: {value:.6f}')
    
    # ==================
    # Save Results (Optional)
    # ==================
    
    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f'\nSaving predictions to: {out_path}')
        np.savez(out_path, X_test=X_test, y_test=y_test, y_pred=y_pred)
        print('Done!')
    
    K.clear_session()
    
    print('\n' + '='*60)
    print('Testing completed!')
    print('='*60)
    
    return y_pred


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Test DeepFilter for ECG baseline removal')
    
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
                        help='LANL: basic, LANLD: dilated+dropout')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights (.weights.h5)')
    
    # Data
    parser.add_argument('--target_snr', type=float, default=5.0,
                        help='Target SNR for baseline wander (dB)')
    parser.add_argument('--fs', type=int, default=config.FS_DEFAULT)
    parser.add_argument('--start_sample', type=int, default=config.START_SAMPLE_DEFAULT)
    parser.add_argument('--duration_sec', type=int, default=config.DURATION_SEC_DEFAULT)
    
    # Inference
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    
    # Output
    parser.add_argument('--out_path', type=str, default=None,
                        help='Path to save predictions (.npz)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test_deepfilter(args)
