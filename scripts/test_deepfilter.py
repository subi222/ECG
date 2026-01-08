"""
DeepFilter Testing/Inference Script

Based on Francisco Perdigon Romero's deepfilter_pipeline
Adapted for ECG baseline wander removal
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import keras
from keras import backend as K
from keras import losses

from models.model_DeepFilter import deepfilter


# ============================================================================
# Custom Loss Functions (same as training)
# ============================================================================

def ssd_loss(y_true, y_pred):
    """Sum of Squared Distance"""
    return K.sum(K.square(y_pred - y_true), axis=-2)


def combined_ssd_mad_loss(y_true, y_pred):
    """Combined SSD + MAD Loss"""
    return K.max(K.square(y_true - y_pred), axis=-2) * 50 + \
           K.sum(K.square(y_true - y_pred), axis=-2)


def mad_loss(y_true, y_pred):
    """Maximum Absolute Distance"""
    return K.max(K.square(y_pred - y_true), axis=-2)


# ============================================================================
# Testing Pipeline
# ============================================================================

def test_deepfilter(X_test, y_test, args):
    """
    Test DeepFilter model
    
    Args:
        X_test: Test noisy signals (N, signal_size, 1)
        y_test: Test clean baseline (N, signal_size, 1)
        args: Test arguments
    
    Returns:
        y_pred: Predicted baseline (N, signal_size, 1)
    """
    print(f'Testing DeepFilter: {args.model_type}')
    print(f'Test samples: {len(X_test)}')
    
    # ==================
    # Load Model
    # ==================
    
    if args.model_type == 'LANL':
        model = deepfilter.deep_filter_I_LANL(signal_size=args.signal_size)
        model_label = 'DeepFilter_LANL'
    elif args.model_type == 'LANLD':
        model = deepfilter.deep_filter_model_I_LANL_dilated(signal_size=args.signal_size)
        model_label = 'DeepFilter_LANLD'
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f'\nModel: {model_label}\n')
    model.summary()
    
    # ==================
    # Compile Model (needed for load_weights)
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
    
    if y_test is not None:
        print('\n' + '='*60)
        print('Evaluating on test set...')
        test_results = model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)
        
        print('\nTest Results:')
        for metric_name, value in zip(model.metrics_names, test_results):
            print(f'  {metric_name}: {value:.6f}')
    
    K.clear_session()
    
    return y_pred


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Test DeepFilter for ECG baseline removal')
    
    # Model
    parser.add_argument('--model_type', type=str, default='LANLD',
                        choices=['LANL', 'LANLD'],
                        help='LANL: basic, LANLD: dilated+dropout')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights (.hdf5)')
    parser.add_argument('--signal_size', type=int, default=512,
                        help='Input signal length')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data (.npz file)')
    
    # Inference
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    
    # Output
    parser.add_argument('--out_path', type=str, default=None,
                        help='Path to save predictions (.npz)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Load data
    print(f'Loading data from: {args.data_path}')
    data = np.load(args.data_path)
    
    X_test = data['X_test']
    y_test = data.get('y_test', None)  # Optional
    
    print(f'X_test shape: {X_test.shape}')
    if y_test is not None:
        print(f'y_test shape: {y_test.shape}')
    
    # Test
    y_pred = test_deepfilter(X_test, y_test, args)
    
    # Save predictions
    if args.out_path:
        print(f'\nSaving predictions to: {args.out_path}')
        np.savez(args.out_path, X_test=X_test, y_test=y_test, y_pred=y_pred)
        print('Done!')
    
    print('\n' + '='*60)
    print('Testing completed!')
    print('='*60)
