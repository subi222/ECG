
import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import wfdb
import math
import yaml
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force CPU for TensorFlow/Keras to avoid CUDA errors
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import models
from models.model_proposed.v37_standalone import v37_baseline_correction
from models.model_DeepFilter import deepfilter

# Use Keras only if needed (lazy import to save memory/startup)
import tensorflow as tf
import keras

# =========================================================
# Model Loading Functions (Adapted from run_benchmark.py)
# =========================================================

# 1. Proposed (v37) - No weights needed
def run_method_proposed(x_in, fs=360):
    # v37 removes baseline wander
    y, _ = v37_baseline_correction(x_in, fs=fs, r_idx=None, adaptive_denoise=True)
    return y

# 2. DeepFilter
def load_deepfilter_model(weights_path):
    print(f"Loading DeepFilter: {weights_path}")
    model = deepfilter.deep_filter_model_I_LANL_dilated(signal_size=512)
    def combined_ssd_mad_loss(y_true, y_pred):
        return tf.reduce_max(tf.square(y_true - y_pred), axis=-2) * 50 + \
               tf.reduce_sum(tf.square(y_true - y_pred), axis=-2)
    model.compile(loss=combined_ssd_mad_loss, optimizer=keras.optimizers.Adam())
    model.load_weights(str(weights_path))
    return model

def run_deepfilter(x_in, model):
    # Window-based inference
    win_len = 512
    hop_len = 256
    N = len(x_in)
    x_pad = np.pad(x_in, (0, win_len - N % hop_len), mode='edge')
    
    windows = []
    starts = []
    for s in range(0, len(x_pad) - win_len + 1, hop_len):
        windows.append(x_pad[s:s + win_len])
        starts.append(s)
    
    if not windows: return np.zeros_like(x_in)
    
    batch = np.array(windows, dtype=np.float32)[:, :, None]
    preds = model.predict(batch, verbose=0)
    
    out_sum = np.zeros(len(x_pad))
    out_cnt = np.zeros(len(x_pad))
    for pred, s in zip(preds, starts):
        out_sum[s:s + win_len] += pred[:, 0]
        out_cnt[s:s + win_len] += 1
        
    out = out_sum / np.maximum(out_cnt, 1.0)
    out = out[:N]
    out = out - np.median(out) # Align baseline
    return out

# 3. FCN+DAE
def load_fcndae_model(weights_path):
    print(f"Loading FCN+DAE: {weights_path}")
    from keras.models import Model
    from keras.layers import Input, Conv1D, Conv1DTranspose, BatchNormalization
    
    input_layer = Input(shape=(512, 1))
    # Basic architecture reconstruction for loading weights
    x = Conv1D(40, 16, 2, 'same', activation='elu')(input_layer); x = BatchNormalization()(x)
    x = Conv1D(20, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1D(20, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1D(20, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1D(40, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1D(1, 16, 1, 'same', activation='elu')(x); x = BatchNormalization()(x)
    
    x = Conv1DTranspose(1, 16, 1, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1DTranspose(40, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1DTranspose(20, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1DTranspose(20, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1DTranspose(20, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    x = Conv1DTranspose(40, 16, 2, 'same', activation='elu')(x); x = BatchNormalization()(x)
    outputs = Conv1DTranspose(1, 16, 1, 'same', activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=outputs)
    model.load_weights(str(weights_path))
    return model

def run_fcndae(x_in, model):
    return run_deepfilter(x_in, model) # Same inference logic

# 4. DRNN
def load_drnn_model(weights_path):
    print(f"Loading DRNN: {weights_path}")
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    model = Sequential()
    model.add(LSTM(64, input_shape=(512, 1), return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.load_weights(str(weights_path))
    return model

def run_drnn(x_in, model):
    # Same window logic but with scaling
    win_len = 512
    hop_len = 256
    N = len(x_in)
    x_pad = np.pad(x_in, (0, win_len - N % hop_len), mode='edge')
    
    windows = []
    starts = []
    for s in range(0, len(x_pad) - win_len + 1, hop_len):
        windows.append(x_pad[s:s + win_len])
        starts.append(s)
    
    if not windows: return np.zeros_like(x_in)
    
    batch = np.array(windows, dtype=np.float32)[:, :, None] / 4.0 # Scale
    preds = model.predict(batch, verbose=0) * 4.0 # Inv Scale
    
    out_sum = np.zeros(len(x_pad))
    out_cnt = np.zeros(len(x_pad))
    for pred, s in zip(preds, starts):
        out_sum[s:s + win_len] += pred[:, 0]
        out_cnt[s:s + win_len] += 1
        
    out = out_sum / np.maximum(out_cnt, 1.0)
    out = out[:N]
    out = out - np.median(out)
    return out

# 5. DeScoD
def load_descod_model(weights_path, device='cuda'):
    print(f"Loading DeScoD: {weights_path}")
    from models.model_DeScoD.small_DeScoD import ConditionalModel
    from models.model_DeScoD.main_DeScoD import DDPM
    
    config_path = weights_path.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    base_model = ConditionalModel(feats=config["train"]["feats"]).to(device)
    model = DDPM(base_model, config, device).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def run_descod(x_in, model, device='cuda', shots=10):
    win_len = 512
    hop_len = 256
    N = len(x_in)
    x_pad = np.pad(x_in, (0, win_len - N % hop_len), mode='edge')
    
    windows = []
    starts = []
    for s in range(0, len(x_pad) - win_len + 1, hop_len):
        windows.append(x_pad[s:s + win_len])
        starts.append(s)
        
    batch = np.array(windows, dtype=np.float32)[:, None, :] / 4.0 # Scale to Torch (B, 1, L)
    batch_t = torch.from_numpy(batch).to(device)
    
    # Averaging shots
    preds_sum = torch.zeros_like(batch_t)
    model.eval()
    with torch.no_grad():
        for _ in range(shots):
            preds_sum += model.denoising(batch_t)
    preds = (preds_sum / shots).cpu().numpy() * 4.0 # Inv scale
    
    out_sum = np.zeros(len(x_pad))
    out_cnt = np.zeros(len(x_pad))
    for i, s in enumerate(starts):
        out_sum[s:s + win_len] += preds[i, 0, :]
        out_cnt[s:s + win_len] += 1
        
    out = out_sum / np.maximum(out_cnt, 1.0)
    out = out[:N]
    # No DC removal for DeScoD usually, but let's allow it if needed. Actually paper plot shows alignment.
    # We won't remove median blindly for DeScoD as it generates full signal, but for comparison aligning baselines is good.
    # Let's keep raw first.
    return out


# =========================================================
# Data Loading
# =========================================================
def load_data(rec_id, noise_type, snr_levels):
    # Define paths
    dataset_dir = ROOT / "data" / "MITDB_data"
    noise_dir = ROOT / "data" / "noise_data"
    
    # Load Record
    rec_path = str(dataset_dir / str(rec_id))
    print(f"Read record: {rec_path}")
    record = wfdb.rdrecord(rec_path)
    clean_sig = record.p_signal[:, 0].astype(np.float32) # Lead 0
    fs = record.fs
    
    # Load Noise
    noise_path = str(noise_dir / noise_type)
    print(f"Read noise: {noise_path}")
    noise_rec = wfdb.rdrecord(noise_path)
    noise_sig = noise_rec.p_signal[:, 0].astype(np.float32)
    
    # Create Noisy Signals
    # Use a specific segment (e.g., 2000 to 3000 samples)
    start_samp = 2000
    dur_samp = 1000 # 3-4 seconds
    
    clean_seg = clean_sig[start_samp : start_samp + dur_samp]
    noise_seg = noise_sig[start_samp : start_samp + dur_samp] # Align noise
    
    
    # RMS func
    rms = lambda x: np.sqrt(np.mean(x**2))
    
    noisy_signals = []
    
    for snr in snr_levels:
        s_rms = rms(clean_seg)
        n_rms = rms(noise_seg)
        if n_rms == 0: alpha = 0
        else: alpha = s_rms / n_rms / (10**(snr/20.0))
        
        noisy = clean_seg + alpha * noise_seg
        pure_noise = alpha * noise_seg
        noisy_signals.append((snr, noisy, pure_noise))
        
    return clean_seg, noisy_signals, fs

# =========================================================
# Plotting
# =========================================================
def plot_grid(clean, noisy_list, results, out_path="paper_comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Colors for paper
    colors = {
        'DeScoD (10-shot)': '#ff7f0e', # Orange
        'Proposed': '#2ca02c', # Green
        'Noisy': 'blue', # Bright Blue
        'Clean': '#d62728' # Red
    }
    
    models = list(results.keys())
    
    for i, (snr, noisy, pure_noise) in enumerate(noisy_list):
        ax = axes[i]
        
        # Plot Noisy Input (Clean + Noise)
        ax.plot(noisy, label=f'Noisy Input', color=colors['Noisy'], alpha=0.8, linewidth=0.8)
        
        # Plot Clean (Reference)
        ax.plot(clean, label='Clean', color=colors['Clean'], linestyle=':', linewidth=2.0, alpha=0.9)
        
        # Plot Models
        for model_name in models:
            output = results[model_name][i]
            c = colors.get(model_name, None)
            ax.plot(output, label=model_name, color=c, linewidth=1.2, alpha=0.9)
            
        ax.set_title(f"(d) noise = {snr} dB") # Following style "(a) noise = ..."
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude (au)")
        
        # Set x-axis range and ticks
        ax.set_xlim(0, 500)
        ax.set_xticks(np.arange(0, 501, 100))
        
        # Set y-axis ticks to interval of 1
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax.grid(True, alpha=0.3)
        
        # Only legend on first plot to save space or layout better
        if i == 0:
            ax.legend(loc='upper right', ncol=2, fontsize='small')
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    # Load Models
    device = "cpu" # Force CPU for stability
    
    # Load Weights Paths
    out_root = ROOT / "outputs"
    
    descod_path = out_root / "train_DeScoD/DeScoD_best.pth"
    deepfilter_path = out_root / "train_DeepFilter/DeepFilter_LANLD_best.weights.h5"
    fcndae_path = out_root / "train_FCN_DAE/FCN_DAE_best.weights.h5"
    drnn_path = out_root / "train_DRNN/DRNN_best.weights.h5"
    
    # Initialize Models
    models = {}
    
    # Note: Loading all might be heavy on GPU.
    # Strategy: Run inference one by one and free memory if possible. Or just load all.
    # Record 230 is small, so inference is fast.
    
    # 1. Load Data first
    clean_seg, noisy_list, fs = load_data('230', 'bw', [0, 5, 10, 15])
    
    # Store results: {ModelName: [out_0dB, out_5dB...]}
    results = {
        # 'DeepFilter': [],
        'DeScoD (10-shot)': [],
        # 'DRNN': [],
        # 'FCN+DAE': [],
        'Proposed': []
    }
    
    # --- PROPOSED (CPU) ---
    print("Running Proposed...")
    print("Running Proposed...")
    for _, noisy, _ in noisy_list:
        results['Proposed'].append(run_method_proposed(noisy, fs))
        
    # --- DEEPFILTER (GPU: Keras) ---
    # try:
    #     df_model = load_deepfilter_model(deepfilter_path)
    #     print("Running DeepFilter...")
    #     for _, noisy in noisy_list:
    #         results['DeepFilter'].append(run_deepfilter(noisy, df_model))
    #     del df_model
    #     keras.backend.clear_session()
    # except Exception as e:
    #     print(f"DeepFilter failed: {e}")
        
    # --- FCN+DAE (GPU: Keras) ---
    # try:
    #     fcn_model = load_fcndae_model(fcndae_path)
    #     print("Running FCN+DAE...")
    #     for _, noisy in noisy_list:
    #         results['FCN+DAE'].append(run_fcndae(noisy, fcn_model))
    #     del fcn_model
    #     keras.backend.clear_session()
    # except Exception as e:
    #      print(f"FCN+DAE failed: {e}")
         
    # --- DRNN (GPU: Keras) ---
    # try:
    #     drnn_model = load_drnn_model(drnn_path)
    #     print("Running DRNN...")
    #     for _, noisy in noisy_list:
    #         results['DRNN'].append(run_drnn(noisy, drnn_model))
    #     del drnn_model
    #     keras.backend.clear_session()
    # except Exception as e:
    #     print(f"DRNN failed: {e}")
        
    # --- DescoD (GPU: PyTorch) ---
    # Run last to avoid VRAM fight with TF
    try:
        descod_model = load_descod_model(descod_path, device=device)
        print("Running DeScoD...")
        print("Running DeScoD...")
        for _, noisy, _ in noisy_list:
            results['DeScoD (10-shot)'].append(run_descod(noisy, descod_model, device, shots=10))
        del descod_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"DeScoD failed: {e}")

    # Plot
    plot_grid(clean_seg, noisy_list, results, out_path=ROOT / "paper_comparison_BW_230.png")

if __name__ == "__main__":
    main()
