# ECG Denoising Performance Metrics
# SNR, RMSE, PRD, PRDN, SSIM

import numpy as np

EPS = 1e-12

def remove_dc(x):
    return x - np.mean(x)

def calculate_snr_db(clean, est, remove_mean=True):
    """
    Signal-to-Noise Ratio in dB
    
    Parameters
    ----------
    clean : array-like
        Reference signal (ground truth)
    est : array-like
        Estimated signal (noisy or processed)
    remove_mean : bool
        If True, remove DC offset from both signals before calculation
    
    Returns
    -------
    snr_db : float
        SNR in decibels (higher is better)
    """
    clean = np.asarray(clean, dtype=np.float64)
    est   = np.asarray(est, dtype=np.float64)

    if remove_mean:
        clean0 = clean - clean.mean()
        est0   = est - est.mean()
    else:
        clean0 = clean
        est0   = est

    noise = est0 - clean0
    ps = np.mean(clean0 ** 2)
    pn = np.mean(noise ** 2) + EPS
    return 10.0 * np.log10(ps / pn)

def calculate_rmse(clean, processed):
    """
    Root Mean Square Error (DC offset removed)
    
    Parameters
    ----------
    clean : array-like
        Reference signal
    processed : array-like
        Processed/denoised signal
    
    Returns
    -------
    rmse : float
        RMSE value (lower is better)
    """
    clean0 = remove_dc(np.asarray(clean, dtype=np.float64))
    proc0  = remove_dc(np.asarray(processed, dtype=np.float64))
    return float(np.sqrt(np.mean((clean0 - proc0) ** 2)))
    

def calculate_prd(clean, processed, remove_mean=True):
    """
    Percent Root-mean-square Difference
    
    PRD (%) = 100 * ||clean - processed|| / ||clean||
    
    Parameters
    ----------
    clean : array-like
        Reference signal
    processed : array-like
        Processed/denoised signal
    remove_mean : bool
        If True, remove DC offset before calculation
    
    Returns
    -------
    prd : float
        PRD percentage (lower is better)
    """
    clean = np.asarray(clean, dtype=np.float64)
    proc  = np.asarray(processed, dtype=np.float64)

    if remove_mean:
        clean = clean - clean.mean()
        proc  = proc - proc.mean()

    num = np.linalg.norm(clean - proc)
    den = np.linalg.norm(clean) + EPS
    return float(100.0 * num / den)


def calculate_prd_normalized(clean, processed, remove_mean=True):
    """
    Normalized PRD (scale-invariant)
    
    Finds optimal gain factor G that minimizes ||clean - G*processed||.
    This ignores amplitude/gain differences between signals.
    
    Parameters
    ----------
    clean : array-like
        Reference signal
    processed : array-like
        Processed/denoised signal
    remove_mean : bool
        If True, remove DC offset before calculation
    
    Returns
    -------
    prdn : float
        Normalized PRD percentage (lower is better)
    """
    clean = np.asarray(clean, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)

    if remove_mean:
        clean = clean - clean.mean()
        proc = proc - proc.mean()

    # Optimal scaling factor G = dot(clean, proc) / dot(proc, proc)
    num_g = np.dot(clean, proc)
    den_g = np.dot(proc, proc) + EPS
    g = num_g / den_g

    # Scaled residual
    res = clean - g * proc
    num = np.linalg.norm(res)
    den = np.linalg.norm(clean) + EPS
    return float(100.0 * num / den)


def calculate_ssim(clean, processed, remove_mean=True, window_size=11):
    """
    Structural Similarity Index for 1D ECG signals
    
    Adapted from image SSIM to 1D using sliding window approach.
    SSIM = (2*μ_x*μ_y + C1)(2*σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
    
    Parameters
    ----------
    clean : array-like
        Reference ECG signal
    processed : array-like
        Processed/denoised ECG signal
    remove_mean : bool
        If True, remove DC offset before calculation
    window_size : int
        Size of the sliding window for local statistics
    
    Returns
    -------
    ssim : float
        SSIM value in range [0, 1] (higher is better, 1 = perfect match)
    """
    clean = np.asarray(clean, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)
    
    if remove_mean:
        clean = clean - clean.mean()
        proc = proc - proc.mean()
    
    # Dynamic range based on data
    L = max(clean.max() - clean.min(), proc.max() - proc.min()) + EPS
    
    # Constants (from original SSIM paper, adapted for 1D)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    # Sliding window for local SSIM
    N = len(clean)
    if N < window_size:
        window_size = N
    
    ssim_values = []
    half_win = window_size // 2
    
    for i in range(half_win, N - half_win):
        start = i - half_win
        end = i + half_win + 1
        
        x_win = clean[start:end]
        y_win = proc[start:end]
        
        # Local statistics
        mu_x = np.mean(x_win)
        mu_y = np.mean(y_win)
        sigma_x_sq = np.var(x_win)
        sigma_y_sq = np.var(y_win)
        sigma_xy = np.cov(x_win, y_win)[0, 1]
        
        # SSIM formula
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        
        ssim_local = numerator / (denominator + EPS)
        ssim_values.append(ssim_local)
    
    if not ssim_values:
        return 1.0  # Perfect match for very short signals
    
    return float(np.mean(ssim_values))