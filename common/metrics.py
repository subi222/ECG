#SNR,RMSE,NRMSE,PRD (채점 기준 통일)

import numpy as np

EPS = 1e-12

def remove_dc(x):
    return x - np.mean(x)

def calculate_snr_db(clean, est, remove_mean=True):
    """
    clean: reference
    est  : estimate (noisy or processed)
    remove_mean=True면 clean/est 각각 DC 제거 후 SNR 계산
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
    RMSE는 DC 제거 후 계산 (run_synthetic_test.py 스타일)
    """
    clean0 = remove_dc(np.asarray(clean, dtype=np.float64))
    proc0  = remove_dc(np.asarray(processed, dtype=np.float64))
    return float(np.sqrt(np.mean((clean0 - proc0) ** 2)))
    

def calculate_prd(clean, processed, remove_mean=True):
    """
    PRD (%): 100 * ||clean - processed|| / ||clean||
    보통 clean에 DC/평균을 제거한 버전으로 계산하는 경우가 많아서
    remove_mean=True를 기본으로 둠.
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
    PRDN (%): Optimized scaling implemented.
    Finds the optimal gain factor G that minimizes ||clean - G*processed||.
    This effectively ignores gain/amplitude differences.
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
    SSIM (Structural Similarity Index) for 1D ECG signals.
    
    Adapted from image SSIM to 1D signals using sliding window approach.
    SSIM = (2*μ_x*μ_y + C1)(2*σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
    
    Parameters
    ----------
    clean : array-like
        Reference ECG signal
    processed : array-like
        Processed/denoised ECG signal
    remove_mean : bool
        Whether to remove DC offset before calculation
    window_size : int
        Size of the sliding window for local statistics
    
    Returns
    -------
    ssim : float
        SSIM value in range [0, 1] (higher is better)
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