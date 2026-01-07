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