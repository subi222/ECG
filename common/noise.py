import numpy as np

from common.metrics import (
    EPS,
    remove_dc,
    calculate_snr_db,
)


EPS = 1e-12

def add_baseline_wander_snr(clean_ecg, bw, target_snr_db):
    """
    reference(raw) + scaled baseline wander (target input SNR)
    - reference는 raw 그대로 유지(DC 제거 X)
    - baseline wander는 DC 제거한 bw0 사용
    - scale 계산/지표 계산에서만 mean 제거 사용
    return: noisy, ref, actual_snr_in
    """
    N = min(len(clean_ecg), len(bw))

    # reference = raw
    ref = np.asarray(clean_ecg[:N], dtype=np.float64)

    # wander = DC 제거한 bw
    bw0 = remove_dc(np.asarray(bw[:N], dtype=np.float64))

    # scale 계산은 DC 제거한 ref0로
    ref0 = remove_dc(ref)
    ps = np.mean(ref0 ** 2)
    pn = np.mean(bw0 ** 2) + EPS

    target_noise_power = ps / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / pn)

    noisy = ref + bw0 * scale
    actual_snr = calculate_snr_db(ref, noisy, remove_mean=True)

    return noisy.astype(np.float32), ref.astype(np.float32), float(actual_snr)