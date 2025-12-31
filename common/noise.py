# common/noise.py
"""
공통 노이즈 믹싱 모듈 (Noise mixing for experiments)

❗️중요 원칙
- 모든 모델(DAE / UNet / Proposed)은 동일한 입력 생성 로직을 사용해야 한다.
- 입력 SNR을 정확히 통제하여 모델 간 비교의 공정성을 보장한다.

이 파일은:
- clean ECG와 baseline wander(NSTDB)를 이용해
- 목표 input SNR을 만족하는 noisy ECG를 생성한다.
"""

import numpy as np

from common.metrics import (
    EPS,
    remove_dc,
    calculate_snr_db,
)


def add_baseline_wander_snr(clean_ecg, bw, target_snr_db):
    """
    Clean ECG에 baseline wander를 추가하여
    목표 입력 SNR(target_snr_db)을 만족하는 noisy ECG를 생성한다.

    설계 원칙
    ----------
    - reference(ref)는 MITDB raw ECG 그대로 사용 (DC 제거 ❌)
    - baseline wander(bw)는 DC 제거 후 사용
    - DC 제거는 "scale 계산과 지표 계산 단계에서만" 적용
    - 입력 신호의 형태 자체는 최대한 보존

    Parameters
    ----------
    clean_ecg : array-like
        MITDB raw ECG 신호 (reference)
    bw : array-like
        Baseline wander 신호 (NSTDB)
    target_snr_db : float
        목표 입력 SNR (dB)

    Returns
    -------
    noisy : np.ndarray
        baseline wander가 추가된 입력 ECG (float32)
    ref : np.ndarray
        기준 ECG (raw reference, float32)
    actual_snr : float
        실제로 달성된 입력 SNR (dB)
    """
    # 길이 맞추기 (겹치는 구간까지만 사용)
    N = min(len(clean_ecg), len(bw))

    # reference = raw ECG (DC 제거 X)
    ref = np.asarray(clean_ecg[:N], dtype=np.float64)

    # baseline wander는 DC 제거 후 사용
    bw0 = remove_dc(np.asarray(bw[:N], dtype=np.float64))

    # SNR 계산을 위한 clean reference (DC 제거)
    ref0 = remove_dc(ref)

    # 신호 / 노이즈 파워
    ps = np.mean(ref0 ** 2)
    pn = np.mean(bw0 ** 2) + EPS  # 수치 안정성 확보

    # 목표 SNR → 목표 노이즈 파워
    target_noise_power = ps / (10 ** (target_snr_db / 10))

    # baseline wander 스케일 계수
    scale = np.sqrt(target_noise_power / pn)

    # noisy ECG 생성
    noisy = ref + bw0 * scale

    # 실제 입력 SNR 계산 (검증용)
    actual_snr = calculate_snr_db(ref, noisy, remove_mean=True)

    return noisy.astype(np.float32), ref.astype(np.float32), float(actual_snr)
