import numpy as np
from fractions import Fraction
from scipy import signal



# 360Hz인 MITDB 데이터를 250Hz로 변환
def _resample_to_target(ecg: np.ndarray, fs_raw: float, fs_target: float) -> np.ndarray:
    fs_raw = float(fs_raw)
    fs_target = float(fs_target)
    if abs(fs_raw - fs_target) < 1e-9:
        return ecg.astype(np.float32, copy=False)

    ratio = float(fs_target) / float(fs_raw)
    frac = Fraction(ratio).limit_denominator(1000)     # 250/360 -> 25/36 로 잘 근사됨
    return signal.resample_poly(ecg, up=frac.numerator, down=frac.denominator).astype(np.float32, copy=False)