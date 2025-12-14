import numpy as np
import wfdb
from scipy.signal import resample
import os


def get_noise(clean_signal, target_snr_db):
    """
    clean_signal: 깨끗한 ECG 신호 (1D array)
    target_snr_db: 목표 SNR (dB)

    Returns:
        noisy_signal: 노이즈가 섞인 신호
        real_noise_scaled: 스케일링된 노이즈
    """
    # ==========================================
    # 1. [안전장치] 입력 데이터 정제 (NaN 제거 & 정밀도 향상)
    # ==========================================
    clean_signal = np.nan_to_num(clean_signal)  # NaN을 0으로 변환
    clean_signal = clean_signal.astype(np.float64)  # 정밀도 float64로 고정

    # ==========================================
    # 2. 노이즈 파일 로드 (bw, ma, em 중 하나)
    # ==========================================
    # 주의: noise_data 폴더 경로가 맞는지 확인하세요.
    # 사용자가 원하는 노이즈 파일 경로로 수정 가능
    noise_path = 'noise_data/bw'

    try:
        # 확장자 없이 경로만 입력 (.dat, .hea 자동 인식)
        record = wfdb.rdrecord(noise_path)
        noise_signal = record.p_signal

        # 2차원이면 1차원으로 변환
        if noise_signal.ndim == 2:
            noise_signal = noise_signal[:, 0]

        # NaN 제거 및 정밀도 향상
        noise_signal = np.nan_to_num(noise_signal)
        noise_signal = noise_signal.astype(np.float64)

        # Hz가 다르면 리샘플링 (교수님 코드 기준 250Hz)
        target_fs = 250
        if record.fs != target_fs:
            num = int(len(noise_signal) * target_fs / record.fs)
            noise_signal = resample(noise_signal, num)

    except Exception as e:
        print(f"⚠️ 노이즈 로드 실패 ({noise_path}): {e}")
        # 로드 실패 시 가짜 노이즈 생성 (테스트용)
        t = np.arange(len(clean_signal))
        noise_signal = 0.5 * np.sin(2 * np.pi * 0.1 * t / 250)

    # ==========================================
    # 3. 믹싱 (SNR 공식 적용)
    # ==========================================
    # 길이 맞추기
    min_len = min(len(clean_signal), len(noise_signal))
    clean = clean_signal[:min_len]
    noise = noise_signal[:min_len]

    # 전력 계산
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    # 노이즈 스케일링
    if noise_power == 0:
        return clean, np.zeros_like(clean)

    target_noise_power = clean_power / (10 ** (target_snr_db / 10))
    scale_factor = np.sqrt(target_noise_power / noise_power)

    real_noise_scaled = noise * scale_factor
    noisy_signal = clean + real_noise_scaled

    # ==========================================
    # 4. [안전장치] 최종 출력 검사
    # ==========================================
    # 혹시 모를 NaN, 무한대 다시 한번 0으로 치환
    noisy_signal = np.nan_to_num(noisy_signal)

    return noisy_signal, real_noise_scaled


# (테스트용 실행 코드)
if __name__ == "__main__":
    # 가짜 데이터로 테스트
    dummy_clean = np.sin(np.linspace(0, 10, 1000))
    noisy, _ = get_noise(dummy_clean, 6)
    print(f"테스트 완료. 데이터 길이: {len(noisy)}, 타입: {noisy.dtype}")