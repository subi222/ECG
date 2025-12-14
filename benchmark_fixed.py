import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample

# ==========================================
# 1. 모듈 임포트 수정 (여기가 핵심!)
# ==========================================
# 기존: from baseline_core import baseline_removal
# 수정: baseline.py 안에 있는 함수를 직접 가져옵니다.
from baseline import process_ecg_array
from metrics import SNR, NRMSE
from noise_mixer import get_noise  # (참고: noise_mixer가 외부 모듈이라면 유지)


def get_data(path, target_fs=250):
    """ 데이터를 읽고 250Hz로 리샘플링 """
    try:
        df = pd.read_csv(path, header=None)

        if df.shape[1] > 1:
            signal = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
        else:
            signal = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values

        src_fs = 360  # MITDB 원본 주파수
        if src_fs != target_fs:
            print(f"[전처리] 리샘플링 실행: {src_fs}Hz -> {target_fs}Hz")
            num_samples = int(len(signal) * target_fs / src_fs)
            signal = resample(signal, num_samples)

        return signal

    except Exception as e:
        print(f"데이터 로드 에러: {e}")
        return np.array([])


def main():
    # 1. 데이터 로드 (파일 경로 확인 필수)
    file_path = '/home/subi/PycharmProjects/ECG/MITDB_data/100.csv'
    original_signal = get_data(file_path, target_fs=250)

    if len(original_signal) == 0:
        print("데이터를 불러오지 못했습니다. 파일 경로를 확인하세요.")
        return

    # 2. 실험용으로 데이터 자르기
    N = 5000
    signal = original_signal[:N]
    signal = signal - np.mean(signal)  # 0점 조절

    # 3. 노이즈 추가 (SNR 10dB)
    target_snr = 10
    # get_noise 함수가 (clean, snr) 또는 (clean, noise, snr) 인지 확인 필요
    # 여기서는 noise_mixer.py가 자동으로 노이즈를 로드한다고 가정
    noisy_signal, _ = get_noise(signal, target_snr)


    # 4. 알고리즘 실행
    # [수정] float64로 변환하고 NaN(빈값)을 한 번 더 제거합니다.
    noisy_signal = np.nan_to_num(noisy_signal).astype(np.float64)

    # 이제 안전해진 데이터를 넘깁니다.
    filtered_signal = process_ecg_array(noisy_signal, fs_raw=250)

    # 5. 성능 평가
    snr_out = SNR(signal, filtered_signal)
    rmse_out = NRMSE(signal, filtered_signal)

    # 6. 결과 출력
    print("=" * 40)
    print(f"Target Input SNR : {target_snr} dB")
    print(f"Output SNR       : {snr_out:.4f} dB")
    print(f"Output RMSE      : {rmse_out:.4f}")
    print("=" * 40)

    # 7. 그래프 확인
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title(f"Noisy Input (SNR {target_snr}dB)")
    plt.plot(noisy_signal, 'gray', label='Noisy', alpha=0.7)
    plt.plot(signal, 'g', label='Clean', linewidth=1.5)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(f"Filtered Result (SNR {snr_out:.2f}dB)")
    plt.plot(filtered_signal, 'orange', label='Filtered')
    plt.plot(signal, 'g--', label='Clean', alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()