import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# --- [1] 경로 설정 및 함수 불러오기 (Import) ---
# 프로젝트 루트: 현재 파일(train_unet.py) 기준 두 단계 위
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# ★ [핵심 수정] 외부 파일 불러오기 및 경로 강제 수정 ★
try:
    # 1. 모듈 자체를 먼저 불러옵니다.
    import run_synthetic_test

    # 2. 모듈 안에 있는 경로 변수들을 '절대 경로'로 강제로 바꿔치기합니다. (Monkey Patching)
    # 이렇게 하면 실행 위치가 어디든 상관없이 무조건 루트 폴더를 찾아갑니다.
    run_synthetic_test.MITDB_DIR = PROJECT_ROOT / "MITDB_data"
    run_synthetic_test.NSTDB_DIR = PROJECT_ROOT / "noise_data"

    # 3. 이제 함수들을 안전하게 가져옵니다.
    from run_synthetic_test import add_baseline_wander_snr, load_mitdb_csv, load_nstdb_bw

    print(" 외부 함수(Mixer) 및 경로 설정 완료!")

except ImportError:
    print(" 오류: run_synthetic_test.py를 찾을 수 없습니다.")
    sys.exit(1)

# 깃허브 모델 불러오기
from UNet.unet1d import UNet


# ---------------------------------------------------------

def generate_data_in_ram(input_len=512, target_snr=6):
    records = [100, 101, 103, 105, 106, 107, 108, 111, 112, 113]
    X_list = []
    Y_list = []

    print(f" 데이터 생성 중... (Import된 함수 사용, SNR: {target_snr}dB)")

    # 노이즈 로드
    # (이제 경로가 수정되었으므로 에러가 나지 않을 것입니다)
    try:
        noise_full, _ = load_nstdb_bw("bw", 0, 60, 360)
    except Exception as e:
        print(f"❌ 노이즈 로드 실패: {e}")
        # 경로 확인용 디버그 메시지
        print(f"확인된 경로: {run_synthetic_test.NSTDB_DIR}")
        return np.array([]), np.array([])

    for rec in records:
        try:
            clean_full, _ = load_mitdb_csv(rec, 0, 30, 360)
            noisy_full, clean_ref, _ = add_baseline_wander_snr(clean_full, noise_full, target_snr)

            for i in range(0, len(clean_ref) - input_len, input_len):
                x_win = noisy_full[i:i + input_len]
                y_win = clean_ref[i:i + input_len]

                denom = np.max(x_win) - np.min(x_win) + 1e-8
                x_norm = (x_win - np.min(x_win)) / denom
                y_norm = (y_win - np.min(x_win)) / denom

                X_list.append(x_norm)
                Y_list.append(y_norm)
        except Exception as e:
            print(f"Pass Record {rec}: {e}")
            continue

    if len(X_list) == 0:
        print(" 데이터가 하나도 생성되지 않았습니다! 경로를 다시 확인해주세요.")
        return np.array([]), np.array([])

    X_arr = np.array(X_list).reshape(-1, 1, input_len).astype(np.float32)
    Y_arr = np.array(Y_list).reshape(-1, 1, input_len).astype(np.float32)

    print(f" 데이터 준비 완료! 총 {len(X_arr)}개 샘플")
    return X_arr, Y_arr


def train_model():
    INPUT_LEN = 512
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    SNR_DB = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 현재 사용 장치: {device}")

    X_train, Y_train = generate_data_in_ram(input_len=INPUT_LEN, target_snr=SNR_DB)

    if len(X_train) == 0:
        print(" 학습을 중단합니다.")
        return

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(in_channels=1, out_classes=1, dimensions=1, padding=True).to(device) #1D로 무조건 고정   (2D,3D 절대 불가)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(" 학습 시작!")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(dataloader):.6f}")

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    torch.save(model.state_dict(), 'outputs/unet_model.pth')
    print(" 학습 완료!")


if __name__ == '__main__':
    train_model()