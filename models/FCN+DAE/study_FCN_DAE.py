import torch
import torch.nn as nn
import torch.optim as optim



class FCN_DAE(nn.Module):
    def __init__(self):
        super(FCN_DAE, self).__init__()

        # ==========================================
        # Encoder (Downsampling)
        # 입력: (Batch, 1, 1024) -> 출력: (Batch, 1, 32)
        # ==========================================
        
        # Layer 1: 40 filters, kernel 16, stride 2 (1024 -> 512)
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=40, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        
        # Layer 2: 20 filters, kernel 16, stride 2 (512 -> 256)
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=20, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # Layer 3: 20 filters, kernel 16, stride 2 (256 -> 128)
        self.enc3 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        
        # Layer 4: 20 filters, kernel 16, stride 2 (128 -> 64)
        self.enc4 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        
        # Layer 5: 40 filters, kernel 16, stride 2 (64 -> 32)
        self.enc5 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=40, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        
        # Layer 6 (Latent): 1 filter, kernel 16, stride 1 (32 -> 32)
        # stride가 1이므로 크기 유지를 위해 padding='same' 또는 적절한 패딩 사용
        self.enc6 = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=1, kernel_size=16, stride=1, padding='same'),
            nn.BatchNorm1d(1),
            nn.ELU(),
            nn.Dropout(0.5)
        )


        # ==========================================
        # Decoder (Upsampling) - Inverse Symmetry
        # 입력: (Batch, 1, 32) -> 출력: (Batch, 1, 1024)
        # ==========================================

        # Layer 1 (Inverse of Enc6): 1 -> 40, stride 1 (32 -> 32)
        self.dec1 = nn.Sequential(
            # ConvTranspose1d는 padding이 'same' 옵션이 없으므로 직접 계산 (kernel=16, stride=1 -> padding=7일 때 1픽셀 큼 -> padding=7, output_padding=0 등 조정 필요)
            # 여기서는 차원 유지를 위해 padding=7로 설정하고 크기 강제 조정이 안전함
            nn.ConvTranspose1d(in_channels=1, out_channels=40, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # Layer 2 (Inverse of Enc5): 40 -> 20, stride 2 (32 -> 64)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=40, out_channels=20, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # Layer 3 (Inverse of Enc4): 20 -> 20, stride 2 (64 -> 128)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # Layer 4 (Inverse of Enc3): 20 -> 20, stride 2 (128 -> 256)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=20, out_channels=20, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # Layer 5 (Inverse of Enc2): 20 -> 40, stride 2 (256 -> 512)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=20, out_channels=40, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # Layer 6 (Output): 40 -> 1, stride 2 (512 -> 1024)
        # 마지막 레이어는 활성화 함수 없음 (Linear)
        self.dec6 = nn.ConvTranspose1d(in_channels=40, out_channels=1, kernel_size=16, stride=2, padding=7)


    def forward(self, x):
        # --- Encoder ---
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        z = self.enc6(x) # Latent vector z (Batch, 1, 32)
        
        # --- Decoder ---
        x = self.dec1(z)
        
        # ConvTranspose1d stride 1 with padding 7 creates size issue (L_in + 16 - 2*7 -1 + 1 = L_in + 2).
        # 보정: 차원이 2픽셀 늘어날 수 있으므로 잘라줍니다.
        if x.shape[2] != z.shape[2]: 
            diff = x.shape[2] - z.shape[2]
            x = x[:, :, diff//2 : -(diff - diff//2)]
            
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        out = self.dec6(x)
        
        return out

# ==========================================
# 학습 세팅 (Training Setup)
# ==========================================

# 모델 인스턴스 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCN_DAE().to(device)

# 손실 함수: 평균 제곱 오차 (MSE)
criterion = nn.MSELoss()

# 최적화 기법: Adam 옵티마이저 (Learning rate는 일반적으로 0.001 등으로 설정)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 모델 구조 검증 (Shape Check)
# ==========================================
if __name__ == "__main__":
    # 가짜 입력 데이터 (Batch Size=8, Channel=1, Length=1024)
    dummy_input = torch.randn(8, 1, 1024).to(device)
    
    # 모델 통과
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")   # torch.Size([8, 1, 1024])
    print(f"Output shape: {output.shape}") # torch.Size([8, 1, 1024])
    
    # 크기가 정확히 복원되었는지 확인
    assert dummy_input.shape == output.shape, "입출력 크기가 다릅니다! 패딩을 확인하세요."
    print("✅ 모델 구조 검증 완료: 입력과 출력의 크기가 동일합니다.")