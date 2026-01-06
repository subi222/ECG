import torch
import torch.nn as nn
import torch.nn.functional as F

"""
UNet1D Study Implementation (Original Paper Style)
- Dimensions: 1D (for ECG)
- Padding: 0 (Unpadded Convolutions)
- Skip Connections: Center Crop & Concatenate
- Architecture: 4 Down-levels, 1 Bottleneck, 4 Up-levels
"""

class DoubleConv(nn.Module):
    """
    [Conv3 -> ReLU -> Conv3 -> ReLU]
    오리지널 논문의 기본 블록입니다. 패딩이 0이므로 거치고 나면 크기가 줄어듭니다.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First Conv: L_out = L_in - (3 - 1) = L_in - 2
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # Second Conv: L_out = L_in - 2
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input x: (Batch, In_Channels, Length)
        return self.double_conv(x)
        # Output: (Batch, Out_Channels, Length - 4)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # -------------------------------------------------------
        # 수축 경로 (Encoder)
        # -------------------------------------------------------
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # -------------------------------------------------------
        # 확장 경로 (Decoder)
        # -------------------------------------------------------
        for feature in reversed(features):
            # Transposed Conv: 크기를 2배로 늘리고 채널을 반으로 줄임
            self.ups.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # -------------------------------------------------------
        # 바닥 구간 (Bottleneck)
        # -------------------------------------------------------
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # -------------------------------------------------------
        # 최종 출력 레이어 (1x1 Conv)
        # -------------------------------------------------------
        self.final_conv = nn.Conv1d(features[0], num_classes, kernel_size=1)

    def crop_and_concat(self, upsampled, bypass):
        """
        Skip Connection을 위한 Center Crop 함수
        - upsampled: 디코더에서 올라온 특징 맵
        - bypass: 엔코더에서 건너온 특징 맵 (크기가 더 큼)
        """
        # upsampled shape: (B, C, L_up)
        # bypass shape: (B, C, L_by)
        diff = bypass.size()[2] - upsampled.size()[2]
        start = diff // 2
        
        # 중앙을 기준으로 슬라이싱 (Center Crop)
        bypass_cropped = bypass[:, :, start : start + upsampled.size()[2]]
        
        # 채널(Dimension 1)을 기준으로 합침
        return torch.cat((bypass_cropped, upsampled), dim=1)

    def forward(self, x):
        # x shape: (Batch, 1, 572) -> 오리지널 논문 권장 입력 크기 예시

        skip_connections = []

        # 1. Encoder 단계
        # (B, 1, 572) -> [DoubleConv] -> (B, 64, 568) -> [Pool] -> (B, 64, 284)
        # (B, 64, 284) -> [DoubleConv] -> (B, 128, 280) -> [Pool] -> (B, 128, 140)
        # ...
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 2. Bottleneck 단계
        # (B, 512, 64) -> [DoubleConv] -> (B, 1024, 60)
        x = self.bottleneck(x)

        # 3. Decoder 단계
        # 역순으로 쌓인 skip_connections를 꺼내기 위해 뒤집음
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            # (1) Up-sampling (ConvTranspose)
            x = self.ups[i](x) 
            
            # (2) Skip connection (Crop & Concat)
            skip_connection = skip_connections[i//2]
            x = self.crop_and_concat(x, skip_connection)
            
            # (3) DoubleConv
            x = self.ups[i+1](x)

        # 4. Final Final Layer (1x1)
        # (B, 64, L_out) -> (B, num_classes, L_out)
        return self.final_conv(x)

if __name__ == "__main__":
    # 간단한 테스트
    model = UNet1D(in_channels=1, num_classes=1)
    
    # 논문의 2D 크기인 572를 1D로 가정하고 테스트
    x = torch.randn((1, 1, 572)) 
    output = model(x)
    
    print(f"입력 크기: {x.shape}")
    print(f"출력 크기: {output.shape}")
    # 출력 크기는 (1, 1, 388)이 될 것입니다. (Unpadded 특성상 줄어듦)