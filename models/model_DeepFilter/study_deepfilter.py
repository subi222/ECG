"""
DeepFilter: An ECG baseline wander removal filter using deep learning techniques
PyTorch Implementation (1D version for ECG signals)

Reference:
Perdigón Laguna, F., Romero Vivo, A., & Laguna, P. (2021).
"DeepFilter: An ECG baseline wander removal filter using deep learning techniques."
Biomedical Signal Processing and Control, 70, 102992.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MKLANL(nn.Module):
    """
    Multi-Kernel Linear And Non-Linear (MKLANL) Module
    
    8개의 병렬 가지(branch)로 구성:
    - 4개 Linear branches (kernel sizes: 3, 5, 9, 15)
    - 4개 Non-linear branches with ReLU (kernel sizes: 3, 5, 9, 15)
    
    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수 (N in paper, must be divisible by 8)
        dilation_rate (int): Dilation rate for all convolutions
    """
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(MKLANL, self).__init__()
        
        assert out_channels % 8 == 0, "out_channels must be divisible by 8"
        
        branch_channels = out_channels // 8
        kernel_sizes = [3, 5, 9, 15]
        
        # Group 1: Linear branches (no activation)
        self.linear_branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k - 1) * dilation_rate // 2
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=k,
                dilation=dilation_rate,
                padding=padding,
                bias=True
            )
            self.linear_branches.append(conv)
        
        # Group 2: Non-linear branches (with ReLU)
        self.nonlinear_branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k - 1) * dilation_rate // 2
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=k,
                dilation=dilation_rate,
                padding=padding,
                bias=True
            )
            self.nonlinear_branches.append(conv)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (Batch, in_channels, Length)
        
        Returns:
            (Batch, out_channels, Length)
        """
        outputs = []
        
        # Linear branches (no activation)
        for conv in self.linear_branches:
            outputs.append(conv(x))
        
        # Non-linear branches (with ReLU)
        for conv in self.nonlinear_branches:
            outputs.append(self.relu(conv(x)))
        
        # Concatenate along channel dimension
        return torch.cat(outputs, dim=1)


class DeepFilter(nn.Module):
    """
    DeepFilter Model for ECG Baseline Wander Removal
    
    Architecture (Figure 3 in paper):
    - 6 sequential MKLANL modules with varying N and dilation rates
    - Final 1x1-like conv to produce single-channel output
    
    Input: (Batch, 1, Signal_Length)
    Output: (Batch, 1, Signal_Length) - estimated baseline
    """
    def __init__(self):
        super(DeepFilter, self).__init__()
        
        # Module 1: N=64, dilation=1
        self.mklanl1 = MKLANL(in_channels=1, out_channels=64, dilation_rate=1)
        
        # Module 2: N=64, dilation=3
        self.mklanl2 = MKLANL(in_channels=64, out_channels=64, dilation_rate=3)
        
        # Module 3: N=32, dilation=1
        self.mklanl3 = MKLANL(in_channels=64, out_channels=32, dilation_rate=1)
        
        # Module 4: N=32, dilation=3
        self.mklanl4 = MKLANL(in_channels=32, out_channels=32, dilation_rate=3)
        
        # Module 5: N=16, dilation=1
        self.mklanl5 = MKLANL(in_channels=32, out_channels=16, dilation_rate=1)
        
        # Module 6: N=16, dilation=3
        self.mklanl6 = MKLANL(in_channels=16, out_channels=16, dilation_rate=3)
        
        # Final layer: reduce to single channel
        self.final_conv = nn.Conv1d(
            in_channels=16,
            out_channels=1,
            kernel_size=9,
            padding='same',
            bias=True
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (Batch, 1, Signal_Length) - Noisy ECG signal
        
        Returns:
            (Batch, 1, Signal_Length) - Estimated baseline wander
        """
        x = self.mklanl1(x)
        x = self.mklanl2(x)
        x = self.mklanl3(x)
        x = self.mklanl4(x)
        x = self.mklanl5(x)
        x = self.mklanl6(x)
        x = self.final_conv(x)
        return x


class DeepFilterLoss(nn.Module):
    """
    Custom Loss Function for DeepFilter
    
    Loss = SSD + lambda * MAD
    
    - SSD (Sum of Squared Distance): Mean Squared Error
    - MAD (Maximum Absolute Distance): Maximum absolute difference
    - lambda: weight for MAD term (논문에서 50 권장)
    
    Reference: Equation 2 in the paper
    """
    def __init__(self, lambda_mad=50.0):
        super(DeepFilterLoss, self).__init__()
        self.lambda_mad = lambda_mad
    
    def forward(self, pred, target):
        """
        Compute loss
        
        Args:
            pred: (Batch, 1, Length) - predicted baseline
            target: (Batch, 1, Length) - ground truth baseline
        
        Returns:
            scalar loss value
        """
        # SSD: Sum of Squared Distance (equivalent to MSE * N)
        # We use mean to make it batch-size independent
        ssd = torch.mean((pred - target) ** 2)
        
        # MAD: Maximum Absolute Distance
        # Take max over the signal length dimension
        abs_diff = torch.abs(pred - target)
        mad = torch.max(abs_diff)
        
        # Combined loss
        loss = ssd + self.lambda_mad * mad
        
        return loss


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepFilter Model Test")
    print("=" * 60)
    
    # Create model
    model = DeepFilter()
    print(f"\n✅ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    # Test with dummy data
    batch_size = 4
    signal_length = 1000
    
    # Random input (noisy ECG)
    x_input = torch.randn(batch_size, 1, signal_length)
    print(f"\n🔢 Input shape: {x_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        baseline_pred = model(x_input)
    
    print(f"✅ Output shape: {baseline_pred.shape}")
    assert baseline_pred.shape == x_input.shape, "Output shape mismatch!"
    
    # Test loss function
    print(f"\n🧮 Testing Loss Function...")
    loss_fn = DeepFilterLoss(lambda_mad=50.0)
    
    # Dummy target (ground truth baseline)
    target_baseline = torch.randn(batch_size, 1, signal_length)
    
    loss = loss_fn(baseline_pred, target_baseline)
    print(f"✅ Loss value: {loss.item():.6f}")
    
    print(f"\n{'='*60}")
    print("✅ All tests passed! DeepFilter is ready to use.")
    print(f"{'='*60}")
