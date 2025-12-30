# -*- coding: utf-8 -*-
"""
model_DAE.py

Improved DAE reimplementation baseline (Xiong et al., 2016)
- Paper: "ECG signal enhancement based on improved denoising auto-encoder"
- Core network: Fully-connected 101-50-50-101 (δ=50 → window_len=101)
- Activations: Sigmoid for hidden and output (outputs in (0,1))
- Objective: Bernoulli distance (cross-entropy) → BCELoss in PyTorch

How this repo trains it (see train_DAE.py):
- Input formation: noisy ECG (MITDB raw + NSTDB bw scaled to target SNR)
- Preprocessing: Wavelet denoise (db6, level=8, soft-threshold) before windowing
- Windowing: radius=50 → length 101
- Training scheme:
  (1) Greedy layer-wise pretraining using SingleLayerAE
      - AE1: 101-50-101 (reconstruct input windows)
      - AE2: 50-50-50  (reconstruct hidden features)
  (2) End-to-end fine-tuning of ImprovedDAE (101-50-50-101) on denoising objective

Important notes / approximations:
- Weight tying (W' = W^T) is mentioned in some AE literature; this implementation does NOT enforce tied weights.
- Normalization follows the paper’s x∈[0,1]^p mapping:
  We use per-window min-max normalization for the input, and apply the SAME min/max to the target window
  (to keep paired training stable under BCELoss).
- Sampling rate: experiments may resample signals to 250 Hz for consistency with the comparison pipeline.
  (Model itself is agnostic; it only sees 101-length windows.)

Usage:
- Use ImprovedDAE for the final 101-50-50-101 denoising network.
- Use SingleLayerAE only as a helper module for layer-wise pretraining.
"""


from __future__ import annotations
import torch
import torch.nn as nn


class ImprovedDAE(nn.Module):
    """Fully-connected DAE: 101 → 50 → 50 → 101 with sigmoid activations."""
    def __init__(self, window_len: int = 101, hidden1: int = 50, hidden2: int = 50):
        super().__init__()
        if window_len != 101:
            # Paper explicitly uses δ=50 ⇒ 2δ+1 = 101 and architecture 101-50-50-101.
            # Paper uses δ=50 ⇒ 2δ+1=101. Override only for ablation; keep default=101 for reproduction.
            pass

        self.window_len = int(window_len)

        self.net = nn.Sequential(
            nn.Linear(self.window_len, hidden1),
            nn.Sigmoid(),
            nn.Linear(hidden1, hidden2),
            nn.Sigmoid(),
            nn.Linear(hidden2, self.window_len),
            nn.Sigmoid(),  # Output is normalized to [0,1], so sigmoid is required
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 101) in [0,1]
        returns: (B, 101) in (0,1) (sigmoid)
        """
        return self.net(x)

class SingleLayerAE(nn.Module):
    """
    Helper for layer-wise pretraining.
    Input -> Hidden -> Output(=Input)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.encoder(x))
        recon = self.act(self.decoder(h))
        return recon, h