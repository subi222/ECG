# -*- coding: utf-8 -*-
"""
model_DAE.py

Improved DAE reimplementation (Xiong et al., 2016)
- Paper: "ECG signal enhancement based on improved denoising auto-encoder"
- DAE structure: 101-50-50-101 (δ=50 → window_len=101)
- Activations: sigmoid for hidden and output
- Loss: Bernoulli distance (cross-entropy) → BCELoss in PyTorch

IMPORTANT (explicit approximation / ablation):
- The paper describes layer-wise pretraining + global fine-tuning ("trained layer by layer... fine-tuning all weights").
  For fair baseline comparison and simplicity, this implementation trains end-to-end from scratch
  (i.e., it omits layer-wise pretraining). This choice should be documented in any report.

Normalization assumption (documented):
- The paper maps each sample vector v to x∈[0,1]^p by min-max normalization.
- Here we standardize that mapping as "per-window min-max" during dataset creation / inference:
  x_norm = (x - min_w) / (max_w - min_w + eps)
  This makes inference possible without access to clean targets.
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
            # Allow override for ablation, but warn via code comments / config.
            pass

        self.window_len = int(window_len)

        self.net = nn.Sequential(
            nn.Linear(self.window_len, hidden1),
            nn.Sigmoid(),
            nn.Linear(hidden1, hidden2),
            nn.Sigmoid(),
            nn.Linear(hidden2, self.window_len)
            #nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 101) in [0,1]
        returns: (B, 101) in (0,1) (sigmoid)
        """
        return self.net(x)