"""
Improved DAE (Xiong et al., 2016)
Reimplementation for ECG baseline wander removal
"""

from .model_DAE import ImprovedDAE, SingleLayerAE

__all__ = [
    "ImprovedDAE",
    "SingleLayerAE",
]
