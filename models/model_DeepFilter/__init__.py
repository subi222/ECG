"""
DeepFilter Model Package
ECG Baseline Wander Removal using Deep Learning

Keras/TensorFlow implementation (original from Francisco Perdigon Romero)
"""

# Keras version uses functional models, not classes
# Import module itself for access to model functions
from . import deepfilter

__all__ = ['deepfilter']
