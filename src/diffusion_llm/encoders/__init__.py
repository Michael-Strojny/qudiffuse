"""
Encoder modules for Diffusion LLM

This package contains text encoders and autoencoder components for converting
between text sequences and binary latent representations compatible with our
QuDiffuse binary diffusion system.
"""

from .bart_autoencoder import BARTBinaryAutoEncoder
from .perceiver_ae import PerceiverBinaryAutoEncoder

__all__ = [
    'BARTBinaryAutoEncoder',
    'PerceiverBinaryAutoEncoder'
] 