"""
QuDiffuse Models - Clean Production Implementation

This module contains the core models for QuDiffuse:
- BinaryAutoEncoder: Binary autoencoder from BinaryLatentDiffusion
- MultiResolutionBinaryAutoEncoder: Enhanced version with multi-resolution support
- HierarchicalDBN: Deep Belief Network for reverse diffusion
- Binary Latent Manager: Efficient latent space management
"""

from .binaryae import BinaryAutoEncoder, BinaryQuantizer
from .multi_resolution_binary_ae import MultiResolutionBinaryAutoEncoder
from .dbn import HierarchicalDBN, RBM
from .binary_latent_manager import BinaryLatentManager

__all__ = [
    'BinaryAutoEncoder',
    'BinaryQuantizer',
    'MultiResolutionBinaryAutoEncoder',
    'HierarchicalDBN', 
    'RBM',
    'BinaryLatentManager'
]