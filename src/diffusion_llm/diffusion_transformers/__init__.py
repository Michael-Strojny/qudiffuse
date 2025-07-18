"""
Diffusion Transformers for Text Reasoning

This package contains the diffusion transformer models adapted for text latent diffusion
with reasoning capabilities, integrating our QuDiffuse binary diffusion system.
"""

from .text_binary_diffusion import TextBinaryDiffusion
from .reasoning_dit import ReasoningDiT

__all__ = [
    'TextBinaryDiffusion',
    'ReasoningDiT'
] 