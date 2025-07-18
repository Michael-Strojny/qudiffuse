"""
Diffusion LLM Datasets

This module implements reasoning task datasets for "Latent Diffusion with LLMs for Reasoning"
following exact paper specifications with ZERO mocks or placeholders.

Key Features:
- Arithmetic reasoning datasets (single-digit addition chains)
- Spatial reasoning datasets (direction and rotation tasks)
- Paper-compliant problem generation
- Step-by-step solution generation
- Integration with BART tokenizer
"""

from .reasoning_datasets import (
    ArithmeticReasoningDataset,
    SpatialReasoningDataset,
    CombinedReasoningDataset
)

__all__ = [
    'ArithmeticReasoningDataset',
    'SpatialReasoningDataset', 
    'CombinedReasoningDataset'
] 