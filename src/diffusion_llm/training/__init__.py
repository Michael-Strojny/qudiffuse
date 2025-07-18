"""
Training Pipeline for Diffusion LLM

This package implements the two-stage training pipeline for "Latent Diffusion with LLMs for Reasoning":

Stage 1: BART Autoencoder Training
- Fine-tune BART encoder-decoder with Perceiver autoencoder
- Learn variable-length to fixed-length compression
- Optimize reconstruction quality

Stage 2: Latent Diffusion Training  
- Train Reasoning DiT in binary latent space
- Use frozen autoencoder from Stage 1
- Enable quantum annealer-compatible reasoning
"""

from .stage1_autoencoder_trainer import Stage1AutoencoderTrainer
from .stage2_diffusion_trainer import Stage2DiffusionTrainer
from .unified_trainer import UnifiedDiffusionLLMTrainer

__all__ = [
    'Stage1AutoencoderTrainer',
    'Stage2DiffusionTrainer', 
    'UnifiedDiffusionLLMTrainer'
] 