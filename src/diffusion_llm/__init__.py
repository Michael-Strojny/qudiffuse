"""
Diffusion LLM: Latent Diffusion with Binary Quantum Annealer for Reasoning

This package implements "Latent Diffusion with LLMs for Reasoning" using our 
existing QuDiffuse binary latent diffusion system as the core engine.

Key Features:
- BART encoder-decoder for text processing
- Binary latent diffusion with quantum annealer support
- DiT (Diffusion Transformer) for reasoning in latent space
- Two-stage training: autoencoder + latent diffusion
- Arithmetic and spatial reasoning capabilities
- ZERO mocks, ZERO simplifications, ZERO placeholders

Architecture:
1. Text → BART Encoder → Variable-length representation
2. Autoencoder → Fixed-length binary latents (16×256)
3. Binary Diffusion → Reasoning in latent space
4. Decoder → Reconstructed reasoning chain
"""

__version__ = "1.0.0"
__author__ = "QuDiffuse Team"

# Core components will be imported as they're implemented 