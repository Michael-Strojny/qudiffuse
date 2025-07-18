"""
Models module for Diffusion LLM implementation.

Contains all model components for the "Latent Diffusion with LLMs for Reasoning" system:
- BART Binary Autoencoder integration
- Perceiver AutoEncoder components  
- Text Binary Diffusion adapters
- Reasoning DiT (Diffusion Transformer)

ZERO mocks, ZERO simplifications, ZERO placeholders.
"""

# Import core model components
from ..encoders.bart_autoencoder import BARTBinaryAutoEncoder
from ..encoders.perceiver_ae import PerceiverBinaryAutoEncoder
from ..diffusion_transformers.text_binary_diffusion import TextBinaryDiffusion
from ..diffusion_transformers.reasoning_dit import ReasoningDiT

# Import utilities and managers
from .tokenizer_wrapper import BARTTokenizerWrapper
from .model_manager import DiffusionLLMModelManager

__all__ = [
    'BARTBinaryAutoEncoder',
    'PerceiverBinaryAutoEncoder', 
    'TextBinaryDiffusion',
    'ReasoningDiT',
    'BARTTokenizerWrapper',
    'DiffusionLLMModelManager'
] 