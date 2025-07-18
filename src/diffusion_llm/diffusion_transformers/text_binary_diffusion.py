"""
Text Binary Diffusion Adapter

This module adapts our QuDiffuse binary diffusion system for text latent diffusion,
enabling reasoning in binary latent space while maintaining quantum annealer compatibility.

Key Features:
- Integration with QuDiffuse binary diffusion system
- Text latent format adaptation (flat tensor conversion)
- Quantum annealer compatibility for reasoning tasks
- Support for arithmetic and spatial reasoning
- Classical fallback via Contrastive Divergence
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
import logging

# Import our QuDiffuse components
from qudiffuse.diffusion import TimestepSpecificBinaryDiffusion, UnifiedReverseProcess
from qudiffuse.models import BinaryLatentManager, HierarchicalDBN
from qudiffuse.models.timestep_specific_dbn_manager import TimestepSpecificDBNManager

logger = logging.getLogger(__name__)


class TextLatentBinaryManager:
    """
    Binary latent manager specifically for text latents.
    
    This adapts our BinaryLatentManager for text latent shapes that come from
    the BART autoencoder (flat tensors rather than spatial).
    """
    
    def __init__(self, latent_dim: int = 256, sequence_length: int = 16):
        """
        Initialize text latent binary manager.
        
        Args:
            latent_dim: Dimension of latents (dae from paper)
            sequence_length: Sequence length of latents (lae from paper)
        """
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # Create flat topology configuration for text latents
        # For text, we need to adjust dimensions to satisfy 0.5 BPP constraint
        # Total bits = sequence_length * latent_dim
        # For 0.5 BPP: total_bits <= 0.5 * img_size^2
        # So img_size^2 >= 2 * total_bits
        total_bits = sequence_length * latent_dim
        required_img_size = int((2 * total_bits) ** 0.5) + 1  # Add buffer for safety
        
        topology_config = {
            'type': 'flat',
            'channels': 1,  # Single channel for text
            'spatial_size': (sequence_length, latent_dim),  # Text sequence as spatial
            'storage_format': 'bool',
            'device': 'cpu',
            'img_size': required_img_size  # Calculated to satisfy BPP constraint
        }
        
        self.binary_manager = BinaryLatentManager(topology_config=topology_config)
        self.topology_type = 'flat'
        self.storage_format = 'bool'
        
        logger.info(f"📝 TextLatentBinaryManager initialized:")
        logger.info(f"   Latent dimension: {latent_dim}")
        logger.info(f"   Sequence length: {sequence_length}")
        logger.info(f"   Total variables: {sequence_length * latent_dim}")
    
    def text_latents_to_binary_format(self, text_latents: torch.Tensor) -> torch.Tensor:
        """
        Convert text latents [B, seq_len, dim] to binary format [B, seq_len, dim, 1].
        
        Args:
            text_latents: Text latents [B, seq_len, latent_dim]
            
        Returns:
            Binary format tensor [B, seq_len, latent_dim, 1]
        """
        # Ensure binary values
        if text_latents.dtype not in [torch.bool, torch.uint8]:
            binary_latents = (text_latents > 0.5).bool()
        else:
            binary_latents = text_latents.bool()
        
        # Add spatial dimension for compatibility
        return binary_latents.unsqueeze(-1)  # [B, seq_len, latent_dim, 1]
    
    def binary_format_to_text_latents(self, binary_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert binary format [B, seq_len, dim, 1] back to text latents [B, seq_len, dim].
        
        Args:
            binary_tensor: Binary tensor [B, seq_len, latent_dim, 1]
            
        Returns:
            Text latents [B, seq_len, latent_dim]
        """
        # Remove spatial dimension
        return binary_tensor.squeeze(-1).float()  # [B, seq_len, latent_dim]
    
    def get_dbn_layer_dimensions(self) -> List[int]:
        """Get layer dimensions for DBN initialization."""
        # Each sequence position becomes a separate layer
        return [self.latent_dim] * self.sequence_length
    
    def extract_hierarchical_channels(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extract hierarchical channels for DBN processing."""
        # Split sequence positions into separate tensors
        channels = []
        for i in range(self.sequence_length):
            channel = tensor[:, i:i+1, :, :]  # [B, 1, latent_dim, 1]
            channels.append(channel)
        return channels
    
    def reconstruct_hierarchical_levels(self, channels: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct tensor from hierarchical channels."""
        # Concatenate channels back to sequence
        return torch.cat(channels, dim=1)  # [B, seq_len, latent_dim, 1]


class TextBinaryDiffusion:
    """
    Text binary diffusion system integrating QuDiffuse with text reasoning.
    
    This class provides a complete diffusion system for text latents using our
    binary diffusion architecture with quantum annealer support for reasoning tasks.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,              # dae from paper
        sequence_length: int = 16,          # lae from paper
        num_timesteps: int = 1000,          # Number of diffusion timesteps
        beta_start: float = 0.0001,         # Start of noise schedule
        beta_end: float = 0.02,             # End of noise schedule
        device: str = 'cpu',                # Computation device
        quantum_enabled: bool = True,       # Enable quantum annealer support
        window_size: int = 4                # Window size for windowed QUBO
    ):
        """
        Initialize text binary diffusion system.
        
        Args:
            latent_dim: Dimension of text latents (dae)
            sequence_length: Sequence length of text latents (lae)
            num_timesteps: Number of diffusion timesteps
            beta_start: Start of noise schedule
            beta_end: End of noise schedule
            device: Computation device
            quantum_enabled: Enable quantum annealer support
            window_size: Window size for windowed QUBO
        """
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.num_timesteps = num_timesteps
        self.device = device
        self.quantum_enabled = quantum_enabled
        
        # Create text-specific binary latent manager
        self.text_binary_manager = TextLatentBinaryManager(latent_dim, sequence_length)
        
        # Create DBN manager for timestep-specific DBNs
        latent_shapes = [(latent_dim, 1, 1)] * sequence_length  # Treat each position as spatial
        total_channels = sum(c for c, h, w in latent_shapes)  # Calculate total channels
        self.dbn_manager = TimestepSpecificDBNManager(
            latent_shapes=latent_shapes,
            hidden_dims=[latent_dim] * total_channels,  # One hidden dim per channel
            timesteps=num_timesteps,
            device=device
        )
        
        # Create noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Initialize binary diffusion system
        self.diffusion = TimestepSpecificBinaryDiffusion(
            timestep_dbn_manager=self.dbn_manager,
            binary_latent_manager=self.text_binary_manager.binary_manager,
            betas=betas.tolist(),
            device=device
        )
        
        # Initialize unified reverse process with quantum support
        if quantum_enabled:
            from qudiffuse.diffusion.unified_reverse_process import ClassicalFallbackMode
            self.unified_process = UnifiedReverseProcess(
                timestep_dbn_manager=self.dbn_manager,
                binary_latent_manager=self.text_binary_manager.binary_manager,
                betas=betas,
                device=device,
                classical_fallback_mode=ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE,
                window_size=window_size
            )
        else:
            self.unified_process = None
        
        logger.info(f"🔥 TextBinaryDiffusion initialized:")
        logger.info(f"   Latent shape: [{sequence_length}, {latent_dim}]")
        logger.info(f"   Timesteps: {num_timesteps}")
        logger.info(f"   Quantum enabled: {quantum_enabled}")
        logger.info(f"   Total DBN layers: {len(latent_shapes)}")
    
    def forward_process(
        self, 
        text_latents: torch.Tensor, 
        timestep: int
    ) -> torch.Tensor:
        """
        Apply forward noising process to text latents.
        
        Args:
            text_latents: Text latents [B, seq_len, latent_dim]
            timestep: Target timestep (1 to T)
            
        Returns:
            Noisy text latents [B, seq_len, latent_dim]
        """
        # Convert to binary format
        binary_format = self.text_binary_manager.text_latents_to_binary_format(text_latents)
        
        # Apply binary diffusion forward process
        noisy_binary = self.diffusion.forward_process(binary_format, timestep)
        
        # Convert back to text format
        return self.text_binary_manager.binary_format_to_text_latents(noisy_binary)
    
    def reverse_process_classical(
        self, 
        noisy_text_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply reverse denoising process using classical methods.
        
        Args:
            noisy_text_latents: Noisy text latents [B, seq_len, latent_dim]
            
        Returns:
            Denoised text latents [B, seq_len, latent_dim]
        """
        # Convert to binary format
        binary_format = self.text_binary_manager.text_latents_to_binary_format(noisy_text_latents)
        
        # Apply binary diffusion reverse process
        clean_binary = self.diffusion.reverse_process(binary_format)
        
        # Convert back to text format
        return self.text_binary_manager.binary_format_to_text_latents(clean_binary)
    
    def reverse_process_quantum(
        self, 
        noisy_text_latents: torch.Tensor,
        mode: str = 'windowed_zephyr',
        **kwargs
    ) -> torch.Tensor:
        """
        Apply reverse denoising process using quantum annealing.
        
        Args:
            noisy_text_latents: Noisy text latents [B, seq_len, latent_dim]
            mode: Quantum sampling mode
            **kwargs: Additional parameters for quantum solver
            
        Returns:
            Denoised text latents [B, seq_len, latent_dim]
        """
        if not self.quantum_enabled or self.unified_process is None:
            logger.warning("Quantum not enabled, falling back to classical")
            return self.reverse_process_classical(noisy_text_latents)
        
        # Convert to binary format
        binary_format = self.text_binary_manager.text_latents_to_binary_format(noisy_text_latents)
        
        # Map mode to unified process
        from qudiffuse.diffusion.unified_reverse_process import SamplingMode
        mode_map = {
            'classical_cd': SamplingMode.CLASSICAL_CD,
            'qubo_classical': SamplingMode.QUBO_CLASSICAL,
            'qubo_quantum': SamplingMode.QUBO_QUANTUM_ZEPHYR,
            'windowed_zephyr': SamplingMode.QUBO_WINDOWED_ZEPHYR
        }
        
        sampling_mode = mode_map.get(mode, SamplingMode.QUBO_WINDOWED_ZEPHYR)
        
        # Apply unified reverse process
        clean_binary = self.unified_process.sample_reverse_process(
            binary_format,
            mode=sampling_mode,
            **kwargs
        )
        
        # Convert back to text format
        return self.text_binary_manager.binary_format_to_text_latents(clean_binary)
    
    def sample_reasoning_latents(
        self, 
        batch_size: int = 1,
        use_quantum: bool = True,
        reasoning_mode: str = 'windowed_zephyr'
    ) -> torch.Tensor:
        """
        Sample text latents for reasoning tasks.
        
        Args:
            batch_size: Number of samples to generate
            use_quantum: Whether to use quantum annealing
            reasoning_mode: Quantum reasoning mode
            
        Returns:
            Sampled text latents [B, seq_len, latent_dim]
        """
        # Create random noise
        noise_shape = (batch_size, self.sequence_length, self.latent_dim)
        noise = torch.randint(0, 2, noise_shape, dtype=torch.float32, device=self.device)
        
        # Apply reverse process
        if use_quantum and self.quantum_enabled:
            clean_latents = self.reverse_process_quantum(noise, mode=reasoning_mode)
        else:
            clean_latents = self.reverse_process_classical(noise)
        
        return clean_latents
    
    def reasoning_diffusion_step(
        self,
        latents: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Apply multiple reasoning diffusion steps for iterative reasoning.
        
        Args:
            latents: Current latents [B, seq_len, latent_dim]
            conditioning: Optional conditioning information
            num_steps: Number of diffusion steps to apply
            
        Returns:
            Refined latents after reasoning steps
        """
        current_latents = latents
        
        # Apply iterative refinement through partial denoising
        for step in range(num_steps):
            # Add small amount of noise
            timestep = max(1, self.num_timesteps // (step + 2))
            noisy_latents = self.forward_process(current_latents, timestep)
            
            # Denoise with quantum reasoning
            if self.quantum_enabled:
                current_latents = self.reverse_process_quantum(
                    noisy_latents, 
                    mode='windowed_zephyr',
                    num_reads=100  # Fewer reads for iterative steps
                )
            else:
                current_latents = self.reverse_process_classical(noisy_latents)
        
        return current_latents
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the diffusion system."""
        stats = self.diffusion.get_performance_stats()
        
        # Add text-specific stats
        stats.update({
            'text_latent_dim': self.latent_dim,
            'text_sequence_length': self.sequence_length,
            'quantum_enabled': self.quantum_enabled,
            'total_binary_variables': self.sequence_length * self.latent_dim
        })
        
        if self.unified_process is not None:
            stats.update({
                'quantum_mode_usage': dict(self.unified_process.mode_usage_count),
                'fallback_usage': dict(self.unified_process.fallback_usage_count)
            })
        
        return stats 