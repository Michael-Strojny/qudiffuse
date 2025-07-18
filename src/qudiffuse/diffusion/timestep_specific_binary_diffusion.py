#!/usr/bin/env python3
"""
Timestep-Specific Binary Diffusion Process

This module implements the binary diffusion process with unique DBN per timestep
as required by the specifications. It handles:
1. Forward noising with Bernoulli-flip noise
2. Reverse denoising with timestep-specific DBNs
3. Proper scheduling alignment
4. Support for all three latent space topologies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from .schedule import BernoulliSchedule
from qudiffuse.utils.error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError

logger = logging.getLogger(__name__)

class TimestepSpecificBinaryDiffusion:
    """
    Binary diffusion process with unique DBN per timestep.
    
    This implements the paper requirement of having a separate DBN
    for each timestep transition (t+1 → t), ensuring proper denoising
    without parameter sharing between timesteps.
    """
    
    def __init__(
        self,
        timestep_dbn_manager,
        binary_latent_manager,
        betas: List[float],
        device: str = 'cpu'
    ):
        """
        Initialize timestep-specific binary diffusion.
        
        Args:
            timestep_dbn_manager: Manager for timestep-specific DBNs
            binary_latent_manager: Manager for binary latent storage
            betas: Noise schedule β_t for forward diffusion
            device: Computation device
        """
        self.timestep_dbn_manager = timestep_dbn_manager
        self.binary_latent_manager = binary_latent_manager
        self.schedule = BernoulliSchedule(betas)
        self.T = self.schedule.T
        self.device = device
        
        logger.info(f"🔥 Initialized TimestepSpecificBinaryDiffusion with {self.T} timesteps")
        logger.info(f"   Topology: {binary_latent_manager.topology_type}")
        logger.info(f"   Storage format: {binary_latent_manager.storage_format}")
    
    def forward_process(self, z_0: Union[torch.Tensor, List[torch.Tensor]], 
                       timestep: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Apply forward noising process with Bernoulli-flip noise.
        
        Args:
            z_0: Clean binary latents (tensor for flat, list for hierarchical)
            timestep: Target timestep (1 to T)
            
        Returns:
            Noisy latents at timestep t
        """
        if timestep < 1 or timestep > self.T:
            raise ValueError(f"Timestep must be in [1, {self.T}], got {timestep}")
        
        # Get noise level for this timestep
        beta_t = self.schedule.betas[timestep - 1]  # 0-indexed
        
        if isinstance(z_0, list):
            # Hierarchical topology
            z_t = []
            for z_level in z_0:
                z_t_level = self._apply_bernoulli_noise(z_level, beta_t)
                z_t.append(z_t_level)
            return z_t
        else:
            # Flat topology
            return self._apply_bernoulli_noise(z_0, beta_t)
    
    def _apply_bernoulli_noise(self, z: torch.Tensor, beta_t: float) -> torch.Tensor:
        """Apply Bernoulli bit-flip noise with probability β_t."""
        # Ensure binary input
        if z.dtype not in [torch.bool, torch.uint8]:
            z_binary = (z > 0.5).bool()
        else:
            z_binary = z.bool()
        
        # Generate flip mask
        flip_mask = torch.bernoulli(torch.full_like(z_binary.float(), beta_t))
        
        # Apply bit-flip: z_t = z_{t-1} ⊕ flip_mask
        z_noisy = z_binary ^ flip_mask.bool()
        
        # Return in same storage format as binary latent manager
        return self.binary_latent_manager.from_float_to_binary(z_noisy.float())
    
    def reverse_process(self, z_T: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run reverse denoising process using timestep-specific DBNs.
        
        Args:
            z_T: Noisy latents at timestep T
            
        Returns:
            Denoised latents at timestep 0 (clean)
        """
        z_current = z_T
        
        # Reverse process: T → T-1 → ... → 1 → 0
        for t in reversed(range(1, self.T + 1)):
            z_current = self._denoise_step(z_current, t)
        
        return z_current
    
    def _denoise_step(self, z_t: Union[torch.Tensor, List[torch.Tensor]], 
                     timestep: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Single denoising step using timestep-specific DBN.
        
        Args:
            z_t: Noisy latents at timestep t
            timestep: Current timestep (DBN trained for t → t-1)
            
        Returns:
            Denoised latents at timestep t-1
        """
        # Get timestep-specific DBN
        dbn = self.timestep_dbn_manager.get_dbn_for_inference(timestep)
        
        # Extract hierarchical channels for DBN processing
        if isinstance(z_t, list):
            # Hierarchical topology
            hierarchical_channels = self.binary_latent_manager.extract_hierarchical_channels(z_t)
        else:
            # Flat topology: split channels
            batch_size = z_t.size(0)
            channels = z_t.size(1)
            hierarchical_channels = []
            for c in range(channels):
                channel_tensor = z_t[:, c:c+1, :, :]  # [B, 1, H, W]
                hierarchical_channels.append(channel_tensor)
        
        # Maintain exact binary storage during inference
        # Use binary tensors directly with DBN processing that respects binary formats
        processed_channels = hierarchical_channels
        
        # Apply DBN denoising with proper visible unit construction
        denoised_channels = self._apply_dbn_denoising(processed_channels, dbn)
        
        # Convert back to binary storage
        binary_channels = []
        for ch in denoised_channels:
            binary_ch = self.binary_latent_manager.from_float_to_binary(ch)
            binary_channels.append(binary_ch)
        
        # Reconstruct output format
        if isinstance(z_t, list):
            # Hierarchical topology
            return self.binary_latent_manager.reconstruct_hierarchical_levels(binary_channels)
        else:
            # Flat topology: concatenate channels
            return torch.cat(binary_channels, dim=1)
    
    def _apply_dbn_denoising(self, hierarchical_channels: List[torch.Tensor], 
                           dbn) -> List[torch.Tensor]:
        """
        Apply DBN denoising with proper visible unit construction v^(ℓ) = [z^(ℓ) || h^(ℓ+1)].
        
        Args:
            hierarchical_channels: List of channel tensors
            dbn: Hierarchical DBN for this timestep
            
        Returns:
            Denoised channel tensors
        """
        num_layers = len(dbn.rbms)
        hidden_states = [None] * num_layers
        denoised_channels = []
        
        # Process layers in reverse order (top-down)
        for layer_idx in reversed(range(num_layers)):
            if layer_idx >= len(hierarchical_channels):
                continue
                
            rbm = dbn.rbms[layer_idx]
            # Maintain binary format throughout processing
            z_channel_tensor = hierarchical_channels[layer_idx]
            if z_channel_tensor.dtype in [torch.bool, torch.uint8]:
                # Convert to float only for computation, maintain exact values
                z_channel = z_channel_tensor.float().flatten(1)  # [B, H*W]
            else:
                z_channel = z_channel_tensor.flatten(1)  # [B, H*W]
            
            # Construct visible units: v^(ℓ) = [z^(ℓ) || h^(ℓ+1)]
            if layer_idx == num_layers - 1:
                # Top layer: only z^(ℓ)
                visible_input = z_channel
            else:
                # Lower layers: [z^(ℓ) || h^(ℓ+1)]
                layer_above_idx = layer_idx + 1
                if layer_above_idx < len(hidden_states) and hidden_states[layer_above_idx] is not None:
                    h_above = hidden_states[layer_above_idx]
                    visible_input = torch.cat([z_channel, h_above], dim=1)
                else:
                    visible_input = z_channel
            
            # Adjust input size to match RBM
            if visible_input.size(1) != rbm.visible_size:
                if visible_input.size(1) < rbm.visible_size:
                    pad_dim = rbm.visible_size - visible_input.size(1)
                    padding = torch.zeros(visible_input.size(0), pad_dim, 
                                        device=self.device, dtype=visible_input.dtype)
                    visible_input = torch.cat([visible_input, padding], dim=1)
                else:
                    visible_input = visible_input[:, :rbm.visible_size]
            
            # Sample from RBM
            with torch.no_grad():
                # One Gibbs step for denoising
                h_sample, _ = rbm._v_to_h(visible_input)
                v_denoised, _ = rbm._h_to_v(h_sample)
                
                # Store hidden state for lower layers
                hidden_states[layer_idx] = h_sample
                
                # Extract denoised latent part (first part of visible units)
                latent_size = z_channel.size(1)
                z_denoised = v_denoised[:, :latent_size]
                
                # Reshape back to spatial dimensions
                original_shape = hierarchical_channels[layer_idx].shape
                z_denoised_spatial = z_denoised.reshape(original_shape)
                denoised_channels.append(z_denoised_spatial)
        
        # Reverse the list to match original order
        denoised_channels.reverse()
        return denoised_channels
    
    def sample(self, shape: Union[Tuple, List[Tuple]], 
              num_samples: int = 1) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples by running full reverse process from noise.
        
        Args:
            shape: Shape of latents (tuple for flat, list of tuples for hierarchical)
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        # Initialize with pure noise
        if isinstance(shape, list):
            # Hierarchical topology
            z_T = []
            for level_shape in shape:
                full_shape = (num_samples,) + level_shape[1:]  # (B, C, H, W)
                z_level = self.binary_latent_manager.create_binary_tensor(full_shape, fill_value=0.5)
                z_T.append(z_level)
        else:
            # Flat topology
            full_shape = (num_samples,) + shape[1:]  # (B, C, H, W)
            z_T = self.binary_latent_manager.create_binary_tensor(full_shape, fill_value=0.5)
        
        # Run reverse process
        return self.reverse_process(z_T)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the diffusion process."""
        return {
            'total_timesteps': self.T,
            'num_dbns': len(self.timestep_dbn_manager.initialized_timesteps),
            'topology_type': self.binary_latent_manager.topology_type,
            'total_channels': self.binary_latent_manager.total_channels,
            'storage_format': self.binary_latent_manager.storage_format
        }
