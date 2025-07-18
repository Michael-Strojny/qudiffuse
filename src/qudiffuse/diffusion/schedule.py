#!/usr/bin/env python3
"""
Bernoulli Noise Schedule for Binary Diffusion

This module implements the noise schedule for binary latent diffusion
using Bernoulli bit-flip noise as specified in the paper.
"""

import torch
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class BernoulliSchedule:
    """
    Bernoulli noise schedule for binary diffusion.
    
    Implements the forward diffusion process:
    q(z_t | z_{t-1}) = Bernoulli(z_t; (1-β_t) * z_{t-1} + β_t * (1-z_{t-1}))
    
    This corresponds to bit-flip noise where each bit has probability β_t
    of being flipped from its previous value.
    """
    
    def __init__(self, betas: Union[List[float], torch.Tensor]):
        """
        Initialize Bernoulli noise schedule.
        
        Args:
            betas: Noise schedule β_t for each timestep
        """
        if isinstance(betas, list):
            self.betas = torch.tensor(betas, dtype=torch.float32)
        else:
            self.betas = betas.clone().detach()
        
        self.num_timesteps = len(self.betas)
        self.T = self.num_timesteps  # Add T attribute for compatibility
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For reverse process
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        logger.info(f"📅 Initialized Bernoulli schedule with {self.num_timesteps} timesteps")
        logger.info(f"   Beta range: [{self.betas.min():.4f}, {self.betas.max():.4f}]")
    
    def add_noise(self, z_0: torch.Tensor, timestep: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add Bernoulli noise to binary latents.
        
        Args:
            z_0: Clean binary latents [B, ...]
            timestep: Timestep (1-indexed)
            noise: Optional pre-generated noise
        
        Returns:
            Noisy binary latents z_t
        """
        if timestep < 1 or timestep > self.num_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [1, {self.num_timesteps}]")
        
        beta_t = self.betas[timestep - 1]  # Convert to 0-indexed
        
        if noise is None:
            # Generate Bernoulli noise with probability β_t
            noise = torch.bernoulli(torch.full_like(z_0.float(), beta_t))
        
        # Apply bit-flip noise: z_t = z_0 ⊕ noise
        z_t = z_0.float() * (1 - noise) + (1 - z_0.float()) * noise
        
        # Ensure binary output
        return z_t.bool().float()
    
    def sample_noise(self, shape: torch.Size, timestep: int, device: torch.device) -> torch.Tensor:
        """
        Sample Bernoulli noise for a given timestep.
        
        Args:
            shape: Shape of noise tensor
            timestep: Timestep (1-indexed)
            device: Device for tensor
        
        Returns:
            Bernoulli noise tensor
        """
        if timestep < 1 or timestep > self.num_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [1, {self.num_timesteps}]")
        
        beta_t = self.betas[timestep - 1]  # Convert to 0-indexed
        return torch.bernoulli(torch.full(shape, beta_t, device=device))
    
    def get_beta(self, timestep: int) -> float:
        """Get β_t for a specific timestep."""
        if timestep < 1 or timestep > self.num_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [1, {self.num_timesteps}]")
        return self.betas[timestep - 1].item()
    
    def get_alpha(self, timestep: int) -> float:
        """Get α_t = 1 - β_t for a specific timestep."""
        return 1.0 - self.get_beta(timestep)
    
    def get_alpha_cumprod(self, timestep: int) -> float:
        """Get cumulative product of alphas up to timestep."""
        if timestep < 1 or timestep > self.num_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [1, {self.num_timesteps}]")
        return self.alphas_cumprod[timestep - 1].item()
    
    def posterior_mean_coeff1(self, timestep: int) -> float:
        """Coefficient for z_{t-1} in posterior mean."""
        if timestep < 1 or timestep > self.num_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [1, {self.num_timesteps}]")
        
        alpha_t = self.alphas[timestep - 1]
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[timestep - 1]
        
        return (alpha_t * alpha_cumprod_t_prev) / self.alphas_cumprod[timestep - 1]
    
    def posterior_mean_coeff2(self, timestep: int) -> float:
        """Coefficient for z_t in posterior mean."""
        if timestep < 1 or timestep > self.num_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [1, {self.num_timesteps}]")
        
        beta_t = self.betas[timestep - 1]
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[timestep - 1]
        
        return beta_t * alpha_cumprod_t_prev / self.alphas_cumprod[timestep - 1]
    
    def to(self, device: torch.device):
        """Move schedule tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        return self


# Alias for compatibility
BernoulliNoiseSchedule = BernoulliSchedule


def create_linear_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> BernoulliSchedule:
    """Create a linear noise schedule."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    return BernoulliSchedule(betas)


def create_cosine_schedule(num_timesteps: int, s: float = 0.008) -> BernoulliSchedule:
    """Create a cosine noise schedule."""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return BernoulliSchedule(torch.clamp(betas, 0.0001, 0.9999))
