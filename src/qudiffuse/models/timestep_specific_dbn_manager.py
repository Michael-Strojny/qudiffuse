#!/usr/bin/env python3
"""
Timestep-Specific DBN Manager for QuDiffuse

This module manages unique DBNs for each timestep transition (t+1 → t),
ensuring that each timestep has its own trained denoising transformation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from .dbn import HierarchicalDBN
from qudiffuse.utils.error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError
from qudiffuse.utils.common_utils import validate_tensor_shape, ensure_device_consistency, cleanup_gpu_memory

logger = logging.getLogger(__name__)

class TimestepSpecificDBNManager:
    """
    Manages unique DBNs for each timestep transition in the diffusion process.
    
    This ensures that:
    1. Each timestep pair (t+1 → t) has its own trained DBN
    2. DBNs are properly initialized and managed
    3. Training and inference use the correct timestep-specific DBN
    4. Memory usage is optimized for large timestep counts
    """
    
    def __init__(self, 
                 latent_shapes: List[Tuple[int, int, int]],
                 hidden_dims: List[int],
                 timesteps: int,
                 device: str = 'cpu',
                 connectivity: str = 'full',
                 max_cached_dbns: int = 50,
                 memory_limit_gb: float = 2.0):
        """
        Args:
            latent_shapes: List of (C, H, W) shapes for each pyramid level
            hidden_dims: Hidden dimensions for each DBN layer (one per channel)
            timesteps: Total number of diffusion timesteps
            device: Device to place DBNs on
            connectivity: DBN connectivity type ('full', 'pegasus', etc.)
            max_cached_dbns: Maximum number of DBNs to keep in memory
            memory_limit_gb: Memory limit for automatic cleanup
        """
        self.latent_shapes = latent_shapes
        self.hidden_dims = hidden_dims
        self.timesteps = timesteps
        self.device = device
        self.connectivity = connectivity
        self.max_cached_dbns = max_cached_dbns
        self.memory_limit_gb = memory_limit_gb
        
        # Calculate total channels for validation
        self.total_channels = sum(c for c, h, w in latent_shapes)
        
        # Validate hidden dimensions
        if len(hidden_dims) != self.total_channels:
            raise ConfigurationError(
                f"Number of hidden dimensions ({len(hidden_dims)}) must match "
                f"total number of channels ({self.total_channels})",
                "timestep_dbn_manager"
            )
        
        # Dictionary to store timestep-specific DBNs
        self.timestep_dbns: Dict[int, HierarchicalDBN] = {}
        
        # Track which timesteps have been initialized and access order
        self.initialized_timesteps = set()
        self.access_order = []  # LRU cache for automatic cleanup
        
        logger.info(f"🕐 Initializing Timestep-Specific DBN Manager:")
        logger.info(f"   Total Timesteps: {timesteps}")
        logger.info(f"   Latent Shapes: {latent_shapes}")
        logger.info(f"   Total Channels: {self.total_channels}")
        logger.info(f"   Hidden Dims per Channel: {hidden_dims}")
        logger.info(f"   Connectivity: {connectivity}")
        logger.info(f"   Device: {device}")
    
    def get_or_create_dbn(self, timestep: int) -> HierarchicalDBN:
        """
        Get the DBN for a specific timestep, creating it if it doesn't exist.
        
        Args:
            timestep: The timestep for which to get/create the DBN
            
        Returns:
            The HierarchicalDBN for the specified timestep
        """
        if timestep < 1 or timestep > self.timesteps:
            raise ValueError(f"Timestep {timestep} out of range [1, {self.timesteps}]")
        
        if timestep not in self.timestep_dbns:
            # Check if we need to free memory before creating new DBN
            self._check_and_cleanup_memory()
            
            logger.info(f"🏗️ Creating DBN for timestep {timestep}")
            
            # Create new DBN for this timestep using proper latent_shapes
            dbn = HierarchicalDBN(
                latent_shapes=self.latent_shapes,
                hidden_dims=self.hidden_dims,
                device=self.device
            )
            
            self.timestep_dbns[timestep] = dbn
            self.initialized_timesteps.add(timestep)
            
            logger.info(f"✅ DBN for timestep {timestep} created successfully")
        
        # Update access order for LRU management
        if timestep in self.access_order:
            self.access_order.remove(timestep)
        self.access_order.append(timestep)
        
        return self.timestep_dbns[timestep]
    
    def _check_and_cleanup_memory(self):
        """Check memory usage and perform cleanup if necessary."""
        # Check if we exceed the maximum number of cached DBNs
        if len(self.timestep_dbns) >= self.max_cached_dbns:
            # Remove least recently used DBNs
            num_to_remove = len(self.timestep_dbns) - self.max_cached_dbns + 1
            for _ in range(num_to_remove):
                if self.access_order:
                    lru_timestep = self.access_order.pop(0)
                    if lru_timestep in self.timestep_dbns:
                        del self.timestep_dbns[lru_timestep]
                        self.initialized_timesteps.discard(lru_timestep)
                        logger.info(f"🧹 Cleaned up LRU DBN for timestep {lru_timestep}")
            
            # Clear GPU cache
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        # Check memory usage if on GPU
        if self.device.startswith('cuda'):
            try:
                import psutil
                memory_usage_gb = psutil.virtual_memory().used / (1024**3)
                if memory_usage_gb > self.memory_limit_gb:
                    # More aggressive cleanup
                    half_cache = max(1, self.max_cached_dbns // 2)
                    while len(self.timestep_dbns) > half_cache and self.access_order:
                        lru_timestep = self.access_order.pop(0)
                        if lru_timestep in self.timestep_dbns:
                            del self.timestep_dbns[lru_timestep]
                            self.initialized_timesteps.discard(lru_timestep)
                    
                    torch.cuda.empty_cache()
                    logger.warning(f"⚠️ Memory cleanup triggered: {memory_usage_gb:.1f}GB > {self.memory_limit_gb}GB")
            except ImportError:
                pass  # psutil not available

    def initialize_all_timesteps(self):
        """Initialize DBNs for all timesteps at once."""
        logger.info(f"🚀 Initializing DBNs for all {self.timesteps} timesteps...")
        
        for timestep in range(1, self.timesteps + 1):
            self.get_or_create_dbn(timestep)
        
        logger.info(f"✅ All {self.timesteps} timestep-specific DBNs initialized")
    
    def get_dbn_for_training(self, timestep: int) -> HierarchicalDBN:
        """
        Get the DBN for training a specific timestep transition.
        
        Args:
            timestep: The target timestep (for t+1 → t transition)
            
        Returns:
            The DBN that learns to denoise from timestep+1 to timestep
        """
        return self.get_or_create_dbn(timestep)
    
    def get_dbn_for_inference(self, current_timestep: int) -> HierarchicalDBN:
        """
        Get the DBN for inference at a specific timestep.
        
        Args:
            current_timestep: Current timestep in reverse diffusion
            
        Returns:
            The DBN that denoises from current_timestep to current_timestep-1
        """
        if current_timestep < 1:
            raise ValueError(f"Cannot perform inference at timestep {current_timestep}")
        
        return self.get_or_create_dbn(current_timestep)
    
    def get_optimizers_for_timestep(self, timestep: int, learning_rate: float = 1e-3) -> List[torch.optim.Optimizer]:
        """
        Get optimizers for all RBMs in a specific timestep's DBN.
        
        Args:
            timestep: The timestep for which to create optimizers
            learning_rate: Learning rate for the optimizers
            
        Returns:
            List of optimizers, one per RBM layer
        """
        dbn = self.get_or_create_dbn(timestep)
        
        optimizers = []
        for rbm in dbn.rbms:
            optimizer = torch.optim.Adam(rbm.parameters(), lr=learning_rate)
            optimizers.append(optimizer)
        
        return optimizers
    
    def save_timestep_dbn(self, timestep: int, filepath: str):
        """Save a specific timestep's DBN to disk."""
        if timestep not in self.timestep_dbns:
            raise ValueError(f"DBN for timestep {timestep} not initialized")
        
        dbn = self.timestep_dbns[timestep]
        torch.save(dbn.state_dict(), filepath)
        logger.info(f"💾 Saved DBN for timestep {timestep} to {filepath}")
    
    def load_timestep_dbn(self, timestep: int, filepath: str):
        """Load a specific timestep's DBN from disk."""
        dbn = self.get_or_create_dbn(timestep)
        dbn.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"📂 Loaded DBN for timestep {timestep} from {filepath}")
    
    def save_all_timesteps(self, base_filepath: str):
        """Save all initialized timestep DBNs to disk."""
        for timestep in self.initialized_timesteps:
            filepath = f"{base_filepath}_timestep_{timestep}.pth"
            self.save_timestep_dbn(timestep, filepath)
        
        logger.info(f"💾 Saved {len(self.initialized_timesteps)} timestep DBNs")
    
    def load_all_timesteps(self, base_filepath: str):
        """Load all timestep DBNs from disk."""
        loaded_count = 0
        
        for timestep in range(1, self.timesteps + 1):
            filepath = f"{base_filepath}_timestep_{timestep}.pth"
            try:
                self.load_timestep_dbn(timestep, filepath)
                loaded_count += 1
            except FileNotFoundError:
                logger.warning(f"⚠️ DBN file for timestep {timestep} not found: {filepath}")
            except Exception as e:
                logger.error(f"❌ Error loading DBN for timestep {timestep}: {e}")
        
        logger.info(f"📂 Loaded {loaded_count}/{self.timesteps} timestep DBNs")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for all timestep DBNs."""
        total_params = 0
        timestep_params = {}
        
        for timestep, dbn in self.timestep_dbns.items():
            params = sum(p.numel() for p in dbn.parameters())
            timestep_params[timestep] = params
            total_params += params
        
        return {
            'total_parameters': total_params,
            'initialized_timesteps': len(self.initialized_timesteps),
            'parameters_per_timestep': timestep_params,
            'average_params_per_timestep': total_params // max(1, len(self.initialized_timesteps))
        }
    
    def cleanup_unused_timesteps(self, keep_timesteps: List[int]):
        """Remove DBNs for timesteps not in the keep list to free memory."""
        to_remove = []
        
        for timestep in self.timestep_dbns.keys():
            if timestep not in keep_timesteps:
                to_remove.append(timestep)
        
        for timestep in to_remove:
            del self.timestep_dbns[timestep]
            self.initialized_timesteps.discard(timestep)
        
        if to_remove:
            logger.info(f"🧹 Cleaned up DBNs for timesteps: {to_remove}")
            torch.cuda.empty_cache()
    
    def get_summary(self) -> str:
        """Get a summary of the timestep-specific DBN manager."""
        memory_info = self.get_memory_usage()
        
        summary = f"Timestep-Specific DBN Manager Summary:\n"
        summary += f"  Total Timesteps: {self.timesteps}\n"
        summary += f"  Latent Shapes: {self.latent_shapes}\n"
        summary += f"  Total Channels: {self.total_channels}\n"
        summary += f"  Hidden Dims: {self.hidden_dims}\n"
        summary += f"  Initialized Timesteps: {len(self.initialized_timesteps)}\n"
        summary += f"  Total Parameters: {memory_info['total_parameters']:,}\n"
        summary += f"  Average Parameters per Timestep: {memory_info['average_params_per_timestep']:,}\n"
        summary += f"  Connectivity: {self.connectivity}\n"
        summary += f"  Device: {self.device}\n"
        
        if self.initialized_timesteps:
            summary += f"  Initialized Timesteps: {sorted(self.initialized_timesteps)}\n"
        
        return summary
