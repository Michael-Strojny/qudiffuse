#!/usr/bin/env python3
"""
Binary Latent Manager for Exact Binary Storage and Topology Support

This module provides exact binary latent storage and supports all three
latent space topologies without code modification:
1. Hierarchical single-channel (multiple resolutions, one channel per level)
2. Flat multi-channel (one resolution, multiple channels)
3. Hierarchical multi-channel (multiple resolutions, multiple channels per level)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union, Optional, Any
import logging
import numpy as np
from qudiffuse.utils.error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError

logger = logging.getLogger(__name__)

class BinaryLatentManager:
    """
    Manages exact binary latent storage and topology configurations.
    
    Key features:
    - Exact binary storage (bool/uint8/packed bits)
    - Support for all three topology configurations
    - Proper spatial downsampling validation
    - Memory-efficient operations
    - Gradient-compatible operations for training
    - Enforces 0.5 bits per pixel constraint
    """
    
    def __init__(self, 
                 latent_shapes: List[Tuple[int, int, int]] = None,
                 topology_type: str = "hierarchical",
                 storage_format: str = "binary_tensor",
                 device: str = "cpu",
                 img_size: int = 32,
                 topology_config: Dict[str, Any] = None):
        """
        Initialize binary latent manager with topology configuration.
        
        Args:
            latent_shapes: List of (C, H, W) shapes for each pyramid level
            topology_type: Type of topology ('hierarchical', 'flat', 'mixed')
            storage_format: Storage format ('binary_tensor', 'bool', 'uint8', 'packed')
            device: Device for tensor operations
            img_size: Original image size for BPP constraint calculation
            topology_config: Legacy config dict (deprecated, use direct params)
        """
        # Handle both new interface (latent_shapes) and legacy interface (topology_config)
        if latent_shapes is not None:
            # New interface - convert to internal format
            self.latent_shapes = latent_shapes
            self.topology_type = topology_type
            self.storage_format = storage_format
            self.device = device
            self.img_size = img_size
            
            # Create internal topology config
            self.topology_config = {
                'type': topology_type,
                'storage_format': storage_format,
                'device': device,
                'levels': []
            }
            
            for c, h, w in latent_shapes:
                self.topology_config['levels'].append({
                    'channels': c,
                    'spatial_size': (h, w)
                })
                
        elif topology_config is not None:
            # Legacy interface
            self.topology_config = topology_config
            self.topology_type = topology_config['type']
            self.storage_format = topology_config.get('storage_format', 'binary_tensor')
            self.device = topology_config.get('device', 'cpu')
            self.img_size = topology_config.get('img_size', 32)
            
            # Extract latent_shapes from config
            if self.topology_type == 'flat':
                channels = topology_config['channels']
                spatial_size = topology_config['spatial_size']
                self.latent_shapes = [(channels, spatial_size[0], spatial_size[1])]
            else:
                self.latent_shapes = []
                for level in topology_config['levels']:
                    c = level['channels']
                    h, w = level['spatial_size']
                    self.latent_shapes.append((c, h, w))
        else:
            raise ValueError("Must provide either latent_shapes or topology_config")
        
        # Validate and parse configuration
        self._validate_topology_config()
        self._validate_bpp_constraint()
        self._parse_topology_structure()
        
        logger.info(f"🔧 Initialized BinaryLatentManager: {self.topology_type} topology")
        logger.info(f"   Total levels: {self.num_levels}")
        logger.info(f"   Total channels: {self.total_channels}")
        logger.info(f"   Storage format: {self.storage_format}")
        logger.info(f"   BPP: {self.actual_bpp:.3f} ≤ 0.5 ✓" if self.actual_bpp <= 0.5 else f"   BPP: {self.actual_bpp:.3f} > 0.5 ✗")
    
    def _validate_bpp_constraint(self):
        """Validate that latent configuration satisfies 0.5 bits per pixel constraint."""
        total_pixels = self.img_size * self.img_size
        total_bits = sum(c * h * w for c, h, w in self.latent_shapes)
        self.actual_bpp = total_bits / total_pixels
        
        if self.actual_bpp > 0.5:
            raise ConfigurationError(
                f"BPP constraint violated: {self.actual_bpp:.4f} > 0.5. "
                f"Total bits: {total_bits}, Total pixels: {total_pixels}. "
                f"Reduce latent dimensions to satisfy constraint.",
                "binary_latent_manager"
            )
    
    def _validate_topology_config(self):
        """Validate topology configuration meets specifications."""
        if self.topology_type not in ['hierarchical', 'flat', 'mixed']:
            raise ValueError(f"Invalid topology type: {self.topology_type}")
        
        if self.topology_type != 'flat' and len(self.latent_shapes) > 1:
            # Validate spatial downsampling for hierarchical/mixed
            for i in range(1, len(self.latent_shapes)):
                prev_shape = self.latent_shapes[i-1]
                curr_shape = self.latent_shapes[i]
                
                prev_h, prev_w = prev_shape[1], prev_shape[2]
                curr_h, curr_w = curr_shape[1], curr_shape[2]
                
                # Check integer downsampling factor >= 2
                if prev_h % curr_h != 0 or prev_w % curr_w != 0:
                    raise ValueError(f"Level {i} spatial size ({curr_h}, {curr_w}) is not integer divisor of level {i-1} size ({prev_h}, {prev_w})")
                
                factor_h = prev_h // curr_h
                factor_w = prev_w // curr_w
                if factor_h < 2 or factor_w < 2:
                    raise ValueError(f"Downsampling factor must be >= 2, got {factor_h}x{factor_w}")
    
    def _parse_topology_structure(self):
        """Parse topology structure into internal representation."""
        self.num_levels = len(self.latent_shapes)
        self.level_info = []
        
        for i, (c, h, w) in enumerate(self.latent_shapes):
            self.level_info.append({
                'channels': c,
                'spatial_size': (h, w),
                'level_idx': i
            })
        
        # Calculate total channels and create channel-to-level mapping
        self.total_channels = sum(c for c, h, w in self.latent_shapes)
        self.channel_to_level = {}
        self.level_to_channels = {}
        
        channel_idx = 0
        for level_idx, (c, h, w) in enumerate(self.latent_shapes):
            self.level_to_channels[level_idx] = list(range(channel_idx, channel_idx + c))
            
            for ch in range(c):
                self.channel_to_level[channel_idx + ch] = level_idx
            
            channel_idx += c
    
    def create_binary_tensor(self, shape: Tuple[int, ...], fill_value: float = 0.5) -> torch.Tensor:
        """
        Create binary tensor with exact binary storage.
        
        Args:
            shape: Tensor shape
            fill_value: Fill probability for random initialization
            
        Returns:
            Binary tensor with exact storage format
        """
        if self.storage_format in ['binary_tensor', 'bool']:
            if fill_value == 0.0:
                return torch.zeros(shape, dtype=torch.bool, device=self.device)
            elif fill_value == 1.0:
                return torch.ones(shape, dtype=torch.bool, device=self.device)
            else:
                return torch.bernoulli(torch.full(shape, fill_value, device=self.device)).bool()
        
        elif self.storage_format == 'uint8':
            if fill_value == 0.0:
                return torch.zeros(shape, dtype=torch.uint8, device=self.device)
            elif fill_value == 1.0:
                return torch.ones(shape, dtype=torch.uint8, device=self.device)
            else:
                return torch.bernoulli(torch.full(shape, fill_value, device=self.device)).to(torch.uint8)
        
        elif self.storage_format == 'packed':
            # Implement true packed format: pack 8 bits per byte
            # Calculate required shape for packed representation
            total_elements = 1
            for dim in shape:
                total_elements *= dim
            
            # Round up to nearest multiple of 8 for bit packing
            packed_size = (total_elements + 7) // 8
            
            if fill_value == 0.0:
                packed_tensor = torch.zeros(packed_size, dtype=torch.uint8, device=self.device)
            elif fill_value == 1.0:
                packed_tensor = torch.full((packed_size,), 255, dtype=torch.uint8, device=self.device)
            else:
                # Generate random bits and pack them
                random_bits = torch.bernoulli(torch.full((total_elements,), fill_value, device=self.device))
                packed_tensor = torch.zeros(packed_size, dtype=torch.uint8, device=self.device)
                
                # Pack 8 bits into each byte
                for i in range(packed_size):
                    byte_val = 0
                    for bit in range(8):
                        idx = i * 8 + bit
                        if idx < total_elements and random_bits[idx] > 0.5:
                            byte_val |= (1 << bit)
                    packed_tensor[i] = byte_val
            
            # Store original shape as metadata
            packed_tensor._original_shape = shape
            return packed_tensor
        
        else:
            raise ValueError(f"Unknown storage format: {self.storage_format}")
    
    def to_float_for_gradients(self, binary_tensor: torch.Tensor) -> torch.Tensor:
        """Convert binary tensor to float for gradient computation while preserving exact values."""
        return binary_tensor.float()
    
    def from_float_to_binary(self, float_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert float tensor back to exact binary storage."""
        binary = (float_tensor > threshold)
        
        if self.storage_format in ['binary_tensor', 'bool']:
            return binary.bool()
        elif self.storage_format == 'uint8':
            return binary.to(torch.uint8)
        else:
            return binary.to(torch.uint8)  # alternative format

    def unpack_binary_tensor(self, packed_tensor: torch.Tensor) -> torch.Tensor:
        """Unpack a packed binary tensor back to original shape."""
        if not hasattr(packed_tensor, '_original_shape'):
            raise ValueError("Packed tensor missing original shape metadata")
        
        original_shape = packed_tensor._original_shape
        total_elements = 1
        for dim in original_shape:
            total_elements *= dim
            
        # Unpack bits from bytes
        unpacked_bits = torch.zeros(total_elements, dtype=torch.bool, device=self.device)
        
        for i in range(len(packed_tensor)):
            byte_val = packed_tensor[i].item()
            for bit in range(8):
                idx = i * 8 + bit
                if idx < total_elements:
                    unpacked_bits[idx] = (byte_val >> bit) & 1
        
        # Reshape to original shape
        return unpacked_bits.reshape(original_shape)
    
    def validate_binary_exact(self, tensor: torch.Tensor) -> bool:
        """Validate that tensor contains only exact binary values."""
        if self.storage_format in ['binary_tensor', 'bool']:
            return tensor.dtype == torch.bool or torch.all((tensor == 0) | (tensor == 1))
        elif self.storage_format == 'uint8':
            return tensor.dtype == torch.uint8 and torch.all((tensor == 0) | (tensor == 1))
        else:
            return torch.all((tensor == 0) | (tensor == 1))
    
    def get_dbn_layer_dimensions(self) -> List[int]:
        """
        Get dimensions for DBN layers (one per channel).
        
        Returns:
            List of latent dimensions, one per channel across all levels
        """
        layer_dims = []
        
        for c, h, w in self.latent_shapes:
            dim_per_channel = h * w
            
            # Add one dimension per channel
            for _ in range(c):
                layer_dims.append(dim_per_channel)
        
        return layer_dims
    
    def extract_hierarchical_channels(self, latents_by_level: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Extract individual channels from hierarchical latent representation.
        
        Args:
            latents_by_level: List of tensors, one per hierarchy level
            
        Returns:
            List of individual channel tensors for DBN processing
        """
        hierarchical_channels = []
        
        for level_idx, level_tensor in enumerate(latents_by_level):
            # level_tensor shape: [B, C, H, W]
            channels = level_tensor.size(1)
            
            for c in range(channels):
                channel_tensor = level_tensor[:, c:c+1, :, :]  # [B, 1, H, W]
                hierarchical_channels.append(channel_tensor)
        
        return hierarchical_channels
    
    def reconstruct_hierarchical_levels(self, hierarchical_channels: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reconstruct hierarchical level representation from individual channels.
        
        Args:
            hierarchical_channels: List of individual channel tensors
            
        Returns:
            List of tensors, one per hierarchy level
        """
        latents_by_level = []
        channel_idx = 0
        
        for c, h, w in self.latent_shapes:
            level_channels = []
            
            for _ in range(c):
                level_channels.append(hierarchical_channels[channel_idx])
                channel_idx += 1
            
            # Concatenate channels for this level
            level_tensor = torch.cat(level_channels, dim=1)  # [B, C_level, H, W]
            latents_by_level.append(level_tensor)
        
        return latents_by_level
    
    def get_topology_summary(self) -> str:
        """Get human-readable summary of topology configuration."""
        summary = [f"Binary Latent Topology: {self.topology_type.upper()}"]
        summary.append(f"Storage Format: {self.storage_format}")
        summary.append(f"Total Levels: {self.num_levels}")
        summary.append(f"Total Channels: {self.total_channels}")
        summary.append(f"BPP: {self.actual_bpp:.3f} ≤ 0.5")
        
        for i, (c, h, w) in enumerate(self.latent_shapes):
            summary.append(f"  Level {i}: {c} channels @ {h}×{w}")
        
        return "\n".join(summary)

    def get_total_latent_shape(self, batch_size: int = 1) -> torch.Size:
        """
        Get total flattened latent shape for a given batch size.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Total latent shape as torch.Size
        """
        total_elements = sum(c * h * w for c, h, w in self.latent_shapes)
        return torch.Size([batch_size, total_elements])

    def get_level_shapes(self) -> List[Tuple[int, int, int]]:
        """
        Get shapes for all levels.
        
        Returns:
            List of (channels, height, width) for each level
        """
        return self.latent_shapes.copy()

    def flatten_latents(self, latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten hierarchical latents into a single tensor.
        
        Args:
            latents: List of latent tensors from each level
            
        Returns:
            Flattened latent tensor
        """
        flattened_list = []
        for latent in latents:
            # Ensure proper dtype and flatten
            latent_float = latent.float()
            flattened_list.append(latent_float.flatten(start_dim=1))
        
        return torch.cat(flattened_list, dim=1)

    def unflatten_latents(self, flat_latents: torch.Tensor) -> List[torch.Tensor]:
        """
        Unflatten a tensor back to hierarchical latents.
        
        Args:
            flat_latents: Flattened latent tensor
            
        Returns:
            List of hierarchical latent tensors
        """
        batch_size = flat_latents.shape[0]
        latents = []
        start_idx = 0
        
        for c, h, w in self.latent_shapes:
            level_size = c * h * w
            
            # Extract and reshape
            level_flat = flat_latents[:, start_idx:start_idx + level_size]
            level_reshaped = level_flat.reshape(batch_size, c, h, w)
            latents.append(level_reshaped)
            
            start_idx += level_size
        
        return latents
