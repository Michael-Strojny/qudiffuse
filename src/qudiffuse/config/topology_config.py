#!/usr/bin/env python3
"""
Topology Configuration System for Binary Latent Diffusion

This module provides configuration classes that support all three latent space
topologies without requiring code modification:
1. Hierarchical single-channel
2. Flat multi-channel  
3. Hierarchical multi-channel
"""

import yaml
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
from qudiffuse.utils.error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError

logger = logging.getLogger(__name__)

@dataclass
class LevelConfig:
    """Configuration for a single hierarchy level."""
    channels: int
    spatial_size: Tuple[int, int]
    
    def __post_init__(self):
        if self.channels < 1:
            raise ValueError(f"Channels must be >= 1, got {self.channels}")
        if len(self.spatial_size) != 2:
            raise ValueError(f"Spatial size must be (H, W), got {self.spatial_size}")
        if self.spatial_size[0] < 1 or self.spatial_size[1] < 1:
            raise ValueError(f"Spatial dimensions must be >= 1, got {self.spatial_size}")

@dataclass
class TopologyConfig:
    """Base configuration for latent space topology."""
    type: str  # 'hierarchical', 'flat', 'mixed'
    storage_format: str = 'bool'  # 'bool', 'uint8', 'packed'
    device: str = 'cpu'
    
    def validate(self):
        """Validate configuration."""
        if self.type not in ['hierarchical', 'flat', 'mixed']:
            raise ValueError(f"Invalid topology type: {self.type}")
        if self.storage_format not in ['bool', 'uint8', 'packed']:
            raise ValueError(f"Invalid storage format: {self.storage_format}")

@dataclass
class HierarchicalTopologyConfig(TopologyConfig):
    """Configuration for hierarchical single-channel topology."""
    type: str = 'hierarchical'
    levels: List[LevelConfig] = field(default_factory=list)
    
    def __post_init__(self):
        self.validate()
        if len(self.levels) < 1:
            raise TopologyError("Hierarchical topology requires at least 1 level", "validation")
        
        # Validate spatial downsampling
        for i in range(1, len(self.levels)):
            prev_size = self.levels[i-1].spatial_size
            curr_size = self.levels[i].spatial_size
            
            if prev_size[0] % curr_size[0] != 0 or prev_size[1] % curr_size[1] != 0:
                raise ValueError(f"Level {i} spatial size {curr_size} is not integer divisor of level {i-1} size {prev_size}")
            
            factor_h = prev_size[0] // curr_size[0]
            factor_w = prev_size[1] // curr_size[1]
            if factor_h < 2 or factor_w < 2:
                raise ValueError(f"Downsampling factor must be >= 2, got {factor_h}x{factor_w}")
    
    @classmethod
    def from_channel_list(cls, channels: List[int], base_spatial_size: Tuple[int, int], 
                         downsample_factor: int = 2) -> 'HierarchicalTopologyConfig':
        """Create hierarchical config from channel list with automatic spatial downsampling."""
        levels = []
        current_size = base_spatial_size
        
        for i, num_channels in enumerate(channels):
            levels.append(LevelConfig(channels=num_channels, spatial_size=current_size))
            # Downsample for next level
            current_size = (current_size[0] // downsample_factor, current_size[1] // downsample_factor)
        
        return cls(levels=levels)

@dataclass
class FlatTopologyConfig(TopologyConfig):
    """Configuration for flat multi-channel topology."""
    type: str = 'flat'
    channels: int = 1
    spatial_size: Tuple[int, int] = (32, 32)
    
    def __post_init__(self):
        self.validate()
        if self.channels < 1:
            raise ValueError(f"Channels must be >= 1, got {self.channels}")
        if len(self.spatial_size) != 2:
            raise ValueError(f"Spatial size must be (H, W), got {self.spatial_size}")

@dataclass
class MixedTopologyConfig(TopologyConfig):
    """Configuration for hierarchical multi-channel topology."""
    type: str = 'mixed'
    levels: List[LevelConfig] = field(default_factory=list)
    
    def __post_init__(self):
        self.validate()
        if len(self.levels) < 1:
            raise TopologyError("Mixed topology requires at least 1 level", "validation")
        
        # Validate spatial downsampling (same as hierarchical)
        for i in range(1, len(self.levels)):
            prev_size = self.levels[i-1].spatial_size
            curr_size = self.levels[i].spatial_size
            
            if prev_size[0] % curr_size[0] != 0 or prev_size[1] % curr_size[1] != 0:
                raise ValueError(f"Level {i} spatial size {curr_size} is not integer divisor of level {i-1} size {prev_size}")
            
            factor_h = prev_size[0] // curr_size[0]
            factor_w = prev_size[1] // curr_size[1]
            if factor_h < 2 or factor_w < 2:
                raise ValueError(f"Downsampling factor must be >= 2, got {factor_h}x{factor_w}")

class TopologyConfigFactory:
    """Factory for creating topology configurations from various input formats."""
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> TopologyConfig:
        """Create topology config from dictionary."""
        topology_type = config_dict.get('type', 'flat')
        
        if topology_type == 'hierarchical':
            levels = []
            for level_dict in config_dict['levels']:
                levels.append(LevelConfig(
                    channels=level_dict['channels'],
                    spatial_size=tuple(level_dict['spatial_size'])
                ))
            return HierarchicalTopologyConfig(
                levels=levels,
                storage_format=config_dict.get('storage_format', 'bool'),
                device=config_dict.get('device', 'cpu')
            )
        
        elif topology_type == 'flat':
            return FlatTopologyConfig(
                channels=config_dict['channels'],
                spatial_size=tuple(config_dict['spatial_size']),
                storage_format=config_dict.get('storage_format', 'bool'),
                device=config_dict.get('device', 'cpu')
            )
        
        elif topology_type == 'mixed':
            levels = []
            for level_dict in config_dict['levels']:
                levels.append(LevelConfig(
                    channels=level_dict['channels'],
                    spatial_size=tuple(level_dict['spatial_size'])
                ))
            return MixedTopologyConfig(
                levels=levels,
                storage_format=config_dict.get('storage_format', 'bool'),
                device=config_dict.get('device', 'cpu')
            )
        
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
    
    @staticmethod
    def from_legacy_config(model_config: Dict[str, Any]) -> TopologyConfig:
        """Create topology config from legacy model configuration."""
        # Detect topology type from legacy config
        if 'resolution_configs' in model_config.get('autoencoder', {}):
            # Multi-resolution flat space -> hierarchical
            resolution_configs = model_config['autoencoder']['resolution_configs']
            levels = []
            for res_config in resolution_configs:
                levels.append(LevelConfig(
                    channels=res_config['channels'],
                    spatial_size=(res_config['spatial_size'], res_config['spatial_size'])
                ))
            return HierarchicalTopologyConfig(levels=levels)
        
        elif 'latent_channels' in model_config.get('autoencoder', {}):
            latent_channels = model_config['autoencoder']['latent_channels']
            
            if isinstance(latent_channels, list):
                # True hierarchical
                base_size = model_config.get('data', {}).get('image_size', 32)
                return HierarchicalTopologyConfig.from_channel_list(
                    channels=latent_channels,
                    base_spatial_size=(base_size // 2, base_size // 2)  # Assume one downsample
                )
            else:
                # Single-resolution flat
                image_size = model_config.get('data', {}).get('image_size', 32)
                # Assume latent is downsampled by factor of 2
                latent_size = image_size // 2
                return FlatTopologyConfig(
                    channels=latent_channels,
                    spatial_size=(latent_size, latent_size)
                )
        
        else:
            raise TopologyError("Cannot determine topology from legacy config", "validation")
    
    @staticmethod
    def from_yaml_file(filepath: str) -> TopologyConfig:
        """Load topology config from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if 'topology' in config_dict:
            return TopologyConfigFactory.from_dict(config_dict['topology'])
        else:
            # Try to parse from legacy format
            return TopologyConfigFactory.from_legacy_config(config_dict)

def create_test_configs() -> Dict[str, TopologyConfig]:
    """Create test configurations for all three topologies."""
    configs = {}
    
    # Test 1: Hierarchical single-channel (L=3, Cₗ=1)
    configs['hierarchical_single'] = HierarchicalTopologyConfig(
        levels=[
            LevelConfig(channels=1, spatial_size=(64, 64)),
            LevelConfig(channels=1, spatial_size=(32, 32)),
            LevelConfig(channels=1, spatial_size=(16, 16))
        ]
    )
    
    # Test 2: Flat multi-channel (L=1, C₁=8)
    configs['flat_multi'] = FlatTopologyConfig(
        channels=8,
        spatial_size=(16, 16)
    )
    
    # Test 3: Hierarchical multi-channel (L=2, C₁=C₂=5)
    configs['hierarchical_multi'] = MixedTopologyConfig(
        levels=[
            LevelConfig(channels=5, spatial_size=(32, 32)),
            LevelConfig(channels=5, spatial_size=(16, 16))
        ]
    )
    
    return configs

def validate_topology_config(config: TopologyConfig) -> bool:
    """Validate topology configuration meets all specifications."""
    try:
        config.validate()
        
        # Additional validation for specifications
        if hasattr(config, 'levels'):
            # Check L >= 1
            if len(config.levels) < 1:
                return False
            
            # Check Cₗ >= 1 for all levels
            for level in config.levels:
                if level.channels < 1:
                    return False
            
            # Check spatial downsampling
            for i in range(1, len(config.levels)):
                prev_size = config.levels[i-1].spatial_size
                curr_size = config.levels[i].spatial_size
                
                factor_h = prev_size[0] // curr_size[0]
                factor_w = prev_size[1] // curr_size[1]
                if factor_h < 2 or factor_w < 2:
                    return False
        
        elif hasattr(config, 'channels'):
            # Flat topology
            if config.channels < 1:
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Topology validation failed: {e}")
        return False
