#!/usr/bin/env python3
"""
Configuration Validator for Binary-Latent Diffusion Model

This module validates configuration files to ensure they meet all specifications
and can support the three required topologies without code modification.
"""

import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from .topology_config import TopologyConfigFactory, validate_topology_config

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration files for binary-latent diffusion model."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate all sections
            self._validate_topology_section(config)
            self._validate_model_section(config)
            self._validate_data_section(config)
            self._validate_diffusion_section(config)
            self._validate_training_section(config)
            self._validate_hardware_section(config)
            
            # Check for consistency between sections
            self._validate_consistency(config)
            
        except FileNotFoundError:
            self.errors.append(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_topology_section(self, config: Dict[str, Any]):
        """Validate topology configuration section."""
        if 'topology' not in config:
            # Check for legacy format
            if 'model' in config and 'autoencoder' in config['model']:
                try:
                    topology_config = TopologyConfigFactory.from_legacy_config(config)
                    self.warnings.append("Using legacy configuration format. Consider updating to new topology format.")
                    return
                except Exception as e:
                    self.errors.append(f"Cannot determine topology from configuration: {e}")
                    return
            else:
                self.errors.append("Missing 'topology' section in configuration")
                return
        
        try:
            topology_config = TopologyConfigFactory.from_dict(config['topology'])
            if not validate_topology_config(topology_config):
                self.errors.append("Invalid topology configuration")
        except Exception as e:
            self.errors.append(f"Topology validation error: {e}")
    
    def _validate_model_section(self, config: Dict[str, Any]):
        """Validate model configuration section."""
        if 'model' not in config:
            self.errors.append("Missing 'model' section in configuration")
            return
        
        model_config = config['model']
        
        # Validate autoencoder section
        if 'autoencoder' not in model_config:
            self.errors.append("Missing 'autoencoder' section in model configuration")
        else:
            ae_config = model_config['autoencoder']
            
            # Check required parameters
            if 'base_channels' not in ae_config:
                self.warnings.append("Missing 'base_channels' in autoencoder config, will use default")
            
            base_channels = ae_config.get('base_channels', 32)
            if not isinstance(base_channels, int) or base_channels < 1:
                self.errors.append("'base_channels' must be a positive integer")
        
        # Validate denoiser section
        if 'denoiser' not in model_config:
            self.errors.append("Missing 'denoiser' section in model configuration")
        else:
            denoiser_config = model_config['denoiser']
            
            # Check hidden dimensions factor
            hidden_factor = denoiser_config.get('hidden_dims_factor', 2.0)
            if not isinstance(hidden_factor, (int, float)) or hidden_factor <= 0:
                self.errors.append("'hidden_dims_factor' must be a positive number")
            
            # Check contrastive divergence steps
            cd_k = denoiser_config.get('cd_k', 1)
            if not isinstance(cd_k, int) or cd_k < 1:
                self.errors.append("'cd_k' must be a positive integer")
    
    def _validate_data_section(self, config: Dict[str, Any]):
        """Validate data configuration section."""
        if 'data' not in config:
            self.errors.append("Missing 'data' section in configuration")
            return
        
        data_config = config['data']
        
        # Check required fields
        required_fields = ['dataset', 'image_size', 'channels']
        for field in required_fields:
            if field not in data_config:
                self.errors.append(f"Missing required field '{field}' in data configuration")
        
        # Validate image size
        if 'image_size' in data_config:
            image_size = data_config['image_size']
            if not isinstance(image_size, int) or image_size < 1:
                self.errors.append("'image_size' must be a positive integer")
            elif image_size % 2 != 0:
                self.warnings.append("'image_size' should be even for proper downsampling")
        
        # Validate channels
        if 'channels' in data_config:
            channels = data_config['channels']
            if not isinstance(channels, int) or channels < 1:
                self.errors.append("'channels' must be a positive integer")
    
    def _validate_diffusion_section(self, config: Dict[str, Any]):
        """Validate diffusion configuration section."""
        if 'diffusion' not in config:
            self.errors.append("Missing 'diffusion' section in configuration")
            return
        
        diffusion_config = config['diffusion']
        
        # Check timesteps
        if 'timesteps' not in diffusion_config:
            self.errors.append("Missing 'timesteps' in diffusion configuration")
        else:
            timesteps = diffusion_config['timesteps']
            if not isinstance(timesteps, int) or timesteps < 1:
                self.errors.append("'timesteps' must be a positive integer")
            elif timesteps > 2000:
                self.warnings.append("Large number of timesteps may require significant memory")
        
        # Check beta schedule
        beta_start = diffusion_config.get('beta_start', 1e-4)
        beta_end = diffusion_config.get('beta_end', 0.02)
        
        if not isinstance(beta_start, (int, float)) or beta_start <= 0:
            self.errors.append("'beta_start' must be a positive number")
        elif beta_start >= 0.5:
            self.errors.append("'beta_start' must be < 0.5 for Bernoulli diffusion")
        
        if not isinstance(beta_end, (int, float)) or beta_end <= 0:
            self.errors.append("'beta_end' must be a positive number")
        elif beta_end >= 0.5:
            self.errors.append("'beta_end' must be < 0.5 for Bernoulli diffusion")
        
        if beta_start >= beta_end:
            self.errors.append("'beta_start' must be less than 'beta_end'")
    
    def _validate_training_section(self, config: Dict[str, Any]):
        """Validate training configuration section."""
        if 'training' not in config:
            self.errors.append("Missing 'training' section in configuration")
            return
        
        training_config = config['training']
        
        # Check learning rates
        ae_lr = training_config.get('autoencoder_lr', 2e-4)
        dbn_lr = training_config.get('dbn_lr', 1e-3)
        
        if not isinstance(ae_lr, (int, float)) or ae_lr <= 0:
            self.errors.append("'autoencoder_lr' must be a positive number")
        elif ae_lr > 0.1:
            self.warnings.append("'autoencoder_lr' is very high, may cause instability")
        
        if not isinstance(dbn_lr, (int, float)) or dbn_lr <= 0:
            self.errors.append("'dbn_lr' must be a positive number")
        elif dbn_lr > 0.1:
            self.warnings.append("'dbn_lr' is very high, may cause instability")
        
        # Check batch size
        batch_size = training_config.get('batch_size', 32)
        if not isinstance(batch_size, int) or batch_size < 1:
            self.errors.append("'batch_size' must be a positive integer")
        elif batch_size > 512:
            self.warnings.append("Large batch size may require significant memory")
        
        # Check epochs
        ae_epochs = training_config.get('ae_epochs', 50)
        if not isinstance(ae_epochs, int) or ae_epochs < 1:
            self.errors.append("'ae_epochs' must be a positive integer")
        
        dbn_epochs = training_config.get('dbn_epochs_per_timestep', 3)
        if not isinstance(dbn_epochs, int) or dbn_epochs < 1:
            self.errors.append("'dbn_epochs_per_timestep' must be a positive integer")
    
    def _validate_hardware_section(self, config: Dict[str, Any]):
        """Validate hardware configuration section."""
        device = config.get('device', 'cpu')
        if device not in ['cpu', 'cuda', 'mps']:
            self.warnings.append(f"Unknown device '{device}', will attempt to use as-is")
        
        compile_models = config.get('compile_models', False)
        if compile_models and device == 'cpu':
            self.warnings.append("Model compilation may not provide benefits on CPU")
        
        mixed_precision = config.get('mixed_precision', False)
        if mixed_precision and device == 'cpu':
            self.warnings.append("Mixed precision not supported on CPU")
    
    def _validate_consistency(self, config: Dict[str, Any]):
        """Validate consistency between different configuration sections."""
        # Check if topology is consistent with model configuration
        if 'topology' in config and 'model' in config:
            topology_config = config['topology']
            
            # For flat topology, ensure appropriate autoencoder settings
            if topology_config.get('type') == 'flat':
                channels = topology_config.get('channels', 0)
                if channels > 64:
                    self.warnings.append("Large number of channels in flat topology may require significant memory")
            
            # For hierarchical topologies, check level consistency
            elif topology_config.get('type') in ['hierarchical', 'mixed']:
                levels = topology_config.get('levels', [])
                if len(levels) > 5:
                    self.warnings.append("Many hierarchy levels may require significant computation")
        
        # Check if training configuration is consistent with diffusion settings
        if 'training' in config and 'diffusion' in config:
            timesteps = config['diffusion'].get('timesteps', 1000)
            dbn_epochs = config['training'].get('dbn_epochs_per_timestep', 3)
            
            total_dbn_training = timesteps * dbn_epochs
            if total_dbn_training > 10000:
                self.warnings.append(f"Total DBN training steps ({total_dbn_training}) is very large")

def validate_config_file(config_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = ConfigValidator()
    return validator.validate_config_file(config_path)

def validate_all_test_configs() -> Dict[str, Tuple[bool, List[str], List[str]]]:
    """Validate all test configuration files."""
    test_configs = [
        "configs/test_hierarchical_single_channel.yaml",
        "configs/test_flat_multi_channel.yaml", 
        "configs/test_hierarchical_multi_channel.yaml"
    ]
    
    results = {}
    for config_path in test_configs:
        if Path(config_path).exists():
            results[config_path] = validate_config_file(config_path)
        else:
            results[config_path] = (False, [f"File not found: {config_path}"], [])
    
    return results
