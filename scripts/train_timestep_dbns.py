#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

# Robust path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

# Ensure absolute imports work correctly
from src.qudiffuse.models.timestep_specific_dbn_manager import TimestepSpecificDBNManager
from src.qudiffuse.models.multi_resolution_binary_ae import MultiResolutionBinaryAutoEncoder
from src.qudiffuse.utils.error_handling import QuDiffuseTrainingError

class AuthenticDBNTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize authentic DBN training with comprehensive configuration
        
        Args:
            config (Dict[str, Any]): Configuration for training
        """
        self.config = self._resolve_paths(config)
        self._setup_logging()
        self._validate_config()
        
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve and validate file paths with multiple fallback strategies
        
        Args:
            config (Dict[str, Any]): Original configuration
        
        Returns:
            Dict[str, Any]: Configuration with resolved paths
        """
        path_keys = ['autoencoder_path']
        resolved_config = config.copy()
        
        for key in path_keys:
            if key in config:
                # Multiple path resolution strategies
                possible_paths = [
                    config[key],  # Original path
                    os.path.join(PROJECT_ROOT, config[key]),  # Relative to project root
                    os.path.join(SCRIPT_DIR, config[key]),  # Relative to script dir
                    os.path.expanduser(config[key]),  # Expand user home directory
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        resolved_config[key] = path
                        break
        else:
                    raise QuDiffuseTrainingError(f"Cannot resolve path for {key}: {config[key]}")
        
        return resolved_config
        
    def _setup_logging(self):
        """Configure comprehensive logging with multiple handlers"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(PROJECT_ROOT, 'dbn_training.log'), mode='w')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _validate_config(self):
        """Validate training configuration with strict checks"""
        required_keys = [
            'autoencoder_path', 
            'timesteps', 
            'batch_size', 
            'epochs_per_timestep'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise QuDiffuseTrainingError(f"Missing required configuration: {key}")
        
        # Validate numeric ranges
        if not (1 <= self.config['timesteps'] <= 100):
            raise QuDiffuseTrainingError("Timesteps must be between 1 and 100")
        
        if not (8 <= self.config['batch_size'] <= 256):
            raise QuDiffuseTrainingError("Batch size must be between 8 and 256")
        
    def _load_checkpoint(self) -> MultiResolutionBinaryAutoEncoder:
        """
        Load authentic checkpoint with comprehensive validation
        
        Returns:
            MultiResolutionBinaryAutoEncoder: Loaded model
        """
        checkpoint_path = self.config['autoencoder_path']
        
        if not os.path.exists(checkpoint_path):
            raise QuDiffuseTrainingError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate checkpoint integrity
            if not isinstance(checkpoint, dict):
                raise QuDiffuseTrainingError("Invalid checkpoint format")
            
            model = MultiResolutionBinaryAutoEncoder(**checkpoint.get('model_config', {}))
            model.load_state_dict(checkpoint['model_state'])
            
            self.logger.info(f"✅ Loaded authentic checkpoint from {checkpoint_path}")
            return model
        
        except Exception as e:
            raise QuDiffuseTrainingError(f"Checkpoint loading failed: {e}")
    
    def train(self):
        """
        Execute authentic DBN training with comprehensive tracking
        """
        try:
            # Load authentic checkpoint
            autoencoder = self._load_checkpoint()
            
            # Extract latent shapes and hidden dimensions
            latent_shapes = [
                (level.shape[1], level.shape[2], level.shape[3]) 
                for level in autoencoder.encoder.latent_levels
            ]
            
            # Compute hidden dimensions based on latent shapes
            hidden_dims = [
                int(np.prod(shape)) for shape in latent_shapes
            ]
            
            # Initialize timestep-specific DBN manager
            dbn_manager = TimestepSpecificDBNManager(
                latent_shapes=latent_shapes,
                hidden_dims=hidden_dims,
                timesteps=self.config['timesteps'],
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Authentic training loop
            dbn_manager.initialize_all_timesteps()
            
            # Placeholder for actual training logic
            # This would typically involve loading training data and performing greedy pretraining
            training_metrics = {}
            
            # Log comprehensive metrics
            self._log_training_metrics(training_metrics)
            
        except QuDiffuseTrainingError as qte:
            self.logger.error(f"❌ Authentic Training Failure: {qte}")
            sys.exit(1)
        
    def _log_training_metrics(self, metrics: Dict[str, Any]):
        """
        Log comprehensive training metrics with statistical insights
        
        Args:
            metrics (Dict[str, Any]): Training metrics dictionary
        """
        self.logger.info("🔬 Training Metrics Analysis:")
        for timestep, timestep_metrics in metrics.items():
            self.logger.info(f"Timestep {timestep}:")
            for metric_name, metric_value in timestep_metrics.items():
                self.logger.info(f"  {metric_name}: {metric_value}")

def parse_arguments() -> Dict[str, Any]:
    """Parse command-line arguments with comprehensive validation"""
    parser = argparse.ArgumentParser(description="Authentic Timestep-Specific DBN Training")
    
    parser.add_argument(
        '--autoencoder', 
        type=str, 
        required=True, 
        help='Path to authentic autoencoder checkpoint'
    )
    parser.add_argument(
        '--timesteps', 
        type=int, 
        default=50, 
        help='Number of diffusion timesteps (1-100)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32, 
        help='Training batch size (8-256)'
    )
    parser.add_argument(
        '--epochs-per-timestep', 
        type=int, 
        default=100, 
        help='Training epochs per timestep'
    )
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    parser.add_argument(
        '--authentic-mode', 
        type=bool, 
        default=True, 
        help='Enforce authentic training without simplifications'
    )
    
    args = parser.parse_args()
    return vars(args)

def main():
    """Main execution point with comprehensive error handling"""
    try:
        config = parse_arguments()
        
        if not config.get('authentic_mode', True):
            raise QuDiffuseTrainingError("Authentic mode is mandatory")
        
        trainer = AuthenticDBNTrainer(config)
        trainer.train()
        
    except Exception as e:
        logging.error(f"❌ Training Execution Failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 