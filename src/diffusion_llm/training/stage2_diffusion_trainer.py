"""
Stage 2 Diffusion Trainer

This module implements the training pipeline for Stage 2 of the diffusion LLM system:
training the Reasoning DiT in binary latent space with frozen autoencoder from Stage 1.

Key Features:
- Reasoning DiT training in binary latent space
- Frozen BART autoencoder from Stage 1
- Binary diffusion loss optimization
- Quantum annealer-compatible training
- Support for arithmetic and spatial reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from tqdm import tqdm
import wandb
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from diffusion_llm.encoders import BARTBinaryAutoEncoder
from diffusion_llm.models import BARTTokenizerWrapper
from diffusion_llm.diffusion_transformers import TextBinaryDiffusion, ReasoningDiT

logger = logging.getLogger(__name__)


class Stage2DiffusionTrainer:
    """
    Trainer for Stage 2: Reasoning DiT in binary latent space.
    
    This trainer implements the second stage of the two-stage training pipeline
    from "Latent Diffusion with LLMs for Reasoning", focusing on training the
    Reasoning DiT to perform reasoning in the binary latent space learned in Stage 1.
    """
    
    def __init__(
        self,
        # Model configuration
        pretrained_autoencoder_path: str,   # Path to Stage 1 checkpoint
        reasoning_dit_config: Dict[str, Any] = None,
        
        # Diffusion configuration
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_epochs: int = 20,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        
        # Loss configuration
        diffusion_loss_weight: float = 1.0,
        reasoning_loss_weight: float = 1.0,
        binary_consistency_weight: float = 0.1,
        
        # Data configuration
        max_seq_len: int = 128,
        validation_split: float = 0.1,
        
        # Saving and logging
        save_dir: str = "./checkpoints/stage2",
        log_interval: int = 50,
        save_interval: int = 1000,
        validate_interval: int = 250,
        use_wandb: bool = False,
        wandb_project: str = "diffusion-llm-stage2",
        
        # Device
        device: str = "auto"
    ):
        """
        Initialize Stage 2 trainer.
        
        Args:
            pretrained_autoencoder_path: Path to trained autoencoder from Stage 1
            reasoning_dit_config: Configuration for Reasoning DiT
            num_timesteps: Number of diffusion timesteps
            beta_start: Start of noise schedule
            beta_end: End of noise schedule
            learning_rate: Learning rate for DiT training
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_epochs: Maximum number of epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            diffusion_loss_weight: Weight for diffusion loss
            reasoning_loss_weight: Weight for reasoning loss
            binary_consistency_weight: Weight for binary consistency loss
            max_seq_len: Maximum sequence length
            validation_split: Fraction of data for validation
            save_dir: Directory to save checkpoints
            log_interval: Logging interval (steps)
            save_interval: Checkpoint saving interval (steps)
            validate_interval: Validation interval (steps)
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.pretrained_autoencoder_path = pretrained_autoencoder_path
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Loss weights
        self.diffusion_loss_weight = diffusion_loss_weight
        self.reasoning_loss_weight = reasoning_loss_weight
        self.binary_consistency_weight = binary_consistency_weight
        
        # Data params
        self.max_seq_len = max_seq_len
        self.validation_split = validation_split
        
        # Logging and saving
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.validate_interval = validate_interval
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Default DiT config
        if reasoning_dit_config is None:
            reasoning_dit_config = {
                'latent_dim': 256,
                'sequence_length': 16,
                'hidden_size': 768,
                'num_heads': 12,
                'num_layers': 12,
                'num_reasoning_types': 4,
                'condition_on_input': True
            }
        self.reasoning_dit_config = reasoning_dit_config
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize components
        self.setup_models()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info("🚀 Stage2DiffusionTrainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Save directory: {self.save_dir}")
        logger.info(f"   Diffusion timesteps: {self.num_timesteps}")
        logger.info(f"   Reasoning DiT config: {self.reasoning_dit_config}")
    
    def setup_models(self):
        """Setup the models for Stage 2 training."""
        logger.info("🔧 Setting up models for Stage 2 training...")
        
        # 1. Load pretrained autoencoder from Stage 1 (frozen)
        logger.info(f"   Loading pretrained autoencoder from: {self.pretrained_autoencoder_path}")
        checkpoint = torch.load(self.pretrained_autoencoder_path, map_location=self.device)
        
        # Extract autoencoder config from checkpoint
        autoencoder_config = checkpoint.get('config', {})
        
        self.autoencoder = BARTBinaryAutoEncoder(
            bart_model_name=autoencoder_config.get('bart_model_name', 'facebook/bart-base'),
            num_encoder_latents=autoencoder_config.get('num_encoder_latents', 16),
            dim_ae=autoencoder_config.get('dim_ae', 256),
            freeze_bart_encoder=True  # Freeze for Stage 2
        ).to(self.device)
        
        # Load weights
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to Stage 2 mode (freeze encoder/decoder)
        self.autoencoder.set_training_stage(2)
        
        # 2. Initialize Reasoning DiT
        logger.info("   Initializing Reasoning DiT...")
        self.reasoning_dit = ReasoningDiT(**self.reasoning_dit_config).to(self.device)
        
        # 3. Initialize text binary diffusion system
        logger.info("   Initializing Text Binary Diffusion...")
        self.text_diffusion = TextBinaryDiffusion(
            latent_dim=self.reasoning_dit_config['latent_dim'],
            sequence_length=self.reasoning_dit_config['sequence_length'],
            num_timesteps=self.num_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=self.device,
            quantum_enabled=True
        )
        
        # 4. Initialize tokenizer
        self.tokenizer = BARTTokenizerWrapper("facebook/bart-base")
        
        logger.info("✅ Models setup complete")
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.reasoning_dit.parameters())
        trainable_params = sum(p.numel() for p in self.reasoning_dit.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model Statistics:")
        logger.info(f"   Reasoning DiT parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Autoencoder frozen: ✅")
    
    def setup_optimizer_and_scheduler(self, total_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer (only DiT parameters are trainable)
        self.optimizer = AdamW(
            self.reasoning_dit.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"   Optimizer: AdamW (lr={self.learning_rate}, wd={self.weight_decay})")
        logger.info(f"   Scheduler: Linear with warmup ({self.warmup_steps} steps)")
        logger.info(f"   Total training steps: {total_steps}")
    
    def compute_diffusion_loss(
        self, 
        binary_latents: torch.Tensor,
        reasoning_type: torch.Tensor,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion loss for binary latents.
        
        Args:
            binary_latents: Binary latents [B, seq_len, latent_dim]
            reasoning_type: Reasoning type IDs [B]
            context_ids: Context token IDs [B, ctx_len]
            context_mask: Context attention mask [B, ctx_len]
            
        Returns:
            Dictionary with loss components
        """
        batch_size = binary_latents.size(0)
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to binary latents
        noisy_latents = self.text_diffusion.forward_process(binary_latents, t)
        
        # Encode context for conditioning
        with torch.no_grad():
            context_encodings = self.autoencoder.get_encoder()(
                input_ids=context_ids,
                attention_mask=context_mask
            ).last_hidden_state
        
        # Predict noise with Reasoning DiT
        predicted_noise = self.reasoning_dit(
            x=noisy_latents,
            t=t,
            reasoning_type=reasoning_type,
            context=context_encodings,
            context_mask=context_mask
        )
        
        # Compute MSE loss for denoising
        target_noise = noisy_latents - binary_latents
        diffusion_loss = F.mse_loss(predicted_noise, target_noise)
        
        # Binary consistency loss (ensure outputs are binary)
        binary_consistency_loss = F.mse_loss(
            torch.round(predicted_noise), 
            predicted_noise
        )
        
        return {
            'diffusion_loss': diffusion_loss,
            'binary_consistency_loss': binary_consistency_loss,
            'predicted_noise': predicted_noise,
            'target_noise': target_noise
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch containing problem_ids, solution_ids, reasoning_types
            
        Returns:
            Dictionary with loss and metrics
        """
        self.reasoning_dit.train()
        
        # Move batch to device
        problem_ids = batch['problem_ids'].to(self.device)
        problem_mask = batch['problem_mask'].to(self.device)
        solution_ids = batch['solution_ids'].to(self.device)
        solution_mask = batch['solution_mask'].to(self.device)
        reasoning_types = batch['reasoning_types'].to(self.device)
        
        # Encode problems and solutions to binary latents (frozen autoencoder)
        with torch.no_grad():
            problem_latents = self.autoencoder.encode_text_to_binary_latents(
                problem_ids, problem_mask
            )
            solution_latents = self.autoencoder.encode_text_to_binary_latents(
                solution_ids, solution_mask
            )
        
        # Compute diffusion loss
        loss_dict = self.compute_diffusion_loss(
            binary_latents=solution_latents,
            reasoning_type=reasoning_types,
            context_ids=problem_ids,
            context_mask=problem_mask
        )
        
        # Combine losses
        total_loss = (
            self.diffusion_loss_weight * loss_dict['diffusion_loss'] +
            self.binary_consistency_weight * loss_dict['binary_consistency_loss']
        )
        
        # Backward pass
        if self.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.gradient_accumulation_steps
        
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.reasoning_dit.parameters(), 
                self.max_grad_norm
            )
        
        # Optimizer step
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Return metrics
        return {
            'total_loss': total_loss.item() * self.gradient_accumulation_steps,
            'diffusion_loss': loss_dict['diffusion_loss'].item(),
            'binary_consistency_loss': loss_dict['binary_consistency_loss'].item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            Dictionary with validation metrics
        """
        self.reasoning_dit.eval()
        
        with torch.no_grad():
            # Same as training step but no gradients
            problem_ids = batch['problem_ids'].to(self.device)
            problem_mask = batch['problem_mask'].to(self.device)
            solution_ids = batch['solution_ids'].to(self.device)
            solution_mask = batch['solution_mask'].to(self.device)
            reasoning_types = batch['reasoning_types'].to(self.device)
            
            # Encode to binary latents
            problem_latents = self.autoencoder.encode_text_to_binary_latents(
                problem_ids, problem_mask
            )
            solution_latents = self.autoencoder.encode_text_to_binary_latents(
                solution_ids, solution_mask
            )
            
            # Compute losses
            loss_dict = self.compute_diffusion_loss(
                binary_latents=solution_latents,
                reasoning_type=reasoning_types,
                context_ids=problem_ids,
                context_mask=problem_mask
            )
            
            total_loss = (
                self.diffusion_loss_weight * loss_dict['diffusion_loss'] +
                self.binary_consistency_weight * loss_dict['binary_consistency_loss']
            )
            
            return {
                'val_total_loss': total_loss.item(),
                'val_diffusion_loss': loss_dict['diffusion_loss'].item(),
                'val_binary_consistency_loss': loss_dict['binary_consistency_loss'].item()
            }
    
    def train(
        self, 
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ):
        """
        Train the Reasoning DiT model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
        """
        logger.info("🚀 Starting Stage 2 training...")
        
        # Calculate total steps
        total_steps = len(train_dataloader) * self.max_epochs // self.gradient_accumulation_steps
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(total_steps)
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config={
                    'stage': 2,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'max_epochs': self.max_epochs,
                    'num_timesteps': self.num_timesteps,
                    **self.reasoning_dit_config
                }
            )
        
        # Training loop
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            logger.info(f"\n📅 Epoch {epoch + 1}/{self.max_epochs}")
            
            # Training phase
            epoch_metrics = []
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'lr': f"{metrics['learning_rate']:.2e}"
                })
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_metrics = {
                        key: np.mean([m[key] for m in epoch_metrics[-self.log_interval:]])
                        for key in metrics.keys()
                    }
                    
                    logger.info(f"Step {self.global_step}: Loss = {avg_metrics['total_loss']:.4f}")
                    
                    if self.use_wandb:
                        wandb.log(avg_metrics, step=self.global_step)
                
                # Validation
                if val_dataloader and self.global_step % self.validate_interval == 0:
                    val_metrics = self.validate(val_dataloader)
                    logger.info(f"Validation: Loss = {val_metrics['val_total_loss']:.4f}")
                    
                    if self.use_wandb:
                        wandb.log(val_metrics, step=self.global_step)
                    
                    # Save best model
                    if val_metrics['val_total_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_total_loss']
                        self.save_checkpoint('best_model')
                
                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
            
            # End of epoch
            epoch_loss = np.mean([m['total_loss'] for m in epoch_metrics])
            logger.info(f"✅ Epoch {epoch + 1} complete. Average loss: {epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
        
        logger.info("🎉 Stage 2 training complete!")
        
        if self.use_wandb:
            wandb.finish()
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Run validation loop."""
        val_metrics = []
        
        for batch in val_dataloader:
            metrics = self.validate_step(batch)
            val_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in val_metrics])
            for key in val_metrics[0].keys()
        }
        
        return avg_metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'reasoning_dit_state_dict': self.reasoning_dit.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.reasoning_dit_config,
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'num_timesteps': self.num_timesteps
            }
        }
        
        save_path = os.path.join(self.save_dir, f'{name}.pt')
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.reasoning_dit.load_state_dict(checkpoint['reasoning_dit_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"📥 Checkpoint loaded: {checkpoint_path}")
        logger.info(f"   Global step: {self.global_step}")
        logger.info(f"   Best val loss: {self.best_val_loss}") 