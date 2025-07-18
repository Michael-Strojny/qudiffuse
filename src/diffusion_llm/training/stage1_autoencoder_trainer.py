"""
Stage 1 Autoencoder Trainer

This module implements the training pipeline for Stage 1 of the diffusion LLM system:
training the BART autoencoder with Perceiver compression to learn variable-length 
to fixed-length binary latent mappings.

Key Features:
- BART encoder-decoder fine-tuning
- Perceiver autoencoder training for compression
- Reconstruction loss optimization
- Binary latent space preparation for Stage 2
- Support for reasoning task datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from tqdm import tqdm
import wandb

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from diffusion_llm.encoders import BARTBinaryAutoEncoder
from diffusion_llm.models import BARTTokenizerWrapper

logger = logging.getLogger(__name__)


class Stage1AutoencoderTrainer:
    """
    Trainer for Stage 1: BART Autoencoder with Perceiver compression.
    
    This trainer implements the first stage of the two-stage training pipeline
    from "Latent Diffusion with LLMs for Reasoning", focusing on learning
    high-quality variable-length to fixed-length binary latent compression.
    """
    
    def __init__(
        self,
        # Model configuration
        bart_model_name: str = "facebook/bart-base",
        num_encoder_latents: int = 16,      # lae from paper
        dim_ae: int = 256,                  # dae from paper
        perceiver_depth: int = 6,
        
        # Training configuration
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_epochs: int = 10,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        
        # Data configuration
        max_seq_len: int = 128,
        validation_split: float = 0.1,
        
        # Saving and logging
        save_dir: str = "./checkpoints/stage1",
        log_interval: int = 100,
        save_interval: int = 1000,
        validate_interval: int = 500,
        use_wandb: bool = False,
        wandb_project: str = "diffusion-llm-stage1",
        
        # Device
        device: str = "auto"
    ):
        """
        Initialize Stage 1 trainer.
        
        Args:
            bart_model_name: BART model to use as base
            num_encoder_latents: Number of encoder latents (lae)
            dim_ae: Autoencoder latent dimension (dae)
            perceiver_depth: Depth of Perceiver transformer
            learning_rate: Learning rate for training
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_epochs: Maximum number of epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
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
        self.bart_model_name = bart_model_name
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.perceiver_depth = perceiver_depth
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Data config
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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components
        self._setup_model()
        self._setup_tokenizer()
        self._setup_directories()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"🚀 Stage1AutoencoderTrainer initialized:")
        logger.info(f"   Model: {bart_model_name}")
        logger.info(f"   Latent shape: [{num_encoder_latents}, {dim_ae}]")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Learning rate: {learning_rate}")
    
    def _setup_model(self):
        """Initialize the BART autoencoder model."""
        self.model = BARTBinaryAutoEncoder(
            bart_model_name=self.bart_model_name,
            num_encoder_latents=self.num_encoder_latents,
            dim_ae=self.dim_ae,
            perceiver_depth=self.perceiver_depth,
            freeze_bart_encoder=False,  # Train BART encoder in Stage 1
            binary_quantization=True
        ).to(self.device)
        
        # Set to Stage 1 training mode
        self.model.set_training_stage(1)
        
        logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_tokenizer(self):
        """Initialize the tokenizer with reasoning tokens."""
        self.tokenizer = BARTTokenizerWrapper(self.bart_model_name)
        
        # Resize model embeddings to account for new special tokens
        self.model.bart_model.resize_token_embeddings(len(self.tokenizer))
    
    def _setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save training configuration
        config = {
            'bart_model_name': self.bart_model_name,
            'num_encoder_latents': self.num_encoder_latents,
            'dim_ae': self.dim_ae,
            'perceiver_depth': self.perceiver_depth,
            'learning_rate': self.learning_rate,
            'max_seq_len': self.max_seq_len,
            'batch_size': self.batch_size
        }
        
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def setup_optimizer_and_scheduler(self, total_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
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
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch containing input_ids, attention_mask, labels
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_binary_latents=True
        )
        
        loss = outputs['loss']
        
        # Backward pass
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        loss.backward()
        
        # Gradient clipping and optimizer step
        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Compute additional metrics
        with torch.no_grad():
            # Perplexity
            logits = outputs['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Mask padding tokens
            mask = (shift_labels != -100)
            
            if mask.sum() > 0:
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none'
                )
                ce_loss = ce_loss.view(shift_labels.shape)
                masked_loss = (ce_loss * mask).sum() / mask.sum()
                perplexity = torch.exp(masked_loss)
            else:
                perplexity = torch.tensor(0.0)
        
        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'perplexity': perplexity.item(),
            'lr': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.learning_rate
        }
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            val_dataloader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Compute perplexity
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                mask = (shift_labels != -100)
                if mask.sum() > 0:
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction='none'
                    )
                    ce_loss = ce_loss.view(shift_labels.shape)
                    masked_loss = (ce_loss * mask).sum() / mask.sum()
                    perplexity = torch.exp(masked_loss)
                    total_perplexity += perplexity.item()
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': avg_perplexity
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'bart_model_name': self.bart_model_name,
                'num_encoder_latents': self.num_encoder_latents,
                'dim_ae': self.dim_ae,
                'perceiver_depth': self.perceiver_depth
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_step_{self.step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Saved best model at step {self.step}")
        
        logger.info(f"💾 Saved checkpoint at step {self.step}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
        """
        # Calculate total steps
        total_steps = len(train_dataloader) * self.max_epochs // self.gradient_accumulation_steps
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(total_steps)
        
        # Setup wandb
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config={
                    'bart_model_name': self.bart_model_name,
                    'num_encoder_latents': self.num_encoder_latents,
                    'dim_ae': self.dim_ae,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'max_epochs': self.max_epochs
                }
            )
        
        logger.info(f"🏁 Starting Stage 1 training for {self.max_epochs} epochs")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Validation: {'Yes' if val_dataloader else 'No'}")
        
        # Training loop
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            
            # Training epoch
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
            
            for batch in pbar:
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                num_batches += 1
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'lr': f"{metrics['lr']:.2e}"
                })
                
                # Logging
                if self.step % self.log_interval == 0:
                    logger.info(f"Step {self.step}: loss={metrics['loss']:.4f}, ppl={metrics['perplexity']:.2f}")
                    
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': metrics['loss'],
                            'train/perplexity': metrics['perplexity'],
                            'train/learning_rate': metrics['lr'],
                            'step': self.step
                        })
                
                # Validation
                if val_dataloader and self.step % self.validate_interval == 0:
                    val_metrics = self.validate(val_dataloader)
                    
                    logger.info(f"Validation: loss={val_metrics['val_loss']:.4f}, ppl={val_metrics['val_perplexity']:.2f}")
                    
                    if self.use_wandb:
                        wandb.log({
                            'val/loss': val_metrics['val_loss'],
                            'val/perplexity': val_metrics['val_perplexity'],
                            'step': self.step
                        })
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.step % self.save_interval == 0:
                    self.save_checkpoint()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info("🎉 Stage 1 training completed!")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"📥 Loaded checkpoint from step {self.step}")
        
        return checkpoint 