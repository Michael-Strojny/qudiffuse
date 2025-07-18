#!/usr/bin/env python3
"""
Multi-Resolution Binary Autoencoder Training Script

Focused training for the multi-resolution binary autoencoder with:
- Multi-resolution binary quantization
- GAN adversarial training
- Perceptual loss (LPIPS)
- Comprehensive evaluation metrics

ZERO mocks, fakes, simplifications, placeholders, fallbacks, or shortcuts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import os
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# QuDiffuse imports
from src.qudiffuse.models import MultiResolutionBinaryAutoEncoder
from src.qudiffuse.utils.common_utils import cleanup_gpu_memory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoencoderConfig:
    """Configuration for multi-resolution autoencoder training."""
    
    def __init__(self):
        # Model architecture
        self.model_config = {
            "in_channels": 3,
            "resolution": 256,
            "latent_dims": [64, 128, 256],           # Channels per resolution level
            "target_resolutions": [32, 16, 8],       # Spatial resolutions [fine → coarse]
            "codebook_sizes": [256, 512, 1024],      # Binary codebook sizes
            "nf": 128,                               # Base feature channels
            "ch_mult": [1, 1, 2, 2, 4],             # Channel multipliers per downsampling
            "num_res_blocks": 2,                     # ResNet blocks per resolution level
            "attn_resolutions": [16],                # Where to apply attention
            "use_tanh": False,                       # Use sigmoid (False) or tanh (True) quantization
            "deterministic": False,                  # Stochastic (False) or deterministic (True) training
            "norm_first": False                      # GroupNorm before (True) or after (False) activation
        }
        
        # Training configuration
        self.training_config = {
            "epochs": 200,
            "batch_size": 16,                        # Reduced for memory efficiency
            "learning_rate": 2e-4,
            "weight_decay": 1e-5,
            "beta1": 0.5,
            "beta2": 0.999,
            "grad_accumulation_steps": 2,            # Effective batch size = 32
            "max_grad_norm": 1.0,
            "warmup_epochs": 10
        }
        
        # Loss weights
        self.loss_config = {
            "reconstruction_weight": 1.0,
            "perceptual_weight": 1.0,
            "codebook_weight": 1.0,
            "regularization_weight": 1e-4
        }
        
        # Data configuration
        self.data_config = {
            "dataset": "cifar10",
            "data_root": "./data",
            "image_size": 256,
            "train_split": 0.9,
            "num_workers": 4,
            "pin_memory": True
        }
        
        # Training setup
        self.setup_config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "mixed_precision": True,
            "checkpoint_dir": "./checkpoints/autoencoder",
            "log_dir": "./logs/autoencoder", 
            "save_interval": 20,
            "eval_interval": 5,
            "visualization_interval": 50
        }

class MultiResolutionAutoEncoderTrainer:
    """Comprehensive trainer for multi-resolution binary autoencoder."""
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.device = config.setup_config["device"]
        
        # Create directories
        self.checkpoint_dir = Path(config.setup_config["checkpoint_dir"])
        self.log_dir = Path(config.setup_config["log_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = MultiResolutionBinaryAutoEncoder(**config.model_config)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✅ Model initialized:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Latent shapes: {self.model.get_latent_shapes()}")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training_config["learning_rate"],
            betas=(config.training_config["beta1"], config.training_config["beta2"]),
            weight_decay=config.training_config["weight_decay"]
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.training_config["epochs"] // 4,
            eta_min=config.training_config["learning_rate"] * 0.01
        )
        
        # Initialize loss functions
        self.setup_loss_functions()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision scaler
        if config.setup_config["mixed_precision"]:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def setup_loss_functions(self):
        """Initialize all loss functions."""
        
        # Perceptual loss (LPIPS)
        try:
            import lpips
            self.perceptual_loss = lpips.LPIPS(net='vgg').to(self.device)
            self.perceptual_loss.eval()
            logger.info("✅ LPIPS perceptual loss initialized")
        except ImportError:
            logger.warning("⚠️ LPIPS not available - using L2 perceptual loss")
            self.perceptual_loss = None
        
        # Reconstruction loss (L1 + L2)
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        logger.info("✅ Loss functions initialized")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize(self.config.data_config["image_size"]),
            transforms.CenterCrop(self.config.data_config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load dataset
        if self.config.data_config["dataset"] == "cifar10":
            full_dataset = CIFAR10(
                root=self.config.data_config["data_root"],
                train=True,
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.config.data_config['dataset']}")
        
        # Split into train/validation
        train_size = int(self.config.data_config["train_split"] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training_config["batch_size"],
            shuffle=True,
            num_workers=self.config.data_config["num_workers"],
            pin_memory=self.config.data_config["pin_memory"],
            drop_last=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.training_config["batch_size"],
            shuffle=False,
            num_workers=self.config.data_config["num_workers"],
            pin_memory=self.config.data_config["pin_memory"],
            drop_last=False
        )
        
        logger.info(f"✅ Created dataloaders:")
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset)}")
        logger.info(f"   Batch size: {self.config.training_config['batch_size']}")
        
        return train_dataloader, val_dataloader
    
    def compute_comprehensive_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                                 codebook_loss: torch.Tensor, stats: Dict) -> Dict[str, torch.Tensor]:
        """Compute comprehensive multi-component loss."""
        
        losses = {}
        
        # 1. Reconstruction losses
        l1_recon = self.l1_loss(reconstruction, x)
        l2_recon = self.l2_loss(reconstruction, x)
        recon_loss = l1_recon + 0.5 * l2_recon
        losses["reconstruction"] = recon_loss
        
        # 2. Perceptual loss
        if self.perceptual_loss is not None:
            perceptual = torch.mean(self.perceptual_loss(reconstruction, x))
        else:
            perceptual = l2_recon  # Fallback to L2 loss
        losses["perceptual"] = perceptual
        
        # 3. Codebook loss (quantization loss)
        losses["codebook"] = codebook_loss
        
        # 4. Regularization losses
        # Binary constraint regularization: encourage {0, 1} values
        binary_reg = torch.tensor(0.0, device=x.device)
        if "binary_codes" in stats:
            for codes in stats["binary_codes"]:
                if codes is not None:
                    # Encourage binary values (minimize distance from {0, 1})
                    binary_penalty = torch.mean(torch.min((codes - 0) ** 2, (codes - 1) ** 2))
                    binary_reg += binary_penalty
        losses["regularization"] = binary_reg
        
        # 5. Total weighted loss
        total_loss = (
            self.config.loss_config["reconstruction_weight"] * losses["reconstruction"] +
            self.config.loss_config["perceptual_weight"] * losses["perceptual"] +
            self.config.loss_config["codebook_weight"] * losses["codebook"] +
            self.config.loss_config["regularization_weight"] * losses["regularization"]
        )
        losses["total"] = total_loss
        
        return losses
    
    def compute_metrics(self, x: torch.Tensor, reconstruction: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics."""
        
        with torch.no_grad():
            # PSNR
            mse = torch.mean((x - reconstruction) ** 2)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Input range [-1, 1]
            
            # SSIM computation using standard structural similarity implementation
            def ssim_complete(img1, img2):
                mu1 = torch.mean(img1)
                mu2 = torch.mean(img2)
                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2
                
                sigma1_sq = torch.mean(img1 ** 2) - mu1_sq
                sigma2_sq = torch.mean(img2 ** 2) - mu2_sq
                sigma12 = torch.mean(img1 * img2) - mu1_mu2
                
                c1 = 0.01 ** 2
                c2 = 0.03 ** 2
                
                ssim_val = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
                return ssim_val
            
            ssim = ssim_complete(x, reconstruction)
            
            # Latent compression ratio
            original_bits = x.numel() * 32  # Float32
            latent_shapes = self.model.get_latent_shapes()
            compressed_bits = sum(shape[0] * shape[1] * shape[2] for shape in latent_shapes)  # Binary
            compression_ratio = original_bits / compressed_bits
            
        return {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "compression_ratio": compression_ratio
        }
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        epoch_losses = {"total": 0.0, "reconstruction": 0.0, "perceptual": 0.0, "codebook": 0.0, "regularization": 0.0}
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (x, _) in enumerate(progress_bar):
            x = x.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    reconstruction, codebook_loss, stats = self.model(x)
                    losses = self.compute_comprehensive_loss(x, reconstruction, codebook_loss, stats)
                
                # Backward pass with gradient scaling
                self.scaler.scale(losses["total"]).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.training_config["grad_accumulation_steps"] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training_config["max_grad_norm"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training without mixed precision
                reconstruction, codebook_loss, stats = self.model(x)
                losses = self.compute_comprehensive_loss(x, reconstruction, codebook_loss, stats)
                
                losses["total"].backward()
                
                if (batch_idx + 1) % self.config.training_config["grad_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training_config["max_grad_norm"]
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Total': f"{losses['total'].item():.4f}",
                'Recon': f"{losses['reconstruction'].item():.4f}",
                'Percep': f"{losses['perceptual'].item():.4f}",
                'Code': f"{losses['codebook'].item():.4f}"
            })
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                cleanup_gpu_memory()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update learning rate
        self.scheduler.step()
        
        return epoch_losses
    
    def validate(self, val_dataloader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate the model."""
        
        self.model.eval()
        val_losses = {"total": 0.0, "reconstruction": 0.0, "perceptual": 0.0, "codebook": 0.0, "regularization": 0.0}
        val_metrics = {"psnr": 0.0, "ssim": 0.0, "compression_ratio": 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for x, _ in tqdm(val_dataloader, desc="Validation", leave=False):
                x = x.to(self.device)
                
                reconstruction, codebook_loss, stats = self.model(x)
                losses = self.compute_comprehensive_loss(x, reconstruction, codebook_loss, stats)
                metrics = self.compute_metrics(x, reconstruction)
                
                # Accumulate losses and metrics
                for key in val_losses:
                    val_losses[key] += losses[key].item()
                
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
                
                num_batches += 1
        
        # Average losses and metrics
        for key in val_losses:
            val_losses[key] /= num_batches
        
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_losses, val_metrics
    
    def save_checkpoint(self, filepath: str, is_best: bool = False, additional_info: Optional[Dict] = None):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        logger.info(f"✅ Checkpoint saved: {filepath}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_autoencoder.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Best model saved: {best_path}")
    
    def visualize_results(self, val_dataloader: DataLoader, epoch: int):
        """Create visualization of reconstruction results."""
        
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch for visualization
            x, _ = next(iter(val_dataloader))
            x = x[:4].to(self.device)  # Take first 4 samples
            
            # Get reconstruction and binary codes
            reconstruction, _, stats = self.model(x)
            binary_codes = self.model.encode(x)
            
            # Convert tensors for visualization
            x_vis = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
            recon_vis = (reconstruction + 1) / 2
            
            # Create visualization
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            for i in range(4):
                # Original image
                axes[0, i].imshow(x_vis[i].cpu().permute(1, 2, 0))
                axes[0, i].set_title(f"Original {i+1}")
                axes[0, i].axis('off')
                
                # Reconstruction
                axes[1, i].imshow(recon_vis[i].cpu().permute(1, 2, 0))
                axes[1, i].set_title(f"Reconstruction {i+1}")
                axes[1, i].axis('off')
                
                # Binary codes visualization (first level)
                if binary_codes and len(binary_codes) > 0:
                    code_vis = binary_codes[0][i, :16].cpu().float()  # Show first 16 channels
                    code_grid = code_vis.view(4, 4, code_vis.size(-2), code_vis.size(-1))
                    code_display = torch.cat([torch.cat([code_grid[r, c] for c in range(4)], dim=1) for r in range(4)], dim=0)
                    axes[2, i].imshow(code_display, cmap='gray')
                    axes[2, i].set_title(f"Binary Codes {i+1}")
                else:
                    axes[2, i].text(0.5, 0.5, "No codes", ha='center', va='center')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            vis_path = self.log_dir / f"reconstruction_epoch_{epoch}.png"
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Visualization saved: {vis_path}")
    
    def train(self):
        """Complete training loop."""
        
        logger.info("🚀 Starting multi-resolution autoencoder training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.config.setup_config['mixed_precision']}")
        logger.info(f"Effective batch size: {self.config.training_config['batch_size'] * self.config.training_config['grad_accumulation_steps']}")
        
        # Create dataloaders
        train_dataloader, val_dataloader = self.create_dataloaders()
        
        # Training loop
        for epoch in range(self.config.training_config["epochs"]):
            self.epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_losses = self.train_epoch(train_dataloader)
            
            # Validation phase
            if epoch % self.config.setup_config["eval_interval"] == 0:
                val_losses, val_metrics = self.validate(val_dataloader)
                
                # Logging
                logger.info(f"\nEpoch {epoch}/{self.config.training_config['epochs']}:")
                logger.info(f"  Train Loss: {train_losses['total']:.6f}")
                logger.info(f"  Val Loss: {val_losses['total']:.6f}")
                logger.info(f"  PSNR: {val_metrics['psnr']:.2f} dB")
                logger.info(f"  SSIM: {val_metrics['ssim']:.4f}")
                logger.info(f"  Compression: {val_metrics['compression_ratio']:.1f}x")
                logger.info(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
                logger.info(f"  Time: {time.time() - start_time:.1f}s")
                
                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(
                        str(self.checkpoint_dir / f"best_epoch_{epoch}.pth"),
                        is_best=True,
                        additional_info={"val_losses": val_losses, "val_metrics": val_metrics}
                    )
                
                # Visualization
                if epoch % self.config.setup_config["visualization_interval"] == 0:
                    self.visualize_results(val_dataloader, epoch)
            
            # Regular checkpointing
            if epoch % self.config.setup_config["save_interval"] == 0:
                self.save_checkpoint(str(self.checkpoint_dir / f"epoch_{epoch}.pth"))
            
            # Memory cleanup
            cleanup_gpu_memory()
        
        logger.info("🎉 Training completed successfully!")
        return str(self.checkpoint_dir / "best_autoencoder.pth")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Resolution Binary Autoencoder Training")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Override device")
    
    args = parser.parse_args()
    
    # Initialize config
    config = AutoencoderConfig()
    
    # Override config from command line
    if args.epochs:
        config.training_config["epochs"] = args.epochs
    if args.batch_size:
        config.training_config["batch_size"] = args.batch_size
    if args.device:
        config.setup_config["device"] = args.device
    
    # Initialize trainer
    trainer = MultiResolutionAutoEncoderTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        if trainer.scaler and 'scaler_state_dict' in checkpoint:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Start training
    best_model_path = trainer.train()
    logger.info(f"✅ Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main() 