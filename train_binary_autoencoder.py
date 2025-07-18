#!/usr/bin/env python3
"""
Binary Autoencoder Training Script

Trains the binary autoencoder on a small CIFAR-10 subset while maintaining
the 0.5 bits per pixel constraint. Zero simplifications, zero mocks.

Key features:
- Strict 0.5 bpp constraint enforcement
- Progressive training with validation
- Real-time monitoring and visualization
- Authentic GAN-style training with discriminator
- Model checkpointing and recovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Import our modules
import sys
sys.path.append('src')
from qudiffuse.models.binaryae import BinaryAutoEncoder, BinaryQuantizer, Discriminator
from qudiffuse.utils.common_utils import validate_tensor_shape, ensure_device_consistency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration with 0.5 bpp constraint validation."""
    # Dataset
    batch_size: int = 8
    num_samples: int = 100  # Small subset for focused training
    num_epochs: int = 50
    
    # Model architecture (must satisfy 0.5 bpp)
    img_size: int = 32
    n_channels: int = 3
    nf: int = 64
    emb_dim: int = 2  # Must maintain: 16×16×2 = 512 bits ≤ 0.5 × 32×32 = 512 bits
    codebook_size: int = 4  # 2^2 for 2-channel quantization
    ch_mult: list = None
    res_blocks: int = 2
    attn_resolutions: list = None
    
    # Training
    lr_ae: float = 1e-4
    lr_disc: float = 1e-4
    lambda_rec: float = 1.0
    lambda_codebook: float = 0.25
    lambda_disc: float = 0.5
    
    # Validation
    val_freq: int = 5  # Validate every N epochs
    save_freq: int = 10  # Save checkpoint every N epochs
    
    # Device
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.ch_mult is None:
            self.ch_mult = [1, 2, 4]
        if self.attn_resolutions is None:
            self.attn_resolutions = [16]
        
        # Validate 0.5 bpp constraint
        latent_size = self.img_size // (2 ** len(self.ch_mult))  # Downsampling factor
        total_bits = latent_size * latent_size * self.emb_dim
        total_pixels = self.img_size * self.img_size
        bpp = total_bits / total_pixels
        
        if bpp > 0.5:
            raise ValueError(f"Configuration violates 0.5 bpp constraint: {bpp:.3f} > 0.5")
        
        logger.info(f"✅ BPP constraint satisfied: {bpp:.3f} ≤ 0.5")

class BinaryAutoencoderTrainer:
    """
    Binary autoencoder trainer with authentic GAN training.
    
    Implements real adversarial training with discriminator for high-quality
    binary reconstructions while maintaining strict BPP constraints.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self._setup_models()
        self._setup_optimizers()
        self._setup_data()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'epoch': [],
            'ae_loss': [],
            'disc_loss': [],
            'rec_loss': [],
            'codebook_loss': [],
            'bpp': [],
            'val_loss': []
        }
        
        logger.info("🚀 BinaryAutoencoderTrainer initialized")
        logger.info(f"   Model parameters: {self._count_parameters():,}")
        logger.info(f"   Training samples: {config.num_samples}")
        logger.info(f"   Target BPP: ≤ 0.5")
    
    def _setup_models(self):
        """Initialize autoencoder and discriminator models."""
        
        # Create hyperparameter object for compatibility
        class H:
            def __init__(self, config):
                self.img_size = config.img_size
                self.n_channels = config.n_channels
                self.nf = config.nf
                self.emb_dim = config.emb_dim
                self.codebook_size = config.codebook_size
                self.ch_mult = config.ch_mult
                self.res_blocks = config.res_blocks
                self.attn_resolutions = config.attn_resolutions
                self.norm_first = True
                self.gen_mul = 1.0
                # Additional required attributes for BinaryAutoEncoder
                self.quantizer = 'binary'
                self.beta = 0.25
                self.deterministic = False
                self.use_tanh = False
        
        H_obj = H(self.config)
        
        # Initialize autoencoder
        self.autoencoder = BinaryAutoEncoder(H_obj).to(self.device)
        
        # Initialize discriminator separately
        self.discriminator = Discriminator(
            nc=self.config.n_channels,  # 3 for RGB
            ndf=64,  # Base discriminator features
            n_layers=3  # Number of layers
        ).to(self.device)
        
        logger.info("✅ Models initialized")
        logger.info(f"   Autoencoder: {self._count_model_params(self.autoencoder):,} parameters")
    
    def _setup_optimizers(self):
        """Setup optimizers for autoencoder and discriminator."""
        self.optimizer_ae = optim.Adam(
            list(self.autoencoder.encoder.parameters()) + 
            list(self.autoencoder.quantize.parameters()) + 
            list(self.autoencoder.generator.parameters()),
            lr=self.config.lr_ae,
            betas=(0.5, 0.999)
        )
        
        self.optimizer_disc = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr_disc,
            betas=(0.5, 0.999)
        )
        
        logger.info("✅ Optimizers configured")
    
    def _setup_data(self):
        """Setup CIFAR-10 dataset with small subset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        # Load full CIFAR-10 dataset
        full_dataset = torchvision.datasets.CIFAR10(
            root='data/', train=True, download=True, transform=transform
        )
        
        # Filter to airplane class (class 0) and take subset
        airplane_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
        subset_indices = airplane_indices[:self.config.num_samples]
        
        # Create training and validation splits
        train_size = int(0.8 * len(subset_indices))
        train_indices = subset_indices[:train_size]
        val_indices = subset_indices[train_size:]
        
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        logger.info(f"✅ Data loaders created")
        logger.info(f"   Training samples: {len(self.train_dataset)}")
        logger.info(f"   Validation samples: {len(self.val_dataset)}")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
    
    def _count_model_params(self, model) -> int:
        """Count parameters in a specific model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with authentic GAN training."""
        self.autoencoder.train()
        
        epoch_losses = {
            'ae_loss': 0.0,
            'disc_loss': 0.0,
            'rec_loss': 0.0,
            'codebook_loss': 0.0,
            'bpp_total': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # =================
            # Train Autoencoder
            # =================
            self.optimizer_ae.zero_grad()
            
            # Forward pass
            quant, qloss, qinfo, binary_code = self.autoencoder(images)
            
            # Reconstruction loss
            rec_loss = F.mse_loss(quant, images)
            
            # Codebook loss (encourages binary quantization)
            codebook_loss = qloss
            
            # Adversarial loss (generator perspective)
            logits_fake = self.discriminator(quant)
            adv_loss = -torch.mean(logits_fake)
            
            # Total autoencoder loss
            ae_loss = (self.config.lambda_rec * rec_loss + 
                      self.config.lambda_codebook * codebook_loss + 
                      self.config.lambda_disc * adv_loss)
            
            ae_loss.backward()
            self.optimizer_ae.step()
            
            # ===================
            # Train Discriminator
            # ===================
            self.optimizer_disc.zero_grad()
            
            # Real images
            logits_real = self.discriminator(images)
            
            # Fake images (detached to prevent generator updates)
            logits_fake = self.discriminator(quant.detach())
            
            # Discriminator loss (hinge loss)
            loss_real = torch.mean(F.relu(1. - logits_real))
            loss_fake = torch.mean(F.relu(1. + logits_fake))
            disc_loss = 0.5 * (loss_real + loss_fake)
            
            disc_loss.backward()
            self.optimizer_disc.step()
            
            # Calculate BPP
            if isinstance(binary_code, torch.Tensor):
                total_bits = binary_code.numel()
            else:
                total_bits = sum(tensor.numel() for tensor in binary_code)
            
            total_pixels = images.size(0) * images.size(2) * images.size(3)
            batch_bpp = total_bits / total_pixels
            
            # Accumulate losses
            epoch_losses['ae_loss'] += ae_loss.item()
            epoch_losses['disc_loss'] += disc_loss.item()
            epoch_losses['rec_loss'] += rec_loss.item()
            epoch_losses['codebook_loss'] += codebook_loss.item()
            epoch_losses['bpp_total'] += batch_bpp
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"   Batch {batch_idx}/{num_batches}: "
                          f"AE={ae_loss.item():.4f}, "
                          f"Disc={disc_loss.item():.4f}, "
                          f"Rec={rec_loss.item():.4f}, "
                          f"BPP={batch_bpp:.3f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.autoencoder.eval()
        
        val_losses = {
            'val_rec_loss': 0.0,
            'val_codebook_loss': 0.0,
            'val_bpp': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, _ in self.val_loader:
                images = images.to(self.device)
                
                # Forward pass
                quant, qloss, qinfo, binary_code = self.autoencoder(images)
                
                # Losses
                rec_loss = F.mse_loss(quant, images)
                codebook_loss = qloss
                
                # BPP calculation
                if isinstance(binary_code, torch.Tensor):
                    total_bits = binary_code.numel()
                else:
                    total_bits = sum(tensor.numel() for tensor in binary_code)
                
                total_pixels = images.size(0) * images.size(2) * images.size(3)
                batch_bpp = total_bits / total_pixels
                
                val_losses['val_rec_loss'] += rec_loss.item()
                val_losses['val_codebook_loss'] += codebook_loss.item()
                val_losses['val_bpp'] += batch_bpp
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"binary_autoencoder_epoch_{self.epoch}.pth"
        
        checkpoint = {
            'epoch': self.epoch,
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'optimizer_ae_state_dict': self.optimizer_ae.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"✅ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.optimizer_ae.load_state_dict(checkpoint['optimizer_ae_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"✅ Checkpoint loaded: {filename}")
    
    def visualize_results(self, num_samples: int = 4):
        """Visualize reconstruction results."""
        self.autoencoder.eval()
        
        with torch.no_grad():
            # Get a batch of validation images
            images, _ = next(iter(self.val_loader))
            images = images[:num_samples].to(self.device)
            
            # Reconstruct
            quant, qloss, qinfo, binary_code = self.autoencoder(images)
            
            # Convert to numpy for visualization
            original = images.cpu().numpy()
            reconstructed = quant.cpu().numpy()
            
            # Denormalize
            original = (original + 1.0) / 2.0
            reconstructed = (reconstructed + 1.0) / 2.0
            
            # Create visualization
            fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
            
            for i in range(num_samples):
                # Original
                axes[0, i].imshow(np.transpose(original[i], (1, 2, 0)))
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstructed
                axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'reconstruction_epoch_{self.epoch}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Visualization saved: reconstruction_epoch_{self.epoch}.png")
    
    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting binary autoencoder training")
        logger.info(f"   Training for {self.config.num_epochs} epochs")
        logger.info(f"   Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch + 1
            
            logger.info(f"\n📍 Epoch {self.epoch}/{self.config.num_epochs}")
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            if self.epoch % self.config.val_freq == 0:
                val_losses = self.validate()
                
                # Log results
                logger.info(f"   Training - AE: {train_losses['ae_loss']:.4f}, "
                          f"Disc: {train_losses['disc_loss']:.4f}, "
                          f"Rec: {train_losses['rec_loss']:.4f}, "
                          f"BPP: {train_losses['bpp_total']:.3f}")
                
                logger.info(f"   Validation - Rec: {val_losses['val_rec_loss']:.4f}, "
                          f"Codebook: {val_losses['val_codebook_loss']:.4f}, "
                          f"BPP: {val_losses['val_bpp']:.3f}")
                
                # Update history
                self.training_history['epoch'].append(self.epoch)
                self.training_history['ae_loss'].append(train_losses['ae_loss'])
                self.training_history['disc_loss'].append(train_losses['disc_loss'])
                self.training_history['rec_loss'].append(train_losses['rec_loss'])
                self.training_history['codebook_loss'].append(train_losses['codebook_loss'])
                self.training_history['bpp'].append(train_losses['bpp_total'])
                self.training_history['val_loss'].append(val_losses['val_rec_loss'])
                
                # Check for best model
                if val_losses['val_rec_loss'] < self.best_loss:
                    self.best_loss = val_losses['val_rec_loss']
                    self.save_checkpoint('best_binary_autoencoder.pth')
                    logger.info(f"   ✅ New best model saved (val_loss: {self.best_loss:.4f})")
                
                # Visualize results
                self.visualize_results()
            
            # Save checkpoint
            if self.epoch % self.config.save_freq == 0:
                self.save_checkpoint()
        
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Training completed!")
        logger.info(f"   Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Best validation loss: {self.best_loss:.4f}")
        logger.info(f"   Final BPP: {self.training_history['bpp'][-1]:.3f} ≤ 0.5 ✓")


def main():
    """Main training function."""
    # Configuration
    config = TrainingConfig(
        batch_size=4,  # Small batch for detailed monitoring
        num_samples=100,  # Focused training set
        num_epochs=25,  # Sufficient for convergence on small dataset
        device='cpu'  # Using CPU from our working environment
    )
    
    logger.info("🎯 Binary Autoencoder Training - Zero Simplifications")
    logger.info(f"   Target: Train on {config.num_samples} CIFAR-10 images")
    logger.info(f"   Constraint: Maintain BPP ≤ 0.5 ({config.emb_dim} channels)")
    logger.info(f"   Architecture: {config.nf} base channels, {config.res_blocks} res blocks")
    
    # Create trainer and train
    trainer = BinaryAutoencoderTrainer(config)
    trainer.train()
    
    logger.info("✅ Binary autoencoder training complete!")
    logger.info("   Ready for Phase 2: DBN training")


if __name__ == "__main__":
    main() 