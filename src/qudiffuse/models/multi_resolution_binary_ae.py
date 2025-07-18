"""
Multi-Resolution Binary Autoencoder

This module extends the BinaryLatentDiffusion autoencoder to support
hierarchical binary latent spaces at multiple resolutions, making it
compatible with our DBN-based diffusion system.

Key Features:
- Multi-scale binary quantization
- Hierarchical latent space organization  
- Compatible with existing DBN structure
- Progressive reconstruction from coarse to fine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
import logging

from .binaryae import ResBlock, AttnBlock, Downsample, Upsample, BinaryQuantizer
from qudiffuse.utils.vqgan_utils import normalize

logger = logging.getLogger(__name__)


class MultiResolutionEncoder(nn.Module):
    """
    Multi-resolution encoder that produces hierarchical binary latents.
    
    This encoder creates multiple binary latent representations at different
    spatial resolutions, enabling hierarchical processing by the DBN system.
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 nf: int = 128, 
                 ch_mult: List[int] = [1, 1, 2, 2, 4],
                 num_res_blocks: int = 2,
                 resolution: int = 256,
                 attn_resolutions: List[int] = [16],
                 latent_dims: List[int] = [64, 128, 256],  # dimensions for each resolution level
                 target_resolutions: List[int] = [32, 16, 8]):  # spatial resolutions for latent levels
        super().__init__()
        
        self.in_channels = in_channels
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions
        self.latent_dims = latent_dims
        self.target_resolutions = target_resolutions
        self.num_latent_levels = len(latent_dims)
        
        # Validate input parameters
        assert len(latent_dims) == len(target_resolutions), \
            f"Number of latent dimensions ({len(latent_dims)}) must match number of target resolutions ({len(target_resolutions)})"
        
        # Build main encoder backbone
        self.encoder_blocks = nn.ModuleList()
        
        # Initial conv
        curr_res = resolution
        block_in_ch = in_channels
        
        # First conv to get to nf channels
        self.initial_conv = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)
        block_in_ch = nf
        
        # Track intermediate features for multi-resolution outputs
        self.feature_extraction_points = []
        
        # Downsampling blocks
        for i in range(self.num_resolutions):
            level_blocks = nn.ModuleList()
            
            block_out_ch = nf * ch_mult[i]
            
            # Add residual blocks
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                
                # Add attention if at the right resolution
                if curr_res in attn_resolutions:
                    level_blocks.append(AttnBlock(block_in_ch))
            
            self.encoder_blocks.append(level_blocks)
            
            # Check if this resolution matches one of our target latent resolutions
            if curr_res in target_resolutions:
                level_idx = target_resolutions.index(curr_res)
                self.feature_extraction_points.append((i, level_idx, curr_res, block_in_ch))
            
            # Downsample (except for last level)
            if i != self.num_resolutions - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample(block_in_ch)]))
                curr_res = curr_res // 2
        
        # Final processing blocks
        final_blocks = nn.ModuleList()
        final_blocks.append(ResBlock(block_in_ch, block_in_ch))
        final_blocks.append(AttnBlock(block_in_ch))
        final_blocks.append(ResBlock(block_in_ch, block_in_ch))
        final_blocks.append(normalize(block_in_ch))
        
        self.encoder_blocks.append(final_blocks)
        
        # Create output projection layers for each latent level
        self.latent_projections = nn.ModuleList()
        for i, (level_idx, target_res, latent_dim) in enumerate(zip(range(len(target_resolutions)), target_resolutions, latent_dims)):
            # Find the corresponding feature extraction point
            feature_channels = None
            for extraction_point in self.feature_extraction_points:
                if extraction_point[1] == level_idx:
                    feature_channels = extraction_point[3]
                    break
            
            if feature_channels is None:
                # Use final features if no specific extraction point
                feature_channels = block_in_ch
            
            projection = nn.Sequential(
                nn.Conv2d(feature_channels, latent_dim, kernel_size=3, stride=1, padding=1),
                normalize(latent_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)
            )
            self.latent_projections.append(projection)
        
        logger.info(f"🏗️ MultiResolutionEncoder initialized:")
        logger.info(f"   Target resolutions: {target_resolutions}")
        logger.info(f"   Latent dimensions: {latent_dims}")
        logger.info(f"   Feature extraction points: {len(self.feature_extraction_points)}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass producing multi-resolution latents.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            List of latent tensors at different resolutions [B, C_i, H_i, W_i]
        """
        # Initial convolution
        h = self.initial_conv(x)
        
        # Store intermediate features
        intermediate_features = []
        
        # Process through encoder blocks
        block_idx = 0
        curr_res = self.resolution
        
        for i in range(self.num_resolutions):
            # Process residual and attention blocks for this level
            level_blocks = self.encoder_blocks[block_idx]
            for block in level_blocks:
                h = block(h)
            block_idx += 1
            
            # Check if we need to extract features at this resolution
            if curr_res in self.target_resolutions:
                level_idx = self.target_resolutions.index(curr_res)
                intermediate_features.append((level_idx, h.clone()))
            
            # Apply downsampling (except for last level)
            if i != self.num_resolutions - 1:
                downsample_blocks = self.encoder_blocks[block_idx]
                for block in downsample_blocks:
                    h = block(h)
                block_idx += 1
                curr_res = curr_res // 2
        
        # Final processing
        final_blocks = self.encoder_blocks[block_idx]
        for block in final_blocks:
            h = block(h)
        
        # Add final features if not already captured
        if curr_res in self.target_resolutions:
            level_idx = self.target_resolutions.index(curr_res)
            intermediate_features.append((level_idx, h.clone()))
        
        # Sort features by level index
        intermediate_features.sort(key=lambda x: x[0])
        
        # Project to latent dimensions
        latent_outputs = []
        for i, (level_idx, features) in enumerate(intermediate_features):
            if i < len(self.latent_projections):
                latent = self.latent_projections[i](features)
                latent_outputs.append(latent)
        
        # Ensure we have outputs for all target resolutions
        while len(latent_outputs) < len(self.target_resolutions):
            # Use final features for missing levels
            latent = self.latent_projections[len(latent_outputs)](h)
            latent_outputs.append(latent)
        
        return latent_outputs


class MultiResolutionQuantizer(nn.Module):
    """
    Multi-resolution binary quantizer for hierarchical latent spaces.
    
    This quantizer applies binary quantization at multiple resolution levels,
    producing hierarchical binary codes suitable for DBN processing.
    """
    
    def __init__(self,
                 codebook_sizes: List[int],
                 latent_dims: List[int],
                 use_tanh: bool = False):
        super().__init__()
        
        self.num_levels = len(codebook_sizes)
        assert len(codebook_sizes) == len(latent_dims), \
            f"Number of codebook sizes ({len(codebook_sizes)}) must match latent dimensions ({len(latent_dims)})"
        
        self.codebook_sizes = codebook_sizes
        self.latent_dims = latent_dims
        self.use_tanh = use_tanh
        
        # Create quantizers for each resolution level
        self.quantizers = nn.ModuleList()
        for i in range(self.num_levels):
            quantizer = BinaryQuantizer(
                codebook_size=codebook_sizes[i],
                emb_dim=latent_dims[i],
                num_hiddens=latent_dims[i],
                use_tanh=use_tanh
            )
            self.quantizers.append(quantizer)
        
        logger.info(f"🔢 MultiResolutionQuantizer initialized with {self.num_levels} levels")
    
    def forward(self, latent_list: List[torch.Tensor], deterministic: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor, Dict, List[torch.Tensor]]:
        """
        Quantize multi-resolution latents.
        
        Args:
            latent_list: List of latent tensors at different resolutions
            deterministic: Whether to use deterministic quantization
            
        Returns:
            Tuple of (quantized_latents, total_codebook_loss, stats, binary_codes)
        """
        quantized_latents = []
        binary_codes = []
        total_codebook_loss = 0.0
        stats = {"binary_codes": []}
        
        for i, (latent, quantizer) in enumerate(zip(latent_list, self.quantizers)):
            if i < len(latent_list):  # Ensure we don't exceed available latents
                z_q, codebook_loss, quant_stats, binary_code = quantizer(latent, deterministic=deterministic)
                
                quantized_latents.append(z_q)
                binary_codes.append(binary_code)
                total_codebook_loss = total_codebook_loss + codebook_loss
                stats["binary_codes"].append(quant_stats["binary_code"])
        
        # Average the codebook loss
        if len(quantized_latents) > 0:
            total_codebook_loss = total_codebook_loss / len(quantized_latents)
        
        return quantized_latents, total_codebook_loss, stats, binary_codes


class MultiResolutionDecoder(nn.Module):
    """
    Multi-resolution decoder that reconstructs from hierarchical binary latents.
    
    This decoder takes hierarchical binary latents and progressively upsamples
    and combines them to reconstruct the final image.
    """
    
    def __init__(self,
                 latent_dims: List[int],
                 nf: int = 128,
                 ch_mult: List[int] = [1, 1, 2, 2, 4],
                 num_res_blocks: int = 2,
                 resolution: int = 256,
                 attn_resolutions: List[int] = [16],
                 out_channels: int = 3,
                 norm_first: bool = False):
        super().__init__()
        
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions
        self.out_channels = out_channels
        self.norm_first = norm_first
        
        # Calculate the combined latent dimension
        self.combined_latent_dim = sum(latent_dims)
        
        # Initial processing of combined latents
        block_in_ch = self.nf * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        
        self.decoder_blocks = nn.ModuleList()
        
        # Initial conv from combined latents
        if norm_first:
            self.decoder_blocks.append(normalize(self.combined_latent_dim))
        
        self.initial_conv = nn.Conv2d(self.combined_latent_dim, block_in_ch, kernel_size=3, stride=1, padding=1)
        
        # Non-local attention block
        initial_blocks = nn.ModuleList()
        initial_blocks.append(ResBlock(block_in_ch, block_in_ch))
        initial_blocks.append(AttnBlock(block_in_ch))
        initial_blocks.append(ResBlock(block_in_ch, block_in_ch))
        self.decoder_blocks.append(initial_blocks)
        
        # Upsampling blocks
        for i in reversed(range(self.num_resolutions)):
            level_blocks = nn.ModuleList()
            block_out_ch = nf * ch_mult[i]
            
            # Add residual blocks
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                
                if curr_res in attn_resolutions:
                    level_blocks.append(AttnBlock(block_in_ch))
            
            self.decoder_blocks.append(level_blocks)
            
            # Upsample (except for last level)
            if i != 0:
                self.decoder_blocks.append(nn.ModuleList([Upsample(block_in_ch)]))
                curr_res = curr_res * 2
        
        # Final output conv
        final_blocks = nn.ModuleList()
        final_blocks.append(normalize(block_in_ch))
        final_blocks.append(nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.decoder_blocks.append(final_blocks)
        
        logger.info(f"🔄 MultiResolutionDecoder initialized:")
        logger.info(f"   Combined latent dim: {self.combined_latent_dim}")
        logger.info(f"   Output resolution: {resolution}")
    
    def forward(self, quantized_latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode from multi-resolution quantized latents.
        
        Args:
            quantized_latents: List of quantized latent tensors
            
        Returns:
            Reconstructed image [B, C, H, W]
        """
        if not quantized_latents:
            raise ValueError("No quantized latents provided")
        
        # Resize all latents to the smallest resolution
        target_size = quantized_latents[-1].shape[-2:]  # Use last (smallest) latent size
        
        resized_latents = []
        for latent in quantized_latents:
            if latent.shape[-2:] != target_size:
                resized = F.interpolate(latent, size=target_size, mode='nearest')
            else:
                resized = latent
            resized_latents.append(resized)
        
        # Combine all latents along channel dimension
        combined_latents = torch.cat(resized_latents, dim=1)
        
        # Initial convolution
        h = self.initial_conv(combined_latents)
        
        # Process through decoder blocks
        for block_group in self.decoder_blocks:
            if isinstance(block_group, nn.ModuleList):
                for block in block_group:
                    h = block(h)
            else:
                h = block_group(h)
        
        return h


class MultiResolutionBinaryAutoEncoder(nn.Module):
    """
    Complete multi-resolution binary autoencoder.
    
    This autoencoder produces hierarchical binary latent representations
    at multiple resolutions, making it compatible with our DBN-based
    diffusion system while leveraging the BinaryLatentDiffusion architecture.
    """
    
    def __init__(self,
                 # Image parameters
                 in_channels: int = 3,
                 resolution: int = 256,
                 
                 # Architecture parameters
                 nf: int = 128,
                 ch_mult: List[int] = [1, 1, 2, 2, 4],
                 num_res_blocks: int = 2,
                 attn_resolutions: List[int] = [16],
                 
                 # Multi-resolution parameters
                 latent_dims: List[int] = [64, 128, 256],
                 target_resolutions: List[int] = [32, 16, 8],
                 codebook_sizes: List[int] = [256, 512, 1024],
                 
                 # Training parameters
                 use_tanh: bool = False,
                 deterministic: bool = False,
                 norm_first: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.resolution = resolution
        self.latent_dims = latent_dims
        self.target_resolutions = target_resolutions
        self.codebook_sizes = codebook_sizes
        self.deterministic = deterministic
        self.num_levels = len(latent_dims)
        
        # Validate parameters
        assert len(latent_dims) == len(target_resolutions) == len(codebook_sizes), \
            "All multi-resolution parameter lists must have the same length"
        
        # Build components
        self.encoder = MultiResolutionEncoder(
            in_channels=in_channels,
            nf=nf,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            latent_dims=latent_dims,
            target_resolutions=target_resolutions
        )
        
        self.quantizer = MultiResolutionQuantizer(
            codebook_sizes=codebook_sizes,
            latent_dims=latent_dims,
            use_tanh=use_tanh
        )
        
        self.decoder = MultiResolutionDecoder(
            latent_dims=latent_dims,
            nf=nf,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            out_channels=in_channels,
            norm_first=norm_first
        )
        
        logger.info(f"🚀 MultiResolutionBinaryAutoEncoder initialized:")
        logger.info(f"   Input resolution: {resolution}x{resolution}")
        logger.info(f"   Latent levels: {len(latent_dims)}")
        logger.info(f"   Total parameters: {sum(p.numel() for p in self.parameters())}")
    
    def forward(self, x: Optional[torch.Tensor] = None, code_only: bool = False, codes: Optional[List[torch.Tensor]] = None) -> Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Forward pass through multi-resolution autoencoder.
        
        Args:
            x: Input image [B, C, H, W] (required if codes is None)
            code_only: If True, return only binary codes
            codes: Pre-computed binary codes for reconstruction
            
        Returns:
            Tuple of (reconstruction, codebook_loss, stats) or binary_codes if code_only=True
        """
        if codes is None:
            if x is None:
                raise ValueError("Input x is required when codes is not provided")
                
            # Encode to multi-resolution latents
            latent_list = self.encoder(x)
            
            # Quantize at each resolution
            quantized_latents, codebook_loss, stats, binary_codes = self.quantizer(
                latent_list, deterministic=self.deterministic
            )
            
            if code_only:
                return binary_codes
        else:
            # Use provided codes - convert binary codes to quantized latents
            quantized_latents = []
            for i, code in enumerate(codes):
                if i < len(self.quantizer.quantizers):
                    # Convert binary code back to quantized latent using embedding weights
                    quantizer = self.quantizer.quantizers[i]
                    z_q = torch.einsum("b n h w, n d -> b d h w", code, quantizer.embed.weight)
                    quantized_latents.append(z_q)
            
            codebook_loss = torch.tensor(0.0, device=codes[0].device if codes else torch.device('cpu'))
            stats = {}
        
        # Decode from quantized latents
        reconstruction = self.decoder(quantized_latents)
        
        return reconstruction, codebook_loss, stats
    
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode input to binary codes."""
        result = self.forward(x, code_only=True)
        return result  # type: ignore  # We know this returns List[torch.Tensor] when code_only=True
    
    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """Decode from binary codes."""
        reconstruction, _, _ = self.forward(codes=codes)
        return reconstruction
    
    def get_latent_shapes(self) -> List[Tuple[int, int, int]]:
        """Get the shapes of hierarchical latents for DBN initialization."""
        shapes = []
        for i, (latent_dim, target_res) in enumerate(zip(self.latent_dims, self.target_resolutions)):
            # Each latent level contributes codebook_size channels
            shapes.append((self.codebook_sizes[i], target_res, target_res))
        return shapes 