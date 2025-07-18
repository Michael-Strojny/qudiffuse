"""
Perceiver AutoEncoder for Binary Latent Compression

This module implements a Perceiver-based autoencoder that compresses variable-length
text sequences to fixed-length binary latent representations compatible with our
QuDiffuse binary diffusion system.

Key Features:
- Variable-length to fixed-length compression (lae=16, dae=256)
- Cross-attention based architecture 
- Binary latent output compatible with QuDiffuse
- Support for reasoning tasks with proper sequence handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""
    
    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           -(math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CrossAttention(nn.Module):
    """Cross-attention module for Perceiver."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        _, ctx_len, _ = context.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Attention
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [b, 1, 1, ctx_len]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feedforward module with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Single Perceiver transformer block."""
    
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.self_attn = CrossAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-attention to context
        x = x + self.cross_attn(self.norm1(x), context, mask)
        
        # Self-attention
        x = x + self.self_attn(self.norm2(x), x)
        
        # Feedforward
        x = x + self.ff(self.norm3(x))
        
        return x


class PerceiverBinaryAutoEncoder(nn.Module):
    """
    Perceiver AutoEncoder for binary latent compression.
    
    This autoencoder compresses variable-length text representations to fixed-length
    binary latents compatible with QuDiffuse, following the paper specifications:
    - Input: Variable-length BART encoder outputs
    - Output: Fixed-length binary latents (lae=16, dae=256)
    """
    
    def __init__(
        self,
        dim_lm: int = 768,           # BART hidden dimension
        dim_ae: int = 256,           # Autoencoder latent dimension (dae)
        num_encoder_latents: int = 16,  # Fixed latent length (lae)
        num_decoder_latents: int = 32,  # Decoder latent length
        depth: int = 6,              # Number of transformer layers
        num_heads: int = 8,          # Number of attention heads
        ff_mult: int = 4,            # Feedforward multiplier
        dropout: float = 0.1,        # Dropout rate
        max_seq_len: int = 512,      # Maximum input sequence length
        binary_quantization: bool = True,  # Enable binary quantization
        l2_normalize_latents: bool = False  # L2 normalize latents
    ):
        super().__init__()
        
        self.dim_lm = dim_lm
        self.dim_ae = dim_ae
        self.num_encoder_latents = num_encoder_latents
        self.num_decoder_latents = num_decoder_latents
        self.binary_quantization = binary_quantization
        self.l2_normalize_latents = l2_normalize_latents
        
        # Learnable latent queries for encoding
        self.latent_queries = nn.Parameter(torch.randn(num_encoder_latents, dim_ae))
        nn.init.normal_(self.latent_queries, std=0.02)
        
        # Input projection from LM dimension to AE dimension
        self.input_proj = nn.Linear(dim_lm, dim_ae)
        
        # Encoder: variable-length → fixed-length
        self.encoder_blocks = nn.ModuleList([
            PerceiverBlock(dim_ae, num_heads, ff_mult, dropout)
            for _ in range(depth)
        ])
        
        # Decoder: fixed-length → variable-length
        self.decoder_queries = nn.Parameter(torch.randn(num_decoder_latents, dim_lm))
        nn.init.normal_(self.decoder_queries, std=0.02)
        
        self.decoder_blocks = nn.ModuleList([
            PerceiverBlock(dim_lm, num_heads, ff_mult, dropout)
            for _ in range(depth)
        ])
        
        # Output projection back to LM dimension
        self.output_proj = nn.Linear(dim_lm, dim_lm)
        
        # Binary quantization layer (for QuDiffuse compatibility)
        if binary_quantization:
            self.binary_quantizer = nn.Sequential(
                nn.Linear(dim_ae, dim_ae),
                nn.Tanh()  # Output in [-1, 1] range for binary conversion
            )
        
        # Layer norms
        self.encoder_norm = nn.LayerNorm(dim_ae)
        self.decoder_norm = nn.LayerNorm(dim_lm)
        
        print(f"🔧 PerceiverBinaryAutoEncoder initialized:")
        print(f"   Input dimension: {dim_lm}")
        print(f"   Latent dimension: {dim_ae}")
        print(f"   Encoder latents: {num_encoder_latents}")
        print(f"   Binary quantization: {binary_quantization}")
    
    def encode(self, encoder_outputs: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode variable-length sequence to fixed-length binary latents.
        
        Args:
            encoder_outputs: BART encoder outputs [B, seq_len, dim_lm]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Binary latents [B, num_encoder_latents, dim_ae]
        """
        batch_size = encoder_outputs.size(0)
        
        # Project to autoencoder dimension
        context = self.input_proj(encoder_outputs)  # [B, seq_len, dim_ae]
        
        # Expand latent queries for batch
        latents = repeat(self.latent_queries, 'n d -> b n d', b=batch_size)
        
        # Cross-attention encoding
        for block in self.encoder_blocks:
            latents = block(latents, context, attention_mask)
        
        latents = self.encoder_norm(latents)
        
        # Apply binary quantization if enabled
        if self.binary_quantization:
            latents = self.binary_quantizer(latents)
            
            # Convert to binary representation for QuDiffuse
            # Tanh outputs [-1, 1] → convert to {0, 1} binary
            binary_latents = (latents > 0).float()
            
            # Use straight-through estimator for gradients
            latents = binary_latents + latents - latents.detach()
        
        # L2 normalization if requested
        if self.l2_normalize_latents:
            latents = F.normalize(latents, p=2, dim=-1)
        
        return latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode fixed-length latents back to variable-length sequence.
        
        Args:
            latents: Binary latents [B, num_encoder_latents, dim_ae]
            
        Returns:
            Decoded sequence [B, num_decoder_latents, dim_lm]
        """
        batch_size = latents.size(0)
        
        # Expand decoder queries for batch
        decoder_queries = repeat(self.decoder_queries, 'n d -> b n d', b=batch_size)
        
        # Cross-attention decoding
        for block in self.decoder_blocks:
            decoder_queries = block(decoder_queries, latents)
        
        output = self.decoder_norm(decoder_queries)
        output = self.output_proj(output)
        
        return output
    
    def forward(self, encoder_outputs: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder forward pass.
        
        Args:
            encoder_outputs: BART encoder outputs [B, seq_len, dim_lm]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Tuple of (decoded_outputs, binary_latents)
        """
        # Encode to binary latents
        latents = self.encode(encoder_outputs, attention_mask)
        
        # Decode back to sequence
        decoded = self.decode(latents)
        
        return decoded, latents
    
    def get_binary_latents_for_diffusion(self, encoder_outputs: torch.Tensor, 
                                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get binary latents specifically formatted for QuDiffuse compatibility.
        
        Args:
            encoder_outputs: BART encoder outputs [B, seq_len, dim_lm]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Binary latents in QuDiffuse format [B, num_encoder_latents, dim_ae]
        """
        with torch.no_grad():
            latents = self.encode(encoder_outputs, attention_mask)
            
            # Ensure exact binary values for QuDiffuse
            if self.binary_quantization:
                binary_latents = (latents > 0).float()
                return binary_latents
            else:
                # If not quantized, threshold at 0.5
                return (torch.sigmoid(latents) > 0.5).float()
    
    def reconstruct_from_binary(self, binary_latents: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct sequence from binary latents (for diffusion output processing).
        
        Args:
            binary_latents: Binary latents [B, num_encoder_latents, dim_ae]
            
        Returns:
            Reconstructed sequence [B, num_decoder_latents, dim_lm]
        """
        # Convert binary back to tanh range for decoder
        if self.binary_quantization:
            latents = binary_latents * 2.0 - 1.0  # {0,1} → {-1,1}
        else:
            latents = binary_latents
        
        return self.decode(latents) 