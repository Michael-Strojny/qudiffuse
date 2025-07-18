"""
Reasoning DiT: Diffusion Transformer for Text Reasoning

This module implements a Diffusion Transformer (DiT) architecture specifically designed
for reasoning tasks in binary latent space. It's adapted from Facebook's DiT architecture
but optimized for text latent diffusion and reasoning capabilities.

Key Features:
- Cross-attention conditioning on input sequences
- Time embeddings with reasoning-specific adaptations
- Support for arithmetic and spatial reasoning tasks
- Integration with QuDiffuse binary diffusion system
- Scalable transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from einops import rearrange, repeat


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Adapted from DiT with reasoning-specific modifications.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: 1-D Tensor of N indices, one per batch element
            dim: Dimension of the output
            max_period: Controls the minimum frequency of the embeddings
            
        Returns:
            (N, D) Tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ReasoningTypeEmbedder(nn.Module):
    """
    Embeds reasoning task types (arithmetic, spatial, etc.) into vector representations.
    """
    
    def __init__(self, num_reasoning_types: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_reasoning_types, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, reasoning_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reasoning_type: Tensor of reasoning type IDs [B]
            
        Returns:
            Reasoning embeddings [B, hidden_size]
        """
        emb = self.embedding(reasoning_type)
        return self.mlp(emb)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero initialization.
    From DiT paper - modulates activations with timestep and conditioning.
    """
    
    def __init__(self, embedding_dim: int, norm_shape: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * norm_shape, bias=True)
        self.norm = nn.LayerNorm(norm_shape, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, N, D]
            c: Conditioning tensor [B, embedding_dim]
            
        Returns:
            Tuple of (modulated_x, gate)
        """
        c = self.silu(c)
        c = self.linear(c)  # [B, 6*D]
        c = c.view(c.size(0), 1, 6, -1)  # [B, 1, 6, D]
        
        # Split into 6 components: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.unbind(dim=2)
        
        x = self.norm(x)
        x = x * (1 + scale_msa) + shift_msa
        
        return x, gate_msa.squeeze(1)


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention for conditioning on input sequences.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
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
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Query tensor [B, N, D]
            context: Context tensor [B, M, D] 
            context_mask: Context mask [B, M]
            
        Returns:
            Output tensor [B, N, D]
        """
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.num_heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.num_heads)
        
        # Compute attention
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale
        
        if context_mask is not None:
            context_mask = context_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, M]
            attn = attn.masked_fill(context_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class ReasoningDiTBlock(nn.Module):
    """
    A DiT block with reasoning-specific modifications.
    Includes self-attention, cross-attention, and MLP with adaptive normalization.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Self-attention
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention for conditioning
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, dropout)
        
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Adaptive layer norms
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, D]
            c: Conditioning tensor [B, D] (timestep + reasoning type)
            context: Context tensor [B, M, D] for cross-attention
            context_mask: Context mask [B, M]
            
        Returns:
            Output tensor [B, N, D]
        """
        # Adaptive modulation
        c = self.adaLN_modulation(c)
        c = c.view(c.size(0), 6, -1)  # [B, 6, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.unbind(dim=1)
        
        # Self-attention with adaptive normalization
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross-attention (if context provided)
        if context is not None:
            x_cross = self.cross_attn(x, context, context_mask)
            x = x + x_cross
        
        # MLP with adaptive normalization
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class ReasoningDiT(nn.Module):
    """
    Reasoning Diffusion Transformer for text latent diffusion.
    
    This model performs reasoning in binary latent space using a transformer
    architecture with cross-attention conditioning and reasoning-specific adaptations.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,              # dae from paper
        sequence_length: int = 16,          # lae from paper
        hidden_size: int = 768,             # Transformer hidden dimension
        num_heads: int = 12,                # Number of attention heads
        num_layers: int = 12,               # Number of transformer layers
        mlp_ratio: float = 4.0,             # MLP expansion ratio
        dropout: float = 0.1,               # Dropout rate
        num_reasoning_types: int = 4,       # Number of reasoning task types
        condition_on_input: bool = True,    # Whether to use cross-attention conditioning
        learn_sigma: bool = False           # Whether to learn noise variance
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.condition_on_input = condition_on_input
        self.learn_sigma = learn_sigma
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, sequence_length, hidden_size) * 0.02)
        
        # Timestep embedder
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        
        # Reasoning type embedder
        self.reasoning_embedder = ReasoningTypeEmbedder(num_reasoning_types, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ReasoningDiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        if learn_sigma:
            # Predict both noise and variance
            self.output_proj = nn.Linear(hidden_size, latent_dim * 2)
        else:
            # Predict only noise
            self.output_proj = nn.Linear(hidden_size, latent_dim)
        
        # Initialize output projection to zero (important for stable training)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)
        
        # Final adaptive layer norm
        self.adaln_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        print(f"🤖 ReasoningDiT initialized:")
        print(f"   Input shape: [{sequence_length}, {latent_dim}]")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Layers: {num_layers}")
        print(f"   Heads: {num_heads}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self,
        x: torch.Tensor,                    # Noisy latents [B, seq_len, latent_dim]
        t: torch.Tensor,                    # Timesteps [B]
        reasoning_type: torch.Tensor,       # Reasoning type IDs [B]
        context: Optional[torch.Tensor] = None,      # Input context [B, ctx_len, hidden_size]
        context_mask: Optional[torch.Tensor] = None  # Context mask [B, ctx_len]
    ) -> torch.Tensor:
        """
        Forward pass for reasoning diffusion.
        
        Args:
            x: Noisy latents [B, seq_len, latent_dim]
            t: Timesteps [B]
            reasoning_type: Reasoning type IDs [B]
            context: Input context for cross-attention [B, ctx_len, hidden_size]
            context_mask: Context mask [B, ctx_len]
            
        Returns:
            Predicted noise (and optionally variance) [B, seq_len, latent_dim (×2)]
        """
        batch_size, seq_len, latent_dim = x.shape
        
        # Input projection and positional embeddings
        x = self.input_proj(x)  # [B, seq_len, hidden_size]
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Timestep embeddings
        t_emb = self.timestep_embedder(t)  # [B, hidden_size]
        
        # Reasoning type embeddings
        r_emb = self.reasoning_embedder(reasoning_type)  # [B, hidden_size]
        
        # Combine timestep and reasoning embeddings
        c = t_emb + r_emb  # [B, hidden_size]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c, context, context_mask)
        
        # Final adaptive layer norm
        c_final = self.adaln_final(c)  # [B, 2*hidden_size]
        shift, scale = c_final.chunk(2, dim=1)  # Each [B, hidden_size]
        
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Output projection
        x = self.output_proj(x)  # [B, seq_len, latent_dim] or [B, seq_len, 2*latent_dim]
        
        return x
    
    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 0.0):
        """Configure optimizers for training."""
        # Separate parameters for different learning rates
        param_groups = [
            {'params': [p for p in self.parameters() if p.requires_grad], 'lr': learning_rate}
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        return optimizer


class ReasoningTaskEmbeddings:
    """Utilities for reasoning task type embeddings."""
    
    ARITHMETIC = 0
    SPATIAL = 1
    LOGICAL = 2
    GENERAL = 3
    
    TASK_NAMES = {
        ARITHMETIC: "arithmetic",
        SPATIAL: "spatial", 
        LOGICAL: "logical",
        GENERAL: "general"
    }
    
    @classmethod
    def get_task_id(cls, task_name: str) -> int:
        """Get task ID from name."""
        name_to_id = {v: k for k, v in cls.TASK_NAMES.items()}
        return name_to_id.get(task_name.lower(), cls.GENERAL)
    
    @classmethod
    def get_task_name(cls, task_id: int) -> str:
        """Get task name from ID."""
        return cls.TASK_NAMES.get(task_id, "general") 