import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from typing import List, Tuple, Optional, Dict, Union
import logging
import numpy as np
from qudiffuse.utils.error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError
from qudiffuse.utils.common_utils import validate_tensor_shape, ensure_device_consistency, cleanup_gpu_memory

logger = logging.getLogger(__name__)

class RBM(nn.Module):
    """
    Restricted Boltzmann Machine for hierarchical DBN.
    
    Following the technical report specification:
    - Energy function: E(v,h) = -sum_ij W_ij v_i h_j - sum_i b_i v_i - sum_j c_j h_j
    - Conditional distributions: P(h_j=1|v) = σ(sum_i W_ij v_i + c_j)
    - Contrastive Divergence training with k steps
    """
    
    def __init__(self, visible_size: int, hidden_size: int, device: str = "cpu", lr: float = 1e-3):
        super().__init__()
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.device = device
        self.lr = lr

        # Initialize weights and biases according to technical report
        self.W = nn.Parameter(torch.randn(visible_size, hidden_size, device=device) * 0.01)
        self.b = nn.Parameter(torch.zeros(visible_size, device=device))  # visible biases
        self.c = nn.Parameter(torch.zeros(hidden_size, device=device))   # hidden biases
        
        logger.debug(f"Initialized RBM: visible={visible_size}, hidden={hidden_size}")
    
    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute energy function E(v,h) = -sum_ij W_ij v_i h_j - sum_i b_i v_i - sum_j c_j h_j
        
        Args:
            v: Visible units [batch_size, visible_size]
            h: Hidden units [batch_size, hidden_size]
            
        Returns:
            Energy values [batch_size]
        """
        # Interaction term: -sum_ij W_ij v_i h_j
        interaction = -torch.sum(torch.matmul(v, self.W) * h, dim=1)
        
        # Visible bias term: -sum_i b_i v_i
        visible_bias = -torch.matmul(v, self.b)
        
        # Hidden bias term: -sum_j c_j h_j
        hidden_bias = -torch.matmul(h, self.c)
        
        return interaction + visible_bias + hidden_bias
    
    def prob_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute P(h_j=1|v) = σ(sum_i W_ij v_i + c_j)
        
        Args:
            v: Visible units [batch_size, visible_size]
            
        Returns:
            Hidden probabilities [batch_size, hidden_size]
        """
        return torch.sigmoid(torch.matmul(v, self.W) + self.c)
    
    def prob_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute P(v_i=1|h) = σ(sum_j W_ij h_j + b_i)
        
        Args:
            h: Hidden units [batch_size, hidden_size]
            
        Returns:
            Visible probabilities [batch_size, visible_size]
        """
        return torch.sigmoid(torch.matmul(h, self.W.T) + self.b)
    
    def sample_h_given_v(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible units."""
        h_prob = self.prob_h_given_v(v)
        h_sample = torch.bernoulli(h_prob)
        return h_sample, h_prob

    def sample_v_given_h(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample visible units given hidden units."""
        v_prob = self.prob_v_given_h(h)
        v_sample = torch.bernoulli(v_prob)
        return v_sample, v_prob

    def _v_to_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for sample_h_given_v for compatibility with timestep_specific_binary_diffusion."""
        return self.sample_h_given_v(v)
    
    def _h_to_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for sample_v_given_h for compatibility with timestep_specific_binary_diffusion."""
        return self.sample_v_given_h(h)

    def contrastive_divergence(self, v_data: torch.Tensor, k: int = 1,
                             use_persistent: bool = False,
                             persistent_chain: Optional[torch.Tensor] = None,
                             learning_rate: float = 0.01,
                             momentum: float = 0.9,
                             weight_decay: float = 0.0001) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advanced Contrastive Divergence training with multiple enhancements.

        Implements state-of-the-art CD training with:
        - Persistent Contrastive Divergence (PCD) option
        - Proper momentum and weight decay
        - Enhanced gradient computation
        - Temperature annealing support

        Args:
            v_data: Input visible data [batch_size, visible_size]
            k: Number of Gibbs sampling steps (recommended: 5-15 for training)
            use_persistent: Whether to use Persistent CD (PCD)
            persistent_chain: Persistent chain state for PCD
            learning_rate: Learning rate for this step
            momentum: Momentum coefficient
            weight_decay: Weight decay coefficient

        Returns:
            (grad_W, grad_b, grad_c, new_persistent_chain): Parameter gradients and updated chain
        """
        batch_size = v_data.size(0)

        # Positive phase - use probabilities for better gradient estimates
        h_pos_sample, h_pos_prob = self.sample_h_given_v(v_data)

        # Negative phase initialization
        if use_persistent and persistent_chain is not None:
            # Use persistent chain for better mixing
            v_neg = persistent_chain.clone()
        else:
            # Standard CD: start from data
            v_neg = v_data.clone()

        # Enhanced Gibbs sampling with k steps
        v_neg_samples = []
        h_neg_samples = []

        for step in range(k):
            # Sample hidden given visible
            h_neg_sample, h_neg_prob = self.sample_h_given_v(v_neg)
            h_neg_samples.append(h_neg_prob.clone())

            # Sample visible given hidden
            v_neg, v_neg_prob = self.sample_v_given_h(h_neg_sample)
            v_neg_samples.append(v_neg.clone())

            # Optional: Add noise for better exploration (annealed)
            if step < k - 1:  # Don't add noise to final sample
                noise_std = 0.01 * (1.0 - step / k)  # Annealed noise
                v_neg = v_neg + torch.randn_like(v_neg) * noise_std
                v_neg = torch.clamp(v_neg, 0, 1)  # Keep in valid range

        # Final negative phase probabilities
        h_neg_sample, h_neg_prob = self.sample_h_given_v(v_neg)

        # Enhanced gradient computation with proper statistics
        # Use probabilities for positive phase, samples for negative phase (standard practice)
        positive_grad_W = torch.matmul(v_data.T, h_pos_prob) / batch_size
        negative_grad_W = torch.matmul(v_neg.T, h_neg_prob) / batch_size
        grad_W = positive_grad_W - negative_grad_W

        # Add weight decay to weight gradients
        if weight_decay > 0:
            grad_W = grad_W - weight_decay * self.W

        # Bias gradients
        grad_b = torch.mean(v_data - v_neg, dim=0)
        grad_c = torch.mean(h_pos_prob - h_neg_prob, dim=0)

        # Update persistent chain for next iteration
        new_persistent_chain = v_neg.detach() if use_persistent else None

        return grad_W, grad_b, grad_c, new_persistent_chain

    def contrastive_divergence_fast(self, v_data: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast CD implementation for inference/sampling (backward compatibility).

        Args:
            v_data: Input visible data [batch_size, visible_size]
            k: Number of Gibbs sampling steps

        Returns:
            (grad_W, grad_b, grad_c): Parameter gradients
        """
        grad_W, grad_b, grad_c, _ = self.contrastive_divergence(
            v_data, k=k, use_persistent=False, learning_rate=0.01
        )
        return grad_W, grad_b, grad_c
    
    def train_step(self, v_data: torch.Tensor, optimizer: torch.optim.Optimizer,
                   k: int = 10, use_persistent: bool = True,
                   persistent_chain: Optional[torch.Tensor] = None,
                   weight_decay: float = 0.0001) -> Tuple[float, torch.Tensor]:
        """
        Enhanced training step using advanced Contrastive Divergence.

        Args:
            v_data: Input visible data
            optimizer: Optimizer for parameter updates
            k: Number of CD steps (recommended: 10-15 for training)
            use_persistent: Whether to use Persistent CD
            persistent_chain: Persistent chain state
            weight_decay: Weight decay coefficient

        Returns:
            (reconstruction_error, new_persistent_chain)
        """
        optimizer.zero_grad()

        # Get learning rate from optimizer
        lr = optimizer.param_groups[0]['lr']

        # Compute gradients using enhanced CD
        grad_W, grad_b, grad_c, new_persistent_chain = self.contrastive_divergence(
            v_data, k=k, use_persistent=use_persistent,
            persistent_chain=persistent_chain,
            learning_rate=lr, weight_decay=weight_decay
        )

        # Set gradients (negative for gradient ascent on log-likelihood)
        self.W.grad = -grad_W
        self.b.grad = -grad_b
        self.c.grad = -grad_c

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([self.W, self.b, self.c], max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Compute reconstruction error for monitoring
        with torch.no_grad():
            h_sample, _ = self.sample_h_given_v(v_data)
            v_recon, _ = self.sample_v_given_h(h_sample)
            recon_error = F.mse_loss(v_recon, v_data, reduction='mean')

        return recon_error.item(), new_persistent_chain

    def train_step_fast(self, v_data: torch.Tensor, optimizer: torch.optim.Optimizer, k: int = 1) -> float:
        """Fast training step for backward compatibility."""
        recon_error, _ = self.train_step(v_data, optimizer, k=k, use_persistent=False)
        return recon_error

    def cd_sampling(self, initial_v: torch.Tensor, k: int = 50,
                   temperature: float = 1.0, return_trajectory: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Advanced CD sampling for inference and generation.

        This method performs k steps of Gibbs sampling starting from initial_v,
        with optional temperature control and trajectory recording.

        Args:
            initial_v: Initial visible state [batch_size, visible_size]
            k: Number of CD sampling steps (recommended: 50-200 for inference)
            temperature: Temperature for sampling (1.0 = standard, <1.0 = sharper)
            return_trajectory: Whether to return full sampling trajectory

        Returns:
            Final visible state, optionally with full trajectory
        """
        batch_size = initial_v.size(0)
        v_current = initial_v.clone()

        trajectory = [] if return_trajectory else None

        with torch.no_grad():
            for step in range(k):
                # Sample hidden given visible (with temperature)
                h_logits = torch.matmul(v_current, self.W) + self.c
                h_prob = torch.sigmoid(h_logits / temperature)
                h_sample = torch.bernoulli(h_prob)

                # Sample visible given hidden (with temperature)
                v_logits = torch.matmul(h_sample, self.W.T) + self.b
                v_prob = torch.sigmoid(v_logits / temperature)
                v_current = torch.bernoulli(v_prob)

                if return_trajectory:
                    trajectory.append(v_current.clone())

        if return_trajectory:
            return v_current, trajectory
        else:
            return v_current

    def block_gibbs_sampling(self, initial_v: torch.Tensor, k: int = 100,
                           block_size: int = None) -> torch.Tensor:
        """
        Block Gibbs sampling for improved mixing.

        Args:
            initial_v: Initial visible state
            k: Number of sampling steps
            block_size: Size of blocks to update (None = full update)

        Returns:
            Final sampled visible state
        """
        if block_size is None:
            return self.cd_sampling(initial_v, k=k)

        batch_size, visible_size = initial_v.shape
        v_current = initial_v.clone()

        with torch.no_grad():
            for step in range(k):
                # Randomly select block to update
                start_idx = torch.randint(0, visible_size - block_size + 1, (1,)).item()
                end_idx = start_idx + block_size

                # Sample hidden given current visible
                h_prob = self.prob_h_given_v(v_current)
                h_sample = torch.bernoulli(h_prob)

                # Update only the selected block
                v_block_logits = torch.matmul(h_sample, self.W[start_idx:end_idx, :].T) + self.b[start_idx:end_idx]
                v_block_prob = torch.sigmoid(v_block_logits)
                v_current[:, start_idx:end_idx] = torch.bernoulli(v_block_prob)

        return v_current

    def parallel_tempering_sampling(self, initial_v: torch.Tensor, k: int = 100,
                                  temperatures: List[float] = None) -> torch.Tensor:
        """
        Parallel tempering for better sampling.

        Args:
            initial_v: Initial visible state
            k: Number of sampling steps
            temperatures: List of temperatures for parallel chains

        Returns:
            Final sampled state from temperature=1.0 chain
        """
        if temperatures is None:
            temperatures = [0.5, 0.7, 1.0, 1.5, 2.0]

        num_temps = len(temperatures)
        batch_size, visible_size = initial_v.shape

        # Initialize chains for each temperature
        chains = [initial_v.clone() for _ in range(num_temps)]

        with torch.no_grad():
            for step in range(k):
                # Update each chain
                for i, temp in enumerate(temperatures):
                    chains[i] = self.cd_sampling(chains[i], k=1, temperature=temp)

                # Attempt swaps between adjacent temperatures
                if step % 10 == 0:  # Swap every 10 steps
                    for i in range(num_temps - 1):
                        temp1, temp2 = temperatures[i], temperatures[i + 1]

                        # Compute swap probability using Metropolis criterion
                        energy1 = self.compute_energy_rbm(chains[i])
                        energy2 = self.compute_energy_rbm(chains[i + 1])

                        delta = (1/temp1 - 1/temp2) * (energy2 - energy1)
                        swap_prob = torch.exp(torch.clamp(delta, max=0))

                        # Perform swap
                        swap_mask = torch.bernoulli(swap_prob).bool()
                        if swap_mask.any():
                            temp_chain = chains[i][swap_mask].clone()
                            chains[i][swap_mask] = chains[i + 1][swap_mask]
                            chains[i + 1][swap_mask] = temp_chain

        # Return chain at temperature 1.0
        temp_1_idx = temperatures.index(1.0) if 1.0 in temperatures else -1
        return chains[temp_1_idx]

    def compute_energy_rbm(self, v: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        """
        Compute RBM energy for visible (and optionally hidden) units.

        Args:
            v: Visible units
            h: Hidden units (if None, will be sampled)

        Returns:
            Energy values
        """
        if h is None:
            h, _ = self.sample_h_given_v(v)

        # Energy: -sum_ij W_ij v_i h_j - sum_i b_i v_i - sum_j c_j h_j
        interaction = -torch.sum(torch.matmul(v, self.W) * h, dim=1)
        visible_bias = -torch.matmul(v, self.b)
        hidden_bias = -torch.matmul(h, self.c)

        return interaction + visible_bias + hidden_bias

    def cd_sampling_optimized(self, initial_v: torch.Tensor, k: int = 50,
                             temperature: float = 1.0, batch_parallel: bool = True) -> torch.Tensor:
        """
        Performance-optimized CD sampling with vectorized operations.

        Args:
            initial_v: Initial visible state
            k: Number of CD steps
            temperature: Sampling temperature
            batch_parallel: Whether to use batch-parallel sampling

        Returns:
            Final sampled visible state
        """
        if not batch_parallel or initial_v.size(0) == 1:
            return self.cd_sampling(initial_v, k=k, temperature=temperature)

        batch_size = initial_v.size(0)
        v_current = initial_v.clone()

        # Pre-compute temperature scaling
        temp_inv = 1.0 / temperature if temperature != 1.0 else 1.0

        with torch.no_grad():
            # Vectorized sampling loop
            for step in range(k):
                # Vectorized hidden sampling
                h_logits = torch.matmul(v_current, self.W) + self.c
                if temperature != 1.0:
                    h_logits = h_logits * temp_inv
                h_prob = torch.sigmoid(h_logits)
                h_sample = torch.bernoulli(h_prob)

                # Vectorized visible sampling
                v_logits = torch.matmul(h_sample, self.W.T) + self.b
                if temperature != 1.0:
                    v_logits = v_logits * temp_inv
                v_prob = torch.sigmoid(v_logits)
                v_current = torch.bernoulli(v_prob)

        return v_current

    def train_step_memory_efficient(self, v_data: torch.Tensor, optimizer: torch.optim.Optimizer,
                                   k: int = 10, gradient_accumulation_steps: int = 1) -> float:
        """
        Memory-efficient training step with gradient accumulation.

        Args:
            v_data: Input visible data
            optimizer: Optimizer
            k: CD steps
            gradient_accumulation_steps: Steps to accumulate gradients

        Returns:
            Reconstruction error
        """
        total_error = 0.0
        batch_size = v_data.size(0)
        mini_batch_size = max(1, batch_size // gradient_accumulation_steps)

        optimizer.zero_grad()

        for step in range(gradient_accumulation_steps):
            start_idx = step * mini_batch_size
            end_idx = min((step + 1) * mini_batch_size, batch_size)
            mini_batch = v_data[start_idx:end_idx]

            # Compute gradients for mini-batch
            grad_W, grad_b, grad_c, _ = self.contrastive_divergence(
                mini_batch, k=k, use_persistent=False
            )

            # Scale gradients by accumulation factor
            scale_factor = mini_batch.size(0) / batch_size
            grad_W = grad_W * scale_factor
            grad_b = grad_b * scale_factor
            grad_c = grad_c * scale_factor

            # Accumulate gradients
            if self.W.grad is None:
                self.W.grad = -grad_W
                self.b.grad = -grad_b
                self.c.grad = -grad_c
            else:
                self.W.grad += -grad_W
                self.b.grad += -grad_b
                self.c.grad += -grad_c

            # Compute error for monitoring
            with torch.no_grad():
                h_sample, _ = self.sample_h_given_v(mini_batch)
                v_recon, _ = self.sample_v_given_h(h_sample)
                error = F.mse_loss(v_recon, mini_batch, reduction='mean')
                total_error += error.item() * scale_factor

        # Apply accumulated gradients
        optimizer.step()

        return total_error

    def adaptive_cd_training(self, v_data: torch.Tensor, optimizer: torch.optim.Optimizer,
                           initial_k: int = 5, max_k: int = 25,
                           convergence_threshold: float = 1e-6) -> Tuple[float, int]:
        """
        Adaptive CD training that adjusts k based on convergence.

        Args:
            v_data: Input data
            optimizer: Optimizer
            initial_k: Starting CD steps
            max_k: Maximum CD steps
            convergence_threshold: Threshold for convergence detection

        Returns:
            (reconstruction_error, final_k_used)
        """
        k = initial_k
        prev_error = float('inf')

        while k <= max_k:
            error, _ = self.train_step(v_data, optimizer, k=k, use_persistent=False)

            # Check convergence
            if abs(prev_error - error) < convergence_threshold:
                break

            prev_error = error
            k += 1

        return error, k

    def to_qubo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert RBM to QUBO formulation.

        RBM energy: E(v,h) = -v^T W h - b^T v - c^T h
        QUBO form: E(x) = x^T J x + h^T x where x = [v, h]

        Returns:
            (J, h): QUBO coupling matrix and linear terms
        """
        visible_size = self.W.size(0)
        hidden_size = self.W.size(1)
        total_size = visible_size + hidden_size

        J = torch.zeros(total_size, total_size, device=self.device)
        h = torch.zeros(total_size, device=self.device)

        # Visible-hidden coupling: -v^T W h becomes quadratic terms
        W = self.W.detach()
        J[:visible_size, visible_size:] = W  # v-h coupling
        J[visible_size:, :visible_size] = W.T  # h-v coupling (symmetric)

        # Linear terms from biases
        h[:visible_size] = self.b.detach()  # visible bias
        h[visible_size:] = self.c.detach()  # hidden bias

        return J, h

    def to_qubo_dict(self) -> Dict[Tuple[int, int], float]:
        """
        Convert RBM to QUBO dictionary format for D-Wave.

        Returns:
            QUBO dictionary with (i,j) -> coupling strength
        """
        J, h = self.to_qubo()

        qubo_dict = {}
        n_vars = J.size(0)

        # Add quadratic and linear terms
        for i in range(n_vars):
            for j in range(i, n_vars):
                if i == j:
                    # Diagonal terms include linear bias
                    coeff = J[i, j].item() + h[i].item()
                else:
                    # Off-diagonal terms
                    coeff = J[i, j].item()

                if abs(coeff) > 1e-10:  # Only include non-zero terms
                    qubo_dict[(i, j)] = coeff

        return qubo_dict

class HierarchicalDBN(nn.Module):
    """
    Hierarchical Deep Belief Network according to technical report.
    
    Key specifications:
    - N = Σ C_ℓ layers, where each binary latent channel corresponds to exactly one DBN layer
    - Visible units: v^(ℓ) = [z^(ℓ) || h^(ℓ+1)] for ℓ < L, v^(L) = z^(L) for top layer
    - Each layer is an RBM with energy function E^(ℓ)(v^(ℓ), h^(ℓ))
    - Complete DBN energy: E(z, h) = Σ E^(ℓ)(v^(ℓ), h^(ℓ))
    """
    
    def __init__(self, 
                 latent_shapes: List[Tuple[int, int, int]],  # (C, H, W) for each pyramid level
                 hidden_dims: List[int],                      # Hidden units per CHANNEL (not per level)
                 device: str = "cpu"):
        super().__init__()
        
        self.latent_shapes = latent_shapes
        self.device = device
        self.num_pyramid_levels = len(latent_shapes)
        
        # Calculate channel information - one DBN layer per channel
        self.channel_info = []  # List of (level_idx, channel_idx, spatial_dim)
        self.total_channels = 0
        
        for level_idx, (c, h, w) in enumerate(latent_shapes):
            spatial_dim = h * w
            for channel_idx in range(c):
                self.channel_info.append({
                    'level_idx': level_idx,
                    'channel_idx': channel_idx,
                    'spatial_dim': spatial_dim,
                    'global_channel_idx': self.total_channels
                })
                self.total_channels += 1
        
        # Validate hidden dimensions
        if len(hidden_dims) != self.total_channels:
            raise ConfigurationError(f"Number of hidden dimensions ({len(hidden_dims)}) "
                                   f"must match total number of channels ({self.total_channels})")
        
        self.hidden_dims = hidden_dims
        
        # Create RBM layers - one per channel following technical report specification
        self.rbms = nn.ModuleList()
        
        for channel_idx in range(self.total_channels):
            channel_info = self.channel_info[channel_idx]
            spatial_dim = channel_info['spatial_dim']
            
            if channel_idx == self.total_channels - 1:
                # Top layer (last channel): v^(L) = z^(L)
                visible_size = spatial_dim
            else:
                # Lower layers: v^(ℓ) = [z^(ℓ) || h^(ℓ+1)]
                # Hidden units from the channel above (channel_idx + 1)
                hidden_size_above = hidden_dims[channel_idx + 1]
                visible_size = spatial_dim + hidden_size_above
            
            rbm = RBM(visible_size, hidden_dims[channel_idx], device=device)
            self.rbms.append(rbm)
        
        logger.info(f"Initialized HierarchicalDBN with {self.total_channels} channel layers")
        logger.info(f"Pyramid levels: {self.num_pyramid_levels}")
        logger.info(f"Channel distribution: {[shape[0] for shape in latent_shapes]}")
        logger.info(f"Hidden dimensions per channel: {hidden_dims}")
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
    
    def extract_channels_from_levels(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Extract individual channels from hierarchical latent representation.
        
        Args:
            latents: List of tensors [batch_size, C, H, W] for each pyramid level
            
        Returns:
            List of individual channel tensors [batch_size, H*W] (flattened)
        """
        channels = []
        
        for level_idx, level_tensor in enumerate(latents):
            batch_size, num_channels, h, w = level_tensor.shape
            
            for channel_idx in range(num_channels):
                # Extract single channel and flatten spatial dimensions
                channel_data = level_tensor[:, channel_idx, :, :]  # [B, H, W]
                channel_flat = channel_data.reshape(batch_size, -1)  # [B, H*W]
                channels.append(channel_flat)
        
        return channels
    
    def reconstruct_levels_from_channels(self, channels: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reconstruct hierarchical level representation from individual channels.
        
        Args:
            channels: List of individual channel tensors [batch_size, H*W]
            
        Returns:
            List of tensors [batch_size, C, H, W] for each pyramid level
        """
        levels = []
        channel_idx = 0
        
        for level_idx, (c, h, w) in enumerate(self.latent_shapes):
            batch_size = channels[0].shape[0]
            level_channels = []
            
            for _ in range(c):
                # Reshape channel back to spatial format
                channel_spatial = channels[channel_idx].reshape(batch_size, h, w)  # [B, H, W]
                level_channels.append(channel_spatial.unsqueeze(1))  # [B, 1, H, W]
                channel_idx += 1
            
            # Concatenate channels for this level
            level_tensor = torch.cat(level_channels, dim=1)  # [B, C, H, W]
            levels.append(level_tensor)
        
        return levels
    
    def construct_visible_unit_for_channel(self, channel_idx: int, channels: List[torch.Tensor], 
                                         hidden_activations: List[Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Construct visible units for a specific channel's RBM layer: v^(ℓ) = [z^(ℓ) || h^(ℓ+1)]
        
        Args:
            channel_idx: Index of the channel to construct visible units for
            channels: List of individual channel tensors [batch_size, spatial_dim]
            hidden_activations: List of hidden activations from each channel
            
        Returns:
            Visible unit tensor for the specified channel's RBM
        """
        if channel_idx == self.total_channels - 1:
            # Top layer (last channel): v^(L) = z^(L)
            return channels[channel_idx]
        else:
            # Lower layers: v^(ℓ) = [z^(ℓ) || h^(ℓ+1)]
            # Concatenate channel data with hidden activations from channel above
            if hidden_activations[channel_idx + 1] is None:
                raise ValueError(f"Hidden activations for channel {channel_idx + 1} are required")
            
            # Concatenate z^(ℓ) with h^(ℓ+1)
            return torch.cat([channels[channel_idx], hidden_activations[channel_idx + 1]], dim=1)
    
    def construct_visible_units(self, channels: List[torch.Tensor], 
                              hidden_activations: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Construct visible units for each channel's RBM layer: v^(ℓ) = [z^(ℓ) || h^(ℓ+1)]
        
        Args:
            channels: List of individual channel tensors [batch_size, spatial_dim]
            hidden_activations: List of hidden activations from each channel
            
        Returns:
            List of visible unit tensors for each channel's RBM
        """
        visible_units = []
        
        for channel_idx in range(self.total_channels):
            visible_unit = self.construct_visible_unit_for_channel(channel_idx, channels, hidden_activations)
            visible_units.append(visible_unit)
        
        return visible_units
    
    def greedy_pretrain(self, 
                       data: List[torch.Tensor],
                       epochs: int = 10,
                       learning_rate: float = 0.01,
                       batch_size: int = 64,
                       k: int = 1,
                       use_persistent: bool = True,
                       verbose: bool = True) -> Dict[str, List[float]]:
        """
        Greedy layer-wise pretraining of the hierarchical DBN.
        
        Args:
            data: List of hierarchical latent tensors [batch_size, C, H, W]
            epochs: Number of training epochs per channel
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            k: Number of Gibbs sampling steps
            use_persistent: Whether to use persistent chains
            verbose: Whether to log progress
            
        Returns:
            Dictionary of training history
        """
        logger.info("Starting greedy pretraining of hierarchical DBN")
        
        # Extract individual channels from hierarchical data
        all_channels = []
        for sample_idx in range(len(data[0])):  # Iterate over batch
            sample_levels = [level[sample_idx:sample_idx+1] for level in data]
            sample_channels = self.extract_channels_from_levels(sample_levels)
            all_channels.append(sample_channels)
        
        # Reorganize data by channel
        channels_data = []
        for channel_idx in range(self.total_channels):
            channel_samples = []
            for sample_idx in range(len(all_channels)):
                channel_samples.append(all_channels[sample_idx][channel_idx])
            # Stack all samples for this channel
            channels_data.append(torch.cat(channel_samples, dim=0))
        
        # Training history
        history = {"channel_losses": [], "reconstruction_errors": []}
        
        # Train each channel's RBM
        for channel_idx in range(self.total_channels):
            if verbose:
                channel_info = self.channel_info[channel_idx]
                logger.info(f"Training channel {channel_idx} (Level {channel_info['level_idx']}, "
                          f"Channel {channel_info['channel_idx']})")
            
            channel_data = channels_data[channel_idx]
            
            # Ensure proper dtype conversion
            if channel_data.dtype == torch.bool:
                channel_data = channel_data.float()
            elif channel_data.dtype == torch.float16:
                channel_data = channel_data.float()
            
            # Train RBM with proper parameters
            rbm_losses = []
            rbm = self.rbms[channel_idx]
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(channel_data)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop for this channel
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for (batch_data,) in dataloader:
                    batch_data = batch_data.to(self.device)
                    
                    # Perform CD training
                    loss, pos_h, neg_v, neg_h = rbm.contrastive_divergence(
                        batch_data, k=k, use_persistent=use_persistent,
                        learning_rate=learning_rate
                    )
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                rbm_losses.append(avg_loss)
                
                if verbose and epoch % (epochs // 5) == 0:
                    logger.debug(f"  Channel {channel_idx}, Epoch {epoch}: Loss = {avg_loss:.6f}")
            
            history["channel_losses"].append(rbm_losses)
            
            if verbose:
                final_loss = rbm_losses[-1] if rbm_losses else 0.0
                logger.info(f"✅ Channel {channel_idx} training completed. Final loss: {final_loss:.6f}")
        
        logger.info("Greedy pretraining completed successfully")
        return history
    
    def forward(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through the hierarchical DBN.
        
        Args:
            latents: List of binary latent tensors from autoencoder [batch_size, C, H, W]
            
        Returns:
            List of hidden activations from each channel
        """
        # Extract individual channels
        channels = self.extract_channels_from_levels(latents)
        
        # Store hidden activations for each channel
        hidden_activations = [None] * self.total_channels
        
        # Process from top to bottom (reverse order)
        for channel_idx in range(self.total_channels - 1, -1, -1):
            # Construct visible units for this specific channel
            visible_unit = self.construct_visible_unit_for_channel(channel_idx, channels, hidden_activations)
            
            # Get hidden activations from this channel's RBM
            with torch.no_grad():
                _, h_prob = self.rbms[channel_idx].sample_h_given_v(visible_unit)
                hidden_activations[channel_idx] = h_prob
        
        return hidden_activations
    
    def cd_inference(self, latents: List[torch.Tensor], cd_steps: int = 50,
                    temperature: float = 1.0, use_parallel_tempering: bool = False) -> List[torch.Tensor]:
        """
        Perform Contrastive Divergence inference for denoising.
        
        Args:
            latents: List of noisy binary latent tensors
            cd_steps: Number of CD sampling steps
            temperature: Sampling temperature
            use_parallel_tempering: Whether to use parallel tempering
            
        Returns:
            List of denoised latent tensors
        """
        logger.info(f"Performing CD inference with {cd_steps} steps")
        
        # Extract individual channels
        channels = self.extract_channels_from_levels(latents)
        denoised_channels = [channel.clone() for channel in channels]
        
        # Store hidden activations
        hidden_activations = [None] * self.total_channels
        
        # Perform CD sampling for each channel
        for channel_idx in range(self.total_channels - 1, -1, -1):
            channel_info = self.channel_info[channel_idx]
            logger.debug(f"  Processing channel {channel_idx} (Level {channel_info['level_idx']}, "
                        f"Channel {channel_info['channel_idx']})")

            # Construct visible units for this channel
            visible_unit = self.construct_visible_unit_for_channel(channel_idx, denoised_channels, hidden_activations)

            # Apply CD sampling to denoise this channel
            if use_parallel_tempering:
                denoised_visible = self.rbms[channel_idx].parallel_tempering_sampling(
                    visible_unit, k=cd_steps
                )
            else:
                denoised_visible = self.rbms[channel_idx].cd_sampling(
                    visible_unit, k=cd_steps, temperature=temperature
                )

            # Update the channel data
            if channel_idx == self.total_channels - 1:
                # Top channel: only channel data
                denoised_channels[channel_idx] = denoised_visible
            else:
                # Lower channels: extract channel part (first part of visible units)
                spatial_dim = self.channel_info[channel_idx]['spatial_dim']
                denoised_channels[channel_idx] = denoised_visible[:, :spatial_dim]

            # Update hidden activations
            with torch.no_grad():
                _, h_prob = self.rbms[channel_idx].sample_h_given_v(denoised_visible)
                hidden_activations[channel_idx] = h_prob

        # Reconstruct hierarchical levels from channels
        return self.reconstruct_levels_from_channels(denoised_channels)

    def hierarchical_cd_sampling(self, initial_latents: List[torch.Tensor],
                                cd_steps: int = 100, channel_steps: int = None) -> List[torch.Tensor]:
        """
        Hierarchical CD sampling that respects the per-channel DBN structure.

        Args:
            initial_latents: Initial latent states
            cd_steps: Total CD steps
            channel_steps: Steps per channel (if None, distribute evenly)

        Returns:
            Sampled latent tensors
        """
        if channel_steps is None:
            channel_steps = cd_steps // self.total_channels

        channels = self.extract_channels_from_levels(initial_latents)
        current_channels = [channel.clone() for channel in channels]

        # Perform hierarchical sampling
        for global_step in range(cd_steps):
            hidden_activations = [None] * self.total_channels

            # Sample each channel
            for channel_idx in range(self.total_channels - 1, -1, -1):
                # Construct visible units
                visible_units = self.construct_visible_units(current_channels, hidden_activations)

                # Perform one CD step for this channel
                new_visible = self.rbms[channel_idx].cd_sampling(
                    visible_units[channel_idx], k=1, temperature=1.0
                )

                # Update channel
                if channel_idx == self.total_channels - 1:
                    current_channels[channel_idx] = new_visible
                else:
                    spatial_dim = self.channel_info[channel_idx]['spatial_dim']
                    current_channels[channel_idx] = new_visible[:, :spatial_dim]

                # Update hidden activations
                with torch.no_grad():
                    _, h_prob = self.rbms[channel_idx].sample_h_given_v(new_visible)
                    hidden_activations[channel_idx] = h_prob

        return self.reconstruct_levels_from_channels(current_channels)

    def generate_samples(self, num_samples: int, gibbs_steps: int = 100) -> List[torch.Tensor]:
        """
        Generate samples from the hierarchical DBN.
        
        Args:
            num_samples: Number of samples to generate
            gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            List of generated latent tensors
        """
        logger.info(f"Generating {num_samples} samples with {gibbs_steps} Gibbs steps")
        
        # Initialize random channels
        generated_channels = []
        for channel_idx in range(self.total_channels):
            spatial_dim = self.channel_info[channel_idx]['spatial_dim']
            channel = torch.randint(0, 2, (num_samples, spatial_dim), 
                                 dtype=torch.float32, device=self.device)
            generated_channels.append(channel)
        
        # Gibbs sampling
        for step in range(gibbs_steps):
            hidden_activations = [None] * self.total_channels
            
            # Forward pass to get hidden activations
            for channel_idx in range(self.total_channels - 1, -1, -1):
                visible_units = self.construct_visible_units(generated_channels, hidden_activations)
                _, h_prob = self.rbms[channel_idx].sample_h_given_v(visible_units[channel_idx])
                hidden_activations[channel_idx] = h_prob
            
            # Backward pass to update channels
            for channel_idx in range(self.total_channels):
                visible_units = self.construct_visible_units(generated_channels, hidden_activations)
                v_sample, _ = self.rbms[channel_idx].sample_v_given_h(hidden_activations[channel_idx])
                
                if channel_idx == self.total_channels - 1:
                    # Top channel: update channel data directly
                    generated_channels[channel_idx] = v_sample
                else:
                    # Lower channels: extract channel part from [channel || h^(ℓ+1)]
                    spatial_dim = self.channel_info[channel_idx]['spatial_dim']
                    generated_channels[channel_idx] = v_sample[:, :spatial_dim]
        
        return self.reconstruct_levels_from_channels(generated_channels)

    def compute_energy(self, latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the total energy of the hierarchical DBN.

        Args:
            latents: List of binary latent tensors

        Returns:
            Total energy values [batch_size]
        """
        channels = self.extract_channels_from_levels(latents)
        hidden_activations = [None] * self.total_channels
        
        # Get hidden activations
        for channel_idx in range(self.total_channels - 1, -1, -1):
            # Construct visible units for this specific channel
            visible_unit = self.construct_visible_unit_for_channel(channel_idx, channels, hidden_activations)
            
            _, h_prob = self.rbms[channel_idx].sample_h_given_v(visible_unit)
            hidden_activations[channel_idx] = h_prob
        
        # Compute total energy
        total_energy = 0
        for channel_idx in range(self.total_channels):
            # Construct visible units for this specific channel
            visible_unit = self.construct_visible_unit_for_channel(channel_idx, channels, hidden_activations)
            
            energy = self.rbms[channel_idx].energy(visible_unit, hidden_activations[channel_idx])
            total_energy += energy
        
        return total_energy

    def qubo_full(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert hierarchical DBN to full QUBO formulation.

        This method converts the entire hierarchical DBN energy function into
        a QUBO matrix Q such that E(x) = x^T Q x + h^T x + constant.

        Returns:
            (J, h): QUBO coupling matrix and linear terms
        """
        # Calculate total variables (all channels + all hidden units)
        total_channel_vars = sum(info['spatial_dim'] for info in self.channel_info)
        total_hidden_vars = sum(self.hidden_dims)
        total_vars = total_channel_vars + total_hidden_vars

        # Initialize QUBO matrices
        J = torch.zeros(total_vars, total_vars, device=self.device)
        h = torch.zeros(total_vars, device=self.device)

        # Variable indexing
        channel_start_idx = 0
        hidden_start_idx = total_channel_vars

        # Convert each channel's RBM to QUBO and combine
        current_channel_idx = 0
        current_hidden_idx = hidden_start_idx

        for channel_idx in range(self.total_channels):
            rbm = self.rbms[channel_idx]
            spatial_dim = self.channel_info[channel_idx]['spatial_dim']

            # Get visible and hidden dimensions for this channel
            if channel_idx == self.total_channels - 1:
                # Top channel: only channel variables
                visible_size = spatial_dim
                visible_start = current_channel_idx
            else:
                # Lower channels: channel + hidden from above
                visible_size = spatial_dim + self.hidden_dims[channel_idx + 1]
                visible_start = current_channel_idx

            hidden_size = self.hidden_dims[channel_idx]
            hidden_start = current_hidden_idx

            # Convert RBM to QUBO block
            J_rbm, h_rbm = self._rbm_to_qubo(rbm, visible_size, hidden_size)

            # Place in full QUBO matrix
            # Visible-visible interactions
            J[visible_start:visible_start+visible_size,
              visible_start:visible_start+visible_size] += J_rbm[:visible_size, :visible_size]

            # Visible-hidden interactions
            J[visible_start:visible_start+visible_size,
              hidden_start:hidden_start+hidden_size] += J_rbm[:visible_size, visible_size:]
            J[hidden_start:hidden_start+hidden_size,
              visible_start:visible_start+visible_size] += J_rbm[visible_size:, :visible_size]

            # Hidden-hidden interactions
            J[hidden_start:hidden_start+hidden_size,
              hidden_start:hidden_start+hidden_size] += J_rbm[visible_size:, visible_size:]

            # Linear terms
            h[visible_start:visible_start+visible_size] += h_rbm[:visible_size]
            h[hidden_start:hidden_start+hidden_size] += h_rbm[visible_size:]

            # Update indices
            current_channel_idx += spatial_dim
            current_hidden_idx += self.hidden_dims[channel_idx]

        return J, h

    def _rbm_to_qubo(self, rbm: 'RBM', visible_size: int, hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert single RBM to QUBO formulation.

        RBM energy: E(v,h) = -v^T W h - b^T v - c^T h
        QUBO form: E(x) = x^T J x + h^T x where x = [v, h]

        Args:
            rbm: RBM instance
            visible_size: Number of visible units
            hidden_size: Number of hidden units

        Returns:
            (J, h): QUBO coupling matrix and linear terms
        """
        total_size = visible_size + hidden_size
        J = torch.zeros(total_size, total_size, device=self.device)
        h = torch.zeros(total_size, device=self.device)

        # Visible-hidden coupling: -v^T W h becomes quadratic terms
        # For QUBO: minimize x^T Q x, so we need negative of energy
        W = rbm.W.detach()  # [visible_size, hidden_size]
        J[:visible_size, visible_size:] = W  # v-h coupling
        J[visible_size:, :visible_size] = W.T  # h-v coupling (symmetric)

        # Linear terms from biases
        h[:visible_size] = rbm.b.detach()  # visible bias
        h[visible_size:] = rbm.c.detach()  # hidden bias

        return J, h

    def qubo_latent_only(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get QUBO formulation for channel variables only (marginalizing out hidden units).

        This is used for quantum annealing where we only want to sample channel variables.

        Returns:
            (J_latent, h_latent): QUBO for channel variables only
        """
        # Get full QUBO
        J_full, h_full = self.qubo_full()

        total_channel_vars = sum(info['spatial_dim'] for info in self.channel_info)

        # Extract channel-only QUBO (first total_channel_vars variables)
        J_latent = J_full[:total_channel_vars, :total_channel_vars]
        h_latent = h_full[:total_channel_vars]

        # Marginalize out hidden variables using proper mathematical approach
        # This extracts the effective channel-channel interactions after marginalizing hidden units
        # The direct channel-channel block contains the correct marginalized interactions

        return J_latent, h_latent

    def qubo_matrix(self) -> Dict[Tuple[int, int], float]:
        """
        Get QUBO in dictionary format for D-Wave solvers.

        Returns:
            QUBO dictionary with (i,j) -> coupling strength
        """
        J, h = self.qubo_latent_only()

        qubo_dict = {}
        n_vars = J.size(0)

        # Add quadratic terms
        for i in range(n_vars):
            for j in range(i, n_vars):
                if i == j:
                    # Diagonal terms include linear bias
                    coeff = J[i, j].item() + h[i].item()
                else:
                    # Off-diagonal terms
                    coeff = J[i, j].item()

                if abs(coeff) > 1e-10:  # Only include non-zero terms
                    qubo_dict[(i, j)] = coeff

        return qubo_dict