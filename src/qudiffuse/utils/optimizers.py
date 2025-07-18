"""
Advanced optimizers for QuDiffuse training.

This module implements state-of-the-art optimizers including:
- AdEMAMix: Dual EMA optimizer from Apple ML Research (2025)
- Enhanced AdamW with better gradient handling
- Optimized schedulers for autoencoder training
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Any, Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class AdEMAMix(optim.Optimizer):
    """
    AdEMAMix optimizer with dual EMA for better gradient utilization.
    
    Based on "The AdEMAMix Optimizer: Better, Faster, Older" (Apple ML Research, 2025).
    
    This optimizer uses two exponential moving averages with different decay rates:
    - Fast EMA for recent gradients (high weight to immediate past)
    - Slow EMA for older gradients (non-negligible weight to distant past)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        alpha: Mixing coefficient between fast and slow EMA (default: 5.0)
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        T_alpha_beta3: Period for alpha and beta3 scheduling (default: None)
        beta3: Decay rate for slow EMA (default: 0.9999)
    """
    
    def __init__(self, 
                 params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 alpha: float = 5.0,
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 T_alpha_beta3: Optional[int] = None,
                 beta3: float = 0.9999):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3 value: {beta3}")
        
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, 
                       weight_decay=weight_decay, T_alpha_beta3=T_alpha_beta3, beta3=beta3)
        super().__init__(params, defaults)
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('T_alpha_beta3', None)
            group.setdefault('beta3', 0.9999)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Fast EMA (exponential moving average of gradient)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Slow EMA (exponential moving average of gradient)
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_slow, exp_avg_sq = state['exp_avg'], state['exp_avg_slow'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                alpha = group['alpha']
                beta3 = group['beta3']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update fast EMA
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update slow EMA
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                
                # Update squared gradient EMA
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected EMAs
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_slow_corrected = exp_avg_slow / bias_correction3
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Adaptive alpha scheduling
                if group['T_alpha_beta3'] is not None:
                    alpha_t = alpha * (1 - (state['step'] % group['T_alpha_beta3']) / group['T_alpha_beta3'])
                else:
                    alpha_t = alpha
                
                # Mix fast and slow EMAs
                mixed_avg = (alpha_t * exp_avg_corrected + exp_avg_slow_corrected) / (alpha_t + 1)
                
                # Compute denominator
                denom = exp_avg_sq_corrected.sqrt().add_(group['eps'])
                
                # Apply update
                step_size = group['lr']
                p.data.addcdiv_(mixed_avg, denom, value=-step_size)
        
        return loss

class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Enhanced Cosine Annealing with Warm Restarts scheduler.
    
    Implements the SGDR (Stochastic Gradient Descent with Warm Restarts) algorithm
    with improvements for autoencoder training.
    
    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart
        T_mult: A factor increases T_i after a restart (default: 1)
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
        warmup_epochs: Number of warmup epochs (default: 0)
        warmup_factor: Warmup factor (default: 0.1)
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, 
                 warmup_epochs=0, warmup_factor=0.1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.T_cur = 0
        self.T_i = T_0
        self.restart_count = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_progress = self.last_epoch / self.warmup_epochs
            return [base_lr * (self.warmup_factor + (1 - self.warmup_factor) * warmup_progress)
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            adjusted_epoch = self.last_epoch - self.warmup_epochs
            
            if adjusted_epoch >= self.T_cur + self.T_i:
                # Restart
                self.restart_count += 1
                self.T_cur += self.T_i
                self.T_i = int(self.T_i * self.T_mult)
            
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * (adjusted_epoch - self.T_cur) / self.T_i)) / 2
                    for base_lr in self.base_lrs]

class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler for stable training start.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            return self.base_lrs

class GradientClipping:
    """
    Gradient clipping utilities for stable training.
    
    Supports both global norm clipping and per-parameter clipping.
    """
    
    @staticmethod
    def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0, 
                       error_if_nonfinite: bool = False) -> float:
        """
        Clip gradients by global norm.
        
        Args:
            parameters: Iterable of parameters or single parameter
            max_norm: Maximum norm of gradients
            norm_type: Type of norm to use (default: 2.0)
            error_if_nonfinite: If True, raise error on non-finite gradients
            
        Returns:
            Total norm of gradients before clipping
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        
        if len(parameters) == 0:
            return torch.tensor(0.0)
        
        device = parameters[0].grad.device
        
        if norm_type == float('inf'):
            norms = [p.grad.detach().abs().max().to(device) for p in parameters]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) 
                                               for p in parameters]), norm_type)
        
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(f'The total norm of order {norm_type} for gradients from '
                             '`parameters` is non-finite, so it cannot be clipped. '
                             'Disable gradient clipping or use a different norm type.')
        
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
        
        return total_norm

class OptimizerFactory:
    """Factory for creating optimized optimizers and schedulers."""
    
    @staticmethod
    def create_optimizer(model, optimizer_type: str = "ademamix", 
                        lr: float = 1e-3, weight_decay: float = 0.01,
                        **kwargs) -> torch.optim.Optimizer:
        """
        Create an optimizer with optimal settings.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer ("ademamix", "adamw", "adam")
            lr: Learning rate
            weight_decay: Weight decay coefficient
            **kwargs: Additional optimizer arguments
            
        Returns:
            Configured optimizer
        """
        if optimizer_type.lower() == "ademamix":
            return AdEMAMix(
                model.parameters(),
                lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                alpha=kwargs.get('alpha', 5.0),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=weight_decay,
                beta3=kwargs.get('beta3', 0.9999)
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adam":
            return optim.Adam(
                model.parameters(),
                lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_scheduler(optimizer, scheduler_type: str = "cosine_warmup",
                        epochs: int = 200, warmup_epochs: int = 10,
                        **kwargs) -> _LRScheduler:
        """
        Create a learning rate scheduler with optimal settings.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler ("cosine_warmup", "cosine", "linear")
            epochs: Total number of epochs
            warmup_epochs: Number of warmup epochs
            **kwargs: Additional scheduler arguments
            
        Returns:
            Configured scheduler
        """
        if scheduler_type.lower() == "cosine_warmup":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=epochs - warmup_epochs,
                T_mult=kwargs.get('T_mult', 1),
                eta_min=kwargs.get('eta_min', 1e-6),
                warmup_epochs=warmup_epochs,
                warmup_factor=kwargs.get('warmup_factor', 0.1)
            )
        elif scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type.lower() == "linear":
            return LinearWarmupScheduler(
                optimizer,
                warmup_epochs=warmup_epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

# Export key classes and functions
__all__ = [
    'AdEMAMix',
    'CosineAnnealingWarmRestarts', 
    'LinearWarmupScheduler',
    'GradientClipping',
    'OptimizerFactory'
] 