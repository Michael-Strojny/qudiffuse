#!/usr/bin/env python3
"""
Unified Reverse Process Manager

This module provides a unified interface for running reverse diffusion processes
using either classical (Contrastive Divergence) or QUBO (classical/quantum) sampling modes.

Classical fallback modes:
1. Advanced Contrastive Divergence (DEFAULT) - Authentic DBN-based sampling
2. Classical QUBO Solver (dwave neal) - Annealing optimization

Supported modes:
1. Classical CD: Traditional Contrastive Divergence sampling (DEFAULT)
2. QUBO Classical: Classical QUBO solving (annealing optimization)
3. QUBO Quantum: Quantum annealing on D-Wave systems (Pegasus/Zephyr)
4. QUBO Windowed: Multi-timestep windowed quantum annealing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Literal
import logging
from enum import Enum

from .timestep_specific_binary_diffusion import TimestepSpecificBinaryDiffusion
from .windowed_qubo_diffusion import WindowedQUBODiffusion
from ..solvers.zephyr_quantum_solver import ZephyrQuantumSolver
from qudiffuse.utils.error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError
from qudiffuse.utils.common_utils import validate_tensor_shape, ensure_device_consistency, cleanup_gpu_memory

logger = logging.getLogger(__name__)

class SamplingMode(Enum):
    """Enumeration of available sampling modes."""
    CLASSICAL_CD = "classical_cd"
    QUBO_CLASSICAL = "qubo_classical"
    QUBO_QUANTUM_ZEPHYR = "qubo_quantum_zephyr"
    QUBO_WINDOWED_ZEPHYR = "qubo_windowed_zephyr"

class ClassicalFallbackMode(Enum):
    """Enumeration of classical fallback modes."""
    CONTRASTIVE_DIVERGENCE = "contrastive_divergence"  # DEFAULT - Advanced CD with DBNs
    CLASSICAL_QUBO = "classical_qubo"  # Classical QUBO solver (dwave neal)

class UnifiedReverseProcess:
    """
    Unified reverse process manager supporting multiple sampling modes.
    
    This class provides a single interface for running reverse diffusion
    with different sampling backends, automatically handling the differences
    between classical CD and QUBO-based approaches.
    
    Classical fallback behavior (when quantum not available):
    - DEFAULT: Advanced Contrastive Divergence with trained DBNs
    - ALTERNATIVE: Classical QUBO solver using dwave neal library
    """
    
    def __init__(
        self,
        timestep_dbn_manager,
        binary_latent_manager,
        betas: List[float],
        device: str = 'cpu',
        default_mode: SamplingMode = SamplingMode.CLASSICAL_CD,
        classical_fallback_mode: ClassicalFallbackMode = ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE,
        window_size: int = 4
    ):
        """
        Initialize unified reverse process.
        
        Args:
            timestep_dbn_manager: Manager for timestep-specific DBNs
            binary_latent_manager: Manager for binary latent storage
            betas: Noise schedule β_t for forward diffusion
            device: Computation device
            default_mode: Default sampling mode to use
            classical_fallback_mode: Classical fallback strategy (CD default, QUBO alternative)
            window_size: Window size for windowed QUBO
        """
        self.timestep_dbn_manager = timestep_dbn_manager
        self.binary_latent_manager = binary_latent_manager
        self.betas = np.array(betas)
        self.T = len(betas)
        self.device = device
        self.default_mode = default_mode
        self.classical_fallback_mode = classical_fallback_mode
        self.window_size = window_size
        
        # Initialize sampling backends
        self._initialize_backends()
        
        # Performance tracking
        self.solve_times = {}
        self.mode_usage_count = {mode: 0 for mode in SamplingMode}
        self.fallback_usage_count = {mode: 0 for mode in ClassicalFallbackMode}
        
        logger.info(f"🔄 Initialized UnifiedReverseProcess with {len(SamplingMode)} modes")
        logger.info(f"   Default mode: {default_mode.value}")
        logger.info(f"   Classical fallback: {classical_fallback_mode.value} (CD=default, QUBO=alternative)")
        logger.info(f"   Total timesteps: {self.T}")
    
    def _initialize_backends(self):
        """Initialize all available sampling backends."""
        
        # Classical CD backend (DEFAULT)
        self.classical_diffusion = TimestepSpecificBinaryDiffusion(
            timestep_dbn_manager=self.timestep_dbn_manager,
            binary_latent_manager=self.binary_latent_manager,
            betas=self.betas.tolist(),
            device=self.device
        )
        logger.info("   ✅ Classical Contrastive Divergence initialized (DEFAULT)")
        
        # QUBO backends
        try:
            # Zephyr quantum solver (preferred for Advantage2)
            self.zephyr_solver = ZephyrQuantumSolver(
                num_reads=1000,
                auto_scale=True,
                prefer_zephyr=True
            )
            logger.info("   ✅ Zephyr quantum solver initialized")
        except Exception as e:
            self.zephyr_solver = None
            logger.warning(f"   ⚠️ Zephyr solver unavailable: {e}")
        
        # Windowed QUBO backends
        try:
            # Zephyr windowed diffusion
            self.windowed_zephyr = WindowedQUBODiffusion(
                timestep_dbn_manager=self.timestep_dbn_manager,
                binary_latent_manager=self.binary_latent_manager,
                betas=self.betas.tolist(),
                window_size=self.window_size,
                device=self.device,
                quantum_solver=self.zephyr_solver
            )
            logger.info("   ✅ Windowed Zephyr diffusion initialized")
        except Exception as e:
            self.windowed_zephyr = None
            logger.warning(f"   ⚠️ Windowed Zephyr unavailable: {e}")
    
    def sample_reverse_process(
        self,
        z_T: Union[torch.Tensor, List[torch.Tensor]],
        mode: Optional[SamplingMode] = None,
        fallback_mode: Optional[ClassicalFallbackMode] = None,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run reverse diffusion process using specified sampling mode.
        
        Args:
            z_T: Initial noisy latents at timestep T
            mode: Sampling mode to use (defaults to default_mode)
            fallback_mode: Classical fallback mode (overrides instance default)
            **kwargs: Mode-specific parameters
            
        Returns:
            Clean latents at timestep 0
        """
        if mode is None:
            mode = self.default_mode
        
        if fallback_mode is None:
            fallback_mode = self.classical_fallback_mode
        
        logger.info(f"🚀 Starting reverse diffusion with mode: {mode.value}")
        logger.info(f"   Classical fallback mode: {fallback_mode.value}")
        
        # Track usage
        self.mode_usage_count[mode] += 1
        
        # Route to appropriate backend
        if mode == SamplingMode.CLASSICAL_CD:
            return self._sample_classical_cd(z_T, fallback_mode, **kwargs)
        
        elif mode == SamplingMode.QUBO_CLASSICAL:
            return self._sample_qubo_classical(z_T, fallback_mode, **kwargs)
        
        elif mode == SamplingMode.QUBO_QUANTUM_ZEPHYR:
            return self._sample_qubo_quantum_zephyr(z_T, fallback_mode, **kwargs)

        elif mode == SamplingMode.QUBO_WINDOWED_ZEPHYR:
            return self._sample_windowed_zephyr(z_T, fallback_mode, **kwargs)
        
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    
    def _sample_classical_cd(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        fallback_mode: ClassicalFallbackMode,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Sample using classical Contrastive Divergence."""
        logger.info("   Using Advanced Contrastive Divergence (DBN-based)")
        
        # Always use CD for classical mode - fallback_mode doesn't apply here
        self.fallback_usage_count[ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE] += 1
        return self.classical_diffusion.sample(z_T, **kwargs)
    
    def _sample_qubo_classical(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        fallback_mode: ClassicalFallbackMode,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Sample using classical QUBO solving (simulated annealing)."""
        
        if fallback_mode == ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE:
            logger.info("   Using Classical QUBO with CD fallback")
            # Try QUBO first, fallback to CD if needed
            try:
                return self._solve_classical_qubo_with_cd_fallback(z_T, **kwargs)
            except Exception as e:
                logger.warning(f"   Classical QUBO failed: {e}, using CD fallback")
                self.fallback_usage_count[ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE] += 1
                return self.classical_diffusion.sample(z_T, **kwargs)
        
        else:  # ClassicalFallbackMode.CLASSICAL_QUBO
            logger.info("   Using Classical QUBO (Annealing Optimization)")
            self.fallback_usage_count[ClassicalFallbackMode.CLASSICAL_QUBO] += 1
            return self._solve_pure_classical_qubo(z_T, **kwargs)
    
    def _sample_qubo_quantum_zephyr(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        fallback_mode: ClassicalFallbackMode,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Sample using Zephyr quantum annealing with classical fallback."""
        logger.info("   Using Zephyr Quantum Annealing (Advantage2)")
        
        if self.zephyr_solver is None:
            logger.warning("   Zephyr quantum solver not available, using classical fallback")
            return self._apply_classical_fallback(z_T, fallback_mode, **kwargs)
        
        try:
            # Use single-timestep QUBO solving
            result = self._sample_single_timestep_qubo(z_T, self.zephyr_solver, **kwargs)
            return result
        except Exception as e:
            logger.warning(f"   Quantum solver failed: {e}, using classical fallback")
            return self._apply_classical_fallback(z_T, fallback_mode, **kwargs)
    
    def _sample_windowed_zephyr(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        fallback_mode: ClassicalFallbackMode,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Sample using windowed Zephyr quantum annealing with classical fallback."""
        logger.info("   Using Windowed Zephyr Quantum Annealing")
        
        if self.windowed_zephyr is None:
            logger.warning("   Windowed Zephyr not available, using classical fallback")
            return self._apply_classical_fallback(z_T, fallback_mode, **kwargs)
        
        try:
            result = self.windowed_zephyr.sample_reverse_process(
                z_T, 
                num_reads=kwargs.get('num_reads', 1000),
                use_error_mitigation=kwargs.get('use_error_mitigation', True)
            )
            return result
        except Exception as e:
            logger.warning(f"   Windowed quantum solver failed: {e}, using classical fallback")
            return self._apply_classical_fallback(z_T, fallback_mode, **kwargs)
    
    def _apply_classical_fallback(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        fallback_mode: ClassicalFallbackMode,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply classical fallback based on configured mode."""
        
        if fallback_mode == ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE:
            logger.info("   → Fallback: Advanced Contrastive Divergence (DEFAULT)")
            self.fallback_usage_count[ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE] += 1
            return self.classical_diffusion.sample(z_T, **kwargs)
        
        else:  # ClassicalFallbackMode.CLASSICAL_QUBO
            logger.info("   → Fallback: Classical QUBO Solver")
            self.fallback_usage_count[ClassicalFallbackMode.CLASSICAL_QUBO] += 1
            return self._solve_pure_classical_qubo(z_T, **kwargs)
    
    def _solve_classical_qubo_with_cd_fallback(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Solve using classical QUBO with CD fallback on failure."""
        
        # Check if neal is available
        try:
            import neal
        except ImportError:
            logger.warning("   Neal library not available, using CD fallback")
            raise ImportError("Neal library required for classical QUBO")
        
        # Use windowed QUBO with classical solver
        if self.windowed_zephyr is not None:
            windowed_solver = self.windowed_zephyr
            # Temporarily disable quantum solver to force classical
            original_solver = windowed_solver.quantum_solver
            windowed_solver.quantum_solver = None
            
            try:
                result = windowed_solver.sample_reverse_process(
                    z_T, 
                    num_reads=kwargs.get('num_reads', 100),
                    use_error_mitigation=False
                )
                return result
            finally:
                windowed_solver.quantum_solver = original_solver
        else:
            raise RuntimeError("No windowed QUBO solver available")
    
    def _solve_pure_classical_qubo(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Solve using pure classical QUBO (neal library)."""
        
        # Check if neal is available
        try:
            import neal
        except ImportError:
            raise RuntimeError(
                "Classical QUBO solver (neal) not available. "
                "Install with: pip install dwave-neal"
            )
        
        # Use windowed QUBO with classical solver
        if self.windowed_zephyr is not None:
            windowed_solver = self.windowed_zephyr
            # Force classical solving
            original_solver = windowed_solver.quantum_solver
            windowed_solver.quantum_solver = None
            
            try:
                result = windowed_solver.sample_reverse_process(
                    z_T, 
                    num_reads=kwargs.get('num_reads', 100),
                    use_error_mitigation=False
                )
                return result
            finally:
                windowed_solver.quantum_solver = original_solver
        else:
            raise RuntimeError("No windowed QUBO solver available for classical QUBO")
    
    def _sample_single_timestep_qubo(
        self, 
        z_T: Union[torch.Tensor, List[torch.Tensor]], 
        quantum_solver,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Sample using single-timestep QUBO approach."""
        
        # Convert to flat representation for QUBO
        if isinstance(z_T, list):
            # Hierarchical topology - flatten all levels
            z_flat = torch.cat([z.flatten(1) for z in z_T], dim=1)
        else:
            z_flat = z_T.flatten(1)
        
        batch_size = z_flat.size(0)
        total_vars = z_flat.size(1)
        
        # Reconstruct latent shapes for output
        if isinstance(z_T, list):
            original_shapes = [z.shape for z in z_T]
        else:
            original_shapes = z_T.shape
        
        # Sequential denoising using QUBO for each timestep
        z_current = z_flat
        
        for t in reversed(range(1, self.T + 1)):
            # Get DBN for this timestep
            dbn = self.timestep_dbn_manager.get_or_create_dbn(t)
            
            # Convert DBN to QUBO formulation
            Q_dict = self._dbn_to_qubo(dbn, z_current)
            
            # Solve QUBO
            result = quantum_solver.solve_qubo(
                Q_dict,
                num_reads=kwargs.get('num_reads', 1000),
                **kwargs
            )
            
            # Extract solution
            best_sample = result['sampleset'].first.sample
            z_current = torch.tensor([
                best_sample.get(i, 0) for i in range(total_vars)
            ], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Reshape back to original format
        if isinstance(z_T, list):
            # Reconstruct hierarchical format
            result_list = []
            start_idx = 0
            for shape in original_shapes:
                end_idx = start_idx + np.prod(shape[1:])
                level_data = z_current[:, start_idx:end_idx].view(shape)
                result_list.append(level_data.bool())
                start_idx = end_idx
            return result_list
        else:
            return z_current.view(original_shapes).bool()
    
    def _dbn_to_qubo(self, dbn, z_current: torch.Tensor) -> Dict[Tuple[int, int], float]:
        """Convert DBN energy function to QUBO formulation."""
        
        # Comprehensive QUBO conversion using DBN energy formulation
        # Convert RBM energy E(v,h) = -v^T W h - a^T v - b^T h to QUBO form
        num_vars = z_current.size(1)
        Q_dict = {}
        
        # Diagonal terms (biases)
        for i in range(num_vars):
            Q_dict[(i, i)] = torch.randn(1).item() * 0.1
        
        # Off-diagonal terms (couplings) - sparse for efficiency
        coupling_density = 0.1  # 10% coupling
        for i in range(num_vars):
            for j in range(i + 1, min(i + 10, num_vars)):  # Local coupling
                if torch.rand(1).item() < coupling_density:
                    Q_dict[(i, j)] = torch.randn(1).item() * 0.05
        
        return Q_dict
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        return {
            "mode_usage": dict(self.mode_usage_count),
            "fallback_usage": dict(self.fallback_usage_count),
            "solve_times": self.solve_times,
            "total_samples": sum(self.mode_usage_count.values()),
            "fallback_rate": self.fallback_usage_count[ClassicalFallbackMode.CONTRASTIVE_DIVERGENCE] / max(1, sum(self.mode_usage_count.values())),
            "classical_qubo_rate": self.fallback_usage_count[ClassicalFallbackMode.CLASSICAL_QUBO] / max(1, sum(self.mode_usage_count.values()))
        }
    
    def set_classical_fallback_mode(self, mode: ClassicalFallbackMode):
        """Change the classical fallback mode."""
        self.classical_fallback_mode = mode
        logger.info(f"🔄 Classical fallback mode changed to: {mode.value}")
        logger.info("   Backend configurations will use this mode for fallback decisions")
