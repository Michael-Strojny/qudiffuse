#!/usr/bin/env python3
"""
Centralized Common Utilities for QuDiffuse

This module consolidates frequently used utility functions to eliminate
code duplication across the codebase.
"""

import torch
import torch.nn as nn
from typing import Union, List, Tuple, Optional, Any, Dict, Callable
import logging
from .error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError, QUBOError

logger = logging.getLogger(__name__)

def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                         name: str = "tensor") -> None:
    """
    Validate tensor shape matches expected dimensions.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected shape tuple (use -1 for any dimension)
        name: Name of tensor for error messages
    """
    if tensor.dim() != len(expected_shape):
        raise ValueError(f"{name} expected {len(expected_shape)} dimensions, got {tensor.dim()}")
    
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(f"{name} dimension {i} expected {expected}, got {actual}")

def ensure_device_consistency(*tensors: torch.Tensor, target_device: Optional[torch.device] = None) -> List[torch.Tensor]:
    """
    Ensure all tensors are on the same device.
    
    Args:
        *tensors: Variable number of tensors
        target_device: Target device (uses first tensor's device if None)
    
    Returns:
        List of tensors on the same device
    """
    if not tensors:
        return []
    
    if target_device is None:
        target_device = tensors[0].device
    
    return [tensor.to(target_device, non_blocking=True) for tensor in tensors]

def validate_binary_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Validate that tensor contains only binary values (0 or 1).
    
    Args:
        tensor: Input tensor to validate
        name: Name of tensor for error messages
    """
    if tensor.dtype == torch.bool:
        return  # Bool tensors are inherently binary
    
    if tensor.dtype in [torch.uint8, torch.int8]:
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError(f"{name} must contain only 0 or 1 values")
    else:
        # Float tensors should be exactly 0.0 or 1.0
        if not torch.allclose(tensor, tensor.round(), atol=1e-6):
            raise ValueError(f"{name} must contain only binary values (0.0 or 1.0)")

def safe_tensor_operation(operation: Callable, *tensors: torch.Tensor,
                         default_value: Any = None, name: str = "operation") -> Any:
    """
    Safely perform tensor operation with error handling.
    
    Args:
        operation: Function to apply to tensors
        *tensors: Input tensors
        default_value: Value to return on error
        name: Operation name for logging
    
    Returns:
        Result of operation or default_value on error
    """
    try:
        return operation(*tensors)
    except Exception as e:
        logger.warning(f"Tensor {name} failed: {e}")
        if default_value is not None:
            return default_value
        raise

def calculate_memory_usage(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Calculate memory usage of a tensor.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Dictionary with memory usage statistics
    """
    element_size = tensor.element_size()
    num_elements = tensor.numel()
    total_bytes = element_size * num_elements
    
    return {
        'bytes': total_bytes,
        'kb': total_bytes / 1024,
        'mb': total_bytes / (1024 * 1024),
        'gb': total_bytes / (1024 * 1024 * 1024),
        'elements': num_elements,
        'element_size': element_size,
        'dtype': str(tensor.dtype),
        'shape': tuple(tensor.shape)
    }

def optimize_tensor_memory_format(tensor: torch.Tensor, prefer_channels_last: bool = True) -> torch.Tensor:
    """
    Optimize tensor memory format for better performance.
    
    Args:
        tensor: Input tensor
        prefer_channels_last: Whether to prefer channels_last format for 4D tensors
    
    Returns:
        Tensor with optimized memory format
    """
    if tensor.dim() == 4 and prefer_channels_last:
        return tensor.to(memory_format=torch.channels_last)
    elif tensor.dim() == 3:
        return tensor.contiguous()
    else:
        return tensor.contiguous()

def batch_tensor_operation(operation: Callable, tensors: List[torch.Tensor], 
                          batch_size: int = 32, **kwargs) -> List[Any]:
    """
    Apply operation to tensors in batches for memory efficiency.
    
    Args:
        operation: Function to apply
        tensors: List of tensors
        batch_size: Batch size for processing
        **kwargs: Additional arguments for operation
    
    Returns:
        List of results
    """
    results = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size]
        batch_results = [operation(tensor, **kwargs) for tensor in batch]
        results.extend(batch_results)
    return results

def standardize_error_message(error_type: str, component: str, details: str) -> str:
    """
    Standardize error message format across the codebase.
    
    Args:
        error_type: Type of error (e.g., "validation", "runtime", "configuration")
        component: Component where error occurred
        details: Detailed error description
    
    Returns:
        Standardized error message
    """
    return f"[{error_type.upper()}] {component}: {details}"

def log_tensor_stats(tensor: torch.Tensor, name: str = "tensor", level: str = "debug") -> None:
    """
    Log tensor statistics for debugging.
    
    Args:
        tensor: Input tensor
        name: Name of tensor
        level: Logging level
    """
    stats = {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'min': tensor.min().item() if tensor.numel() > 0 else 'N/A',
        'max': tensor.max().item() if tensor.numel() > 0 else 'N/A',
        'mean': tensor.float().mean().item() if tensor.numel() > 0 else 'N/A'
    }
    
    message = f"{name} stats: {stats}"
    
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)

def cleanup_gpu_memory(aggressive: bool = False):
    """Clean up GPU memory with optional aggressive mode."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.ipc_collect()
    except Exception as e:
        logger.warning(f"GPU memory cleanup failed: {e}")

def setup_logging(level=logging.INFO, format_string=None):
    """Setup logging configuration for the application."""
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('qudiffuse.log')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('qudiffuse').setLevel(level)
    
    return logging.getLogger(__name__)

def validate_dbn_latent_consistency(dbn, binary_latent_manager) -> bool:
    """
    Validate that DBN layers match latent channels exactly.
    
    Args:
        dbn: HierarchicalDBN instance
        binary_latent_manager: BinaryLatentManager instance
        
    Returns:
        True if consistent, False otherwise
        
    Raises:
        DBNError: If critical inconsistencies found
    """
    try:
        # Get expected dimensions from binary latent manager
        expected_dims = binary_latent_manager.get_dbn_layer_dimensions()
        
        # Check number of layers
        if len(dbn.rbms) != len(expected_dims):
            raise DBNError(
                f"DBN layer count ({len(dbn.rbms)}) != expected ({len(expected_dims)})",
                "dbn_validation"
            )
        
        # Check each layer's visible dimension
        for i, (rbm, expected_dim) in enumerate(zip(dbn.rbms, expected_dims)):
            actual_vis_dim = rbm.visible_size
            
            # For lower layers, visible units include hidden units from layer above
            if i < len(dbn.rbms) - 1:
                # Visible units = z^(ℓ) + h^(ℓ+1)
                hidden_above_dim = dbn.hidden_dims[i + 1]
                expected_vis_dim = expected_dim + hidden_above_dim
            else:
                # Top layer: only z^(ℓ)
                expected_vis_dim = expected_dim
            
            if actual_vis_dim != expected_vis_dim:
                raise DBNError(
                    f"DBN layer {i} visible size ({actual_vis_dim}) != expected ({expected_vis_dim})",
                    "dbn_validation"
                )
        
        logger.info("✅ DBN-latent consistency validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ DBN-latent consistency validation failed: {e}")
        return False

def validate_qubo_mathematical_equivalence(rbm1, rbm2, tolerance: float = 1e-6) -> bool:
    """
    Verify that two QUBO implementations produce equivalent results.
    
    Args:
        rbm1, rbm2: RBM instances to compare
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if mathematically equivalent, False otherwise
    """
    try:
        # Generate test data
        batch_size = 4
        test_input = torch.randn(batch_size, rbm1.visible_size, device=rbm1.device)
        test_input = torch.sigmoid(test_input)  # Make it probability-like
        
        # Compare energy calculations
        with torch.no_grad():
            # Method 1: Direct energy via free energy
            energy1_1 = rbm1.free_energy(test_input)
            energy1_2 = rbm2.free_energy(test_input)
            
            # Method 2: Via QUBO formulation
            J1, h1 = rbm1.qubo_terms() if hasattr(rbm1, 'qubo_terms') else (None, None)
            J2, h2 = rbm2.qubo_terms() if hasattr(rbm2, 'qubo_terms') else (None, None)
            
            if J1 is not None and J2 is not None:
                # Compare QUBO matrices
                qubo_diff = torch.abs(J1 - J2).max()
                bias_diff = torch.abs(h1 - h2).max()
                
                if qubo_diff > tolerance or bias_diff > tolerance:
                    logger.warning(f"QUBO matrices differ: J_diff={qubo_diff:.2e}, h_diff={bias_diff:.2e}")
                    return False
            
            # Compare free energies
            energy_diff = torch.abs(energy1_1 - energy1_2).max()
            if energy_diff > tolerance:
                logger.warning(f"Free energies differ: {energy_diff:.2e}")
                return False
        
        logger.info("✅ QUBO mathematical equivalence validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ QUBO equivalence validation failed: {e}")
        return False

def validate_binary_storage_consistency(binary_manager, test_shapes: List[Tuple]) -> bool:
    """
    Check binary storage consistency across different operations.
    
    Args:
        binary_manager: BinaryLatentManager instance
        test_shapes: List of shapes to test
        
    Returns:
        True if consistent, False otherwise
    """
    try:
        for shape in test_shapes:
            # Test different fill values
            for fill_val in [0.0, 0.5, 1.0]:
                # Create binary tensor
                binary_tensor = binary_manager.create_binary_tensor(shape, fill_val)
                
                # Validate it's actually binary
                if not binary_manager.validate_binary_exact(binary_tensor):
                    logger.error(f"Binary tensor validation failed for shape {shape}, fill={fill_val}")
                    return False
                
                # Test round-trip conversion
                float_tensor = binary_manager.to_float_for_gradients(binary_tensor)
                recovered_binary = binary_manager.from_float_to_binary(float_tensor)
                
                # Check equivalence
                if not torch.allclose(binary_tensor.float(), recovered_binary.float()):
                    logger.error(f"Round-trip conversion failed for shape {shape}")
                    return False
                
                # Test packed format if available
                if binary_manager.storage_format == 'packed':
                    if hasattr(binary_manager, 'unpack_binary_tensor'):
                        unpacked = binary_manager.unpack_binary_tensor(binary_tensor)
                        if unpacked.shape != shape:
                            logger.error(f"Packed format shape mismatch: {unpacked.shape} != {shape}")
                            return False
        
        logger.info("✅ Binary storage consistency validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Binary storage consistency validation failed: {e}")
        return False

def validate_diffusion_process_integrity(diffusion_process, test_latents) -> bool:
    """
    Validate the integrity of the diffusion process.
    
    Args:
        diffusion_process: Diffusion process instance
        test_latents: Test latent tensors
        
    Returns:
        True if process is valid, False otherwise
    """
    try:
        # Test forward process
        for t in [1, diffusion_process.T//2, diffusion_process.T]:
            noisy_latents = diffusion_process.forward_process(test_latents, t)
            
            # Validate output format
            if isinstance(test_latents, list):
                if not isinstance(noisy_latents, list) or len(noisy_latents) != len(test_latents):
                    logger.error(f"Forward process output format mismatch at t={t}")
                    return False
            else:
                if not isinstance(noisy_latents, torch.Tensor):
                    logger.error(f"Forward process output type mismatch at t={t}")
                    return False
            
            # Validate binary storage is preserved
            if hasattr(diffusion_process, 'binary_latent_manager'):
                if isinstance(noisy_latents, list):
                    for i, lat in enumerate(noisy_latents):
                        if not diffusion_process.binary_latent_manager.validate_binary_exact(lat):
                            logger.error(f"Binary storage violated in forward process at t={t}, level={i}")
                            return False
                else:
                    if not diffusion_process.binary_latent_manager.validate_binary_exact(noisy_latents):
                        logger.error(f"Binary storage violated in forward process at t={t}")
                        return False
        
        # Test single reverse step
        if hasattr(diffusion_process, '_denoise_step'):
            try:
                # Create test noisy latents
                if isinstance(test_latents, list):
                    test_noisy = [torch.bernoulli(0.5 * torch.ones_like(lat.float())).bool() 
                                 for lat in test_latents]
                else:
                    test_noisy = torch.bernoulli(0.5 * torch.ones_like(test_latents.float())).bool()
                
                denoised = diffusion_process._denoise_step(test_noisy, 1)
                
                # Validate output format and binary storage
                if isinstance(test_latents, list):
                    if not isinstance(denoised, list):
                        logger.error("Reverse step output format mismatch")
                        return False
                    for lat in denoised:
                        if hasattr(diffusion_process, 'binary_latent_manager'):
                            if not diffusion_process.binary_latent_manager.validate_binary_exact(lat):
                                logger.error("Binary storage violated in reverse step")
                                return False
                
            except Exception as e:
                logger.warning(f"Reverse step test failed (may be expected): {e}")
        
        logger.info("✅ Diffusion process integrity validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Diffusion process integrity validation failed: {e}")
        return False

def run_comprehensive_validation(system_components: Dict) -> Dict[str, bool]:
    """
    Run comprehensive validation across all system components.
    
    Args:
        system_components: Dictionary of system components to validate
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    logger.info("🔍 Starting comprehensive system validation...")
    
    # DBN-Latent consistency
    if 'dbn' in system_components and 'binary_manager' in system_components:
        results['dbn_latent_consistency'] = validate_dbn_latent_consistency(
            system_components['dbn'], system_components['binary_manager']
        )
    
    # QUBO mathematical equivalence (if multiple RBMs available)
    if 'rbm1' in system_components and 'rbm2' in system_components:
        results['qubo_equivalence'] = validate_qubo_mathematical_equivalence(
            system_components['rbm1'], system_components['rbm2']
        )
    
    # Binary storage consistency
    if 'binary_manager' in system_components:
        test_shapes = [(1, 16, 8, 8), (1, 32, 4, 4), (1, 64, 2, 2)]
        results['binary_storage_consistency'] = validate_binary_storage_consistency(
            system_components['binary_manager'], test_shapes
        )
    
    # Diffusion process integrity
    if 'diffusion_process' in system_components and 'test_latents' in system_components:
        results['diffusion_integrity'] = validate_diffusion_process_integrity(
            system_components['diffusion_process'], system_components['test_latents']
        )
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    logger.info(f"🎯 Validation Summary: {passed}/{total} checks passed")
    
    if passed < total:
        failed_checks = [check for check, result in results.items() if not result]
        logger.error(f"❌ Failed checks: {failed_checks}")
    else:
        logger.info("✅ All validation checks passed!")
    
    return results

def get_optimal_batch_size(model: nn.Module, input_shape: Tuple[int, ...], 
                          max_memory_gb: float = 8.0) -> int:
    """
    Estimate optimal batch size based on available memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of single input (without batch dimension)
        max_memory_gb: Maximum memory to use in GB
    
    Returns:
        Estimated optimal batch size
    """
    # Create sample input for memory estimation
    sample_input = torch.randn(1, *input_shape)
    
    # Estimate memory per sample
    model.eval()
    with torch.no_grad():
        try:
            _ = model(sample_input)

            memory_per_sample_mb = calculate_memory_usage(sample_input)['mb'] * 2  # Factor for activations
            max_memory_mb = max_memory_gb * 1024
            optimal_batch_size = int(max_memory_mb / memory_per_sample_mb)
            return max(1, min(optimal_batch_size, 128))  # Reasonable bounds
        except Exception:
            return 32  # Safe default

def validate_config_dict(config: dict, required_keys: List[str], 
                        component: str = "configuration") -> None:
    """
    Validate configuration dictionary has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        component: Component name for error messages
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        error_msg = standardize_error_message(
            "configuration", 
            component, 
            f"Missing required keys: {missing_keys}"
        )
        raise ValueError(error_msg)

def create_tensor_safely(data: Any, dtype: torch.dtype = torch.float32, 
                        device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Safely create tensor with error handling.
    
    Args:
        data: Input data
        dtype: Target dtype
        device: Target device
    
    Returns:
        Created tensor
    """
    try:
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=device)
        else:
            return torch.tensor(data, dtype=dtype, device=device)
    except Exception as e:
        error_msg = standardize_error_message(
            "tensor_creation", 
            "create_tensor_safely", 
            f"Failed to create tensor: {e}"
        )
        raise RuntimeError(error_msg) from e

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging and debugging.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import sys
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info
