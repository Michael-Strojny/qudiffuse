"""Utility functions for QuDiffuse."""

from .common_utils import validate_tensor_shape, ensure_device_consistency, cleanup_gpu_memory
from .error_handling import TopologyError, BinaryLatentError, ConfigurationError, TrainingError, DBNError

__all__ = [
    "validate_tensor_shape",
    "ensure_device_consistency",
    "cleanup_gpu_memory",
    "TopologyError",
    "BinaryLatentError",
    "ConfigurationError",
    "TrainingError",
    "DBNError"
]