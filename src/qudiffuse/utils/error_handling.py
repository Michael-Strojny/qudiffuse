#!/usr/bin/env python3
"""
Comprehensive Error Handling Module for QuDiffuse Training Pipeline

This module provides custom exception classes and error handling utilities
to ensure authentic, traceable, and informative error reporting.
"""

import logging
import sys
from typing import Optional, Dict, Any

class QuDiffuseBaseError(Exception):
    """Base exception for all QuDiffuse-specific errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize a QuDiffuse base error with optional context.
        
        Args:
            message (str): Error description
            context (Optional[Dict[str, Any]]): Additional error context
        """
        super().__init__(message)
        self.context = context or {}
        self._log_error()
    
    def _log_error(self):
        """Log error with comprehensive details."""
        logger = logging.getLogger('QuDiffuseErrorHandler')
        logger.error(f"❌ {self.__class__.__name__}: {self}")
        
        if self.context:
            logger.error("Error Context:")
            for key, value in self.context.items():
                logger.error(f"  {key}: {value}")

class QuDiffuseTrainingError(QuDiffuseBaseError):
    """Specific error for training pipeline failures."""
    
    def __init__(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None
    ):
        """
        Initialize a training-specific error.
        
        Args:
            message (str): Error description
            context (Optional[Dict[str, Any]]): Additional error context
            recovery_suggestion (Optional[str]): Potential solution
        """
        context = context or {}
        if recovery_suggestion:
            context['recovery_suggestion'] = recovery_suggestion
        
        super().__init__(message, context)

class CheckpointIntegrityError(QuDiffuseTrainingError):
    """Error raised when checkpoint fails integrity checks."""
    pass

class ConfigurationError(QuDiffuseTrainingError):
    """Error raised for invalid configuration parameters."""
    pass

def global_exception_handler(
    exc_type, 
    exc_value, 
    exc_traceback
):
    """
    Global exception handler for unhandled exceptions.
    
    Provides comprehensive logging and potential recovery strategies.
    
    Args:
        exc_type (Type[Exception]): Exception type
        exc_value (Exception): Exception instance
        exc_traceback (TracebackType): Traceback object
    """
    logger = logging.getLogger('QuDiffuseGlobalHandler')
    
    logger.critical(
        "🚨 Unhandled Exception in QuDiffuse Training Pipeline 🚨"
    )
    logger.critical(f"Type: {exc_type.__name__}")
    logger.critical(f"Message: {exc_value}")
    
    # Optional: Log traceback for debugging
    import traceback
    logger.critical("Traceback:\n" + 
        ''.join(traceback.format_tb(exc_traceback))
    )
    
    # Potential system-wide recovery or notification mechanism
    sys.exit(1)

# Set global exception handler
sys.excepthook = global_exception_handler

def configure_error_logging(log_level: str = 'INFO'):
    """
    Configure comprehensive error logging.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('qudiffuse_error.log', mode='w')
        ]
    )

# Configure default error logging
configure_error_logging()
