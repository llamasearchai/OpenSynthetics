"""MLX utilities for OpenSynthetics."""

import os
from typing import Any, Dict, List, Optional, Union, Tuple, TypeVar, cast, Type

import numpy as np
from loguru import logger

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import MLXAcceleratorError
import platform


T = TypeVar('T', bound=np.ndarray)

class MLXAccelerator:
    """MLX acceleration for Apple Silicon devices.
    
    This class provides utilities for accelerating computation on Apple Silicon
    devices using MLX. It falls back gracefully to NumPy when MLX is not available.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MLX accelerator.

        Args:
            config: Configuration with MLX settings
        """
        self.enabled = self._check_config_enabled(config)
        self.mx = None
        self._initialized = False
        
        if self.enabled:
            try:
                self._initialize_mlx()
                self._initialized = True
                logger.info("MLX accelerator initialized successfully")
            except ImportError:
                logger.warning("MLX is enabled but not installed. Install with 'pip install mlx'")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize MLX: {e}")
                self.enabled = False
    
    def _check_config_enabled(self, config: Optional[Any] = None) -> bool:
        """Check if MLX is enabled in config.
        
        Args:
            config: Configuration object
            
        Returns:
            bool: Whether MLX is enabled
        """
        # Default to enabled if running on Apple Silicon
        is_apple_silicon = self._is_apple_silicon()
        
        if config is None:
            return is_apple_silicon
        
        # Check if enabled in config
        try:
            if hasattr(config, 'settings') and isinstance(config.settings, dict):
                return config.settings.get('mlx', {}).get('enabled', is_apple_silicon)
            elif hasattr(config, 'mlx') and hasattr(config.mlx, 'enabled'):
                return bool(config.mlx.enabled)
            else:
                return is_apple_silicon
        except Exception as e:
            logger.debug(f"Error checking MLX config, defaulting to {is_apple_silicon}: {e}")
            return is_apple_silicon
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon.
        
        Returns:
            bool: Whether running on Apple Silicon
        """
        try:
            return platform.system() == "Darwin" and platform.processor() == "arm"
        except Exception as e:
            logger.debug(f"Error detecting Apple Silicon: {e}")
            return False
    
    def _initialize_mlx(self) -> None:
        """Initialize MLX library.
        
        Raises:
            ImportError: If MLX is not installed
        """
        try:
            import mlx.core as mx
            self.mx = mx
            # Test basic operation to verify MLX works
            test = mx.array([1, 2, 3])
            _ = test + test
        except ImportError:
            raise ImportError("MLX is not installed. Install with 'pip install mlx'")
        except Exception as e:
            raise RuntimeError(f"MLX initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if MLX is available and initialized.
        
        Returns:
            bool: True if MLX is available
        """
        return self.enabled and self.mx is not None and self._initialized
    
    def array(self, data: Union[list, np.ndarray, float, int]) -> Any:
        """Convert data to MLX array.
        
        Args:
            data: Input data (list, NumPy array, float, or int)
            
        Returns:
            MLX array
            
        Raises:
            MLXAcceleratorError: If MLX is not available or conversion fails
        """
        if not self.is_available():
            raise MLXAcceleratorError("MLX accelerator is not available")
        
        try:
            # Handle different input types efficiently
            if isinstance(data, (float, int)):
                return self.mx.array(data)
            elif isinstance(data, np.ndarray):
                return self.mx.array(data)
            elif isinstance(data, list):
                return self.mx.array(data)
            else:
                # Try to convert to numpy first for other types
                return self.mx.array(np.array(data))
        except Exception as e:
            raise MLXAcceleratorError(f"Failed to convert to MLX array: {e}")
    
    def to_numpy(self, arr: Any) -> np.ndarray:
        """Convert MLX array to NumPy array.
        
        Args:
            arr: MLX array or any object convertible to NumPy array
            
        Returns:
            np.ndarray: NumPy array
            
        Raises:
            MLXAcceleratorError: If conversion fails
        """
        if not self.is_available():
            raise MLXAcceleratorError("MLX accelerator is not available")
        
        try:
            # Check if this is an MLX array (has tolist method)
            if hasattr(arr, 'tolist'):
                return np.array(arr.tolist())
            # Return NumPy arrays directly
            elif isinstance(arr, np.ndarray):
                return arr
            # Handle scalars (float, int)
            elif isinstance(arr, (float, int)):
                return np.array(arr)
            # Handle lists
            elif isinstance(arr, list):
                return np.array(arr)
            # Try generic conversion for other types
            else:
                return np.array(arr)
        except Exception as e:
            raise MLXAcceleratorError(f"Failed to convert to NumPy array: {e}")
    
    def cosine_similarity(self, a: Union[np.ndarray, list], b: Union[np.ndarray, list]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector (NumPy array or list)
            b: Second vector (NumPy array or list)
            
        Returns:
            float: Cosine similarity (-1 to 1)
            
        Raises:
            MLXAcceleratorError: If calculation fails
            ValueError: If inputs have incompatible shapes
        """
        # Convert inputs to numpy arrays if they aren't already
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=np.float32)
        if not isinstance(b, np.ndarray):
            b = np.array(b, dtype=np.float32)
            
        # Check dimensions
        if a.shape != b.shape:
            raise ValueError(f"Vectors must have the same shape: {a.shape} vs {b.shape}")
            
        if not self.is_available():
            # Fall back to NumPy
            try:
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0  # Handle zero vectors
                return float(dot_product / (norm_a * norm_b))
            except Exception as e:
                raise MLXAcceleratorError(f"NumPy cosine similarity failed: {e}")
        
        try:
            # Use MLX for computation
            a_mx = self.array(a)
            b_mx = self.array(b)
            
            dot_product = self.mx.sum(a_mx * b_mx)
            norm_a = self.mx.sqrt(self.mx.sum(a_mx * a_mx))
            norm_b = self.mx.sqrt(self.mx.sum(b_mx * b_mx))
            
            # Handle division by zero
            if self.to_numpy(norm_a) == 0 or self.to_numpy(norm_b) == 0:
                return 0.0
                
            similarity = dot_product / (norm_a * norm_b)
            return float(self.to_numpy(similarity))
        except Exception as e:
            # Fall back to NumPy
            logger.warning(f"MLX cosine similarity failed, falling back to NumPy: {e}")
            try:
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0  # Handle zero vectors
                return float(dot_product / (norm_a * norm_b))
            except Exception as e2:
                raise MLXAcceleratorError(f"Cosine similarity calculation failed: {e2} (after MLX failure: {e})")
                
    def vector_norm(self, a: Union[np.ndarray, list]) -> float:
        """Calculate L2 norm of a vector.
        
        Args:
            a: Vector (NumPy array or list)
            
        Returns:
            float: L2 norm
            
        Raises:
            MLXAcceleratorError: If calculation fails
        """
        # Convert input to numpy array if it isn't already
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=np.float32)
            
        if not self.is_available():
            # Fall back to NumPy
            try:
                return float(np.linalg.norm(a))
            except Exception as e:
                raise MLXAcceleratorError(f"NumPy vector norm failed: {e}")
        
        try:
            # Use MLX for computation
            a_mx = self.array(a)
            norm = self.mx.sqrt(self.mx.sum(a_mx * a_mx))
            return float(self.to_numpy(norm))
        except Exception as e:
            # Fall back to NumPy
            logger.warning(f"MLX vector norm failed, falling back to NumPy: {e}")
            try:
                return float(np.linalg.norm(a))
            except Exception as e2:
                raise MLXAcceleratorError(f"Vector norm calculation failed: {e2} (after MLX failure: {e})")