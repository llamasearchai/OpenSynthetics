"""Custom exceptions for OpenSynthetics."""

class OpenSyntheticsError(Exception):
    """Base exception class for all OpenSynthetics errors."""
    pass

class ConfigError(OpenSyntheticsError):
    """Exception raised for configuration errors."""
    pass

class WorkspaceError(OpenSyntheticsError):
    """Exception raised for workspace operation errors."""
    pass

class DatasetError(OpenSyntheticsError):
    """Exception raised for dataset operation errors."""
    pass

class AuthenticationError(OpenSyntheticsError):
    """Exception raised for authentication errors."""
    pass

class RateLimitError(OpenSyntheticsError):
    """Exception raised when rate limits are exceeded."""
    pass

class APIError(OpenSyntheticsError):
    """Exception raised for API operation errors."""
    pass

class APIRateLimitError(RateLimitError):
    """Exception raised when API rate limits are exceeded."""
    pass

class APIAuthError(AuthenticationError):
    """Exception raised for API authentication errors."""
    pass

class ValidationError(OpenSyntheticsError):
    """Exception raised for data validation errors."""
    pass

class LLMError(OpenSyntheticsError):
    """Exception raised for LLM provider errors."""
    pass

class GenerationError(OpenSyntheticsError):
    """Exception raised for data generation errors."""
    pass

class EmbeddingError(OpenSyntheticsError):
    """Exception raised for embedding generation or search errors."""
    pass

class TrainingError(OpenSyntheticsError):
    """Exception raised for model training errors."""
    pass

class EvaluationError(OpenSyntheticsError):
    """Exception raised for evaluation errors."""
    pass

class MLXAcceleratorError(OpenSyntheticsError):
    """Exception raised for MLX acceleration errors."""
    pass

class AgentError(OpenSyntheticsError):
    """Exception raised for agent execution errors."""
    pass

class ResourceExhaustedError(OpenSyntheticsError):
    """Exception raised when system resources are exhausted."""
    pass