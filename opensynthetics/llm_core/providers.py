"""LLM provider integrations for OpenSynthetics."""

import os
import time
from typing import Any, Dict, List, Optional, Union, Callable

import openai
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from loguru import logger
from pydantic import BaseModel, Field, root_validator

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import LLMError, APIRateLimitError, APIAuthError
from opensynthetics.core.monitoring import monitor


class LLMResponse(BaseModel):
    """Response from an LLM."""

    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics",
    )
    finish_reason: Optional[str] = Field(None, description="Reason generation finished")
    raw_response: Optional[Any] = Field(None, description="Raw response object")


class TokenUsageTracker:
    """Track token usage across providers."""
    
    def __init__(self) -> None:
        """Initialize token usage tracker."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.calls_by_model: Dict[str, int] = {}
        self.tokens_by_model: Dict[str, Dict[str, int]] = {}
        self.cost_by_model: Dict[str, float] = {}
        self.total_cost: float = 0.0
        
        # Default pricing per 1000 tokens (can be updated)
        self.pricing: Dict[str, Dict[str, float]] = {
            "gpt-4o": {"prompt": 5.0, "completion": 15.0},
            "gpt-4": {"prompt": 10.0, "completion": 30.0},
            "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
            "default": {"prompt": 1.0, "completion": 2.0},
        }
    
    def add_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Add token usage.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        total_tokens = prompt_tokens + completion_tokens
        
        # Update global counters
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        
        # Update model-specific counters
        if model not in self.calls_by_model:
            self.calls_by_model[model] = 0
            self.tokens_by_model[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            self.cost_by_model[model] = 0.0
            
        self.calls_by_model[model] += 1
        self.tokens_by_model[model]["prompt_tokens"] += prompt_tokens
        self.tokens_by_model[model]["completion_tokens"] += completion_tokens
        self.tokens_by_model[model]["total_tokens"] += total_tokens
        
        # Calculate cost
        model_pricing = self.pricing.get(model, self.pricing["default"])
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        self.cost_by_model[model] += total_cost
        self.total_cost += total_cost
        
        # Send metrics to monitoring
        monitor.count("llm_tokens_used", value=total_tokens, tags={"model": model})
        monitor.gauge("llm_cost", value=total_cost, tags={"model": model})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary.
        
        Returns:
            Dict[str, Any]: Usage summary
        """
        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "calls_by_model": self.calls_by_model,
            "tokens_by_model": self.tokens_by_model,
            "cost_by_model": self.cost_by_model,
            "total_cost": self.total_cost,
        }
    
    def update_pricing(self, model: str, prompt_price: float, completion_price: float) -> None:
        """Update pricing for a model.
        
        Args:
            model: Model name
            prompt_price: Price per 1000 prompt tokens
            completion_price: Price per 1000 completion tokens
        """
        self.pricing[model] = {
            "prompt": prompt_price,
            "completion": completion_price,
        }


# Global token usage tracker
token_tracker = TokenUsageTracker()


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize LLM provider.

        Args:
            config: Configuration, if None uses default
        """
        self.config = config or Config.load()
        self.token_tracker = token_tracker
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Load configuration if available
        if hasattr(self.config, 'settings') and 'llm' in self.config.settings:
            llm_config = self.config.settings.get('llm', {})
            self.max_retries = llm_config.get('max_retries', 3)
            self.retry_delay = llm_config.get('retry_delay', 1.0)

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            model: Model to use, defaults to config
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            LLMResponse: Generated response
            
        Raises:
            LLMError: If generation fails
        """
        raise NotImplementedError("Subclasses must implement generate()")
        
    def with_retries(
        self, 
        func: Callable, 
        *args, 
        retries: Optional[int] = None, 
        retry_delay: Optional[float] = None, 
        **kwargs
    ) -> Any:
        """Execute a function with retries.
        
        Args:
            func: Function to execute
            *args: Function arguments
            retries: Number of retries (default from config)
            retry_delay: Delay between retries in seconds (default from config)
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            LLMError: If all retries fail
        """
        max_retries = retries if retries is not None else self.max_retries
        delay = retry_delay if retry_delay is not None else self.retry_delay
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (openai.RateLimitError, APIRateLimitError) as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise APIRateLimitError(f"Rate limit exceeded after {max_retries} retries: {e}")
            except (openai.APIError, LLMError) as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API error, retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise LLMError(f"API error after {max_retries} retries: {e}")
            except Exception as e:
                # Don't retry unexpected errors
                raise LLMError(f"Unexpected error: {e}")
        
        # If we got here, all retries failed
        raise LLMError(f"All retries failed: {last_error}")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize OpenAI provider.

        Args:
            config: Configuration, if None uses default
            
        Raises:
            LLMError: If API key is not configured
        """
        super().__init__(config)
        
        try:
            if not self.config.api_keys.get("openai"):
                raise APIAuthError("OpenAI API key not configured. Use 'opensynthetics config set api_keys.openai YOUR_KEY'")
                
            self.client = openai.OpenAI(api_key=self.config.get_api_key("openai"))
            # Test connection with a minimal API call
            self._test_connection()
        except openai.AuthenticationError as e:
            raise APIAuthError(f"OpenAI authentication failed: {e}")
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI provider: {e}")
            
    def _test_connection(self) -> None:
        """Test API connection with a minimal call.
        
        Raises:
            APIAuthError: If authentication fails
            LLMError: If connection test fails
        """
        try:
            # Use models.list for a lightweight API call
            self.client.models.list(limit=1)
        except openai.AuthenticationError as e:
            raise APIAuthError(f"OpenAI authentication failed: {e}")
        except Exception as e:
            logger.warning(f"OpenAI connection test failed: {e}")
            # Don't raise here, just log the warning

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Generate text using OpenAI API.

        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            LLMResponse: Generated response
            
        Raises:
            LLMError: If API call fails
        """
        # Get default values from config
        default_model = "gpt-3.5-turbo"
        default_temperature = 0.7
        default_max_tokens = 1000
        fallback_models = ["gpt-3.5-turbo"]
        
        # Check if config has llm settings
        if hasattr(self.config, 'settings') and 'llm' in self.config.settings:
            llm_config = self.config.settings.get('llm', {})
            default_model = llm_config.get('default_model', "gpt-3.5-turbo")
            default_temperature = llm_config.get('temperature', 0.7)
            default_max_tokens = llm_config.get('max_tokens', 1000)
            fallback_models = llm_config.get('fallback_models', ["gpt-3.5-turbo"])
        
        # Use provided values or defaults
        model = model or default_model
        temperature = temperature if temperature is not None else default_temperature
        max_tokens = max_tokens or default_max_tokens

        with monitor.timer("openai_api_call", tags={"model": model}):
            try:
                def api_call():
                    return self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                
                # Call with retries
                response = self.with_retries(api_call)
                
                # Track token usage
                self.token_tracker.add_usage(
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

                return LLMResponse(
                    text=response.choices[0].message.content,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    finish_reason=response.choices[0].finish_reason,
                    raw_response=response,
                )
            except openai.RateLimitError as e:
                logger.error(f"OpenAI rate limit exceeded: {e}")
                # Try fallback models if configured
                if model != fallback_models[0] and fallback_models:
                    fallback = fallback_models[0]
                    logger.info(f"Falling back to {fallback}")
                    return self.generate(
                        prompt=prompt,
                        model=fallback,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                raise APIRateLimitError(f"Rate limit exceeded: {e}")
            except openai.AuthenticationError as e:
                logger.error(f"OpenAI authentication error: {e}")
                raise APIAuthError(f"Authentication error: {e}")
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                # Try fallback models if configured
                if model != fallback_models[0] and fallback_models:
                    fallback = fallback_models[0]
                    logger.info(f"Falling back to {fallback}")
                    return self.generate(
                        prompt=prompt,
                        model=fallback,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                raise LLMError(f"API error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during OpenAI API call: {e}")
                raise LLMError(f"Unexpected error: {e}")
                
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics.
        
        Returns:
            Dict[str, Any]: Token usage statistics
        """
        return self.token_tracker.get_summary()


class ProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers: Dict[str, type] = {
        "openai": OpenAIProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a provider.
        
        Args:
            name: Provider name
            provider_class: Provider class
        """
        cls._providers[name] = provider_class
        
    @classmethod
    def get_provider(cls, name: str, config: Optional[Config] = None) -> LLMProvider:
        """Get a provider instance.
        
        Args:
            name: Provider name
            config: Configuration, if None uses default
            
        Returns:
            LLMProvider: Provider instance
            
        Raises:
            LLMError: If provider not found
        """
        if name not in cls._providers:
            raise LLMError(f"Provider not found: {name}")
            
        provider_class = cls._providers[name]
        return provider_class(config)