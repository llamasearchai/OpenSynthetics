"""Generator agent for text generation."""

from typing import Any, Dict, List, Optional, Union

from loguru import logger

from opensynthetics.core.config import Config
from opensynthetics.core.monitoring import monitor
from opensynthetics.llm_core.agents.base import Agent, Tool
from opensynthetics.llm_core.providers import LLMProvider, ProviderFactory


class GeneratorAgent(Agent):
    """Agent for generating text content."""
    
    def __init__(
        self,
        name: str = "generator",
        description: str = "Generates text content",
        config: Optional[Config] = None,
        provider: Optional[Union[str, LLMProvider]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> None:
        """Initialize generator agent.
        
        Args:
            name: Agent name
            description: Agent description
            config: Configuration, if None uses default
            provider: LLM provider or provider name
            model: Model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
        """
        super().__init__(name, description, config)
        
        # Get default provider from config or use openai
        default_provider = "openai"
        default_model = "gpt-3.5-turbo"
        
        # Check if config has llm settings
        if hasattr(self.config, 'settings') and 'llm' in self.config.settings:
            llm_config = self.config.settings.get('llm', {})
            default_provider = llm_config.get('default_provider', "openai")
            default_model = llm_config.get('default_model', "gpt-3.5-turbo")
        
        # Set up provider
        if provider is None:
            self.provider = ProviderFactory.get_provider(default_provider, self.config)
        elif isinstance(provider, str):
            self.provider = ProviderFactory.get_provider(provider, self.config)
        else:
            self.provider = provider
            
        # Set generation parameters
        self.model = model or default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Add default tools
        self._add_default_tools()
        
    def _add_default_tools(self) -> None:
        """Add default tools."""
        self.add_tool(
            Tool(
                name="search",
                description="Search for information",
                function=lambda query: f"Search results for '{query}'",
            )
        )
        
        self.add_tool(
            Tool(
                name="calculate",
                description="Perform a calculation",
                function=lambda expression: eval(expression),
            )
        )
        
    def process(self, input_text: str) -> str:
        """Process an input text.
        
        Args:
            input_text: Input text
            
        Returns:
            str: Generated response
        """
        with monitor.timer("generator_process", tags={"agent": self.name}):
            try:
                # Generate response using LLM
                response = self.provider.generate(
                    prompt=input_text,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                return response.text
            except Exception as e:
                logger.error(f"Error in generator agent: {e}")
                return f"Error generating response: {e}"
                
    def generate(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate content.
        
        Args:
            prompt: Generation prompt
            system_prompt: System prompt
            model: Model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated content
        """
        with monitor.timer("generator_generate", tags={"agent": self.name}):
            try:
                # Use provided parameters or fall back to instance defaults
                effective_model = model or self.model
                effective_temperature = temperature if temperature is not None else self.temperature
                effective_max_tokens = max_tokens or self.max_tokens
                
                # Build final prompt
                final_prompt = ""
                if system_prompt:
                    final_prompt += f"{system_prompt}\n\n"
                if prompt:
                    final_prompt += prompt
                else:
                    final_prompt += "Please generate content."
                    
                # Generate response
                response = self.provider.generate(
                    prompt=final_prompt,
                    model=effective_model,
                    temperature=effective_temperature,
                    max_tokens=effective_max_tokens,
                )
                
                return response.text
            except Exception as e:
                logger.error(f"Error in generator agent: {e}")
                return f"Error generating content: {e}" 