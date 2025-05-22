"""Base classes for agents in OpenSynthetics."""

from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel, Field

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import OpenSyntheticsError


class Tool(BaseModel):
    """Tool that can be used by agents."""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    function: Callable = Field(..., description="Tool function")
    
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Tool result
        """
        return self.function(*args, **kwargs)


class Memory:
    """Memory for agents to store information."""
    
    def __init__(self) -> None:
        """Initialize memory."""
        self.items: Dict[str, Any] = {}
        
    def add(self, key: str, value: Any) -> None:
        """Add an item to memory.
        
        Args:
            key: Item key
            value: Item value
        """
        self.items[key] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from memory.
        
        Args:
            key: Item key
            default: Default value if key not found
            
        Returns:
            Any: Item value or default
        """
        return self.items.get(key, default)
        
    def clear(self) -> None:
        """Clear memory."""
        self.items.clear()


class Agent:
    """Base class for agents."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        config: Optional[Config] = None,
    ) -> None:
        """Initialize agent.
        
        Args:
            name: Agent name
            description: Agent description
            config: Configuration, if None uses default
        """
        self.name = name
        self.description = description
        self.config = config or Config.load()
        self.memory = Memory()
        self.tools: List[Tool] = []
        
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        
    def get_tool(self, tool_name: str) -> Tool:
        """Get a tool by name.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool: Tool
            
        Raises:
            ValueError: If tool not found
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
                
        raise ValueError(f"Tool not found: {tool_name}")
        
    def process(self, input_text: str) -> str:
        """Process an input text.
        
        Args:
            input_text: Input text
            
        Returns:
            str: Response
        """
        raise NotImplementedError("Subclasses must implement process()")
        
    def generate(self, **kwargs: Any) -> Any:
        """Generate content.
        
        Args:
            **kwargs: Generation parameters
            
        Returns:
            Any: Generated content
        """
        raise NotImplementedError("Subclasses must implement generate()")
        
    def reset(self) -> None:
        """Reset agent state."""
        self.memory.clear() 