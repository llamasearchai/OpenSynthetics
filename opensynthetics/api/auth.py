"""Authentication utilities for the API."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import AuthenticationError

# API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# User model
class User(BaseModel):
    """User model."""
    
    username: str
    is_admin: bool = False


async def get_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Get API key from header.
    
    Args:
        api_key: API key from header
        
    Returns:
        str: Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        # Default development API key
        config = Config.load()
        if config.environment == "development":
            return "default-key"
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Validate API key (in a real app, you would check against a database)
    if api_key not in ["default-key", "test-key", "admin-key"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


async def get_current_user(api_key: str = Depends(get_api_key)) -> User:
    """Get current user from API key.
    
    Args:
        api_key: Validated API key
        
    Returns:
        User: Current user
        
    Raises:
        HTTPException: If user is not found
    """
    # In a real app, you would look up the user in a database
    # Here we just return a mock user based on the API key
    
    if api_key == "admin-key":
        return User(username="admin", is_admin=True)
    
    if api_key in ["default-key", "test-key"]:
        return User(username="user")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "ApiKey"},
    )