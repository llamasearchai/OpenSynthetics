"""Middleware for OpenSynthetics API."""

import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from opensynthetics.core.exceptions import RateLimitError
from opensynthetics.core.monitoring import monitor


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware.
        
        Args:
            app: ASGI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details.
        
        Args:
            request: Request object
            call_next: Next middleware or route handler
            
        Returns:
            Response: Response from next middleware or route handler
        """
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Call next middleware or route handler
        try:
            response = await call_next(request)
            
            # Log response
            process_time = (time.time() - start_time) * 1000
            status_code = response.status_code
            logger.info(f"Response: {status_code} {request.method} {request.url.path} - {process_time:.2f}ms")
            
            return response
        except Exception as e:
            # Log exception
            process_time = (time.time() - start_time) * 1000
            logger.error(f"Error: {request.method} {request.url.path} - {process_time:.2f}ms - {str(e)}")
            raise


class ClientCategory:
    """Category for rate limiting."""
    
    STANDARD = "standard"
    PREMIUM = "premium"
    INTERNAL = "internal"
    ADMIN = "admin"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app: ASGIApp, rate_limit: int = 100) -> None:
        """Initialize middleware.
        
        Args:
            app: ASGI application
            rate_limit: Maximum number of requests per minute
        """
        super().__init__(app)
        self.rate_limit = rate_limit
        self.requests: Dict[str, Dict[str, int]] = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting.
        
        Args:
            request: Request object
            call_next: Next middleware or route handler
            
        Returns:
            Response: Response from next middleware or route handler
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get current minute
        current_minute = int(time.time() / 60)
        
        # Initialize request counter for client
        if client_ip not in self.requests:
            self.requests[client_ip] = {}
        
        # Clear old entries
        for minute in list(self.requests[client_ip].keys()):
            if minute != current_minute:
                del self.requests[client_ip][minute]
        
        # Initialize request counter for current minute
        if current_minute not in self.requests[client_ip]:
            self.requests[client_ip][current_minute] = 0
        
        # Increment request counter
        self.requests[client_ip][current_minute] += 1
        
        # Check rate limit
        if self.requests[client_ip][current_minute] > self.rate_limit:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content={"error": "Rate limit exceeded"},
                status_code=429,
                media_type="application/json",
            )
        
        # Call next middleware or route handler
        response = await call_next(request)
        return response

    def clear_old_data(self) -> None:
        """Clear old rate limit data to prevent memory growth."""
        current_minute = int(time.time() / 60)
        
        for client_ip in list(self.requests.keys()):
            for minute in list(self.requests[client_ip].keys()):
                if minute < current_minute - 10:  # Keep only last 10 minutes
                    del self.requests[client_ip][minute]
                    
            # Remove empty clients
            if not self.requests[client_ip]:
                del self.requests[client_ip]