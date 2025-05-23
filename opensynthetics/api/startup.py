"""Startup utilities for OpenSynthetics API server."""

import os
from pathlib import Path
from loguru import logger

from opensynthetics.core.security import create_default_api_key, get_security_manager


def initialize_api_server():
    """Initialize the API server with default configuration."""
    logger.info("Initializing OpenSynthetics API server...")
    
    # Ensure security manager is initialized
    security_manager = get_security_manager()
    
    # Create default API key if none exist
    try:
        api_key = create_default_api_key()
        logger.info(f"API server initialized with API key: {security_manager.mask_api_key(api_key)}")
        
        # Store the API key in environment variable for easy access
        os.environ['OPENSYNTHETICS_DEFAULT_API_KEY'] = api_key
        
    except Exception as e:
        logger.error(f"Failed to initialize default API key: {e}")
    
    # Clean up expired keys
    try:
        security_manager.cleanup_expired_keys()
    except Exception as e:
        logger.warning(f"Failed to cleanup expired keys: {e}")
    
    logger.info("API server initialization complete")


def create_demo_data():
    """Create demo API keys and configurations for testing."""
    security_manager = get_security_manager()
    
    try:
        # Create demo API keys for different services
        demo_keys = [
            {
                "name": "Google Drive Integration",
                "service": "google",
                "scope": ["read", "write"],
                "description": "Demo key for Google Drive integration"
            },
            {
                "name": "Postman API Testing",
                "service": "postman",
                "scope": ["read"],
                "description": "Demo key for Postman API testing"
            },
            {
                "name": "AWS S3 Storage",
                "service": "aws",
                "scope": ["read", "write"],
                "description": "Demo key for AWS S3 storage"
            }
        ]
        
        for key_config in demo_keys:
            # Check if key already exists
            existing_keys = [
                key for key in security_manager.list_api_keys()
                if key.name == key_config["name"] and key.status == "active"
            ]
            
            if not existing_keys:
                api_key = security_manager.create_api_key(**key_config)
                logger.info(f"Created demo API key: {key_config['name']}")
        
        # Store demo cloud credentials
        demo_credentials = {
            "google_drive": {
                "client_id": "demo-client-id.apps.googleusercontent.com",
                "api_key": "demo-api-key",
                "status": "demo"
            },
            "aws": {
                "access_key_id": "DEMO-ACCESS-KEY",
                "secret_access_key": "demo-secret-key",
                "region": "us-east-1",
                "status": "demo"
            }
        }
        
        for service, credentials in demo_credentials.items():
            existing = security_manager.get_credentials(service)
            if not existing:
                security_manager.store_credentials(service, credentials)
                logger.info(f"Created demo credentials for: {service}")
        
    except Exception as e:
        logger.error(f"Failed to create demo data: {e}")


def print_startup_info():
    """Print startup information for the API server."""
    default_key = os.environ.get("OPENSYNTHETICS_API_KEY", "os_FuxLl1noXE5z2Sw4xmKONd95iydY9G5-FRjzEuCzppk")
    masked_key = f"{default_key[:6]}{'*' * 36}{default_key[-4:]}"
    
    print("=" * 80)
    print("OpenSynthetics API Server Started Successfully!")
    print("=" * 80)
    
    # API Key Information
    print(f"Default API Key: {masked_key}")
    print(f"Full API Key: {default_key}")
    
    print("\nAvailable Endpoints:")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Web UI: http://localhost:8000/ui/")
    print("   - Health Check: http://localhost:8000/health")
    print("   - API Base: http://localhost:8000/api/v1/")
    
    print("\nIntegration Features:")
    print("   - Google Drive: Full OAuth2 integration with file management")
    print("   - Postman API: Collection generation and API testing")
    print("   - API Keys: Comprehensive key management with usage analytics")
    print("   - MCP Servers: Model Context Protocol server integration")
    print("   - Cloud Storage: Multi-provider storage integration")
    
    print("\nKeyboard Shortcuts (in Web UI):")
    print("   - Ctrl+U: Open file upload modal")
    print("   - Ctrl+Shift+G: Navigate to Google Drive")
    print("   - Ctrl+Shift+P: Navigate to Postman integration")
    print("   - Ctrl+Shift+K: Navigate to API key management")
    print("   - Ctrl+N: Create new workspace")
    print("   - Ctrl+G: Navigate to data generation")
    
    print("\nSecurity Features:")
    print("   - Rate limiting and usage tracking")
    print("   - API key expiration and scopes")
    print("   - Secure credential storage")
    print("   - Request logging and analytics")
    
    print("\nQuick Start:")
    print("   1. Visit the Web UI at http://localhost:8000/ui/")
    print("   2. Create your first workspace")
    print("   3. Generate synthetic data")
    print("   4. Connect cloud storage integrations")
    print("   5. Export and manage your datasets")
    
    print("=" * 80)
    print("Happy data generation!")
    print("=" * 80) 