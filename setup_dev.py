#!/usr/bin/env python3
"""
OpenSynthetics Development Setup Script
This script sets up the development environment with sample data and configurations.
"""

import os
import sys
import json
from pathlib import Path

# Add the opensynthetics package to the Python path
sys.path.insert(0, str(Path(__file__).parent / "opensynthetics"))

from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace
from opensynthetics.core.security import get_security_manager, create_default_api_key

def create_sample_workspace():
    """Create a sample workspace with demo data."""
    print("[INFO] Creating sample workspace...")
    
    config = Config.load()
    workspace_path = config.base_dir / "demo-workspace"
    
    if workspace_path.exists():
        print(f"   [OK] Sample workspace already exists at: {workspace_path}")
        return workspace_path
    
    # Create workspace
    workspace = Workspace.create(
        name="demo-workspace",
        path=workspace_path,
        description="Demo workspace with sample synthetic data"
    )
    
    # Create sample dataset
    dataset = workspace.create_dataset(
        name="customer-data",
        description="Sample customer data for testing"
    )
    
    # Add some sample data
    sample_data = [
        {
            "customer_id": "C001",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 30,
            "city": "New York",
            "purchase_amount": 150.75
        },
        {
            "customer_id": "C002", 
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "age": 25,
            "city": "Los Angeles",
            "purchase_amount": 89.50
        },
        {
            "customer_id": "C003",
            "name": "Bob Johnson",
            "email": "bob.johnson@example.com", 
            "age": 35,
            "city": "Chicago",
            "purchase_amount": 220.00
        }
    ]
    
    dataset.add_data(sample_data, "customers")
    workspace.close()
    
    print(f"   [OK] Sample workspace created at: {workspace_path}")
    return workspace_path

def setup_api_keys():
    """Set up default API keys for development."""
    print("[INFO] Setting up API keys...")
    
    security_manager = get_security_manager()
    
    # Create default API key if none exists
    api_keys = security_manager.list_api_keys()
    if not api_keys:
        default_key = create_default_api_key()
        print(f"   [OK] Default API key created: {default_key}")
    else:
        print(f"   [OK] Found {len(api_keys)} existing API keys")
    
    # Create sample service API keys
    sample_keys = [
        {
            "name": "OpenAI Integration",
            "service": "openai",
            "description": "API key for OpenAI GPT integration"
        },
        {
            "name": "Google Cloud",
            "service": "google_cloud",
            "description": "API key for Google Cloud services"
        },
        {
            "name": "External API",
            "service": "external_api",
            "description": "Sample external API key"
        }
    ]
    
    for key_config in sample_keys:
        existing = [k for k in api_keys if k.name == key_config["name"]]
        if not existing:
            new_key = security_manager.create_api_key(**key_config)
            print(f"   [OK] Created sample API key: {key_config['name']}")

def create_config_files():
    """Create sample configuration files."""
    print("[INFO] Creating configuration files...")
    
    config_dir = Path.home() / ".opensynthetics"
    config_dir.mkdir(exist_ok=True)
    
    # Create sample Google Drive config
    google_config = {
        "configured": False,
        "instructions": {
            "step1": "Go to Google Cloud Console (console.cloud.google.com)",
            "step2": "Create a new project or select existing one",
            "step3": "Enable Google Drive API",
            "step4": "Create OAuth2 credentials",
            "step5": "Add authorized redirect URIs",
            "step6": "Copy client ID and client secret to the web interface"
        }
    }
    
    google_config_file = config_dir / "google_drive_setup.json"
    if not google_config_file.exists():
        with open(google_config_file, 'w') as f:
            json.dump(google_config, f, indent=2)
        print(f"   [OK] Google Drive setup guide created: {google_config_file}")
    
    # Create sample MCP server configurations
    mcp_config = {
        "servers": [
            {
                "name": "file-manager",
                "endpoint": "http://localhost:3001",
                "description": "Local file management server",
                "capabilities": ["read", "write", "list", "sync"]
            },
            {
                "name": "data-processor", 
                "endpoint": "http://localhost:3002",
                "description": "Data processing and transformation server",
                "capabilities": ["transform", "validate", "analyze", "export"]
            }
        ]
    }
    
    mcp_config_file = config_dir / "mcp_servers.json"
    if not mcp_config_file.exists():
        with open(mcp_config_file, 'w') as f:
            json.dump(mcp_config, f, indent=2)
        print(f"   [OK] MCP server configuration created: {mcp_config_file}")

def setup_development():
    """Set up OpenSynthetics for development."""
    print("[INFO] OpenSynthetics Development Setup")
    print("=" * 50)
    
    # 1. Initialize configuration
    config = Config.load()
    print(f"[INFO] Configuration loaded from: {Config.get_config_path()}")
    
    # 2. Create sample workspace
    workspace_path = create_sample_workspace()
    
    # 3. Set up API keys
    setup_api_keys()
    
    # 4. Create config files
    create_config_files()
    
    # 5. Print summary
    print("\n[OK] Setup complete!")
    print("=" * 50)
    print(f"[INFO] Config directory: {Path.home() / '.opensynthetics'}")
    print("\n[INFO] Next steps:")
    print("   1. Start the API server: python start_server.py")
    print("   2. Visit http://localhost:8000/docs for API documentation")
    print("   3. Use the web interface at http://localhost:8000/ui")
    print("   4. Configure Google Drive integration if needed")
    print("   5. Create your first synthetic dataset!")

if __name__ == "__main__":
    try:
        setup_development()
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")
        sys.exit(1) 