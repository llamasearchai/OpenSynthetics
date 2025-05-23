#!/usr/bin/env python3
"""
OpenSynthetics Server Startup Script
Run this script to start the OpenSynthetics API server with all integrated features.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def start_server():
    """Start the OpenSynthetics API server."""
    # Print startup information
    print("Starting OpenSynthetics API Server...")
    print("Features available:")
    print("   - Synthetic Data Generation")
    print("   - Workspace Management")
    print("   - Google Drive Integration")
    print("   - API Key Management")
    print("   - Postman Collection Generation")
    print("   - MCP Server Support")
    print("   - File Upload/Download")
    print("   - Usage Analytics")
    print("")
    print("Server will be available at:")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Web Interface: http://localhost:8000/ui")
    print("   - Health Check: http://localhost:8000/health")
    print("")
    print("Quick Start:")
    print("   1. Visit http://localhost:8000/docs for API documentation")
    print("   2. Use the web interface at http://localhost:8000/ui")
    print("   3. Configure Google Drive in the Integrations section")
    print("   4. Generate Postman collections for API testing")
    print("   5. Manage API keys for external services")
    print("")
    print("Press Ctrl+C to stop the server")
    print("============================================================")
    
    # Start the server using uvicorn
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "opensynthetics.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--reload-dir", str(project_root / "opensynthetics")
        ])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()