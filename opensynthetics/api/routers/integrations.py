"""API router for integrations."""

import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from opensynthetics.core.security import get_security_manager, APIKey
from opensynthetics.api.auth import get_api_key
from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace
from loguru import logger

router = APIRouter()

# Pydantic models
class GoogleDriveConfig(BaseModel):
    """Google Drive configuration."""
    client_id: str = Field(..., description="Google OAuth2 client ID")
    client_secret: str = Field(..., description="Google OAuth2 client secret")
    redirect_uri: str = Field(..., description="OAuth2 redirect URI")
    scope: List[str] = Field(default=["https://www.googleapis.com/auth/drive"], description="OAuth2 scopes")

class CloudProviderStatus(BaseModel):
    """Cloud provider connection status."""
    provider: str = Field(..., description="Provider name")
    status: str = Field(..., description="Connection status")
    configured: bool = Field(..., description="Whether provider is configured")
    last_check: Optional[datetime] = Field(None, description="Last status check")

class APIKeyRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., description="Key name")
    service: str = Field(..., description="Service name")
    key_value: Optional[str] = Field(None, description="API key value (if existing)")
    expiration_days: Optional[int] = Field(365, description="Expiration in days")
    rate_limit: Optional[int] = Field(1000, description="Rate limit per hour")
    scope: Optional[List[str]] = Field(["read"], description="Key permissions")
    description: Optional[str] = Field(None, description="Key description")

class APIKeyResponse(BaseModel):
    """API key response."""
    id: str
    name: str
    service: str
    masked_key: str
    status: str
    created_at: datetime
    expires_at: Optional[datetime]
    rate_limit: Optional[int]
    scope: List[str]
    description: Optional[str]
    usage_24h: int
    usage_month: int

class PostmanCollection(BaseModel):
    """Postman collection structure."""
    info: Dict[str, Any]
    item: List[Dict[str, Any]]
    variable: List[Dict[str, Any]]

class MCPServerConfig(BaseModel):
    """MCP server configuration."""
    name: str = Field(..., description="Server name")
    endpoint: str = Field(..., description="Server endpoint URL")
    description: Optional[str] = Field(None, description="Server description")
    auth_required: bool = Field(False, description="Whether authentication is required")
    capabilities: List[str] = Field([], description="Server capabilities")

# Google Drive endpoints
@router.get("/integrations/google-drive/config", tags=["Integrations"])
async def get_google_drive_config(api_key: str = Depends(get_api_key)):
    """Get Google Drive configuration."""
    security_manager = get_security_manager()
    credentials = security_manager.get_credentials("google_drive")
    
    if not credentials:
        return JSONResponse(
            status_code=200,
            content={
                "configured": False,
                "message": "Google Drive not configured"
            }
        )
    
    return {
        "configured": True,
        "client_id": credentials.get("client_id", ""),
        "redirect_uri": credentials.get("redirect_uri", ""),
        "scope": credentials.get("scope", ["https://www.googleapis.com/auth/drive"])
    }

@router.post("/integrations/google-drive/config", tags=["Integrations"])
async def set_google_drive_config(
    config: GoogleDriveConfig,
    api_key: str = Depends(get_api_key)
):
    """Set Google Drive configuration."""
    security_manager = get_security_manager()
    
    credentials = {
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "redirect_uri": config.redirect_uri,
        "scope": config.scope,
        "configured_at": datetime.utcnow().isoformat()
    }
    
    security_manager.store_credentials("google_drive", credentials)
    
    return {
        "success": True,
        "message": "Google Drive configuration saved successfully"
    }

# Cloud provider endpoints
@router.get("/integrations/cloud-providers/status", tags=["Integrations"])
async def get_cloud_providers_status(api_key: str = Depends(get_api_key)):
    """Get status of all cloud providers."""
    security_manager = get_security_manager()
    
    providers = [
        {"name": "google_drive", "display_name": "Google Drive"},
        {"name": "aws_s3", "display_name": "AWS S3"},
        {"name": "azure_blob", "display_name": "Azure Blob Storage"},
        {"name": "gcp_storage", "display_name": "Google Cloud Storage"},
        {"name": "dropbox", "display_name": "Dropbox"},
        {"name": "onedrive", "display_name": "OneDrive"}
    ]
    
    status_list = []
    for provider in providers:
        credentials = security_manager.get_credentials(provider["name"])
        configured = credentials is not None
        
        status_list.append(CloudProviderStatus(
            provider=provider["display_name"],
            status="connected" if configured else "disconnected",
            configured=configured,
            last_check=datetime.utcnow()
        ))
    
    return {"providers": status_list}

# API Key management endpoints
@router.get("/integrations/api-keys", tags=["API Keys"])
async def list_api_keys(api_key: str = Depends(get_api_key)):
    """List all API keys."""
    security_manager = get_security_manager()
    api_keys = security_manager.list_api_keys()
    
    response_keys = []
    for key in api_keys:
        response_keys.append(APIKeyResponse(
            id=key.id,
            name=key.name,
            service=key.service,
            masked_key=security_manager.mask_api_key(key.key_value),
            status=key.status,
            created_at=key.created_at,
            expires_at=key.expires_at,
            rate_limit=key.rate_limit,
            scope=key.scope or [],
            description=key.description,
            usage_24h=key.usage_24h,
            usage_month=key.usage_month
        ))
    
    return {"api_keys": response_keys}

@router.post("/integrations/api-keys", tags=["API Keys"])
async def create_api_key(
    request: APIKeyRequest,
    api_key: str = Depends(get_api_key)
):
    """Create a new API key."""
    security_manager = get_security_manager()
    
    try:
        new_key = security_manager.create_api_key(
            name=request.name,
            service=request.service,
            key_value=request.key_value,
            expiration_days=request.expiration_days,
            rate_limit=request.rate_limit,
            scope=request.scope,
            description=request.description
        )
        
        return {
            "success": True,
            "api_key": APIKeyResponse(
                id=new_key.id,
                name=new_key.name,
                service=new_key.service,
                masked_key=security_manager.mask_api_key(new_key.key_value),
                status=new_key.status,
                created_at=new_key.created_at,
                expires_at=new_key.expires_at,
                rate_limit=new_key.rate_limit,
                scope=new_key.scope or [],
                description=new_key.description,
                usage_24h=new_key.usage_24h,
                usage_month=new_key.usage_month
            ),
            "revealed_key": new_key.key_value if not request.key_value else None
        }
    
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}"
        )

@router.delete("/integrations/api-keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(key_id: str, api_key: str = Depends(get_api_key)):
    """Revoke an API key."""
    security_manager = get_security_manager()
    
    success = security_manager.revoke_api_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    return {"success": True, "message": "API key revoked successfully"}

@router.post("/integrations/api-keys/{key_id}/regenerate", tags=["API Keys"])
async def regenerate_api_key(key_id: str, api_key: str = Depends(get_api_key)):
    """Regenerate an API key."""
    security_manager = get_security_manager()
    
    existing_key = security_manager.get_api_key(key_id)
    if not existing_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    # Generate new key value
    new_key_value = security_manager.generate_api_key()
    existing_key.key_value = new_key_value
    existing_key.created_at = datetime.utcnow()
    existing_key.usage_24h = 0
    existing_key.usage_month = 0
    existing_key.failed_requests_24h = 0
    
    security_manager.save_api_keys()
    
    return {
        "success": True,
        "new_key_value": new_key_value,
        "message": "API key regenerated successfully"
    }

# Postman integration endpoints
@router.get("/integrations/postman/collection", tags=["Postman"])
async def generate_postman_collection(
    include_auth: bool = True,
    base_url: str = "http://localhost:8000",
    api_key: str = Depends(get_api_key)
):
    """Generate Postman collection for OpenSynthetics API."""
    
    collection = create_postman_collection(base_url, include_auth)
    
    return collection

def create_postman_collection(base_url: str, include_auth: bool = True) -> Dict[str, Any]:
    """Create a Postman collection for the OpenSynthetics API."""
    
    collection = {
        "info": {
            "name": "OpenSynthetics API",
            "description": "Complete API collection for OpenSynthetics platform",
            "version": "1.0.0",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "variable": [
            {
                "key": "base_url",
                "value": base_url,
                "type": "string"
            },
            {
                "key": "api_key",
                "value": "your-api-key-here",
                "type": "string"
            }
        ],
        "auth": {
            "type": "apikey",
            "apikey": [
                {
                    "key": "key",
                    "value": "X-API-Key",
                    "type": "string"
                },
                {
                    "key": "value",
                    "value": "{{api_key}}",
                    "type": "string"
                }
            ]
        } if include_auth else None,
        "item": []
    }
    
    # Health and Info endpoints
    health_folder = {
        "name": "Health & Info",
        "item": [
            {
                "name": "Health Check",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/health",
                        "host": ["{{base_url}}"],
                        "path": ["health"]
                    }
                }
            }
        ]
    }
    
    # Workspace endpoints
    workspace_folder = {
        "name": "Workspaces",
        "item": [
            {
                "name": "List Workspaces",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/workspaces",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "workspaces"]
                    }
                }
            },
            {
                "name": "Get Workspace",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/workspaces/{{workspace_path}}",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "workspaces", "{{workspace_path}}"]
                    }
                }
            },
            {
                "name": "Create Workspace",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": "{\n  \"name\": \"test-workspace\",\n  \"description\": \"Test workspace\"\n}"
                    },
                    "url": {
                        "raw": "{{base_url}}/api/v1/workspaces",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "workspaces"]
                    }
                }
            }
        ]
    }
    
    # Dataset endpoints
    dataset_folder = {
        "name": "Datasets",
        "item": [
            {
                "name": "List Datasets",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/workspaces/{{workspace_path}}/datasets",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "workspaces", "{{workspace_path}}", "datasets"]
                    }
                }
            },
            {
                "name": "Get Dataset",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/workspaces/{{workspace_path}}/datasets/{{dataset_name}}",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "workspaces", "{{workspace_path}}", "datasets", "{{dataset_name}}"]
                    }
                }
            }
        ]
    }
    
    # Generation endpoints
    generation_folder = {
        "name": "Data Generation",
        "item": [
            {
                "name": "Create Generation Job",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": "{\n  \"strategy\": \"random\",\n  \"parameters\": {\n    \"count\": 100\n  },\n  \"workspace_path\": \"test-workspace\",\n  \"output_dataset\": \"generated-data\"\n}"
                    },
                    "url": {
                        "raw": "{{base_url}}/api/v1/generate/jobs",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "generate", "jobs"]
                    }
                }
            },
            {
                "name": "List Strategies",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/strategies",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "strategies"]
                    }
                }
            }
        ]
    }
    
    # Integration endpoints
    integration_folder = {
        "name": "Integrations",
        "item": [
            {
                "name": "Google Drive Config",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/integrations/google-drive/config",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "integrations", "google-drive", "config"]
                    }
                }
            },
            {
                "name": "List API Keys",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/integrations/api-keys",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "integrations", "api-keys"]
                    }
                }
            },
            {
                "name": "Create API Key",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": "{\n  \"name\": \"test-key\",\n  \"service\": \"external-api\",\n  \"description\": \"Test API key\"\n}"
                    },
                    "url": {
                        "raw": "{{base_url}}/api/v1/integrations/api-keys",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "integrations", "api-keys"]
                    }
                }
            }
        ]
    }
    
    collection["item"] = [
        health_folder,
        workspace_folder,
        dataset_folder,
        generation_folder,
        integration_folder
    ]
    
    return collection

@router.get("/integrations/postman/collection/download", tags=["Postman"])
async def download_postman_collection(
    include_auth: bool = True,
    base_url: str = "http://localhost:8000",
    api_key: str = Depends(get_api_key)
):
    """Download Postman collection as JSON file."""
    
    collection = create_postman_collection(base_url, include_auth)
    
    # Create temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(collection, f, indent=2)
        temp_path = f.name
    
    return FileResponse(
        path=temp_path,
        filename="opensynthetics-api-collection.json",
        media_type="application/json"
    )

# MCP Server endpoints
@router.get("/integrations/mcp/servers", tags=["MCP"])
async def list_mcp_servers(api_key: str = Depends(get_api_key)):
    """List available MCP servers."""
    # For now, return a static list of example servers
    servers = [
        {
            "name": "file-manager",
            "endpoint": "http://localhost:3001",
            "description": "File management server",
            "capabilities": ["read", "write", "list"],
            "status": "available"
        },
        {
            "name": "data-processor",
            "endpoint": "http://localhost:3002", 
            "description": "Data processing server",
            "capabilities": ["transform", "validate", "analyze"],
            "status": "available"
        },
        {
            "name": "ml-trainer",
            "endpoint": "http://localhost:3003",
            "description": "Machine learning training server",
            "capabilities": ["train", "evaluate", "predict"],
            "status": "available"
        }
    ]
    
    return {"servers": servers}

@router.post("/integrations/mcp/servers/discover", tags=["MCP"])
async def discover_mcp_servers(api_key: str = Depends(get_api_key)):
    """Discover available MCP servers on the network."""
    # This would normally scan for available MCP servers
    # For now, return the same static list
    discovered_servers = [
        {
            "name": "local-file-server",
            "endpoint": "http://localhost:3001",
            "description": "Local file management server",
            "capabilities": ["read", "write", "list", "sync"],
            "discovered_at": datetime.utcnow().isoformat()
        }
    ]
    
    return {"discovered_servers": discovered_servers}

# File upload endpoint
@router.post("/integrations/upload", tags=["File Upload"])
async def upload_file(
    file: UploadFile = File(...),
    workspace_path: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """Upload a file to the workspace."""
    
    try:
        # Determine workspace
        if workspace_path:
            workspace = Workspace.load(workspace_path)
        else:
            # Use default workspace
            config = Config.load()
            workspaces = list(config.base_dir.iterdir())
            if workspaces:
                workspace = Workspace.load(workspaces[0])
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No workspace available"
                )
        
        # Create uploads directory if it doesn't exist
        uploads_dir = workspace.path / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "success": True,
            "filename": file.filename,
            "size": len(content),
            "path": str(file_path),
            "workspace": workspace.name
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )

# Usage analytics endpoint
@router.get("/integrations/analytics/usage", tags=["Analytics"])
async def get_usage_analytics(api_key: str = Depends(get_api_key)):
    """Get usage analytics for integrations."""
    security_manager = get_security_manager()
    stats = security_manager.get_usage_stats()
    
    return {
        "period": "24h",
        "total_requests": stats.get("total_usage_24h", 0),
        "failed_requests": stats.get("total_failed_24h", 0),
        "success_rate": stats.get("success_rate", 0.0),
        "active_keys": stats.get("active_keys", 0),
        "top_services": stats.get("top_services", [])
    } 