"""API server for OpenSynthetics."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, validator

from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace, WorkspaceError, DatasetError
from opensynthetics.core.exceptions import OpenSyntheticsError, AuthenticationError
from opensynthetics.api.auth import get_api_key, get_current_user
from opensynthetics.api.middleware import LoggingMiddleware, RateLimitMiddleware
from opensynthetics.api.routers import integrations_router, projects_router
from opensynthetics.api.startup import initialize_api_server, create_demo_data, print_startup_info

# API Models
class GenerationParameters(BaseModel):
    """Parameters for data generation."""
    
    strategy: str = Field(..., description="Generation strategy")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    workspace_path: str = Field(..., description="Path to workspace")
    output_dataset: str = Field(..., description="Dataset name for output")
    
    @validator("workspace_path")
    def validate_workspace_path(cls, v):
        # Expand user home directory if needed
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(f"Workspace path does not exist: {path}")
        return str(path)

class GenerationResponse(BaseModel):
    """Response for data generation."""
    
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    dataset: str = Field(..., description="Output dataset name")
    count: int = Field(..., description="Number of items generated")
    
class WorkspaceInfo(BaseModel):
    """Workspace information."""
    
    name: str = Field(..., description="Workspace name")
    path: str = Field(..., description="Workspace path")
    description: str = Field("", description="Workspace description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    datasets: List[Dict[str, Any]] = Field([], description="Datasets in workspace")

class DatasetInfo(BaseModel):
    """Dataset information."""
    
    name: str = Field(..., description="Dataset name")
    description: str = Field("", description="Dataset description")
    tags: List[str] = Field([], description="Dataset tags")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    tables: Dict[str, Any] = Field({}, description="Tables in dataset")

# Create FastAPI app
app = FastAPI(
    title="OpenSynthetics API",
    description="API for OpenSynthetics data generation platform",
    version="0.1.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, rate_limit=100)

# Include routers
app.include_router(integrations_router, prefix="/api/v1")
app.include_router(projects_router, prefix="/api/v1/projects")

# Get the directory where the web UI assets are stored
web_ui_dir = Path(__file__).parent.parent / "web_ui" / "dist"

# Mount static files if the directory exists
if web_ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(web_ui_dir), html=True), name="ui")

# Exception handler
@app.exception_handler(OpenSyntheticsError)
async def opensynthetics_exception_handler(request: Request, exc: OpenSyntheticsError):
    """Handle OpenSynthetics exceptions."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    if isinstance(exc, AuthenticationError):
        status_code = status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, (WorkspaceError, DatasetError)):
        status_code = status.HTTP_404_NOT_FOUND
    
    return JSONResponse(
        status_code=status_code,
        content={"error": str(exc), "type": type(exc).__name__},
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint with additional information."""
    from opensynthetics.core.security import get_security_manager
    
    security_manager = get_security_manager()
    stats = security_manager.get_usage_stats()
    
    return {
        "status": "ok", 
        "version": "0.1.0",
        "features": {
            "google_drive": True,
            "postman_integration": True,
            "api_key_management": True,
            "mcp_servers": True,
            "cloud_storage": True,
            "file_upload": True
        },
        "api_stats": stats
    }

# Root redirect to UI
@app.get("/", tags=["UI"])
async def root():
    """Redirect to UI."""
    return JSONResponse(
        content={
            "message": "OpenSynthetics API is running",
            "docs_url": "/docs",
            "ui_url": "/ui"
        }
    )

# API routes
@app.get("/api/v1/workspaces", tags=["Workspaces"])
async def list_workspaces(api_key: str = Depends(get_api_key)):
    """List available workspaces."""
    try:
        config = Config.load()
        base_dir = config.base_dir
        
        # Ensure base directory exists
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created base directory: {base_dir}")
            return {"workspaces": []}
        
        # Find all directories in the base directory that have a metadata.json file
        workspaces = []
        for path in base_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                try:
                    workspace = Workspace.load(path)
                    workspaces.append({
                        "name": workspace.name,
                        "path": str(workspace.path),
                        "description": workspace.metadata.description,
                        "created_at": workspace.metadata.created_at,
                        "updated_at": workspace.metadata.updated_at,
                    })
                except Exception as e:
                    logger.warning(f"Skipping invalid workspace at {path}: {e}")
                    continue
        
        return {"workspaces": workspaces}
    except Exception as e:
        logger.error(f"Error listing workspaces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workspaces: {str(e)}"
        )

@app.get("/api/v1/workspaces/{workspace_path:path}", tags=["Workspaces"])
async def get_workspace(workspace_path: str, api_key: str = Depends(get_api_key)):
    """Get workspace information."""
    try:
        # Handle both absolute and relative paths
        if not Path(workspace_path).is_absolute():
            config = Config.load()
            workspace_path = str(config.base_dir / workspace_path)
        
        workspace_path_obj = Path(workspace_path)
        if not workspace_path_obj.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_path}"
            )
        
        workspace = Workspace.load(workspace_path_obj)
        datasets = workspace.list_datasets()
        
        return {
            "name": workspace.name,
            "path": str(workspace.path),
            "description": workspace.metadata.description,
            "created_at": workspace.metadata.created_at,
            "updated_at": workspace.metadata.updated_at,
            "datasets": datasets,
        }
    except HTTPException:
        raise
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting workspace {workspace_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workspace: {str(e)}"
        )

@app.post("/api/v1/workspaces", tags=["Workspaces"])
async def create_workspace(
    request: dict,
    api_key: str = Depends(get_api_key),
):
    """Create a new workspace."""
    try:
        name = request.get("name")
        path = request.get("path")
        description = request.get("description", "")
        tags = request.get("tags", [])
        
        if not name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workspace name is required"
            )
        
        # Use default path if not provided
        if not path:
            config = Config.load()
            path = str(config.base_dir / name)
        
        workspace = Workspace.create(
            name=name,
            path=path,
            description=description,
            tags=tags,
        )
        
        return {
            "name": workspace.name,
            "path": str(workspace.path),
            "description": workspace.metadata.description,
            "created_at": workspace.metadata.created_at,
            "updated_at": workspace.metadata.updated_at,
            "message": "Workspace created successfully"
        }
        
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating workspace: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workspace: {str(e)}"
        )

@app.get("/api/v1/workspaces/{workspace_path:path}/datasets", tags=["Datasets"])
async def list_datasets(workspace_path: str, api_key: str = Depends(get_api_key)):
    """List datasets in a workspace."""
    try:
        # Handle both absolute and relative paths
        if not Path(workspace_path).is_absolute():
            config = Config.load()
            workspace_path = str(config.base_dir / workspace_path)
        
        workspace_path_obj = Path(workspace_path)
        if not workspace_path_obj.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_path}"
            )
        
        workspace = Workspace.load(workspace_path_obj)
        datasets = workspace.list_datasets()
        
        return {"datasets": datasets}
        
    except HTTPException:
        raise
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error listing datasets in workspace {workspace_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )

@app.get("/api/v1/workspaces/{workspace_path:path}/datasets/{dataset_name}", tags=["Datasets"])
async def get_dataset(workspace_path: str, dataset_name: str, api_key: str = Depends(get_api_key)):
    """Get dataset information."""
    try:
        # Handle both absolute and relative paths
        if not Path(workspace_path).is_absolute():
            config = Config.load()
            workspace_path = str(config.base_dir / workspace_path)
        
        workspace_path_obj = Path(workspace_path)
        if not workspace_path_obj.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_path}"
            )
        
        workspace = Workspace.load(workspace_path_obj)
        
        # Check if dataset exists
        if not workspace.has_dataset(dataset_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset '{dataset_name}' not found in workspace"
            )
        
        dataset_info = workspace.get_dataset_info(dataset_name)
        
        return {
            "name": dataset_name,
            "workspace": workspace.name,
            "info": dataset_info,
        }
        
    except HTTPException:
        raise
    except (WorkspaceError, DatasetError) as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_name} in workspace {workspace_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset: {str(e)}"
        )

@app.post("/api/v1/generate", tags=["Generation"])
async def generate_data(
    request: dict,
    api_key: str = Depends(get_api_key),
):
    """Generate synthetic data."""
    try:
        workspace_name = request.get("workspace")
        strategy = request.get("strategy")
        parameters = request.get("parameters", {})
        dataset_name = request.get("dataset")
        
        if not all([workspace_name, strategy, dataset_name]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="workspace, strategy, and dataset are required"
            )
        
        # Load config and find workspace
        config = Config.load()
        workspace_path = config.base_dir / workspace_name
        
        if not workspace_path.exists():
            # Create workspace if it doesn't exist
            workspace = Workspace.create(name=workspace_name)
        else:
            workspace = Workspace.load(workspace_path)
        
        # Generate mock data for demonstration
        import random
        import time
        
        # Simulate data generation based on strategy
        if strategy == "tabular_random":
            num_rows = parameters.get("num_rows", 10)
            num_columns = parameters.get("num_columns", 3)
            
            data = []
            for i in range(num_rows):
                row = {"id": i + 1}
                for j in range(num_columns - 1):
                    if j % 3 == 0:
                        row[f"col_{j+1}"] = random.randint(1, 100)
                    elif j % 3 == 1:
                        row[f"col_{j+1}"] = random.choice(["alpha", "beta", "gamma", "delta"])
                    else:
                        row[f"col_{j+1}"] = round(random.uniform(0, 1), 3)
                data.append(row)
        else:
            # Default mock data
            data = [
                {"id": 1, "name": "Sample Item 1", "value": 42},
                {"id": 2, "name": "Sample Item 2", "value": 87},
                {"id": 3, "name": "Sample Item 3", "value": 15},
            ]
        
        # Create dataset and add data
        try:
            dataset = workspace.get_dataset(dataset_name)
        except:
            dataset = workspace.create_dataset(name=dataset_name, description=f"Generated using {strategy}")
        
        dataset.add_data(data)
        
        return {
            "count": len(data),
            "strategy": strategy,
            "output_dataset": dataset_name,
            "workspace": workspace_name,
            "timestamp": int(time.time()),
            "sample_items": data[:3] if len(data) > 3 else data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data generation failed"
        )

@app.get("/api/v1/strategies", tags=["Generation"])
async def list_strategies(api_key: str = Depends(get_api_key)):
    """List available generation strategies."""
    try:
        strategies = {
            "tabular_random": {
                "name": "tabular_random",
                "description": "Generate random tabular data with configurable rows and columns",
                "schema": {
                    "type": "object",
                    "properties": {
                        "num_rows": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10000,
                            "default": 100,
                            "description": "Number of rows to generate"
                        },
                        "num_columns": {
                            "type": "integer", 
                            "minimum": 1,
                            "maximum": 50,
                            "default": 5,
                            "description": "Number of columns to generate"
                        }
                    },
                    "required": ["num_rows", "num_columns"]
                }
            },
            "customer_data": {
                "name": "customer_data",
                "description": "Generate realistic customer data with names, emails, addresses",
                "schema": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                            "default": 50,
                            "description": "Number of customers to generate"
                        }
                    }
                }
            },
            "sales_data": {
                "name": "sales_data", 
                "description": "Generate sales transaction data with products, amounts, dates",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transactions": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5000,
                            "default": 200,
                            "description": "Number of transactions to generate"
                        }
                    }
                }
            }
        }
        
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        return {"strategies": {}}

@app.get("/api/v1/config", tags=["Config"])
async def get_config(api_key: str = Depends(get_api_key)):
    """Get configuration."""
    try:
        config = Config.load()
        
        return {
            "base_dir": str(config.base_dir),
            "environment": "development",
            "has_openai_key": False,  # Don't expose sensitive info
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return {
            "base_dir": "/tmp/opensynthetics",
            "environment": "development", 
            "has_openai_key": False,
            "version": "1.0.0"
        }

# Add startup initialization
initialize_api_server()
create_demo_data()
print_startup_info()

# Initialize server on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the API server on startup."""
    initialize_api_server()
    create_demo_data()
    print_startup_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 