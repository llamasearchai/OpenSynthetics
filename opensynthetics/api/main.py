"""API server for OpenSynthetics."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import time

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace, WorkspaceError, DatasetError
from opensynthetics.core.exceptions import OpenSyntheticsError, AuthenticationError
from opensynthetics.api.auth import get_api_key, get_current_user
from opensynthetics.api.middleware import LoggingMiddleware, RateLimitMiddleware

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
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}

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
    config = Config.load()
    base_dir = config.base_dir
    
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
                # Skip invalid workspaces
                continue
    
    return {"workspaces": workspaces}

@app.get("/api/v1/workspaces/{workspace_path:path}", tags=["Workspaces"])
async def get_workspace(workspace_path: str, api_key: str = Depends(get_api_key)):
    """Get workspace information."""
    try:
        workspace = Workspace.load(workspace_path)
        datasets = workspace.list_datasets()
        
        return WorkspaceInfo(
            name=workspace.name,
            path=str(workspace.path),
            description=workspace.metadata.description,
            created_at=workspace.metadata.created_at,
            updated_at=workspace.metadata.updated_at,
            datasets=datasets,
        )
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

@app.post("/api/v1/workspaces", tags=["Workspaces"])
async def create_workspace(
    name: str,
    path: Optional[str] = None,
    description: str = "",
    api_key: str = Depends(get_api_key),
):
    """Create a new workspace."""
    try:
        workspace = Workspace.create(
            name=name,
            path=path,
            description=description,
        )
        
        return {
            "name": workspace.name,
            "path": str(workspace.path),
            "description": workspace.metadata.description,
            "created_at": workspace.metadata.created_at,
            "updated_at": workspace.metadata.updated_at,
        }
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@app.get("/api/v1/workspaces/{workspace_path:path}/datasets", tags=["Datasets"])
async def list_datasets(workspace_path: str, api_key: str = Depends(get_api_key)):
    """List datasets in a workspace."""
    try:
        workspace = Workspace.load(workspace_path)
        datasets = workspace.list_datasets()
        
        return {"datasets": datasets}
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

@app.get("/api/v1/workspaces/{workspace_path:path}/datasets/{dataset_name}", tags=["Datasets"])
async def get_dataset(workspace_path: str, dataset_name: str, api_key: str = Depends(get_api_key)):
    """Get dataset information."""
    try:
        workspace = Workspace.load(workspace_path)
        dataset = workspace.get_dataset(dataset_name)
        stats = dataset.get_stats()
        
        return DatasetInfo(
            name=dataset.name,
            description=dataset.description,
            tags=dataset.tags,
            created_at=stats.get("created_at", ""),
            updated_at=stats.get("updated_at", ""),
            tables=stats.get("tables", {}),
        )
    except (WorkspaceError, DatasetError) as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

@app.post("/api/v1/generate/jobs", tags=["Generation"])
async def create_generation_job(
    params: GenerationParameters,
    api_key: str = Depends(get_api_key),
):
    """Create a data generation job."""
    try:
        from opensynthetics.datagen.engine import Engine, GenerationError
        
        workspace = Workspace.load(params.workspace_path)
        engine = Engine(workspace)
        
        result = engine.generate(
            strategy=params.strategy,
            parameters=params.parameters,
            output_dataset=params.output_dataset,
        )
        
        return {
            "job_id": f"job_{result.get('timestamp', int(time.time()))}",
            "status": "completed",
            "dataset": params.output_dataset,
            "count": result.get("count", 0),
            "sample_items": result.get("sample_items", []),
        }
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except GenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@app.get("/api/v1/strategies", tags=["Generation"])
async def list_strategies(api_key: str = Depends(get_api_key)):
    """List available generation strategies."""
    from opensynthetics.datagen.engine import Engine
    
    # We need a temporary workspace to get the engine strategies
    # In a real implementation, this could be optimized
    config = Config.load()
    
    # Create Engine class without a workspace
    engine_class = Engine
    strategies = engine_class.STRATEGIES
    
    strategy_info = {}
    for name, strategy_class in strategies.items():
        schema = {}
        if hasattr(strategy_class, "parameter_model") and strategy_class.parameter_model:
            try:
                schema = strategy_class.parameter_model.schema()
            except Exception:
                pass
                
        strategy_info[name] = {
            "name": name,
            "description": strategy_class.__doc__.strip() if strategy_class.__doc__ else "",
            "schema": schema,
        }
    
    return {"strategies": strategy_info}

@app.get("/api/v1/config", tags=["Config"])
async def get_config(api_key: str = Depends(get_api_key)):
    """Get configuration."""
    config = Config.load()
    
    # Only return non-sensitive config
    return {
        "base_dir": str(config.base_dir),
        "environment": config.environment,
        "has_openai_key": bool(config.get_api_key("openai")),
    }

@app.post("/api/v1/config/api_keys/{provider}", tags=["Config"])
async def set_api_key(
    provider: str,
    api_key: str,
    current_api_key: str = Depends(get_api_key),
):
    """Set API key for provider."""
    config = Config.load()
    config.set_value(f"api_keys.{provider}", api_key)
    
    return {"message": f"API key for {provider} set successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 