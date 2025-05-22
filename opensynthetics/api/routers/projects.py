"""API endpoints for project management."""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from opensynthetics.api.auth import get_api_key
from opensynthetics.api.models.requests import ProjectCreate
from opensynthetics.api.models.responses import Project, ProjectList
from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import WorkspaceError
from opensynthetics.core.workspace import Workspace

router = APIRouter()


@router.post("/", response_model=Project)
async def create_project(
    project: ProjectCreate,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Create a new project.

    Args:
        project: Project parameters
        api_key: API key

    Returns:
        Dict[str, Any]: Created project
    """
    try:
        workspace = Workspace.create(
            name=project.name,
            path=project.path,
            description=project.description,
            tags=project.tags,
        )
        
        return {
            "name": workspace.name,
            "description": workspace.metadata.description,
            "path": str(workspace.path),
            "created_at": workspace.metadata.created_at.isoformat(),
            "updated_at": workspace.metadata.updated_at.isoformat(),
            "tags": workspace.metadata.tags,
        }
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@router.get("/", response_model=ProjectList)
async def list_projects(
    limit: int = Query(100, description="Maximum number of projects"),
    offset: int = Query(0, description="Starting offset"),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """List available projects.

    Args:
        limit: Maximum number of projects
        offset: Starting offset
        api_key: API key

    Returns:
        Dict[str, Any]: List of projects
    """
    try:
        config = Config.load()
        base_dir = config.storage.base_dir
        
        # List all subdirectories in base_dir
        projects = []
        count = 0
        
        for path in base_dir.glob("*"):
            if path.is_dir() and (path / "metadata.json").exists():
                count += 1
                
                # Apply offset and limit
                if count <= offset:
                    continue
                    
                if len(projects) >= limit:
                    break
                    
                try:
                    workspace = Workspace.load(path)
                    projects.append({
                        "name": workspace.name,
                        "description": workspace.metadata.description,
                        "path": str(workspace.path),
                        "created_at": workspace.metadata.created_at.isoformat(),
                        "updated_at": workspace.metadata.updated_at.isoformat(),
                        "tags": workspace.metadata.tags,
                    })
                except Exception as e:
                    # Skip invalid workspaces
                    pass
                    
        return {
            "count": count,
            "projects": projects,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@router.get("/{project_name}", response_model=Project)
async def get_project(
    project_name: str = Path(..., description="Project name"),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Get project details.

    Args:
        project_name: Project name
        api_key: API key

    Returns:
        Dict[str, Any]: Project details
    """
    try:
        config = Config.load()
        project_path = config.storage.base_dir / project_name
        
        if not project_path.exists() or not (project_path / "metadata.json").exists():
            raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
        workspace = Workspace.load(project_path)
        
        return {
            "name": workspace.name,
            "description": workspace.metadata.description,
            "path": str(workspace.path),
            "created_at": workspace.metadata.created_at.isoformat(),
            "updated_at": workspace.metadata.updated_at.isoformat(),
            "tags": workspace.metadata.tags,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@router.delete("/{project_name}")
async def delete_project(
    project_name: str = Path(..., description="Project name"),
    confirm: bool = Query(False, description="Confirmation flag"),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Delete a project.

    Args:
        project_name: Project name
        confirm: Confirmation flag
        api_key: API key

    Returns:
        Dict[str, Any]: Operation result
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Deletion not confirmed. Set confirm=true to delete project.")
        
    try:
        config = Config.load()
        project_path = config.storage.base_dir / project_name
        
        if not project_path.exists() or not (project_path / "metadata.json").exists():
            raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
        workspace = Workspace.load(project_path)
        workspace.delete(confirm=True)
        
        return {"status": "success", "message": f"Project {project_name} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")