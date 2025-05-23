"""Request models for OpenSynthetics API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    """Request model for creating a project."""

    name: str = Field(..., description="Project name", min_length=1, max_length=100)
    description: str = Field("", description="Project description", max_length=500)
    path: Optional[str] = Field(None, description="Project path (optional, will be generated if not provided)")
    tags: List[str] = Field(default_factory=list, description="Project tags")


class WorkspaceCreate(BaseModel):
    """Request model for creating a workspace."""

    name: str = Field(..., description="Workspace name", min_length=1, max_length=100)
    description: str = Field("", description="Workspace description", max_length=500)
    path: Optional[str] = Field(None, description="Workspace path (optional)")
    tags: List[str] = Field(default_factory=list, description="Workspace tags")


class DatasetCreate(BaseModel):
    """Request model for creating a dataset."""

    name: str = Field(..., description="Dataset name", min_length=1, max_length=100)
    description: str = Field("", description="Dataset description", max_length=500)
    tags: List[str] = Field(default_factory=list, description="Dataset tags")


class GenerationJobCreate(BaseModel):
    """Request model for creating a generation job."""

    workspace: str = Field(..., description="Workspace path")
    strategy: str = Field(..., description="Generation strategy")
    parameters: dict = Field(..., description="Strategy parameters")
    output_dataset: str = Field(..., description="Output dataset name")


class AgentTaskCreate(BaseModel):
    """Request model for creating an agent task."""

    agent_type: str = Field(..., description="Agent type")
    input: str = Field(..., description="Task input")
    parameters: dict = Field(default_factory=dict, description="Task parameters")


class ExperimentCreate(BaseModel):
    """Request model for creating an evaluation experiment."""

    workspace: str = Field(..., description="Workspace path")
    dataset_id: str = Field(..., description="Dataset ID")
    model: str = Field(..., description="Model to evaluate")
    metrics: List[str] = Field(..., description="Metrics to measure")
    parameters: dict = Field(default_factory=dict, description="Experiment parameters") 