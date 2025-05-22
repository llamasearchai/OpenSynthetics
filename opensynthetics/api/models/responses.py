"""Response models for OpenSynthetics API."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Project(BaseModel):
    """Project information."""

    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    path: str = Field(..., description="Project path")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Project tags")


class ProjectList(BaseModel):
    """List of projects."""

    count: int = Field(..., description="Total count")
    projects: List[Project] = Field(..., description="Projects")


class GenerationJob(BaseModel):
    """Generation job information."""

    id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    workspace: str = Field(..., description="Workspace path")
    strategy: str = Field(..., description="Generation strategy")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    output_dataset: Optional[str] = Field(None, description="Output dataset")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result")


class GenerationJobList(BaseModel):
    """List of generation jobs."""

    count: int = Field(..., description="Total count")
    jobs: List[GenerationJob] = Field(..., description="Jobs")


class Dataset(BaseModel):
    """Dataset information."""

    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Dataset tags")
    tables: Dict[str, Dict[str, Any]] = Field(..., description="Tables information")
    size_bytes: int = Field(..., description="Dataset size in bytes")


class DatasetList(BaseModel):
    """List of datasets."""

    count: int = Field(..., description="Total count")
    datasets: List[Dataset] = Field(..., description="Datasets")


class AgentTask(BaseModel):
    """Agent task information."""

    id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    agent_type: str = Field(..., description="Agent type")
    input: str = Field(..., description="Task input")
    parameters: Dict[str, Any] = Field(..., description="Task parameters")
    result: Optional[Any] = Field(None, description="Task result")


class AgentTaskList(BaseModel):
    """List of agent tasks."""

    count: int = Field(..., description="Total count")
    tasks: List[AgentTask] = Field(..., description="Tasks")


class Experiment(BaseModel):
    """Evaluation experiment information."""

    id: str = Field(..., description="Experiment ID")
    status: str = Field(..., description="Experiment status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    workspace: str = Field(..., description="Workspace path")
    dataset_id: str = Field(..., description="Dataset ID")
    model: str = Field(..., description="Model evaluated")
    metrics: List[str] = Field(..., description="Metrics measured")
    parameters: Dict[str, Any] = Field(..., description="Experiment parameters")
    results: Optional[Dict[str, Any]] = Field(None, description="Experiment results")


class ExperimentList(BaseModel):
    """List of experiments."""

    count: int = Field(..., description="Total count")
    experiments: List[Experiment] = Field(..., description="Experiments")


class LLMResponse(BaseModel):
    """Response from LLM."""

    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(..., description="Token usage")
    finish_reason: Optional[str] = Field(None, description="Finish reason")