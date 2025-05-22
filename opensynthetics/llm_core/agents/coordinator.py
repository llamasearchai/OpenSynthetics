"""Coordinator for multi-agent workflows."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

from loguru import logger
from pydantic import BaseModel, Field

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import OpenSyntheticsError
from opensynthetics.llm_core.agents.base import Agent, Memory, Tool
from opensynthetics.llm_core.providers import LLMProvider, ProviderFactory


class WorkflowTask(BaseModel):
    """Workflow task."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str
    action: str
    input: Optional[Any] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    store_as: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentCoordinator:
    """Coordinator for multi-agent workflows."""

    def __init__(
        self,
        config: Optional[Config] = None,
        provider: Optional[Union[str, LLMProvider]] = None,
    ) -> None:
        """Initialize coordinator.

        Args:
            config: Configuration, if None uses default
            provider: LLM provider or provider name
        """
        self.config = config or Config.load()
        
        # Set up provider
        if provider is None:
            self.provider = ProviderFactory.get_provider(self.config.llm.default_provider, self.config)
        elif isinstance(provider, str):
            self.provider = ProviderFactory.get_provider(provider, self.config)
        else:
            self.provider = provider
            
        self.agents: Dict[str, Agent] = {}
        self.workflow_memory = Memory()
        self.running_tasks: Dict[str, WorkflowTask] = {}
        self.task_callbacks: Dict[str, List[Callable[[WorkflowTask], None]]] = {}
        
    def register_agent(self, agent: Agent) -> None:
        """Register an agent.

        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        logger.debug(f"Registered agent: {agent.name}")
        
    def get_agent(self, agent_name: str) -> Agent:
        """Get an agent by name.

        Args:
            agent_name: Agent name

        Returns:
            Agent: Agent

        Raises:
            ValueError: If agent not found
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        return self.agents[agent_name]
    
    def add_task_callback(self, task_id: str, callback: Callable[[WorkflowTask], None]) -> None:
        """Add a callback for a task.

        Args:
            task_id: Task ID
            callback: Callback function
        """
        if task_id not in self.task_callbacks:
            self.task_callbacks[task_id] = []
        self.task_callbacks[task_id].append(callback)
    
    def _notify_task_update(self, task: WorkflowTask) -> None:
        """Notify task update.

        Args:
            task: Task
        """
        callbacks = self.task_callbacks.get(task.id, [])
        for callback in callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")
    
    async def execute_task_async(self, task: WorkflowTask) -> Any:
        """Execute a task asynchronously.

        Args:
            task: Task to execute

        Returns:
            Any: Task result

        Raises:
            ValueError: If agent not found
            OpenSyntheticsError: If task execution fails
        """
        if task.agent_name not in self.agents:
            raise ValueError(f"Agent not found: {task.agent_name}")
        
        agent = self.agents[task.agent_name]
        task.status = "running"
        self._notify_task_update(task)
        
        try:
            if task.action == "process":
                task.result = agent.process(task.input)
            elif task.action == "generate":
                task.result = agent.generate(**task.parameters)
            else:
                raise ValueError(f"Unsupported action: {task.action}")
            
            task.status = "completed"
            return task.result
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Task {task.id} failed: {e}")
            raise OpenSyntheticsError(f"Task execution failed: {e}")
        finally:
            self._notify_task_update(task)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            bool: Whether task was cancelled
        """
        task = self.running_tasks.get(task_id)
        if task and task.status == "running":
            task.status = "cancelled"
            self._notify_task_update(task)
            return True
        return False
        
    def execute_workflow(
        self,
        workflow: List[Dict[str, Any]],
        initial_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow.

        Args:
            workflow: Workflow steps
            initial_input: Initial input

        Returns:
            Dict[str, Any]: Workflow results

        Raises:
            OpenSyntheticsError: If workflow execution fails
        """
        results = {}
        current_input = initial_input
        
        try:
            for step in workflow:
                agent_name = step.get("agent")
                action = step.get("action", "process")
                store_as = step.get("store_as")
                
                if agent_name not in self.agents:
                    raise ValueError(f"Agent not found: {agent_name}")
                    
                agent = self.agents[agent_name]
                
                # Get input
                step_input = step.get("input", current_input)
                if isinstance(step_input, str) and step_input.startswith("$result."):
                    result_key = step_input[8:]  # Remove "$result."
                    step_input = results.get(result_key, "")
                    
                # Create task
                task = WorkflowTask(
                    agent_name=agent_name,
                    action=action,
                    input=step_input,
                    parameters=step.get("parameters", {}),
                    store_as=store_as,
                )
                
                # Store task
                self.running_tasks[task.id] = task
                
                # Execute action
                try:
                    if action == "process":
                        task.result = agent.process(step_input)
                    elif action == "generate":
                        params = step.get("parameters", {})
                        task.result = agent.generate(**params)
                    else:
                        raise ValueError(f"Unsupported action: {action}")
                    
                    task.status = "completed"
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"Task {task.id} failed: {e}")
                    raise OpenSyntheticsError(f"Workflow step failed: {e}")
                finally:
                    self._notify_task_update(task)
                    
                # Store result
                if store_as:
                    results[store_as] = task.result
                    
                # Update current input for next step
                current_input = task.result
            
            return results
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise OpenSyntheticsError(f"Workflow execution failed: {e}")
    
    async def execute_workflow_async(
        self,
        workflow: List[Dict[str, Any]],
        initial_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow asynchronously.

        Args:
            workflow: Workflow steps
            initial_input: Initial input

        Returns:
            Dict[str, Any]: Workflow results

        Raises:
            OpenSyntheticsError: If workflow execution fails
        """
        results = {}
        current_input = initial_input
        
        try:
            for step in workflow:
                agent_name = step.get("agent")
                action = step.get("action", "process")
                store_as = step.get("store_as")
                
                if agent_name not in self.agents:
                    raise ValueError(f"Agent not found: {agent_name}")
                    
                # Get input
                step_input = step.get("input", current_input)
                if isinstance(step_input, str) and step_input.startswith("$result."):
                    result_key = step_input[8:]  # Remove "$result."
                    step_input = results.get(result_key, "")
                    
                # Create task
                task = WorkflowTask(
                    agent_name=agent_name,
                    action=action,
                    input=step_input,
                    parameters=step.get("parameters", {}),
                    store_as=store_as,
                )
                
                # Store task
                self.running_tasks[task.id] = task
                
                # Execute task asynchronously
                task_result = await self.execute_task_async(task)
                    
                # Store result
                if store_as:
                    results[store_as] = task_result
                    
                # Update current input for next step
                current_input = task_result
            
            return results
        except Exception as e:
            logger.error(f"Async workflow execution failed: {e}")
            raise OpenSyntheticsError(f"Async workflow execution failed: {e}")
        
    def reset(self) -> None:
        """Reset coordinator and all agents."""
        self.workflow_memory = Memory()
        
        # Cancel any running tasks
        for task_id, task in list(self.running_tasks.items()):
            if task.status == "running":
                self.cancel_task(task_id)
        
        # Clear task data
        self.running_tasks.clear()
        self.task_callbacks.clear()
        
        # Reset agents
        for agent in self.agents.values():
            agent.reset()
            
        logger.info("Coordinator reset complete")