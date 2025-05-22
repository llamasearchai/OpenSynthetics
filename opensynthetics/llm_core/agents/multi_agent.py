"""Multi-agent system for collaborative problem solving."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable

from loguru import logger
from pydantic import BaseModel, Field

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import OpenSyntheticsError
from opensynthetics.core.monitoring import monitor
from opensynthetics.llm_core.agents.base import Agent, Memory, Tool
from opensynthetics.llm_core.agents.coordinator import AgentCoordinator
from opensynthetics.llm_core.agents.generator import GeneratorAgent
from opensynthetics.llm_core.providers import LLMProvider, ProviderFactory


class AgentRole(BaseModel):
    """Role in a multi-agent system."""

    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    agent_type: str = Field(..., description="Agent type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Agent parameters")


class AgentAction:
    """Agent action with retry capabilities."""
    
    def __init__(
        self,
        agent: Agent,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        error_handler: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Initialize agent action.
        
        Args:
            agent: Agent to use
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            error_handler: Optional error handler
        """
        self.agent = agent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_handler = error_handler
    
    def execute(self, action: str, *args: Any, **kwargs: Any) -> Any:
        """Execute an agent action with retries.
        
        Args:
            action: Action to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Action result
            
        Raises:
            OpenSyntheticsError: If action fails after all retries
        """
        method = getattr(self.agent, action, None)
        if not method:
            raise ValueError(f"Action '{action}' not supported by agent '{self.agent.name}'")
            
        retry_count = 0
        last_exception = None
        
        while retry_count <= self.max_retries:
            try:
                with monitor.timer(f"agent_action_{action}", tags={"agent": self.agent.name}):
                    result = method(*args, **kwargs)
                    
                # Success - return the result
                return result
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Log the error
                logger.warning(f"Agent '{self.agent.name}' action '{action}' failed (attempt {retry_count}/{self.max_retries+1}): {e}")
                
                # Call error handler if provided
                if self.error_handler:
                    try:
                        self.error_handler(e)
                    except Exception as handler_e:
                        logger.error(f"Error in agent error handler: {handler_e}")
                
                # If we've reached max retries, break
                if retry_count > self.max_retries:
                    break
                    
                # Wait before retrying
                time.sleep(self.retry_delay)
                
        # If we get here, all retries failed
        error_msg = f"Agent '{self.agent.name}' action '{action}' failed after {self.max_retries+1} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
            
        logger.error(error_msg)
        raise OpenSyntheticsError(error_msg)


class MultiAgentSystem:
    """Multi-agent system for collaborative problem solving."""

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Config] = None,
        provider: Optional[Union[str, LLMProvider]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize multi-agent system.

        Args:
            name: System name
            description: System description
            config: Configuration, if None uses default
            provider: LLM provider or provider name
            max_retries: Maximum number of retries for agent actions
            retry_delay: Delay between retries in seconds
        """
        self.name = name
        self.description = description
        self.config = config or Config.load()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up coordinator
        self.coordinator = AgentCoordinator(config=self.config, provider=provider)
        
        # Initialize roles and agents
        self.roles: Dict[str, AgentRole] = {}
        self.workflows: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_actions: Dict[str, AgentAction] = {}
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        
    def _log_error(self, agent_name: str, error: Exception) -> None:
        """Log an agent error.
        
        Args:
            agent_name: Agent name
            error: Error
        """
        if agent_name not in self.error_counts:
            self.error_counts[agent_name] = 0
            
        self.error_counts[agent_name] += 1
        
        monitor.count(
            name="multi_agent_errors",
            value=1,
            tags={"agent": agent_name, "error_type": type(error).__name__},
        )
        
    def add_role(self, role: AgentRole) -> None:
        """Add a role to the system.

        Args:
            role: Role to add

        Raises:
            ValueError: If role already exists
        """
        if role.name in self.roles:
            raise ValueError(f"Role already exists: {role.name}")
            
        self.roles[role.name] = role
        logger.debug(f"Added role {role.name} to system {self.name}")
        
        # Create agent for role
        self._create_agent_for_role(role)
        
    def _create_agent_for_role(self, role: AgentRole) -> None:
        """Create an agent for a role.

        Args:
            role: Role to create agent for

        Raises:
            ValueError: If agent type not supported
        """
        try:
            if role.agent_type == "generator":
                agent = GeneratorAgent(
                    config=self.config,
                    **role.parameters,
                )
            # Add other agent types here
            else:
                raise ValueError(f"Unsupported agent type: {role.agent_type}")
                
            # Set name and description from role
            agent.name = role.name
            agent.description = role.description
            
            # Create agent action wrapper with error logging
            self.agent_actions[role.name] = AgentAction(
                agent=agent,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                error_handler=lambda e: self._log_error(role.name, e),
            )
            
            # Register agent with coordinator
            self.coordinator.register_agent(agent)
            
            logger.info(f"Created agent for role {role.name} with type {role.agent_type}")
        except Exception as e:
            logger.error(f"Failed to create agent for role {role.name}: {e}")
            raise OpenSyntheticsError(f"Failed to create agent: {e}")
        
    def define_workflow(self, name: str, steps: List[Dict[str, Any]]) -> None:
        """Define a workflow.

        Args:
            name: Workflow name
            steps: Workflow steps

        Raises:
            ValueError: If workflow already exists or contains invalid roles
        """
        if name in self.workflows:
            raise ValueError(f"Workflow already exists: {name}")
            
        # Validate steps
        for step in steps:
            agent_name = step.get("agent")
            if agent_name not in self.roles:
                raise ValueError(f"Invalid role in workflow: {agent_name}")
                
        self.workflows[name] = steps
        logger.debug(f"Defined workflow {name} with {len(steps)} steps")
        
    def execute_workflow(
        self,
        workflow_name: str,
        initial_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow.

        Args:
            workflow_name: Workflow name
            initial_input: Initial input

        Returns:
            Dict[str, Any]: Workflow results

        Raises:
            ValueError: If workflow not found
            OpenSyntheticsError: If workflow execution fails
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_name}")
            
        workflow = self.workflows[workflow_name]
        
        # Track execution metrics
        start_time = time.time()
        
        try:
            # Execute workflow using coordinator
            results = self.coordinator.execute_workflow(workflow, initial_input)
            
            # Track successful execution
            execution_time = time.time() - start_time
            monitor.gauge(
                name="workflow_execution_time",
                value=execution_time,
                tags={"workflow": workflow_name, "status": "success"},
            )
            
            return results
        except Exception as e:
            # Track failed execution
            execution_time = time.time() - start_time
            monitor.gauge(
                name="workflow_execution_time",
                value=execution_time,
                tags={"workflow": workflow_name, "status": "failed"},
            )
            
            logger.error(f"Error executing workflow {workflow_name}: {e}")
            raise OpenSyntheticsError(f"Workflow execution failed: {e}")
        
    def generate_engineering_problem_solution(self, problem_description: str) -> Dict[str, Any]:
        """Generate a solution to an engineering problem.

        Args:
            problem_description: Problem description

        Returns:
            Dict[str, Any]: Solution with explanation

        Raises:
            OpenSyntheticsError: If solution generation fails
        """
        # Set up roles and workflow if not already defined
        try:
            if "problem_analyst" not in self.roles:
                self.add_role(AgentRole(
                    name="problem_analyst",
                    description="Analyzes engineering problems and identifies key components",
                    agent_type="generator",
                ))
                
            if "solution_generator" not in self.roles:
                self.add_role(AgentRole(
                    name="solution_generator",
                    description="Generates solutions to engineering problems",
                    agent_type="generator",
                ))
                
            if "solution_reviewer" not in self.roles:
                self.add_role(AgentRole(
                    name="solution_reviewer",
                    description="Reviews and improves solutions",
                    agent_type="generator",
                ))
                
            if "solve_engineering_problem" not in self.workflows:
                self.define_workflow("solve_engineering_problem", [
                    {
                        "agent": "problem_analyst",
                        "action": "process",
                        "input": "$input",
                        "store_as": "analysis",
                    },
                    {
                        "agent": "solution_generator",
                        "action": "process",
                        "input": "Based on this analysis, generate a step-by-step solution:\n\n$result.analysis",
                        "store_as": "solution",
                    },
                    {
                        "agent": "solution_reviewer",
                        "action": "process",
                        "input": "Review and improve this solution if needed:\n\n$result.solution",
                        "store_as": "final_solution",
                    },
                ])
                
            results = self.execute_workflow(
                workflow_name="solve_engineering_problem",
                initial_input=f"Analyze this engineering problem in detail, identifying the key principles and quantities involved:\n\n{problem_description}",
            )
            
            return {
                "problem": problem_description,
                "analysis": results.get("analysis", ""),
                "solution": results.get("solution", ""),
                "final_solution": results.get("final_solution", ""),
            }
            
        except Exception as e:
            logger.error(f"Failed to generate engineering problem solution: {e}")
            raise OpenSyntheticsError(f"Failed to generate engineering problem solution: {e}")
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dict[str, Any]: Error statistics
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "errors_by_agent": self.error_counts,
        }