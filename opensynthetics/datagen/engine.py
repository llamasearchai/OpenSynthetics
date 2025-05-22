"""Data generation engine for OpenSynthetics."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

from loguru import logger
from pydantic import BaseModel, Field, validator

from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace, Dataset
from opensynthetics.core.exceptions import WorkspaceError, DatasetError, OpenSyntheticsError, GenerationError
from opensynthetics.llm_core.agents.generator import GeneratorAgent
# Import advanced strategies when they're ready
try:
    from opensynthetics.datagen.advanced_strategies import (
        ConversationalDataStrategy,
        ResearchPaperStrategy,
        CodeGenerationStrategy,
        MultiModalDataStrategy
    )
    ADVANCED_STRATEGIES_AVAILABLE = True
except ImportError:
    ADVANCED_STRATEGIES_AVAILABLE = False


class EngineeringProblemParams(BaseModel):
    """Parameters for engineering problem generation."""
    
    domain: str = Field(..., description="Engineering domain (e.g., mechanical, electrical, civil)")
    count: int = Field(..., description="Number of problems to generate", gt=0)
    difficulty: int = Field(..., description="Difficulty level (1-10)", ge=1, le=10)
    constraints: Optional[str] = Field(None, description="Optional constraints for the problems")
    
    @validator('domain')
    def validate_domain(cls, v):
        """Validate domain."""
        valid_domains = ['mechanical', 'electrical', 'civil', 'chemical', 'software', 'aerospace']
        if v.lower() not in valid_domains:
            raise ValueError(f"Domain must be one of: {', '.join(valid_domains)}")
        return v


class DesignSystemParams(BaseModel):
    """Parameters for system design generation."""
    
    requirements: str = Field(..., description="System requirements")
    constraints: str = Field(..., description="Design constraints")


class GenerationStrategy:
    """Base class for data generation strategies."""
    
    # Class variable to define parameter model (to be overridden by subclasses)
    parameter_model = None
    
    def __init__(self, parameters: Dict[str, Any], config: Config) -> None:
        """Initialize a generation strategy.
        
        Args:
            parameters: Strategy parameters
            config: Global configuration
        """
        self.config = config
        self.validate_and_set_parameters(parameters)
        
    def validate_and_set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate and set parameters using Pydantic model.
        
        Args:
            parameters: Strategy parameters
            
        Raises:
            GenerationError: If parameters are invalid
        """
        if not self.parameter_model:
            # Base implementation does minimal validation
            if not isinstance(parameters, dict):
                raise GenerationError("Parameters must be a dictionary")
            self.parameters = parameters
            return
            
        try:
            # Use the strategy's parameter model for validation
            validated_params = self.parameter_model(**parameters)
            self.parameters = validated_params.dict()
        except Exception as e:
            raise GenerationError(f"Invalid parameters: {e}")
        
    def generate(self) -> List[Dict[str, Any]]:
        """Generate data items.
        
        Returns:
            List[Dict[str, Any]]: Generated data items
        
        Raises:
            GenerationError: If generation fails
        """
        raise NotImplementedError("Subclasses must implement generate()")


class EngineeringProblemsStrategy(GenerationStrategy):
    """Strategy for generating engineering problems."""
    
    # Define parameter model for this strategy
    parameter_model = EngineeringProblemParams
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate engineering problems.
        
        Returns:
            List[Dict[str, Any]]: Generated problems
            
        Raises:
            GenerationError: If generation fails
        """
        domain = self.parameters["domain"]
        count = self.parameters["count"]
        difficulty = self.parameters["difficulty"]
        constraints = self.parameters.get("constraints", "")
        
        try:
            # Create a generator agent
            agent = GeneratorAgent(config=self.config)
            
            # Generate problems using LLM
            problems = []
            for i in range(count):
                # Construct a detailed prompt for the LLM
                prompt = f"""
                Generate a challenging {domain} engineering problem with difficulty level {difficulty}/10.
                
                {constraints if constraints else ""}
                
                Provide the output in the following JSON format:
                {{
                    "id": "unique_id",
                    "problem_description": "Detailed description of the problem",
                    "difficulty_level": {difficulty},
                    "domain": "{domain}",
                    "expected_solution": "Detailed solution to the problem",
                    "metadata": {{
                        "generated_timestamp": "<current_timestamp>",
                        "constraints_used": "{constraints}"
                    }}
                }}
                
                Return only valid JSON with no additional text or explanation.
                """
                
                # Generate the problem
                logger.info(f"Generating {domain} problem #{i+1} with difficulty {difficulty}")
                response = agent.generate(prompt=prompt)
                
                try:
                    # Parse the generated JSON
                    problem_json = response.strip()
                    
                    # Find the JSON part if there's additional text
                    if problem_json.find('{') > 0:
                        problem_json = problem_json[problem_json.find('{'):]
                    if problem_json.rfind('}') < len(problem_json) - 1:
                        problem_json = problem_json[:problem_json.rfind('}')+1]
                    
                    problem = json.loads(problem_json)
                    
                    # Add timestamp if not present
                    if "metadata" not in problem:
                        problem["metadata"] = {}
                    if "generated_timestamp" not in problem["metadata"]:
                        problem["metadata"]["generated_timestamp"] = time.time()
                    
                    problems.append(problem)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing generated problem JSON: {e}")
                    logger.debug(f"Generated content: {response}")
                    # Create a problem anyway with the raw response
                    problem = {
                        "id": f"{domain}_{i+1}",
                        "problem_description": response,
                        "difficulty_level": difficulty,
                        "domain": domain,
                        "expected_solution": "Error parsing generated content",
                        "metadata": {
                            "generated_timestamp": time.time(),
                            "constraints_used": constraints,
                            "error": f"JSON parse error: {str(e)}"
                        }
                    }
                    problems.append(problem)
                
            logger.info(f"Generated {len(problems)} {domain} problems with difficulty {difficulty}")
            return problems
            
        except Exception as e:
            logger.error(f"Error generating engineering problems: {e}")
            raise GenerationError(f"Failed to generate engineering problems: {e}")


class DesignSystemStrategy(GenerationStrategy):
    """Strategy for generating system designs."""
    
    # Define parameter model for this strategy
    parameter_model = DesignSystemParams
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate system designs.
        
        Returns:
            List[Dict[str, Any]]: Generated system designs
            
        Raises:
            GenerationError: If generation fails
        """
        requirements = self.parameters["requirements"]
        constraints = self.parameters["constraints"]
        
        try:
            # Create a generator agent
            agent = GeneratorAgent(config=self.config)
            
            # Construct a detailed prompt for the LLM
            prompt = f"""
            Generate a detailed system design based on the following requirements and constraints:
            
            Requirements:
            {requirements}
            
            Constraints:
            {constraints}
            
            Provide the output in the following JSON format:
            {{
                "id": "design_1",
                "requirements": "{requirements}",
                "constraints": "{constraints}",
                "design_overview": "Detailed overview of the system design",
                "components": [
                    {{"name": "Component Name", "purpose": "Component purpose", "specifications": "Component specifications"}},
                    // Additional components
                ],
                "interfaces": [
                    {{"source": "Component A", "target": "Component B", "description": "Interface description"}},
                    // Additional interfaces
                ],
                "metadata": {{
                    "generated_timestamp": "<current_timestamp>"
                }}
            }}
            
            Return only valid JSON with no additional text or explanation.
            """
            
            # Generate the system design
            logger.info(f"Generating system design based on requirements")
            response = agent.generate(prompt=prompt)
            
            try:
                # Parse the generated JSON
                design_json = response.strip()
                
                # Find the JSON part if there's additional text
                if design_json.find('{') > 0:
                    design_json = design_json[design_json.find('{'):]
                if design_json.rfind('}') < len(design_json) - 1:
                    design_json = design_json[:design_json.rfind('}')+1]
                
                design = json.loads(design_json)
                
                # Add timestamp if not present
                if "metadata" not in design:
                    design["metadata"] = {}
                if "generated_timestamp" not in design["metadata"]:
                    design["metadata"]["generated_timestamp"] = time.time()
                
                return [design]
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing generated design JSON: {e}")
                logger.debug(f"Generated content: {response}")
                # Create a design anyway with the raw response
                design = {
                    "id": "design_1",
                    "requirements": requirements,
                    "constraints": constraints,
                    "design_overview": response,
                    "components": [],
                    "interfaces": [],
                    "metadata": {
                        "generated_timestamp": time.time(),
                        "error": f"JSON parse error: {str(e)}"
                    }
                }
                return [design]
                
        except Exception as e:
            logger.error(f"Error generating system design: {e}")
            raise GenerationError(f"Failed to generate system design: {e}")


class GenerationParameters(BaseModel):
    """Parameters for data generation."""
    
    strategy: str = Field(..., description="Generation strategy name")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    output_dataset: str = Field(..., description="Dataset name for output")
    
    @validator('strategy')
    def validate_strategy(cls, v):
        """Validate strategy."""
        if v not in Engine.STRATEGIES:
            raise ValueError(f"Strategy must be one of: {', '.join(Engine.STRATEGIES.keys())}")
        return v
        
    @validator('output_dataset')
    def validate_output_dataset(cls, v):
        """Validate output dataset."""
        if not v:
            raise ValueError("Output dataset name cannot be empty")
        return v


class Engine:
    """Data generation engine."""
    
    STRATEGIES = {
        "engineering_problems": EngineeringProblemsStrategy,
        "system_design": DesignSystemStrategy,
    }
    
    # Add advanced strategies if available
    if ADVANCED_STRATEGIES_AVAILABLE:
        STRATEGIES.update({
            "conversational_data": ConversationalDataStrategy,
            "research_papers": ResearchPaperStrategy,
            "code_generation": CodeGenerationStrategy,
            "multimodal_data": MultiModalDataStrategy,
        })
    
    def __init__(self, workspace: Workspace) -> None:
        """Initialize engine.
        
        Args:
            workspace: Workspace to use
        """
        self.workspace = workspace
        self.config = Config.load()
        
    def generate(
        self,
        strategy: str,
        parameters: Dict[str, Any],
        output_dataset: str,
    ) -> Dict[str, Any]:
        """Generate data.
        
        Args:
            strategy: Generation strategy
            parameters: Strategy parameters
            output_dataset: Dataset name for output
            
        Returns:
            Dict[str, Any]: Generation result with metadata
            
        Raises:
            GenerationError: If generation fails
        """
        # Validate parameters
        try:
            params = GenerationParameters(
                strategy=strategy,
                parameters=parameters,
                output_dataset=output_dataset,
            )
        except Exception as e:
            raise GenerationError(f"Invalid parameters: {e}")
            
        # Get strategy class
        if strategy not in self.STRATEGIES:
            raise GenerationError(f"Strategy not found: {strategy}")
            
        strategy_class = self.STRATEGIES[strategy]
        
        # Create strategy instance
        strategy_instance = strategy_class(parameters, self.config)
        
        # Generate data
        try:
            data = strategy_instance.generate()
        except Exception as e:
            raise GenerationError(f"Data generation failed: {e}")
            
        # Store in dataset
        try:
            # Create dataset if it doesn't exist
            try:
                dataset = self.workspace.get_dataset(output_dataset)
            except WorkspaceError:
                dataset = self.workspace.create_dataset(
                    name=output_dataset,
                    description=f"Generated data using {strategy}",
                    tags=[strategy],
                )
                
            # Add data to dataset
            dataset.add_data(data)
            
            # Return result metadata
            result = {
                "count": len(data),
                "strategy": strategy,
                "output_dataset": output_dataset,
                "workspace": str(self.workspace.path),
                "timestamp": time.time(),
            }
            
            # Add sample items (up to 3)
            sample_count = min(3, len(data))
            if sample_count > 0:
                result["sample_items"] = data[:sample_count]
                
            return result
        except Exception as e:
            raise GenerationError(f"Failed to store generated data: {e}")
            
    def list_strategies(self) -> Dict[str, Dict[str, Any]]:
        """List available strategies.
        
        Returns:
            Dict[str, Dict[str, Any]]: Available strategies with metadata
        """
        strategies = {}
        
        for name, strategy_class in self.STRATEGIES.items():
            # Get parameter model
            parameter_model = strategy_class.parameter_model
            
            # Get schema if available
            schema = {}
            if parameter_model:
                try:
                    schema = parameter_model.schema()
                except Exception:
                    pass
                    
            strategies[name] = {
                "name": name,
                "description": strategy_class.__doc__.strip() if strategy_class.__doc__ else "",
                "schema": schema,
            }
            
        return strategies

# For future implementation: Multi-agent generation strategies
# class MultiAgentStrategy(GenerationStrategy):
#     """Strategy using multiple agents to generate and validate data."""
#     pass