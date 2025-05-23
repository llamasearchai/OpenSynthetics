"""Data generation engine for OpenSynthetics."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace, Dataset
from opensynthetics.core.exceptions import WorkspaceError, DatasetError, OpenSyntheticsError, GenerationError, ProcessingError, EvaluationError
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

# Import scientific strategies
try:
    from opensynthetics.datagen.scientific_strategies import (
        ScientificLiteratureStrategy,
        ScientificDatasetStrategy
    )
    SCIENTIFIC_STRATEGIES_AVAILABLE = True
except ImportError:
    SCIENTIFIC_STRATEGIES_AVAILABLE = False

from opensynthetics.datagen.synthetic_datasets import (
    SyntheticDatasetFactory, 
    DatasetTemplate,
    SyntheticDatasetConfig
)
from opensynthetics.data_ops.export_utils import DataExporter, ExportConfig, BatchExporter
from opensynthetics.training_eval.benchmark import SyntheticDatasetBenchmark, BenchmarkConfig


class EngineeringProblemParams(BaseModel):
    """Parameters for engineering problem generation."""
    
    domain: str = Field(..., description="Engineering domain (e.g., mechanical, electrical, civil)")
    count: int = Field(..., description="Number of problems to generate", gt=0)
    difficulty: int = Field(..., description="Difficulty level (1-10)", ge=1, le=10)
    constraints: Optional[str] = Field(None, description="Optional constraints for the problems")
    
    @field_validator('domain')
    @classmethod
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
            self.parameters = validated_params.model_dump()
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
    
    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        """Validate strategy."""
        if v not in Engine.STRATEGIES:
            raise ValueError(f"Strategy must be one of: {', '.join(Engine.STRATEGIES.keys())}")
        return v
        
    @field_validator('output_dataset')
    @classmethod
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
    
    # Add scientific strategies if available
    if SCIENTIFIC_STRATEGIES_AVAILABLE:
        STRATEGIES.update({
            "scientific_literature": ScientificLiteratureStrategy,
            "scientific_dataset": ScientificDatasetStrategy,
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
                    schema = parameter_model.model_json_schema()
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

class OpenSyntheticsEngine:
    """Main engine for OpenSynthetics data generation."""
    
    def __init__(self, config: Config = None) -> None:
        """Initialize the engine."""
        self.config = config or Config.load()
        self.strategies = {}
        self.results_cache = {}
        
        # Initialize new components
        self.synthetic_factory = SyntheticDatasetFactory()
        self.data_exporter = DataExporter()
        self.benchmarker = SyntheticDatasetBenchmark()
        
        # Register default strategies
        self._register_default_strategies()
        
        logger.info("OpenSynthetics engine initialized")
    
    def _register_default_strategies(self) -> None:
        """Register default data generation strategies."""
        # Register existing strategies (commented out until they're implemented)
        # self.register_strategy("conversation", ConversationStrategy())
        # self.register_strategy("instruction", InstructionStrategy())
        # self.register_strategy("scientific_literature", ScientificLiteratureStrategy())
        # self.register_strategy("scientific_dataset", ScientificDatasetStrategy())
        
        # Register new synthetic dataset strategies
        self.strategies["synthetic_customer"] = self._create_customer_strategy
        self.strategies["synthetic_sales"] = self._create_sales_strategy
        self.strategies["synthetic_iot"] = self._create_iot_strategy
        self.strategies["synthetic_custom"] = self._create_custom_strategy
    
    def _create_customer_strategy(self):
        """Create customer data strategy."""
        return lambda num_examples, **kwargs: self.synthetic_factory.create_dataset(
            config="customer_data",
            num_rows=num_examples,
            **kwargs
        )
    
    def _create_sales_strategy(self):
        """Create sales data strategy."""
        return lambda num_examples, **kwargs: self.synthetic_factory.create_dataset(
            config="sales_data",
            num_rows=num_examples,
            **kwargs
        )
    
    def _create_iot_strategy(self):
        """Create IoT sensor data strategy."""
        return lambda num_examples, **kwargs: self.synthetic_factory.create_dataset(
            config="iot_sensor_data",
            num_rows=num_examples,
            **kwargs
        )
    
    def _create_custom_strategy(self):
        """Create custom synthetic data strategy."""
        return lambda config, **kwargs: self.synthetic_factory.create_dataset(
            config=config,
            **kwargs
        )

    def generate_synthetic_dataset(
        self,
        template_name: str,
        num_rows: int = 1000,
        export_format: str = "parquet",
        benchmark: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate synthetic dataset using templates.
        
        Args:
            template_name: Name of the dataset template
            num_rows: Number of rows to generate
            export_format: Export format (parquet, csv, json, etc.)
            benchmark: Whether to run quality benchmarking
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary containing dataset and metadata
        """
        logger.info(f"Generating synthetic dataset: {template_name}")
        
        try:
            # Create export configuration
            export_config = ExportConfig(
                format=export_format,
                include_metadata=True,
                create_checksums=True
            )
            
            # Generate dataset
            result = self.synthetic_factory.create_dataset(
                config=template_name,
                num_rows=num_rows,
                benchmark=benchmark,
                export=True,
                export_config=export_config,
                **kwargs
            )
            
            # Cache result
            cache_key = f"synthetic_{template_name}_{num_rows}"
            self.results_cache[cache_key] = result
            
            logger.info(f"Generated synthetic dataset: {result['metadata']['num_rows']} rows, "
                       f"{result['metadata']['num_columns']} columns")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic dataset: {e}")
            raise GenerationError(f"Synthetic dataset generation failed: {e}")
    
    def generate_multiple_synthetic_datasets(
        self,
        dataset_configs: Dict[str, Union[str, Dict[str, Any]]],
        output_dir: str = "./synthetic_datasets",
        export_format: str = "parquet",
        benchmark: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Generate multiple synthetic datasets in batch.
        
        Args:
            dataset_configs: Dictionary mapping dataset names to configurations
            output_dir: Output directory for datasets
            export_format: Export format for all datasets
            benchmark: Whether to run quality benchmarking
            
        Returns:
            Dictionary of results for each dataset
        """
        logger.info(f"Generating {len(dataset_configs)} synthetic datasets in batch")
        
        try:
            # Prepare configurations
            configs = {}
            for name, config in dataset_configs.items():
                if isinstance(config, str):
                    # Template name
                    configs[name] = config
                else:
                    # Custom configuration
                    if "export_config" not in config:
                        config["export_config"] = {
                            "format": export_format,
                            "include_metadata": True,
                            "create_checksums": True
                        }
                    configs[name] = SyntheticDatasetConfig.parse_obj(config)
            
            # Generate datasets
            results = self.synthetic_factory.create_multiple_datasets(
                configs=configs,
                benchmark=benchmark,
                export=True
            )
            
            # Export datasets if needed
            if output_dir:
                export_config = ExportConfig(format=export_format, include_metadata=True)
                batch_exporter = BatchExporter()
                
                datasets_to_export = {}
                for name, result in results.items():
                    if not result.get("error") and result.get("dataset") is not None:
                        datasets_to_export[name] = result["dataset"]
                
                if datasets_to_export:
                    batch_exporter.export_multiple(
                        datasets=datasets_to_export,
                        output_dir=output_dir,
                        default_config=export_config
                    )
            
            # Cache results
            cache_key = f"batch_synthetic_{len(dataset_configs)}"
            self.results_cache[cache_key] = results
            
            successful = sum(1 for r in results.values() if not r.get("error"))
            logger.info(f"Batch generation completed: {successful}/{len(dataset_configs)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic datasets: {e}")
            raise GenerationError(f"Batch synthetic dataset generation failed: {e}")
    
    def benchmark_dataset(
        self,
        dataset: Union[pd.DataFrame, str, Path],
        reference_dataset: Optional[Union[pd.DataFrame, str, Path]] = None,
        target_column: Optional[str] = None,
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """Benchmark dataset quality and utility.
        
        Args:
            dataset: Dataset to benchmark (DataFrame or file path)
            reference_dataset: Optional reference dataset for comparison
            target_column: Target column for ML evaluation
            config: Benchmark configuration
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Running dataset quality benchmark")
        
        try:
            # Load datasets if needed
            if isinstance(dataset, (str, Path)):
                dataset_path = Path(dataset)
                if dataset_path.suffix == '.parquet':
                    synthetic_df = pd.read_parquet(dataset_path)
                elif dataset_path.suffix == '.csv':
                    synthetic_df = pd.read_csv(dataset_path)
                elif dataset_path.suffix == '.json':
                    synthetic_df = pd.read_json(dataset_path)
                else:
                    raise ProcessingError(f"Unsupported file format: {dataset_path.suffix}")
            else:
                synthetic_df = dataset
            
            reference_df = None
            if reference_dataset is not None:
                if isinstance(reference_dataset, (str, Path)):
                    ref_path = Path(reference_dataset)
                    if ref_path.suffix == '.parquet':
                        reference_df = pd.read_parquet(ref_path)
                    elif ref_path.suffix == '.csv':
                        reference_df = pd.read_csv(ref_path)
                    elif ref_path.suffix == '.json':
                        reference_df = pd.read_json(ref_path)
                    else:
                        raise ProcessingError(f"Unsupported reference file format: {ref_path.suffix}")
                else:
                    reference_df = reference_dataset
            
            # Use provided config or create default
            benchmark_config = config or BenchmarkConfig()
            benchmarker = SyntheticDatasetBenchmark(benchmark_config)
            
            # Auto-detect target column if not provided
            if not target_column:
                categorical_cols = synthetic_df.select_dtypes(include=['object', 'category']).columns
                numeric_cols = synthetic_df.select_dtypes(include=['number']).columns
                
                if len(categorical_cols) > 0:
                    target_column = categorical_cols[0]
                elif len(numeric_cols) > 0:
                    target_column = numeric_cols[0]
            
            # Run benchmark
            metrics = benchmarker.benchmark_dataset(
                synthetic_data=synthetic_df,
                reference_data=reference_df,
                target_column=target_column
            )
            
            results = {
                "metrics": metrics.dict(),
                "dataset_info": {
                    "synthetic_shape": synthetic_df.shape,
                    "reference_shape": reference_df.shape if reference_df is not None else None,
                    "target_column": target_column
                },
                "config": benchmark_config.dict()
            }
            
            logger.info(f"Benchmark completed. Overall quality score: {metrics.overall_quality_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Dataset benchmarking failed: {e}")
            raise EvaluationError(f"Failed to benchmark dataset: {e}")
    
    def export_dataset(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        output_path: Union[str, Path],
        format: str = "parquet",
        **export_options
    ) -> Dict[str, Any]:
        """Export dataset to specified format.
        
        Args:
            data: Data to export
            output_path: Output file path
            format: Export format
            **export_options: Additional export options
            
        Returns:
            Export metadata
        """
        logger.info(f"Exporting dataset to {output_path} in {format} format")
        
        try:
            # Create export configuration
            export_config = ExportConfig(
                format=format,
                include_metadata=True,
                create_checksums=True,
                **export_options
            )
            
            # Export dataset
            exporter = DataExporter(export_config)
            metadata = exporter.export_dataset(data, output_path)
            
            logger.info(f"Export completed: {metadata.total_records} records, "
                       f"{metadata.total_size_bytes} bytes")
            
            return metadata.dict()
            
        except Exception as e:
            logger.error(f"Dataset export failed: {e}")
            raise ProcessingError(f"Failed to export dataset: {e}")
    
    def create_synthetic_dataset_config(
        self,
        num_rows: int,
        columns: List[Dict[str, Any]],
        **options
    ) -> SyntheticDatasetConfig:
        """Create synthetic dataset configuration.
        
        Args:
            num_rows: Number of rows to generate
            columns: Column specifications
            **options: Additional configuration options
            
        Returns:
            SyntheticDatasetConfig object
        """
        try:
            # Parse column specifications
            column_schemas = []
            for col_spec in columns:
                from opensynthetics.datagen.synthetic_datasets import ColumnSchema, DataDistribution
                
                # Create distribution if specified
                distribution = DataDistribution()
                if "distribution" in col_spec:
                    distribution = DataDistribution.parse_obj(col_spec["distribution"])
                
                # Create column schema
                column_schema = ColumnSchema(
                    name=col_spec["name"],
                    data_type=col_spec.get("data_type", "numeric"),
                    distribution=distribution,
                    **{k: v for k, v in col_spec.items() 
                       if k not in ["name", "data_type", "distribution"]}
                )
                column_schemas.append(column_schema)
            
            # Create configuration
            config = SyntheticDatasetConfig(
                num_rows=num_rows,
                columns=column_schemas,
                **options
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create synthetic dataset config: {e}")
            raise GenerationError(f"Config creation failed: {e}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available synthetic dataset templates.
        
        Returns:
            List of template names
        """
        return self.synthetic_factory.list_templates()
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template information
        """
        if template_name == "customer_data":
            return {
                "name": "customer_data",
                "description": "Customer demographic and profile data",
                "columns": ["customer_id", "age", "income", "gender", "region", "registration_date", "is_premium"],
                "use_cases": ["Customer segmentation", "Marketing analysis", "Demographic studies"]
            }
        elif template_name == "sales_data":
            return {
                "name": "sales_data",
                "description": "E-commerce transaction and sales data",
                "columns": ["transaction_id", "customer_id", "product_category", "amount", "quantity", "transaction_date", "payment_method", "discount_applied"],
                "use_cases": ["Sales forecasting", "Customer behavior analysis", "Revenue optimization"]
            }
        elif template_name == "iot_sensor_data":
            return {
                "name": "iot_sensor_data",
                "description": "IoT sensor monitoring and telemetry data",
                "columns": ["sensor_id", "timestamp", "temperature", "humidity", "pressure", "light_level", "battery_level", "status"],
                "use_cases": ["Environmental monitoring", "Predictive maintenance", "Anomaly detection"]
            }
        else:
            raise ValueError(f"Unknown template: {template_name}")