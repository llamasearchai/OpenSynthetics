"""Unit tests for data generation engine."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import GenerationError
from opensynthetics.core.workspace import Workspace
from opensynthetics.datagen.engine import (
    Engine,
    EngineeringProblemsStrategy,
    DesignSystemStrategy,
    GenerationStrategy,
    EngineeringProblemParams,
    DesignSystemParams,
)


class TestGenerationStrategy:
    """Tests for GenerationStrategy base class."""

    def test_init_without_parameter_model(self):
        """Test init without parameter_model should raise error."""
        class TestStrategy(GenerationStrategy):
            pass  # No parameter_model
            
        config = Config()
        
        with pytest.raises(GenerationError, match="parameter_model"):
            TestStrategy({}, config)

    def test_init_with_invalid_parameters_dict(self):
        """Test init with invalid parameters dict."""
        config = Config()
        
        with pytest.raises(GenerationError, match="Invalid parameters"):
            EngineeringProblemsStrategy("not a dict", config)

    def test_generate_not_implemented(self):
        """Test that generate method must be implemented."""
        config = Config()
        strategy = GenerationStrategy({}, config)
        
        with pytest.raises(NotImplementedError):
            strategy.generate()


class TestEngineeringProblemParams:
    """Tests for EngineeringProblemParams validation."""

    def test_valid_parameters(self):
        """Test valid engineering problem parameters."""
        params = EngineeringProblemParams(
            domain="mechanical",
            count=5,
            difficulty=7,
            constraints="Static equilibrium problems"
        )
        
        assert params.domain == "mechanical"
        assert params.count == 5
        assert params.difficulty == 7
        assert params.constraints == "Static equilibrium problems"

    def test_invalid_domain(self):
        """Test invalid domain validation."""
        with pytest.raises(ValidationError, match="Domain must be one of"):
            EngineeringProblemParams(
                domain="invalid_domain",
                count=5,
                difficulty=7
            )

    def test_invalid_count_negative(self):
        """Test invalid count (negative)."""
        with pytest.raises(ValidationError):
            EngineeringProblemParams(
                domain="mechanical",
                count=-1,
                difficulty=7
            )

    def test_invalid_count_zero(self):
        """Test invalid count (zero)."""
        with pytest.raises(ValidationError):
            EngineeringProblemParams(
                domain="mechanical",
                count=0,
                difficulty=7
            )

    def test_invalid_difficulty_low(self):
        """Test invalid difficulty (too low)."""
        with pytest.raises(ValidationError):
            EngineeringProblemParams(
                domain="mechanical",
                count=5,
                difficulty=0
            )

    def test_invalid_difficulty_high(self):
        """Test invalid difficulty (too high)."""
        with pytest.raises(ValidationError):
            EngineeringProblemParams(
                domain="mechanical",
                count=5,
                difficulty=11
            )

    def test_optional_constraints(self):
        """Test that constraints are optional."""
        params = EngineeringProblemParams(
            domain="electrical",
            count=3,
            difficulty=5
        )
        
        assert params.constraints is None


class TestDesignSystemParams:
    """Tests for DesignSystemParams validation."""

    def test_valid_parameters(self):
        """Test valid design system parameters."""
        params = DesignSystemParams(
            requirements="High-performance web application",
            constraints="Must support 10,000 concurrent users"
        )
        
        assert params.requirements == "High-performance web application"
        assert params.constraints == "Must support 10,000 concurrent users"

    def test_missing_requirements(self):
        """Test missing requirements field."""
        with pytest.raises(ValidationError):
            DesignSystemParams(constraints="Some constraints")

    def test_missing_constraints(self):
        """Test missing constraints field."""
        with pytest.raises(ValidationError):
            DesignSystemParams(requirements="Some requirements")


class TestEngineeringProblemsStrategy:
    """Tests for EngineeringProblemsStrategy."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def mock_agent(self):
        """Create mock generator agent."""
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent') as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.generate.return_value = json.dumps({
                "id": "mech_001",
                "problem_description": "Calculate the stress in a steel beam",
                "difficulty_level": 5,
                "domain": "mechanical",
                "expected_solution": "Maximum stress = 150 MPa",
                "metadata": {
                    "generated_timestamp": 1234567890,
                    "constraints_used": "Static equilibrium"
                }
            })
            yield mock_agent

    def test_init_with_valid_parameters(self, config):
        """Test initializing strategy with valid parameters."""
        parameters = {
            "domain": "mechanical",
            "count": 3,
            "difficulty": 6,
            "constraints": "Static analysis"
        }
        
        strategy = EngineeringProblemsStrategy(parameters, config)
        assert strategy.parameters == parameters

    def test_init_with_invalid_parameters(self, config):
        """Test initializing strategy with invalid parameters."""
        parameters = {
            "domain": "invalid",
            "count": 3,
            "difficulty": 6
        }
        
        with pytest.raises(GenerationError, match="Invalid parameters"):
            EngineeringProblemsStrategy(parameters, config)

    def test_generate_single_problem(self, config, mock_agent):
        """Test generating a single engineering problem."""
        parameters = {
            "domain": "mechanical",
            "count": 1,
            "difficulty": 5,
            "constraints": "Static equilibrium"
        }
        
        strategy = EngineeringProblemsStrategy(parameters, config)
        
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent', return_value=mock_agent):
            problems = strategy.generate()
        
        assert len(problems) == 1
        problem = problems[0]
        assert problem["id"] == "mech_001"
        assert problem["domain"] == "mechanical"
        assert problem["difficulty_level"] == 5

    def test_generate_multiple_problems(self, config, mock_agent):
        """Test generating multiple engineering problems."""
        parameters = {
            "domain": "electrical",
            "count": 3,
            "difficulty": 7
        }
        
        strategy = EngineeringProblemsStrategy(parameters, config)
        
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent', return_value=mock_agent):
            problems = strategy.generate()
        
        assert len(problems) == 3
        for problem in problems:
            assert "id" in problem
            assert "problem_description" in problem
            assert "expected_solution" in problem

    def test_generate_with_invalid_json_response(self, config):
        """Test handling invalid JSON response from LLM."""
        parameters = {
            "domain": "mechanical",
            "count": 1,
            "difficulty": 5
        }
        
        strategy = EngineeringProblemsStrategy(parameters, config)
        
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent') as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.generate.return_value = "Invalid JSON response"
            
            problems = strategy.generate()
        
        assert len(problems) == 1
        problem = problems[0]
        assert problem["id"] == "mechanical_1"
        assert problem["problem_description"] == "Invalid JSON response"
        assert "error" in problem["metadata"]

    def test_generate_with_partial_json_response(self, config):
        """Test handling partial JSON in response."""
        parameters = {
            "domain": "civil",
            "count": 1,
            "difficulty": 4
        }
        
        strategy = EngineeringProblemsStrategy(parameters, config)
        
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent') as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            # Response with extra text before and after JSON
            mock_agent.generate.return_value = '''
            Here is your problem:
            {"id": "civil_001", "problem_description": "Design a bridge", "difficulty_level": 4, "domain": "civil", "expected_solution": "Use steel truss design"}
            Hope this helps!
            '''
            
            problems = strategy.generate()
        
        assert len(problems) == 1
        problem = problems[0]
        assert problem["id"] == "civil_001"
        assert problem["domain"] == "civil"


class TestDesignSystemStrategy:
    """Tests for DesignSystemStrategy."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def mock_agent(self):
        """Create mock generator agent."""
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent') as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.generate.return_value = json.dumps({
                "id": "design_1",
                "requirements": "Scalable web application",
                "constraints": "Cloud deployment",
                "design_overview": "Microservices architecture",
                "components": [
                    {"name": "API Gateway", "purpose": "Route requests", "specifications": "Handle 10k RPS"},
                    {"name": "Database", "purpose": "Store data", "specifications": "PostgreSQL cluster"}
                ],
                "interfaces": [
                    {"source": "API Gateway", "target": "Database", "description": "SQL queries"}
                ],
                "metadata": {"generated_timestamp": 1234567890}
            })
            yield mock_agent

    def test_init_with_valid_parameters(self, config):
        """Test initializing strategy with valid parameters."""
        parameters = {
            "requirements": "High-performance system",
            "constraints": "Limited budget"
        }
        
        strategy = DesignSystemStrategy(parameters, config)
        assert strategy.parameters == parameters

    def test_init_with_invalid_parameters(self, config):
        """Test initializing strategy with invalid parameters."""
        parameters = {
            "requirements": "Some requirements"
            # Missing constraints
        }
        
        with pytest.raises(GenerationError, match="Invalid parameters"):
            DesignSystemStrategy(parameters, config)

    def test_generate_system_design(self, config, mock_agent):
        """Test generating a system design."""
        parameters = {
            "requirements": "Scalable web application",
            "constraints": "Cloud deployment"
        }
        
        strategy = DesignSystemStrategy(parameters, config)
        
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent', return_value=mock_agent):
            designs = strategy.generate()
        
        assert len(designs) == 1
        design = designs[0]
        assert design["id"] == "design_1"
        assert design["requirements"] == "Scalable web application"
        assert design["constraints"] == "Cloud deployment"
        assert "components" in design
        assert "interfaces" in design

    def test_generate_with_invalid_json_response(self, config):
        """Test handling invalid JSON response from LLM."""
        parameters = {
            "requirements": "Simple system",
            "constraints": "Low complexity"
        }
        
        strategy = DesignSystemStrategy(parameters, config)
        
        with mock.patch('opensynthetics.datagen.engine.GeneratorAgent') as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.generate.return_value = "Invalid JSON response"
            
            designs = strategy.generate()
        
        assert len(designs) == 1
        design = designs[0]
        assert design["id"] == "design_1"
        assert design["design_overview"] == "Invalid JSON response"
        assert "error" in design["metadata"]


class TestEngine:
    """Tests for Engine class."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create workspace structure
        (temp_path / "datasets").mkdir()
        (temp_path / "models").mkdir()
        (temp_path / "embeddings").mkdir()
        
        # Create metadata file
        metadata = {
            "name": "test_workspace",
            "description": "Test workspace",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "version": "0.1.0",
            "tags": []
        }
        
        with open(temp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        workspace = Workspace(temp_path)
        yield workspace
        
        # Cleanup
        workspace.close()
        import shutil
        shutil.rmtree(temp_dir)

    def test_init(self, temp_workspace):
        """Test engine initialization."""
        engine = Engine(temp_workspace)
        assert engine.workspace == temp_workspace
        assert isinstance(engine.config, Config)

    def test_list_strategies(self, temp_workspace):
        """Test listing available strategies."""
        engine = Engine(temp_workspace)
        strategies = engine.list_strategies()
        
        assert "engineering_problems" in strategies
        assert "system_design" in strategies
        
        eng_strategy = strategies["engineering_problems"]
        assert "name" in eng_strategy
        assert "description" in eng_strategy
        assert "schema" in eng_strategy

    def test_generate_with_valid_parameters(self, temp_workspace):
        """Test data generation with valid parameters."""
        engine = Engine(temp_workspace)
        
        # Create dataset first
        temp_workspace.create_dataset("test_dataset")
        
        parameters = {
            "domain": "mechanical",
            "count": 2,
            "difficulty": 5
        }
        
        with mock.patch.object(EngineeringProblemsStrategy, 'generate') as mock_generate:
            mock_generate.return_value = [
                {"id": "prob1", "problem_description": "Problem 1"},
                {"id": "prob2", "problem_description": "Problem 2"}
            ]
            
            result = engine.generate(
                strategy="engineering_problems",
                parameters=parameters,
                output_dataset="test_dataset"
            )
        
        assert result["count"] == 2
        assert result["strategy"] == "engineering_problems"
        assert result["output_dataset"] == "test_dataset"
        assert "sample_items" in result

    def test_generate_with_invalid_strategy(self, temp_workspace):
        """Test generation with invalid strategy."""
        engine = Engine(temp_workspace)
        
        with pytest.raises(GenerationError, match="Invalid parameters"):
            engine.generate(
                strategy="invalid_strategy",
                parameters={},
                output_dataset="test_dataset"
            )

    def test_generate_with_invalid_parameters(self, temp_workspace):
        """Test generation with invalid parameters."""
        engine = Engine(temp_workspace)
        
        # Invalid parameters for engineering_problems strategy
        parameters = {
            "domain": "invalid_domain",
            "count": 2,
            "difficulty": 5
        }
        
        with pytest.raises(GenerationError, match="Invalid parameters"):
            engine.generate(
                strategy="engineering_problems",
                parameters=parameters,
                output_dataset="test_dataset"
            )

    def test_generate_creates_new_dataset(self, temp_workspace):
        """Test that generation creates a new dataset if it doesn't exist."""
        engine = Engine(temp_workspace)
        
        # Create dataset first to avoid dataset creation failure
        temp_workspace.create_dataset("new_dataset")
        
        parameters = {
            "domain": "electrical",
            "count": 1,
            "difficulty": 3
        }
        
        with mock.patch.object(EngineeringProblemsStrategy, 'generate') as mock_generate:
            mock_generate.return_value = [
                {"id": "prob1", "problem_description": "Test problem"}
            ]
            
            result = engine.generate(
                strategy="engineering_problems",
                parameters=parameters,
                output_dataset="new_dataset"
            )
        
        # Check that dataset exists
        datasets = temp_workspace.list_datasets()
        dataset_names = [d["name"] for d in datasets]
        assert "new_dataset" in dataset_names

    def test_generate_uses_existing_dataset(self, temp_workspace):
        """Test that generation uses existing dataset if it exists."""
        engine = Engine(temp_workspace)
        
        # Create dataset first
        temp_workspace.create_dataset("existing_dataset")
        
        parameters = {
            "domain": "civil",
            "count": 1,
            "difficulty": 4
        }
        
        with mock.patch.object(EngineeringProblemsStrategy, 'generate') as mock_generate:
            mock_generate.return_value = [
                {"id": "prob1", "problem_description": "Test problem"}
            ]
            
            result = engine.generate(
                strategy="engineering_problems",
                parameters=parameters,
                output_dataset="existing_dataset"
            )
        
        assert result["output_dataset"] == "existing_dataset"

    def test_generate_with_system_design_strategy(self, temp_workspace):
        """Test generation with system design strategy."""
        engine = Engine(temp_workspace)
        
        # Create dataset first
        temp_workspace.create_dataset("design_dataset")
        
        parameters = {
            "requirements": "Scalable system",
            "constraints": "Cloud deployment"
        }
        
        with mock.patch.object(DesignSystemStrategy, 'generate') as mock_generate:
            mock_generate.return_value = [
                {
                    "id": "design1",
                    "requirements": "Scalable system",
                    "design_overview": "Microservices architecture"
                }
            ]
            
            result = engine.generate(
                strategy="system_design",
                parameters=parameters,
                output_dataset="design_dataset"
            )
        
        assert result["count"] == 1
        assert result["strategy"] == "system_design"

    def test_generate_handles_strategy_exception(self, temp_workspace):
        """Test that engine handles strategy generation exceptions."""
        engine = Engine(temp_workspace)
        
        parameters = {
            "domain": "mechanical",
            "count": 1,
            "difficulty": 5
        }
        
        with mock.patch.object(EngineeringProblemsStrategy, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("Strategy failed")
            
            with pytest.raises(GenerationError, match="Data generation failed"):
                engine.generate(
                    strategy="engineering_problems",
                    parameters=parameters,
                    output_dataset="test_dataset"
                )

    def test_generate_handles_dataset_exception(self, temp_workspace):
        """Test that engine handles dataset creation/storage exceptions."""
        engine = Engine(temp_workspace)
        
        parameters = {
            "domain": "mechanical",
            "count": 1,
            "difficulty": 5
        }
        
        with mock.patch.object(EngineeringProblemsStrategy, 'generate') as mock_generate:
            mock_generate.return_value = [{"id": "prob1"}]
            
            with mock.patch.object(temp_workspace, 'create_dataset') as mock_create:
                mock_create.side_effect = Exception("Dataset creation failed")
                
                with pytest.raises(GenerationError, match="Failed to store generated data"):
                    engine.generate(
                        strategy="engineering_problems",
                        parameters=parameters,
                        output_dataset="test_dataset"
                    )

    def test_generate_includes_sample_items(self, temp_workspace):
        """Test that generation result includes sample items."""
        engine = Engine(temp_workspace)
        
        # Create dataset first
        temp_workspace.create_dataset("sample_dataset")
        
        parameters = {
            "domain": "software",
            "count": 5,
            "difficulty": 6
        }
        
        generated_data = [
            {"id": f"prob{i}", "problem_description": f"Problem {i}"}
            for i in range(1, 6)
        ]
        
        with mock.patch.object(EngineeringProblemsStrategy, 'generate') as mock_generate:
            mock_generate.return_value = generated_data
            
            result = engine.generate(
                strategy="engineering_problems",
                parameters=parameters,
                output_dataset="sample_dataset"
            )
        
        assert "sample_items" in result
        assert len(result["sample_items"]) == 3  # Should include first 3 items
        assert result["sample_items"][0]["id"] == "prob1"
        assert result["sample_items"][2]["id"] == "prob3" 