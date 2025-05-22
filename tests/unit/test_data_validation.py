"""Unit tests for data validation module."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from opensynthetics.core.exceptions import ValidationError as OpenSyntheticsValidationError
from opensynthetics.data_ops.validation import DataValidator


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a test validator."""
        return DataValidator()

    @pytest.fixture
    def sample_schema(self):
        """Create a sample JSON schema."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age"]
        }

    @pytest.fixture
    def temp_schema_file(self, sample_schema):
        """Create a temporary schema file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_schema, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)

    @pytest.fixture
    def temp_schema_dir(self, sample_schema):
        """Create a temporary directory with schema files."""
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        
        # Create multiple schema files
        schemas = {
            "user.json": sample_schema,
            "product.json": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "price": {"type": "number", "minimum": 0}
                },
                "required": ["id", "price"]
            },
            "order.json": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["order_id"]
            }
        }
        
        for filename, schema in schemas.items():
            schema_file = temp_dir_path / filename
            with open(schema_file, 'w') as f:
                json.dump(schema, f)
        
        yield temp_dir_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_register_schema(self, validator, sample_schema):
        """Test registering a schema."""
        validator.register_schema("user", sample_schema)
        
        assert "user" in validator.list_schemas()
        retrieved_schema = validator.get_schema("user")
        assert retrieved_schema == sample_schema

    def test_register_schema_overwrites_existing(self, validator, sample_schema):
        """Test that registering a schema overwrites existing one."""
        # Register initial schema
        validator.register_schema("user", sample_schema)
        
        # Register new schema with same name
        new_schema = {"type": "string"}
        validator.register_schema("user", new_schema)
        
        retrieved_schema = validator.get_schema("user")
        assert retrieved_schema == new_schema
        assert retrieved_schema != sample_schema

    def test_load_schema_from_file(self, validator, temp_schema_file, sample_schema):
        """Test loading a schema from file."""
        validator.load_schema_from_file("user", temp_schema_file)
        
        assert "user" in validator.list_schemas()
        retrieved_schema = validator.get_schema("user")
        assert retrieved_schema == sample_schema

    def test_load_schema_from_file_nonexistent(self, validator):
        """Test loading schema from non-existent file."""
        with pytest.raises(OpenSyntheticsValidationError, match="Failed to load schema"):
            validator.load_schema_from_file("user", "/nonexistent/file.json")

    def test_load_schema_from_file_invalid_json(self, validator):
        """Test loading schema from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(OpenSyntheticsValidationError, match="Failed to load schema"):
                validator.load_schema_from_file("user", temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_schemas_from_directory(self, validator, temp_schema_dir):
        """Test loading schemas from directory."""
        validator.load_schemas_from_directory(str(temp_schema_dir))
        
        schemas = validator.list_schemas()
        assert "user" in schemas
        assert "product" in schemas
        assert "order" in schemas
        assert len(schemas) == 3

    def test_load_schemas_from_directory_with_extension(self, validator, temp_schema_dir):
        """Test loading schemas from directory with specific extension."""
        # Create a file with different extension
        other_file = temp_schema_dir / "config.yaml"
        other_file.write_text("not a json schema")
        
        validator.load_schemas_from_directory(str(temp_schema_dir), file_extension=".json")
        
        schemas = validator.list_schemas()
        assert "config" not in schemas  # Should not load .yaml file
        assert len(schemas) == 3  # Only .json files

    def test_load_schemas_from_directory_invalid_path(self, validator):
        """Test loading schemas from invalid directory."""
        with pytest.raises(OpenSyntheticsValidationError, match="Invalid directory path"):
            validator.load_schemas_from_directory("/nonexistent/directory")

    def test_load_schemas_from_directory_no_files(self, validator):
        """Test loading schemas from directory with no schema files."""
        temp_dir = tempfile.mkdtemp()
        try:
            # This should not raise an error, just log a warning
            validator.load_schemas_from_directory(temp_dir)
            assert len(validator.list_schemas()) == 0
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_validate_valid_data(self, validator, sample_schema):
        """Test validating valid data."""
        validator.register_schema("user", sample_schema)
        
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        is_valid, error_message = validator.validate(valid_data, "user")
        assert is_valid is True
        assert error_message is None

    def test_validate_invalid_data_missing_required(self, validator, sample_schema):
        """Test validating data with missing required field."""
        validator.register_schema("user", sample_schema)
        
        invalid_data = {
            "name": "John Doe"
            # Missing required 'age' field
        }
        
        is_valid, error_message = validator.validate(invalid_data, "user")
        assert is_valid is False
        assert error_message is not None
        assert "age" in error_message

    def test_validate_invalid_data_wrong_type(self, validator, sample_schema):
        """Test validating data with wrong type."""
        validator.register_schema("user", sample_schema)
        
        invalid_data = {
            "name": "John Doe",
            "age": "thirty"  # Should be integer
        }
        
        is_valid, error_message = validator.validate(invalid_data, "user")
        assert is_valid is False
        assert error_message is not None
        assert "age" in error_message

    def test_validate_invalid_data_constraint_violation(self, validator, sample_schema):
        """Test validating data that violates constraints."""
        validator.register_schema("user", sample_schema)
        
        invalid_data = {
            "name": "John Doe",
            "age": -5  # Violates minimum constraint
        }
        
        is_valid, error_message = validator.validate(invalid_data, "user")
        assert is_valid is False
        assert error_message is not None
        assert "age" in error_message

    def test_validate_with_path_in_error(self, validator):
        """Test that validation errors include path information."""
        nested_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "age": {"type": "integer", "minimum": 0}
                            },
                            "required": ["age"]
                        }
                    },
                    "required": ["profile"]
                }
            },
            "required": ["user"]
        }
        
        validator.register_schema("nested", nested_schema)
        
        invalid_data = {
            "user": {
                "profile": {
                    "age": -1  # Invalid age
                }
            }
        }
        
        is_valid, error_message = validator.validate(invalid_data, "nested")
        assert is_valid is False
        assert "user/profile/age" in error_message

    def test_validate_schema_not_found(self, validator):
        """Test validating with non-existent schema."""
        data = {"name": "John"}
        
        with pytest.raises(OpenSyntheticsValidationError, match="Schema not found: nonexistent"):
            validator.validate(data, "nonexistent")

    def test_validate_batch_all_valid(self, validator, sample_schema):
        """Test batch validation with all valid data."""
        validator.register_schema("user", sample_schema)
        
        data_list = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
            {"name": "Bob", "age": 35}
        ]
        
        results = validator.validate_batch(data_list, "user")
        
        assert len(results) == 3
        for is_valid, error_message in results:
            assert is_valid is True
            assert error_message is None

    def test_validate_batch_mixed_validity(self, validator, sample_schema):
        """Test batch validation with mixed valid/invalid data."""
        validator.register_schema("user", sample_schema)
        
        data_list = [
            {"name": "John", "age": 30},      # Valid
            {"name": "Jane"},                 # Invalid - missing age
            {"name": "Bob", "age": "old"}     # Invalid - wrong type
        ]
        
        results = validator.validate_batch(data_list, "user")
        
        assert len(results) == 3
        assert results[0] == (True, None)
        assert results[1][0] is False
        assert results[2][0] is False

    def test_validate_batch_empty_list(self, validator, sample_schema):
        """Test batch validation with empty list."""
        validator.register_schema("user", sample_schema)
        
        results = validator.validate_batch([], "user")
        assert results == []

    def test_get_schema_existing(self, validator, sample_schema):
        """Test getting an existing schema."""
        validator.register_schema("user", sample_schema)
        
        retrieved_schema = validator.get_schema("user")
        assert retrieved_schema == sample_schema

    def test_get_schema_nonexistent(self, validator):
        """Test getting a non-existent schema."""
        with pytest.raises(OpenSyntheticsValidationError, match="Schema not found: nonexistent"):
            validator.get_schema("nonexistent")

    def test_list_schemas_empty(self, validator):
        """Test listing schemas when none are registered."""
        schemas = validator.list_schemas()
        assert schemas == []

    def test_list_schemas_multiple(self, validator, sample_schema):
        """Test listing multiple schemas."""
        validator.register_schema("user", sample_schema)
        validator.register_schema("product", {"type": "object"})
        validator.register_schema("order", {"type": "array"})
        
        schemas = validator.list_schemas()
        assert set(schemas) == {"user", "product", "order"}

    def test_unregister_schema_existing(self, validator, sample_schema):
        """Test unregistering an existing schema."""
        validator.register_schema("user", sample_schema)
        assert "user" in validator.list_schemas()
        
        validator.unregister_schema("user")
        assert "user" not in validator.list_schemas()

    def test_unregister_schema_nonexistent(self, validator):
        """Test unregistering a non-existent schema."""
        with pytest.raises(OpenSyntheticsValidationError, match="Schema not found: nonexistent"):
            validator.unregister_schema("nonexistent")

    def test_multiple_operations(self, validator, sample_schema, temp_schema_file):
        """Test multiple operations in sequence."""
        # Register a schema
        validator.register_schema("user1", sample_schema)
        
        # Load from file
        validator.load_schema_from_file("user2", temp_schema_file)
        
        # Validate some data
        data = {"name": "John", "age": 30}
        is_valid1, _ = validator.validate(data, "user1")
        is_valid2, _ = validator.validate(data, "user2")
        
        assert is_valid1 is True
        assert is_valid2 is True
        assert len(validator.list_schemas()) == 2
        
        # Unregister one schema
        validator.unregister_schema("user1")
        assert len(validator.list_schemas()) == 1
        assert "user2" in validator.list_schemas()

    def test_schema_validation_with_complex_data(self, validator):
        """Test validation with complex nested data structures."""
        complex_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "scores": {
                            "type": "object",
                            "patternProperties": {
                                "^[a-z]+$": {"type": "number", "minimum": 0, "maximum": 100}
                            }
                        }
                    },
                    "required": ["tags"]
                }
            },
            "required": ["id", "metadata"]
        }
        
        validator.register_schema("complex", complex_schema)
        
        # Valid complex data
        valid_data = {
            "id": "item123",
            "metadata": {
                "tags": ["important", "urgent"],
                "scores": {
                    "quality": 95,
                    "relevance": 87
                }
            }
        }
        
        is_valid, error_message = validator.validate(valid_data, "complex")
        assert is_valid is True
        assert error_message is None
        
        # Invalid complex data
        invalid_data = {
            "id": "item123",
            "metadata": {
                "tags": "not an array",  # Should be array
                "scores": {
                    "quality": 150  # Exceeds maximum
                }
            }
        }
        
        is_valid, error_message = validator.validate(invalid_data, "complex")
        assert is_valid is False
        assert error_message is not None 