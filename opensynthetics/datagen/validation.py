"""Data validation utilities for OpenSynthetics."""

import json
from typing import Any, Dict, List, Optional, Union, Tuple

import jsonschema
from loguru import logger

from opensynthetics.core.exceptions import ValidationError


class DataValidator:
    """Validator for generated data.
    
    This class provides utilities for validating generated data against JSON schemas.
    It supports registering schemas, loading schemas from files, and validating
    data against registered schemas.
    """

    def __init__(self) -> None:
        """Initialize validator with an empty schema registry."""
        self._schemas: Dict[str, Dict[str, Any]] = {}
        
    def register_schema(self, schema_name: str, schema: Dict[str, Any]) -> None:
        """Register a JSON schema.

        Args:
            schema_name: Schema name for reference
            schema: JSON schema definition
        """
        self._schemas[schema_name] = schema
        logger.debug(f"Registered schema: {schema_name}")
        
    def load_schema_from_file(self, schema_name: str, schema_path: str) -> None:
        """Load a JSON schema from file.

        Args:
            schema_name: Schema name for reference
            schema_path: Path to schema file

        Raises:
            ValidationError: If schema loading fails
        """
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
            self.register_schema(schema_name, schema)
        except Exception as e:
            logger.error(f"Error loading schema from {schema_path}: {e}")
            raise ValidationError(f"Failed to load schema: {e}")
            
    def validate(self, data: Dict[str, Any], schema_name: str) -> Tuple[bool, Optional[str]]:
        """Validate data against a schema.

        Args:
            data: Data to validate
            schema_name: Schema name to validate against

        Returns:
            Tuple[bool, Optional[str]]: (valid, error_message)
            
        Raises:
            ValidationError: If schema not found
        """
        if schema_name not in self._schemas:
            raise ValidationError(f"Schema not found: {schema_name}")
            
        try:
            jsonschema.validate(data, self._schemas[schema_name])
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)
            
    def validate_batch(self, data_list: List[Dict[str, Any]], schema_name: str) -> List[Tuple[bool, Optional[str]]]:
        """Validate a batch of data against a schema.

        Args:
            data_list: List of data items to validate
            schema_name: Schema name to validate against

        Returns:
            List[Tuple[bool, Optional[str]]]: List of (valid, error_message) for each item
            
        Raises:
            ValidationError: If schema not found
        """
        return [self.validate(data, schema_name) for data in data_list]
        
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """Get a schema by name.

        Args:
            schema_name: Schema name

        Returns:
            Dict[str, Any]: Schema definition

        Raises:
            ValidationError: If schema not found
        """
        if schema_name not in self._schemas:
            raise ValidationError(f"Schema not found: {schema_name}")
        return self._schemas[schema_name]