"""Data validation utilities for OpenSynthetics."""

import json
import os
import glob
from typing import Any, Dict, List, Optional, Union, Tuple

import jsonschema
from loguru import logger

from opensynthetics.core.exceptions import ValidationError


class DataValidator:
    """Validator for generated data."""

    def __init__(self) -> None:
        """Initialize validator."""
        self._schemas: Dict[str, Dict[str, Any]] = {}
        
    def register_schema(self, schema_name: str, schema: Dict[str, Any]) -> None:
        """Register a JSON schema.

        Args:
            schema_name: Schema name
            schema: JSON schema
        """
        self._schemas[schema_name] = schema
        logger.debug(f"Registered schema: {schema_name}")
        
    def load_schema_from_file(self, schema_name: str, schema_path: str) -> None:
        """Load a JSON schema from file.

        Args:
            schema_name: Schema name
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
            
    def load_schemas_from_directory(self, directory_path: str, file_extension: str = ".json") -> None:
        """Load all JSON schemas from a directory.
        Schema names will be derived from filenames (without extension).

        Args:
            directory_path: Path to the directory containing schema files.
            file_extension: The file extension of schema files (default: .json).

        Raises:
            ValidationError: If directory is invalid or schema loading fails for any file.
        """
        if not os.path.isdir(directory_path):
            msg = f"Invalid directory path: {directory_path}"
            logger.error(msg)
            raise ValidationError(msg)

        schema_files = glob.glob(os.path.join(directory_path, f"*{file_extension}"))
        if not schema_files:
            logger.warning(f"No schema files found in directory: {directory_path} with extension {file_extension}")
            return

        for schema_file_path in schema_files:
            schema_name = os.path.splitext(os.path.basename(schema_file_path))[0]
            try:
                self.load_schema_from_file(schema_name, schema_file_path)
                logger.info(f"Successfully loaded schema '{schema_name}' from '{schema_file_path}'")
            except ValidationError as e:
                # Log and continue, or re-raise to stop on first error?
                # For now, log and continue to try loading other schemas.
                logger.error(f"Failed to load schema '{schema_name}' from '{schema_file_path}': {e}")
                # Optionally, collect errors and raise a summary error at the end.

    def validate(self, data: Dict[str, Any], schema_name: str) -> Tuple[bool, Optional[str]]:
        """Validate data against a schema.

        Args:
            data: Data to validate
            schema_name: Schema name

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
            path = "/".join(map(str, e.path))
            error_message = f"Validation error at path '{path}': {e.message}" if e.path else e.message
            return False, error_message
            
    def validate_batch(self, data_list: List[Dict[str, Any]], schema_name: str) -> List[Tuple[bool, Optional[str]]]:
        """Validate a batch of data against a schema.

        Args:
            data_list: List of data to validate
            schema_name: Schema name

        Returns:
            List[Tuple[bool, Optional[str]]]: List of (valid, error_message)
        """
        return [self.validate(data, schema_name) for data in data_list]
        
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """Get a schema by name.

        Args:
            schema_name: Schema name

        Returns:
            Dict[str, Any]: Schema

        Raises:
            ValidationError: If schema not found
        """
        if schema_name not in self._schemas:
            raise ValidationError(f"Schema not found: {schema_name}")
        return self._schemas[schema_name]

    def list_schemas(self) -> List[str]:
        """List the names of all registered schemas.

        Returns:
            List[str]: A list of schema names.
        """
        return list(self._schemas.keys())

    def unregister_schema(self, schema_name: str) -> None:
        """Unregister a schema.

        Args:
            schema_name: The name of the schema to unregister.

        Raises:
            ValidationError: If the schema is not found.
        """
        if schema_name not in self._schemas:
            raise ValidationError(f"Schema not found: {schema_name}")
        del self._schemas[schema_name]
        logger.debug(f"Unregistered schema: {schema_name}") 