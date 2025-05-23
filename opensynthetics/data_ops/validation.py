"""Advanced data validation utilities for OpenSynthetics."""

import json
import os
import glob
import statistics
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import jsonschema
from loguru import logger

from opensynthetics.core.exceptions import ValidationError


class DataQualityMetrics:
    """Calculate comprehensive data quality metrics."""
    
    @staticmethod
    def calculate_completeness(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data completeness metrics."""
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        
        return {
            "overall_completeness": (total_cells - missing_cells) / total_cells,
            "column_completeness": {
                col: (len(data) - data[col].isnull().sum()) / len(data)
                for col in data.columns
            }
        }
    
    @staticmethod
    def calculate_uniqueness(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data uniqueness metrics."""
        return {
            col: data[col].nunique() / len(data) if len(data) > 0 else 0
            for col in data.columns
        }
    
    @staticmethod
    def calculate_consistency(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data consistency metrics."""
        consistency_metrics = {}
        
        for col in data.columns:
            if data[col].dtype in ['object', 'string']:
                # String consistency
                avg_length = data[col].str.len().mean() if not data[col].isnull().all() else 0
                std_length = data[col].str.len().std() if not data[col].isnull().all() else 0
                consistency_metrics[col] = {
                    "avg_length": avg_length,
                    "length_std": std_length,
                    "format_consistency": 1.0 - (std_length / avg_length) if avg_length > 0 else 1.0
                }
            elif np.issubdtype(data[col].dtype, np.number):
                # Numeric consistency
                if not data[col].isnull().all():
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    consistency_metrics[col] = {
                        "mean": mean_val,
                        "std": std_val,
                        "coefficient_of_variation": std_val / mean_val if mean_val != 0 else 0
                    }
        
        return consistency_metrics
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, method: str = "iqr") -> Dict[str, List[int]]:
        """Detect outliers in numeric columns."""
        outliers = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
            elif method == "zscore":
                z_scores = np.abs(statistics.zscore(data[col].dropna()))
                outlier_indices = data[z_scores > 3].index.tolist()
            else:
                outlier_indices = []
            
            outliers[col] = outlier_indices
        
        return outliers


class StatisticalValidator:
    """Advanced statistical validation for synthetic data."""
    
    def __init__(self):
        self.tolerance = 0.1  # 10% tolerance for statistical tests
    
    def validate_distribution(self, data: pd.Series, expected_distribution: str, **params) -> Tuple[bool, str]:
        """Validate if data follows expected statistical distribution."""
        try:
            if expected_distribution == "normal":
                from scipy import stats
                statistic, p_value = stats.normaltest(data.dropna())
                is_valid = p_value > 0.05  # Null hypothesis: data is normally distributed
                message = f"Normal distribution test: p-value={p_value:.4f}"
                
            elif expected_distribution == "uniform":
                from scipy import stats
                statistic, p_value = stats.kstest(data.dropna(), 'uniform')
                is_valid = p_value > 0.05
                message = f"Uniform distribution test: p-value={p_value:.4f}"
                
            else:
                is_valid = False
                message = f"Unknown distribution: {expected_distribution}"
                
            return is_valid, message
            
        except Exception as e:
            return False, f"Distribution validation failed: {str(e)}"
    
    def validate_correlation(self, data: pd.DataFrame, expected_correlations: Dict[str, Dict[str, float]]) -> Tuple[bool, str]:
        """Validate correlations between columns."""
        try:
            correlation_matrix = data.corr()
            
            for col1, correlations in expected_correlations.items():
                for col2, expected_corr in correlations.items():
                    if col1 in correlation_matrix.columns and col2 in correlation_matrix.columns:
                        actual_corr = correlation_matrix.loc[col1, col2]
                        if abs(actual_corr - expected_corr) > self.tolerance:
                            return False, f"Correlation between {col1} and {col2}: expected {expected_corr}, got {actual_corr}"
            
            return True, "All correlations within tolerance"
            
        except Exception as e:
            return False, f"Correlation validation failed: {str(e)}"


class MLAnomalyDetector:
    """Machine learning-based anomaly detection for data validation."""
    
    def __init__(self):
        self.models = {}
    
    def detect_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Prepare numeric data
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return {"anomalies": [], "anomaly_score": 0.0, "message": "No numeric data for anomaly detection"}
            
            # Handle missing values
            numeric_data = numeric_data.fillna(numeric_data.mean())
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.score_samples(scaled_data)
            
            # Get anomaly indices
            anomaly_indices = np.where(anomaly_labels == -1)[0].tolist()
            
            return {
                "anomalies": anomaly_indices,
                "anomaly_scores": anomaly_scores.tolist(),
                "anomaly_count": len(anomaly_indices),
                "anomaly_percentage": len(anomaly_indices) / len(data) * 100,
                "message": f"Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(data)*100:.2f}%)"
            }
            
        except ImportError:
            return {"anomalies": [], "message": "scikit-learn not available for anomaly detection"}
        except Exception as e:
            return {"anomalies": [], "message": f"Anomaly detection failed: {str(e)}"}


class AdvancedDataValidator:
    """Advanced data validator with comprehensive validation capabilities."""

    def __init__(self) -> None:
        """Initialize validator."""
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self.quality_metrics = DataQualityMetrics()
        self.statistical_validator = StatisticalValidator()
        self.anomaly_detector = MLAnomalyDetector()
        
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
                logger.error(f"Failed to load schema '{schema_name}' from '{schema_file_path}': {e}")

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
    
    def comprehensive_validation(self, data: Union[pd.DataFrame, List[Dict[str, Any]]], 
                               schema_name: Optional[str] = None,
                               include_quality_metrics: bool = True,
                               include_statistical_tests: bool = True,
                               include_anomaly_detection: bool = True) -> Dict[str, Any]:
        """Perform comprehensive validation including quality metrics, statistical tests, and anomaly detection.
        
        Args:
            data: Data to validate (DataFrame or list of dictionaries)
            schema_name: Optional schema name for JSON schema validation
            include_quality_metrics: Whether to include data quality metrics
            include_statistical_tests: Whether to include statistical validation
            include_anomaly_detection: Whether to include anomaly detection
            
        Returns:
            Dict[str, Any]: Comprehensive validation report
        """
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": None,
            "schema_validation": None,
            "quality_metrics": None,
            "statistical_validation": None,
            "anomaly_detection": None,
            "overall_score": 0.0
        }
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        validation_report["data_shape"] = {"rows": len(df), "columns": len(df.columns)}
        
        scores = []
        
        # Schema validation
        if schema_name and isinstance(data, list):
            try:
                schema_results = self.validate_batch(data, schema_name)
                valid_count = sum(1 for valid, _ in schema_results if valid)
                schema_score = valid_count / len(data)
                validation_report["schema_validation"] = {
                    "valid_records": valid_count,
                    "total_records": len(data),
                    "validation_score": schema_score,
                    "errors": [error for valid, error in schema_results if not valid]
                }
                scores.append(schema_score)
            except Exception as e:
                validation_report["schema_validation"] = {"error": str(e)}
        
        # Quality metrics
        if include_quality_metrics:
            try:
                completeness = self.quality_metrics.calculate_completeness(df)
                uniqueness = self.quality_metrics.calculate_uniqueness(df)
                consistency = self.quality_metrics.calculate_consistency(df)
                outliers = self.quality_metrics.detect_outliers(df)
                
                quality_score = completeness["overall_completeness"]
                
                validation_report["quality_metrics"] = {
                    "completeness": completeness,
                    "uniqueness": uniqueness,
                    "consistency": consistency,
                    "outliers": outliers,
                    "quality_score": quality_score
                }
                scores.append(quality_score)
            except Exception as e:
                validation_report["quality_metrics"] = {"error": str(e)}
        
        # Statistical validation
        if include_statistical_tests:
            try:
                statistical_results = {}
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    if not df[col].isnull().all():
                        # Test for normality
                        is_normal, normal_msg = self.statistical_validator.validate_distribution(
                            df[col], "normal"
                        )
                        statistical_results[col] = {
                            "normality_test": {"valid": is_normal, "message": normal_msg}
                        }
                
                validation_report["statistical_validation"] = statistical_results
            except Exception as e:
                validation_report["statistical_validation"] = {"error": str(e)}
        
        # Anomaly detection
        if include_anomaly_detection:
            try:
                anomaly_results = self.anomaly_detector.detect_anomalies(df)
                anomaly_score = 1.0 - (anomaly_results.get("anomaly_percentage", 0) / 100)
                anomaly_results["anomaly_score"] = anomaly_score
                validation_report["anomaly_detection"] = anomaly_results
                scores.append(anomaly_score)
            except Exception as e:
                validation_report["anomaly_detection"] = {"error": str(e)}
        
        # Calculate overall score
        if scores:
            validation_report["overall_score"] = sum(scores) / len(scores)
        
        return validation_report
        
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


# Backward compatibility
DataValidator = AdvancedDataValidator 