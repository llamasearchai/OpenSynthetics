"""Comprehensive synthetic dataset creation module for OpenSynthetics."""

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Literal
import warnings

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    logger.warning("Faker not available. Some synthetic data features will be limited.")

from opensynthetics.core.exceptions import GenerationError
from opensynthetics.data_ops.export_utils import DataExporter, ExportConfig
from opensynthetics.training_eval.benchmark import SyntheticDatasetBenchmark, BenchmarkConfig


class DataDistribution(BaseModel):
    """Configuration for data distribution."""
    
    distribution_type: Literal["normal", "uniform", "exponential", "gamma", "beta", "lognormal"] = Field(
        "normal", description="Type of distribution"
    )
    parameters: Dict[str, float] = Field(default_factory=dict, description="Distribution parameters")
    
    # Common parameters
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    mean: Optional[float] = Field(None, description="Mean value")
    std: Optional[float] = Field(None, description="Standard deviation")
    
    # Category-specific parameters
    categories: Optional[List[str]] = Field(None, description="Categories for categorical data")
    weights: Optional[List[float]] = Field(None, description="Weights for categorical data")


class ColumnSchema(BaseModel):
    """Schema definition for a synthetic data column."""
    
    name: str = Field(..., description="Column name")
    data_type: Literal["numeric", "categorical", "datetime", "text", "boolean", "id"] = Field(
        "numeric", description="Data type"
    )
    distribution: DataDistribution = Field(default_factory=DataDistribution, description="Data distribution")
    
    # Constraints
    nullable: bool = Field(False, description="Allow null values")
    null_probability: float = Field(0.0, description="Probability of null values", ge=0.0, le=1.0)
    unique: bool = Field(False, description="Ensure unique values")
    
    # Relationships
    correlation_with: Optional[str] = Field(None, description="Column to correlate with")
    correlation_strength: float = Field(0.0, description="Correlation strength", ge=-1.0, le=1.0)
    
    # Text-specific parameters
    text_pattern: Optional[str] = Field(None, description="Pattern for text generation")
    text_length_range: Tuple[int, int] = Field((5, 50), description="Text length range")
    
    # Datetime-specific parameters
    datetime_start: Optional[datetime] = Field(None, description="Start datetime")
    datetime_end: Optional[datetime] = Field(None, description="End datetime")
    
    # ID-specific parameters
    id_prefix: Optional[str] = Field(None, description="ID prefix")
    id_format: Optional[str] = Field(None, description="ID format pattern")


class SyntheticDatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation."""
    
    # Dataset structure
    num_rows: int = Field(1000, description="Number of rows to generate", ge=1)
    columns: List[ColumnSchema] = Field(..., description="Column schemas")
    
    # Quality parameters
    consistency_level: float = Field(0.9, description="Data consistency level", ge=0.0, le=1.0)
    completeness_level: float = Field(0.95, description="Data completeness level", ge=0.0, le=1.0)
    
    # Advanced features
    add_outliers: bool = Field(False, description="Add outliers to numeric data")
    outlier_probability: float = Field(0.05, description="Probability of outliers", ge=0.0, le=1.0)
    add_noise: bool = Field(False, description="Add noise to data")
    noise_level: float = Field(0.1, description="Noise level", ge=0.0, le=1.0)
    
    # Temporal patterns
    add_temporal_patterns: bool = Field(False, description="Add temporal patterns")
    seasonality: Optional[str] = Field(None, description="Seasonality pattern: daily, weekly, monthly, yearly")
    trend: Optional[str] = Field(None, description="Trend pattern: increasing, decreasing, cyclical")
    
    # Business rules
    business_rules: List[str] = Field(default_factory=list, description="Business rules to apply")
    constraints: List[str] = Field(default_factory=list, description="Data constraints")
    
    # Output configuration
    export_config: Optional[ExportConfig] = Field(None, description="Export configuration")
    include_metadata: bool = Field(True, description="Include generation metadata")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SyntheticDataGenerator:
    """Advanced synthetic data generator."""
    
    def __init__(self, config: SyntheticDatasetConfig) -> None:
        """Initialize the generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config
        
        # Set random seed for reproducibility
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)
        
        # Initialize faker if available
        if FAKER_AVAILABLE:
            self.fake = faker.Faker()
            if config.seed is not None:
                self.fake.seed_instance(config.seed)
        else:
            self.fake = None
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate synthetic dataset.
        
        Returns:
            Generated pandas DataFrame
            
        Raises:
            GenerationError: If generation fails
        """
        logger.info(f"Generating synthetic dataset with {self.config.num_rows} rows and {len(self.config.columns)} columns")
        
        try:
            # Generate base data
            data = {}
            
            # First pass: generate independent columns
            for column in self.config.columns:
                if column.correlation_with is None:
                    data[column.name] = self._generate_column_data(column)
            
            # Second pass: generate correlated columns
            for column in self.config.columns:
                if column.correlation_with is not None and column.correlation_with in data:
                    data[column.name] = self._generate_correlated_column(
                        column, data[column.correlation_with]
                    )
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Apply post-processing
            df = self._apply_post_processing(df)
            
            # Apply business rules
            df = self._apply_business_rules(df)
            
            # Add temporal patterns if requested
            if self.config.add_temporal_patterns:
                df = self._add_temporal_patterns(df)
            
            # Validate dataset
            self._validate_dataset(df)
            
            logger.info(f"Successfully generated dataset with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            raise GenerationError(f"Failed to generate synthetic dataset: {e}")
    
    def _generate_column_data(self, column: ColumnSchema) -> np.ndarray:
        """Generate data for a single column."""
        
        if column.data_type == "numeric":
            return self._generate_numeric_data(column)
        elif column.data_type == "categorical":
            return self._generate_categorical_data(column)
        elif column.data_type == "datetime":
            return self._generate_datetime_data(column)
        elif column.data_type == "text":
            return self._generate_text_data(column)
        elif column.data_type == "boolean":
            return self._generate_boolean_data(column)
        elif column.data_type == "id":
            return self._generate_id_data(column)
        else:
            raise GenerationError(f"Unsupported data type: {column.data_type}")
    
    def _generate_numeric_data(self, column: ColumnSchema) -> np.ndarray:
        """Generate numeric data."""
        dist = column.distribution
        size = self.config.num_rows
        
        if dist.distribution_type == "normal":
            mean = dist.mean or 0.0
            std = dist.std or 1.0
            data = np.random.normal(mean, std, size)
        
        elif dist.distribution_type == "uniform":
            low = dist.min_value or 0.0
            high = dist.max_value or 1.0
            data = np.random.uniform(low, high, size)
        
        elif dist.distribution_type == "exponential":
            scale = dist.parameters.get("scale", 1.0)
            data = np.random.exponential(scale, size)
        
        elif dist.distribution_type == "gamma":
            shape = dist.parameters.get("shape", 2.0)
            scale = dist.parameters.get("scale", 1.0)
            data = np.random.gamma(shape, scale, size)
        
        elif dist.distribution_type == "beta":
            alpha = dist.parameters.get("alpha", 2.0)
            beta = dist.parameters.get("beta", 2.0)
            data = np.random.beta(alpha, beta, size)
        
        elif dist.distribution_type == "lognormal":
            mean = dist.mean or 0.0
            sigma = dist.std or 1.0
            data = np.random.lognormal(mean, sigma, size)
        
        else:
            raise GenerationError(f"Unsupported distribution: {dist.distribution_type}")
        
        # Apply bounds if specified
        if dist.min_value is not None:
            data = np.maximum(data, dist.min_value)
        if dist.max_value is not None:
            data = np.minimum(data, dist.max_value)
        
        # Add outliers if requested
        if self.config.add_outliers:
            self._add_outliers(data, column)
        
        # Add noise if requested
        if self.config.add_noise:
            noise = np.random.normal(0, self.config.noise_level * np.std(data), size)
            data += noise
        
        return data
    
    def _generate_categorical_data(self, column: ColumnSchema) -> np.ndarray:
        """Generate categorical data."""
        dist = column.distribution
        categories = dist.categories or ["A", "B", "C", "D", "E"]
        weights = dist.weights or [1.0] * len(categories)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        data = np.random.choice(categories, size=self.config.num_rows, p=weights)
        return data
    
    def _generate_datetime_data(self, column: ColumnSchema) -> np.ndarray:
        """Generate datetime data."""
        start_date = column.datetime_start or datetime.now() - timedelta(days=365)
        end_date = column.datetime_end or datetime.now()
        
        # Generate random timestamps
        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()
        
        random_timestamps = np.random.uniform(start_timestamp, end_timestamp, self.config.num_rows)
        dates = [datetime.fromtimestamp(ts) for ts in random_timestamps]
        
        return np.array(dates)
    
    def _generate_text_data(self, column: ColumnSchema) -> np.ndarray:
        """Generate text data."""
        min_length, max_length = column.text_length_range
        
        if self.fake and not column.text_pattern:
            # Use faker for realistic text
            data = [self.fake.text(max_nb_chars=random.randint(min_length, max_length)) 
                   for _ in range(self.config.num_rows)]
        else:
            # Generate simple text patterns
            pattern = column.text_pattern or "text_{}"
            data = [pattern.format(i) for i in range(self.config.num_rows)]
        
        return np.array(data)
    
    def _generate_boolean_data(self, column: ColumnSchema) -> np.ndarray:
        """Generate boolean data."""
        prob_true = column.distribution.parameters.get("prob_true", 0.5)
        data = np.random.choice([True, False], size=self.config.num_rows, p=[prob_true, 1-prob_true])
        return data
    
    def _generate_id_data(self, column: ColumnSchema) -> np.ndarray:
        """Generate ID data."""
        prefix = column.id_prefix or "ID"
        id_format = column.id_format or "{prefix}_{:06d}"
        
        if column.unique:
            data = [id_format.format(i, prefix=prefix) for i in range(self.config.num_rows)]
        else:
            # Allow some duplicates
            unique_ids = [id_format.format(i, prefix=prefix) for i in range(int(self.config.num_rows * 0.8))]
            data = np.random.choice(unique_ids, size=self.config.num_rows)
        
        return np.array(data)
    
    def _generate_correlated_column(self, column: ColumnSchema, reference_data: np.ndarray) -> np.ndarray:
        """Generate data correlated with reference column."""
        correlation = column.correlation_strength
        
        if column.data_type == "numeric" and reference_data.dtype in [np.float64, np.int64]:
            # Generate correlated numeric data
            independent_data = self._generate_column_data(column)
            
            # Apply correlation using Cholesky decomposition method
            ref_normalized = (reference_data - np.mean(reference_data)) / np.std(reference_data)
            ind_normalized = (independent_data - np.mean(independent_data)) / np.std(independent_data)
            
            correlated_data = correlation * ref_normalized + np.sqrt(1 - correlation**2) * ind_normalized
            
            # Scale back to original distribution
            correlated_data = correlated_data * np.std(independent_data) + np.mean(independent_data)
            
            return correlated_data
        
        else:
            # For non-numeric data, create weak correlation by conditional generation
            data = self._generate_column_data(column)
            
            if abs(correlation) > 0.3:
                # Apply some conditional logic for categorical correlation
                # This is a simplified approach
                unique_refs = np.unique(reference_data)
                for i, unique_ref in enumerate(unique_refs):
                    mask = reference_data == unique_ref
                    if column.data_type == "categorical" and hasattr(column.distribution, 'categories'):
                        # Bias towards specific categories
                        biased_category = column.distribution.categories[i % len(column.distribution.categories)]
                        data[mask] = biased_category
            
            return data
    
    def _add_outliers(self, data: np.ndarray, column: ColumnSchema) -> None:
        """Add outliers to numeric data."""
        n_outliers = int(len(data) * self.config.outlier_probability)
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        
        # Generate outliers as extreme values
        std = np.std(data)
        mean = np.mean(data)
        
        for idx in outlier_indices:
            # Random outlier: either very high or very low
            if np.random.random() > 0.5:
                data[idx] = mean + np.random.uniform(3, 5) * std
            else:
                data[idx] = mean - np.random.uniform(3, 5) * std
    
    def _apply_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to the dataset."""
        
        # Add null values based on column configuration
        for column in self.config.columns:
            if column.nullable and column.null_probability > 0:
                null_mask = np.random.random(len(df)) < column.null_probability
                df.loc[null_mask, column.name] = None
        
        # Apply consistency constraints
        if self.config.consistency_level < 1.0:
            # Introduce some inconsistencies
            inconsistency_prob = 1 - self.config.consistency_level
            for col in df.select_dtypes(include=['object']).columns:
                mask = np.random.random(len(df)) < inconsistency_prob
                # Add slight variations to some text values
                df.loc[mask, col] = df.loc[mask, col].astype(str) + "_var"
        
        return df
    
    def _apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules to the dataset."""
        
        for rule in self.config.business_rules:
            try:
                # Simple rule engine - in practice this would be more sophisticated
                if "if" in rule and "then" in rule:
                    condition, action = rule.split(" then ")
                    condition = condition.replace("if ", "")
                    
                    # Example: "if age > 65 then category = 'senior'"
                    if ">" in condition and "=" in action:
                        condition_parts = condition.split(">")
                        if len(condition_parts) == 2:
                            col_name = condition_parts[0].strip()
                            threshold = float(condition_parts[1].strip())
                            
                            action_parts = action.split("=")
                            if len(action_parts) == 2:
                                target_col = action_parts[0].strip()
                                value = action_parts[1].strip().strip("'\"")
                                
                                if col_name in df.columns and target_col in df.columns:
                                    mask = df[col_name] > threshold
                                    
                                    # Get the expected dtype for the target column
                                    target_column_schema = next((c for c in self.config.columns if c.name == target_col), None)
                                    if target_column_schema:
                                        if target_column_schema.data_type == "boolean":
                                            # Convert string to boolean
                                            bool_value = value.lower() in ('true', '1', 'yes', 'on')
                                            df.loc[mask, target_col] = bool_value
                                        else:
                                            df.loc[mask, target_col] = value
                                    else:
                                        df.loc[mask, target_col] = value
                                    
            except Exception as e:
                logger.warning(f"Failed to apply business rule '{rule}': {e}")
        
        return df
    
    def _add_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal patterns to the dataset."""
        
        # Find datetime columns
        datetime_cols = [col for col in df.columns 
                        if any(c.name == col and c.data_type == "datetime" for c in self.config.columns)]
        
        if not datetime_cols:
            logger.warning("No datetime columns found for temporal patterns")
            return df
        
        main_date_col = datetime_cols[0]
        dates = pd.to_datetime(df[main_date_col])
        
        # Add seasonality
        if self.config.seasonality:
            seasonal_col = f"{main_date_col}_seasonal_factor"
            
            if self.config.seasonality == "daily":
                # Hour of day pattern
                df[seasonal_col] = np.sin(2 * np.pi * dates.dt.hour / 24)
            elif self.config.seasonality == "weekly":
                # Day of week pattern
                df[seasonal_col] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
            elif self.config.seasonality == "monthly":
                # Day of month pattern
                df[seasonal_col] = np.sin(2 * np.pi * dates.dt.day / 30)
            elif self.config.seasonality == "yearly":
                # Day of year pattern
                df[seasonal_col] = np.sin(2 * np.pi * dates.dt.dayofyear / 365)
        
        # Add trend
        if self.config.trend:
            trend_col = f"{main_date_col}_trend_factor"
            
            if self.config.trend == "increasing":
                df[trend_col] = np.linspace(0, 1, len(df))
            elif self.config.trend == "decreasing":
                df[trend_col] = np.linspace(1, 0, len(df))
            elif self.config.trend == "cyclical":
                # Long-term cyclical pattern
                df[trend_col] = np.sin(2 * np.pi * np.arange(len(df)) / (len(df) / 3))
        
        return df
    
    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate the generated dataset."""
        
        # Check uniqueness constraints
        for column in self.config.columns:
            if column.unique and column.name in df.columns:
                if df[column.name].duplicated().any():
                    logger.warning(f"Column '{column.name}' should be unique but contains duplicates")
        
        # Check data types
        for column in self.config.columns:
            if column.name in df.columns:
                expected_dtype = self._get_expected_dtype(column)
                if expected_dtype and not pd.api.types.is_dtype_equal(df[column.name].dtype, expected_dtype):
                    logger.debug(f"Column '{column.name}' dtype mismatch: expected {expected_dtype}, got {df[column.name].dtype}")
        
        # Check completeness
        actual_completeness = 1 - (df.isnull().sum().sum() / df.size)
        if actual_completeness < self.config.completeness_level - 0.05:  # Allow small tolerance
            logger.warning(f"Dataset completeness {actual_completeness:.3f} is below target {self.config.completeness_level:.3f}")
    
    def _get_expected_dtype(self, column: ColumnSchema) -> Optional[str]:
        """Get expected pandas dtype for a column."""
        if column.data_type == "numeric":
            return "float64"
        elif column.data_type == "categorical":
            return "object"
        elif column.data_type == "datetime":
            return "datetime64[ns]"
        elif column.data_type == "text":
            return "object"
        elif column.data_type == "boolean":
            return "bool"
        elif column.data_type == "id":
            return "object"
        return None


class DatasetTemplate:
    """Pre-defined templates for common dataset types."""
    
    @staticmethod
    def customer_data(num_rows: int = 1000) -> SyntheticDatasetConfig:
        """Template for customer data."""
        columns = [
            ColumnSchema(
                name="customer_id",
                data_type="id",
                unique=True,
                id_prefix="CUST",
                id_format="{prefix}_{:08d}"
            ),
            ColumnSchema(
                name="age",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="normal",
                    mean=35.0,
                    std=12.0,
                    min_value=18.0,
                    max_value=80.0
                )
            ),
            ColumnSchema(
                name="income",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="lognormal",
                    mean=10.5,
                    std=0.8,
                    min_value=20000.0
                ),
                correlation_with="age",
                correlation_strength=0.3
            ),
            ColumnSchema(
                name="gender",
                data_type="categorical",
                distribution=DataDistribution(
                    categories=["Male", "Female", "Other"],
                    weights=[0.48, 0.48, 0.04]
                )
            ),
            ColumnSchema(
                name="region",
                data_type="categorical",
                distribution=DataDistribution(
                    categories=["North", "South", "East", "West", "Central"],
                    weights=[0.2, 0.25, 0.15, 0.25, 0.15]
                )
            ),
            ColumnSchema(
                name="registration_date",
                data_type="datetime",
                datetime_start=datetime.now() - timedelta(days=365*3),
                datetime_end=datetime.now()
            ),
            ColumnSchema(
                name="is_premium",
                data_type="boolean",
                distribution=DataDistribution(
                    parameters={"prob_true": 0.3}
                ),
                correlation_with="income",
                correlation_strength=0.4
            )
        ]
        
        return SyntheticDatasetConfig(
            num_rows=num_rows,
            columns=columns,
            add_outliers=True,
            outlier_probability=0.02,
            business_rules=[
                "if age > 65 then is_premium = True",
                "if income > 100000 then is_premium = True"
            ]
        )
    
    @staticmethod
    def sales_data(num_rows: int = 5000) -> SyntheticDatasetConfig:
        """Template for sales transaction data."""
        columns = [
            ColumnSchema(
                name="transaction_id",
                data_type="id",
                unique=True,
                id_prefix="TXN",
                id_format="{prefix}_{:010d}"
            ),
            ColumnSchema(
                name="customer_id",
                data_type="id",
                id_prefix="CUST",
                id_format="{prefix}_{:08d}",
                unique=False
            ),
            ColumnSchema(
                name="product_category",
                data_type="categorical",
                distribution=DataDistribution(
                    categories=["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"],
                    weights=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
                )
            ),
            ColumnSchema(
                name="amount",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="gamma",
                    parameters={"shape": 2.0, "scale": 50.0},
                    min_value=10.0,
                    max_value=2000.0
                )
            ),
            ColumnSchema(
                name="quantity",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="gamma",
                    parameters={"shape": 1.5, "scale": 2.0},
                    min_value=1.0,
                    max_value=20.0
                ),
                correlation_with="amount",
                correlation_strength=0.6
            ),
            ColumnSchema(
                name="transaction_date",
                data_type="datetime",
                datetime_start=datetime.now() - timedelta(days=365),
                datetime_end=datetime.now()
            ),
            ColumnSchema(
                name="payment_method",
                data_type="categorical",
                distribution=DataDistribution(
                    categories=["Credit Card", "Debit Card", "PayPal", "Cash", "Bank Transfer"],
                    weights=[0.35, 0.25, 0.20, 0.10, 0.10]
                )
            ),
            ColumnSchema(
                name="discount_applied",
                data_type="boolean",
                distribution=DataDistribution(
                    parameters={"prob_true": 0.25}
                )
            )
        ]
        
        return SyntheticDatasetConfig(
            num_rows=num_rows,
            columns=columns,
            add_temporal_patterns=True,
            seasonality="monthly",
            trend="increasing",
            business_rules=[
                "if amount > 500 then discount_applied = True",
                "if quantity > 10 then discount_applied = True"
            ]
        )
    
    @staticmethod
    def iot_sensor_data(num_rows: int = 10000) -> SyntheticDatasetConfig:
        """Template for IoT sensor data."""
        columns = [
            ColumnSchema(
                name="sensor_id",
                data_type="id",
                id_prefix="SENSOR",
                id_format="{prefix}_{:04d}",
                unique=False
            ),
            ColumnSchema(
                name="timestamp",
                data_type="datetime",
                datetime_start=datetime.now() - timedelta(days=30),
                datetime_end=datetime.now()
            ),
            ColumnSchema(
                name="temperature",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="normal",
                    mean=22.0,
                    std=5.0,
                    min_value=-10.0,
                    max_value=50.0
                )
            ),
            ColumnSchema(
                name="humidity",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="beta",
                    parameters={"alpha": 2.0, "beta": 2.0},
                    min_value=0.0,
                    max_value=100.0
                ),
                correlation_with="temperature",
                correlation_strength=-0.3
            ),
            ColumnSchema(
                name="pressure",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="normal",
                    mean=1013.25,
                    std=20.0,
                    min_value=950.0,
                    max_value=1050.0
                )
            ),
            ColumnSchema(
                name="light_level",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="gamma",
                    parameters={"shape": 2.0, "scale": 200.0},
                    min_value=0.0,
                    max_value=2000.0
                )
            ),
            ColumnSchema(
                name="battery_level",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="beta",
                    parameters={"alpha": 3.0, "beta": 1.5},
                    min_value=0.0,
                    max_value=100.0
                )
            ),
            ColumnSchema(
                name="status",
                data_type="categorical",
                distribution=DataDistribution(
                    categories=["Online", "Offline", "Maintenance", "Error"],
                    weights=[0.85, 0.10, 0.03, 0.02]
                )
            )
        ]
        
        return SyntheticDatasetConfig(
            num_rows=num_rows,
            columns=columns,
            add_temporal_patterns=True,
            seasonality="daily",
            add_noise=True,
            noise_level=0.05,
            add_outliers=True,
            outlier_probability=0.01,
            business_rules=[
                "if battery_level < 10 then status = 'Error'",
                "if temperature > 40 then status = 'Error'"
            ]
        )


class SyntheticDatasetFactory:
    """Factory for creating and managing synthetic datasets."""
    
    def __init__(self) -> None:
        """Initialize the factory."""
        self.templates = {
            "customer_data": DatasetTemplate.customer_data,
            "sales_data": DatasetTemplate.sales_data,
            "iot_sensor_data": DatasetTemplate.iot_sensor_data
        }
        self.generated_datasets = {}
    
    def create_dataset(
        self,
        config: Union[SyntheticDatasetConfig, str],
        name: Optional[str] = None,
        benchmark: bool = True,
        export: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a synthetic dataset.
        
        Args:
            config: Dataset configuration or template name
            name: Dataset name
            benchmark: Whether to benchmark the dataset
            export: Whether to export the dataset
            **kwargs: Additional arguments for templates
            
        Returns:
            Dictionary with dataset and metadata
        """
        # Handle template names
        if isinstance(config, str):
            if config not in self.templates:
                raise GenerationError(f"Unknown template: {config}")
            config = self.templates[config](**kwargs)
        
        # Generate dataset
        generator = SyntheticDataGenerator(config)
        df = generator.generate_dataset()
        
        # Create result dictionary
        result = {
            "dataset": df,
            "config": config.model_dump(),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "generation_time": time.time()  # This would be actual generation time
            }
        }
        
        # Benchmark if requested
        if benchmark:
            try:
                benchmark_config = BenchmarkConfig(
                    ml_evaluation=True,
                    create_visualizations=False,
                    generate_report=False
                )
                benchmarker = SyntheticDatasetBenchmark(benchmark_config)
                
                # Find a suitable target column for ML evaluation
                target_column = None
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                if len(categorical_cols) > 0:
                    target_column = categorical_cols[0]
                elif len(numeric_cols) > 0:
                    target_column = numeric_cols[0]
                
                if target_column:
                    metrics = benchmarker.benchmark_dataset(df, target_column=target_column)
                    result["benchmark_metrics"] = metrics.model_dump()
                    
            except Exception as e:
                logger.warning(f"Benchmarking failed: {e}")
                result["benchmark_metrics"] = None
        
        # Export if requested
        if export and config.export_config:
            try:
                exporter = DataExporter(config.export_config)
                export_path = Path(f"synthetic_dataset_{name or 'default'}.{config.export_config.format}")
                export_metadata = exporter.export_dataset(df, export_path)
                result["export_metadata"] = export_metadata.model_dump()
                
            except Exception as e:
                logger.warning(f"Export failed: {e}")
                result["export_metadata"] = None
        
        # Store in factory
        dataset_name = name or f"dataset_{len(self.generated_datasets)}"
        self.generated_datasets[dataset_name] = result
        
        logger.info(f"Created synthetic dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns")
        return result
    
    def create_multiple_datasets(
        self,
        configs: Dict[str, Union[SyntheticDatasetConfig, str]],
        benchmark: bool = True,
        export: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Create multiple synthetic datasets.
        
        Args:
            configs: Dictionary of dataset name -> config
            benchmark: Whether to benchmark datasets
            export: Whether to export datasets
            
        Returns:
            Dictionary of dataset name -> result
        """
        results = {}
        
        for name, config in configs.items():
            try:
                result = self.create_dataset(
                    config=config,
                    name=name,
                    benchmark=benchmark,
                    export=export
                )
                results[name] = result
                
            except Exception as e:
                logger.error(f"Failed to create dataset '{name}': {e}")
                results[name] = {"error": str(e)}
        
        # Create comparative report if benchmarking was done
        if benchmark:
            self._create_comparative_benchmark_report(results)
        
        return results
    
    def _create_comparative_benchmark_report(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Create comparative benchmark report across multiple datasets."""
        comparison_data = []
        
        for name, result in results.items():
            if "benchmark_metrics" in result and result["benchmark_metrics"]:
                metrics = result["benchmark_metrics"]
                comparison_data.append({
                    "dataset_name": name,
                    "num_rows": result["metadata"]["num_rows"],
                    "num_columns": result["metadata"]["num_columns"],
                    "overall_quality": metrics.get("overall_quality_score", 0.0),
                    "completeness": metrics.get("completeness_score", 0.0),
                    "consistency": metrics.get("consistency_score", 0.0),
                    "uniqueness": metrics.get("uniqueness_score", 0.0),
                    "validity": metrics.get("validity_score", 0.0),
                    "ml_utility": metrics.get("ml_utility_score", 0.0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Save comparative report
            report_path = Path("synthetic_datasets_comparison.csv")
            df.to_csv(report_path, index=False)
            
            # Create summary
            summary = {
                "total_datasets": len(comparison_data),
                "avg_quality_score": df["overall_quality"].mean(),
                "best_dataset": df.loc[df["overall_quality"].idxmax(), "dataset_name"],
                "total_rows": df["num_rows"].sum(),
                "generation_timestamp": datetime.now().isoformat()
            }
            
            summary_path = Path("synthetic_datasets_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Comparative reports saved to: {report_path} and {summary_path}")
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a generated dataset by name."""
        return self.generated_datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all generated dataset names."""
        return list(self.generated_datasets.keys())
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys()) 