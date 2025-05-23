"""Tests for synthetic dataset creation and benchmarking."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from opensynthetics.datagen.synthetic_datasets import (
    SyntheticDatasetFactory,
    SyntheticDataGenerator,
    SyntheticDatasetConfig,
    ColumnSchema,
    DataDistribution,
    DatasetTemplate
)
from opensynthetics.data_ops.export_utils import DataExporter, ExportConfig, ExportMetadata
from opensynthetics.training_eval.benchmark import (
    SyntheticDatasetBenchmark,
    BenchmarkConfig,
    DatasetQualityMetrics
)
from opensynthetics.core.exceptions import GenerationError, ProcessingError, EvaluationError


class TestDataDistribution:
    """Test data distribution configuration."""
    
    def test_default_distribution(self):
        """Test default distribution creation."""
        dist = DataDistribution()
        assert dist.distribution_type == "normal"
        assert dist.parameters == {}
        assert dist.min_value is None
        assert dist.max_value is None
    
    def test_normal_distribution(self):
        """Test normal distribution configuration."""
        dist = DataDistribution(
            distribution_type="normal",
            mean=100.0,
            std=15.0,
            min_value=0.0,
            max_value=200.0
        )
        assert dist.distribution_type == "normal"
        assert dist.mean == 100.0
        assert dist.std == 15.0
        assert dist.min_value == 0.0
        assert dist.max_value == 200.0
    
    def test_categorical_distribution(self):
        """Test categorical distribution configuration."""
        dist = DataDistribution(
            categories=["A", "B", "C"],
            weights=[0.5, 0.3, 0.2]
        )
        assert dist.categories == ["A", "B", "C"]
        assert dist.weights == [0.5, 0.3, 0.2]


class TestColumnSchema:
    """Test column schema configuration."""
    
    def test_numeric_column(self):
        """Test numeric column schema."""
        schema = ColumnSchema(
            name="age",
            data_type="numeric",
            distribution=DataDistribution(
                distribution_type="normal",
                mean=35.0,
                std=12.0
            )
        )
        assert schema.name == "age"
        assert schema.data_type == "numeric"
        assert schema.distribution.distribution_type == "normal"
    
    def test_categorical_column(self):
        """Test categorical column schema."""
        schema = ColumnSchema(
            name="category",
            data_type="categorical",
            distribution=DataDistribution(
                categories=["A", "B", "C"],
                weights=[0.4, 0.4, 0.2]
            )
        )
        assert schema.name == "category"
        assert schema.data_type == "categorical"
        assert schema.distribution.categories == ["A", "B", "C"]
    
    def test_correlated_column(self):
        """Test correlated column schema."""
        schema = ColumnSchema(
            name="income",
            data_type="numeric",
            correlation_with="age",
            correlation_strength=0.3
        )
        assert schema.correlation_with == "age"
        assert schema.correlation_strength == 0.3


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    @pytest.fixture
    def basic_config(self):
        """Create basic dataset configuration."""
        columns = [
            ColumnSchema(
                name="id",
                data_type="id",
                unique=True,
                id_prefix="TEST"
            ),
            ColumnSchema(
                name="value",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="normal",
                    mean=100.0,
                    std=15.0
                )
            ),
            ColumnSchema(
                name="category",
                data_type="categorical",
                distribution=DataDistribution(
                    categories=["A", "B", "C"],
                    weights=[0.5, 0.3, 0.2]
                )
            )
        ]
        
        return SyntheticDatasetConfig(
            num_rows=100,
            columns=columns,
            seed=42,
            consistency_level=1.0  # Set to 1.0 to avoid _var suffixes in test
        )
    
    def test_generate_basic_dataset(self, basic_config):
        """Test basic dataset generation."""
        generator = SyntheticDataGenerator(basic_config)
        df = generator.generate_dataset()
        
        assert len(df) == 100
        assert len(df.columns) == 3
        assert "id" in df.columns
        assert "value" in df.columns
        assert "category" in df.columns
        
        # Check data types
        assert df["value"].dtype in [np.float64, np.int64]
        assert df["category"].dtype == object
        
        # Check uniqueness
        assert df["id"].nunique() == 100
        
        # Check categories
        assert set(df["category"].unique()).issubset({"A", "B", "C"})
    
    def test_generate_correlated_data(self):
        """Test correlated data generation."""
        columns = [
            ColumnSchema(
                name="x",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="normal",
                    mean=0.0,
                    std=1.0
                )
            ),
            ColumnSchema(
                name="y",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="normal",
                    mean=0.0,
                    std=1.0
                ),
                correlation_with="x",
                correlation_strength=0.8
            )
        ]
        
        config = SyntheticDatasetConfig(
            num_rows=1000,
            columns=columns,
            seed=42,
            consistency_level=1.0  # Set to 1.0 to avoid _var suffixes in test
        )
        
        generator = SyntheticDataGenerator(config)
        df = generator.generate_dataset()
        
        # Check correlation
        correlation = df["x"].corr(df["y"])
        assert abs(correlation - 0.8) < 0.2  # Allow some tolerance
    
    def test_generate_with_outliers(self):
        """Test data generation with outliers."""
        columns = [
            ColumnSchema(
                name="value",
                data_type="numeric",
                distribution=DataDistribution(
                    distribution_type="normal",
                    mean=100.0,
                    std=10.0
                )
            )
        ]
        
        config = SyntheticDatasetConfig(
            num_rows=1000,
            columns=columns,
            add_outliers=True,
            outlier_probability=0.05,
            seed=42
        )
        
        generator = SyntheticDataGenerator(config)
        df = generator.generate_dataset()
        
        # Check for outliers (values beyond 3 standard deviations)
        mean_val = df["value"].mean()
        std_val = df["value"].std()
        outliers = df[(df["value"] < mean_val - 3*std_val) | (df["value"] > mean_val + 3*std_val)]
        
        assert len(outliers) > 0
    
    @patch('opensynthetics.datagen.synthetic_datasets.FAKER_AVAILABLE', False)
    def test_generate_without_faker(self, basic_config):
        """Test generation when Faker is not available."""
        generator = SyntheticDataGenerator(basic_config)
        df = generator.generate_dataset()
        
        assert len(df) == 100
        assert len(df.columns) == 3


class TestDatasetTemplate:
    """Test dataset templates."""
    
    def test_customer_data_template(self):
        """Test customer data template."""
        config = DatasetTemplate.customer_data(num_rows=500)
        
        assert config.num_rows == 500
        assert len(config.columns) == 7
        
        # Check column names
        column_names = [col.name for col in config.columns]
        expected_names = ["customer_id", "age", "income", "gender", "region", "registration_date", "is_premium"]
        assert set(column_names) == set(expected_names)
        
        # Check business rules
        assert len(config.business_rules) == 2
    
    def test_sales_data_template(self):
        """Test sales data template."""
        config = DatasetTemplate.sales_data(num_rows=1000)
        
        assert config.num_rows == 1000
        assert len(config.columns) == 8
        assert config.add_temporal_patterns is True
        assert config.seasonality == "monthly"
        assert config.trend == "increasing"
    
    def test_iot_sensor_data_template(self):
        """Test IoT sensor data template."""
        config = DatasetTemplate.iot_sensor_data(num_rows=2000)
        
        assert config.num_rows == 2000
        assert len(config.columns) == 8
        assert config.add_temporal_patterns is True
        assert config.seasonality == "daily"
        assert config.add_noise is True
        assert config.add_outliers is True


class TestSyntheticDatasetFactory:
    """Test synthetic dataset factory."""
    
    def test_create_dataset_from_template(self):
        """Test creating dataset from template."""
        factory = SyntheticDatasetFactory()
        
        result = factory.create_dataset(
            config="customer_data",
            num_rows=100,
            benchmark=False,
            export=False
        )
        
        assert "dataset" in result
        assert "config" in result
        assert "metadata" in result
        
        df = result["dataset"]
        assert len(df) == 100
        assert len(df.columns) == 7
    
    def test_create_multiple_datasets(self):
        """Test creating multiple datasets."""
        factory = SyntheticDatasetFactory()
        
        configs = {
            "customers": "customer_data",
            "sales": "sales_data"
        }
        
        results = factory.create_multiple_datasets(
            configs=configs,
            benchmark=False,
            export=False
        )
        
        assert len(results) == 2
        assert "customers" in results
        assert "sales" in results
        
        # Check both datasets were created successfully
        assert "dataset" in results["customers"]
        assert "dataset" in results["sales"]
    
    def test_list_templates(self):
        """Test listing available templates."""
        factory = SyntheticDatasetFactory()
        templates = factory.list_templates()
        
        expected_templates = ["customer_data", "sales_data", "iot_sensor_data"]
        assert set(templates) == set(expected_templates)
    
    def test_invalid_template(self):
        """Test creating dataset with invalid template."""
        factory = SyntheticDatasetFactory()
        
        with pytest.raises(GenerationError):
            factory.create_dataset(config="invalid_template")


class TestDataExporter:
    """Test data export functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            "id": range(100),
            "value": np.random.normal(100, 15, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "flag": np.random.choice([True, False], 100)
        })
    
    def test_export_to_json(self, sample_data):
        """Test exporting to JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.json"
            
            config = ExportConfig(format="json")
            exporter = DataExporter(config)
            
            metadata = exporter.export_dataset(sample_data, output_path)
            
            assert output_path.exists()
            assert metadata.total_records == 100
            assert metadata.file_count == 1
            
            # Verify exported data
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
            assert len(exported_data) == 100
    
    @pytest.mark.skipif(not pd.api.types.is_extension_array_dtype, reason="PyArrow not available")
    def test_export_to_parquet(self, sample_data):
        """Test exporting to Parquet format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.parquet"
            
            config = ExportConfig(format="parquet", parquet_compression="snappy")
            exporter = DataExporter(config)
            
            metadata = exporter.export_dataset(sample_data, output_path)
            
            assert output_path.exists()
            assert metadata.total_records == 100
            assert metadata.compression_ratio is not None
    
    def test_export_to_csv(self, sample_data):
        """Test exporting to CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.csv"
            
            config = ExportConfig(
                format="csv",
                csv_separator=",",
                csv_quoting="minimal"
            )
            exporter = DataExporter(config)
            
            metadata = exporter.export_dataset(sample_data, output_path)
            
            assert output_path.exists()
            assert metadata.total_records == 100
            
            # Verify exported data
            exported_df = pd.read_csv(output_path)
            assert len(exported_df) == 100
            assert list(exported_df.columns) == list(sample_data.columns)
    
    def test_export_with_compression(self, sample_data):
        """Test exporting with compression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.json"
            
            config = ExportConfig(format="json", compression="gzip")
            exporter = DataExporter(config)
            
            metadata = exporter.export_dataset(sample_data, output_path)
            
            assert Path(f"{output_path}.gz").exists()
            assert metadata.total_records == 100


class TestSyntheticDatasetBenchmark:
    """Test dataset benchmarking functionality."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for benchmarking."""
        np.random.seed(42)
        return pd.DataFrame({
            "id": range(1000),
            "feature1": np.random.normal(100, 15, 1000),
            "feature2": np.random.uniform(0, 100, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
            "target": np.random.choice([0, 1], 1000)
        })
    
    @pytest.fixture
    def reference_dataset(self):
        """Create reference dataset for comparison."""
        np.random.seed(123)
        return pd.DataFrame({
            "id": range(1000, 2000),
            "feature1": np.random.normal(105, 12, 1000),
            "feature2": np.random.uniform(5, 95, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.4, 0.4, 0.2]),
            "target": np.random.choice([0, 1], 1000)
        })
    
    def test_benchmark_basic_quality(self, sample_dataset):
        """Test basic quality assessment."""
        config = BenchmarkConfig(
            ml_evaluation=False,
            create_visualizations=False,
            generate_report=False
        )
        
        benchmarker = SyntheticDatasetBenchmark(config)
        metrics = benchmarker.benchmark_dataset(sample_dataset)
        
        assert isinstance(metrics, DatasetQualityMetrics)
        assert 0 <= metrics.completeness_score <= 1
        assert 0 <= metrics.consistency_score <= 1
        assert 0 <= metrics.uniqueness_score <= 1
        assert 0 <= metrics.validity_score <= 1
        assert 0 <= metrics.overall_quality_score <= 1
    
    def test_benchmark_with_reference(self, sample_dataset, reference_dataset):
        """Test benchmarking with reference dataset."""
        config = BenchmarkConfig(
            ml_evaluation=False,
            test_distributions=True,
            test_correlations=True,
            create_visualizations=False,
            generate_report=False
        )
        
        benchmarker = SyntheticDatasetBenchmark(config)
        metrics = benchmarker.benchmark_dataset(
            sample_dataset,
            reference_data=reference_dataset
        )
        
        assert metrics.distribution_similarity is not None
        assert metrics.correlation_preservation >= 0
        assert metrics.statistical_fidelity >= 0
    
    def test_ml_utility_evaluation(self, sample_dataset):
        """Test ML utility evaluation."""
        config = BenchmarkConfig(
            ml_evaluation=True,
            create_visualizations=False,
            generate_report=False
        )
        
        benchmarker = SyntheticDatasetBenchmark(config)
        metrics = benchmarker.benchmark_dataset(
            sample_dataset,
            target_column="target"
        )
        
        assert metrics.ml_utility_score >= 0
        assert len(metrics.predictive_performance) > 0
    
    def test_benchmark_with_missing_data(self):
        """Test benchmarking dataset with missing values."""
        # Create dataset with missing values
        df = pd.DataFrame({
            "feature1": [1, 2, None, 4, 5],
            "feature2": [10, None, 30, 40, None],
            "category": ["A", "B", None, "A", "B"]
        })
        
        config = BenchmarkConfig(
            ml_evaluation=False,
            create_visualizations=False,
            generate_report=False
        )
        
        benchmarker = SyntheticDatasetBenchmark(config)
        metrics = benchmarker.benchmark_dataset(df)
        
        assert metrics.completeness_score < 1.0  # Should detect missing values
    
    def test_privacy_evaluation(self, sample_dataset, reference_dataset):
        """Test privacy evaluation."""
        config = BenchmarkConfig(
            privacy_evaluation=True,
            ml_evaluation=False,
            create_visualizations=False,
            generate_report=False
        )
        
        benchmarker = SyntheticDatasetBenchmark(config)
        metrics = benchmarker.benchmark_dataset(
            sample_dataset,
            reference_data=reference_dataset
        )
        
        assert 0 <= metrics.privacy_score <= 1
        assert 0 <= metrics.disclosure_risk <= 1


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Create dataset using factory
            factory = SyntheticDatasetFactory()
            
            result = factory.create_dataset(
                config="customer_data",
                num_rows=200,
                benchmark=True,
                export=False
            )
            
            assert "dataset" in result
            assert "benchmark_metrics" in result
            
            # 2. Export dataset
            export_config = ExportConfig(format="csv")
            exporter = DataExporter(export_config)
            
            output_path = Path(temp_dir) / "test_customers.csv"
            export_metadata = exporter.export_dataset(result["dataset"], output_path)
            
            assert output_path.exists()
            assert export_metadata.total_records == 200
            
            # 3. Load and benchmark exported dataset
            loaded_df = pd.read_csv(output_path)
            
            benchmark_config = BenchmarkConfig(
                create_visualizations=False,
                generate_report=False
            )
            benchmarker = SyntheticDatasetBenchmark(benchmark_config)
            
            metrics = benchmarker.benchmark_dataset(loaded_df)
            
            assert metrics.overall_quality_score > 0
    
    def test_batch_generation_and_export(self):
        """Test batch generation and export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = SyntheticDatasetFactory()
            
            # Create multiple datasets
            configs = {
                "customers": "customer_data",
                "sales": "sales_data"
            }
            
            results = factory.create_multiple_datasets(
                configs=configs,
                benchmark=False,
                export=False
            )
            
            # Export all datasets
            from opensynthetics.data_ops.export_utils import BatchExporter
            
            export_config = ExportConfig(format="json")
            batch_exporter = BatchExporter()
            
            datasets_to_export = {
                name: result["dataset"]
                for name, result in results.items()
                if "dataset" in result
            }
            
            export_results = batch_exporter.export_multiple(
                datasets=datasets_to_export,
                output_dir=temp_dir,
                default_config=export_config
            )
            
            # Verify exports
            assert len(export_results) == 2
            assert (Path(temp_dir) / "customers.json").exists()
            assert (Path(temp_dir) / "sales.json").exists()


if __name__ == "__main__":
    pytest.main([__file__]) 