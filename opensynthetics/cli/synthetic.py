"""CLI commands for synthetic dataset creation and benchmarking."""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import click
import pandas as pd
from loguru import logger

from opensynthetics.datagen.synthetic_datasets import (
    SyntheticDatasetFactory, 
    DatasetTemplate,
    SyntheticDatasetConfig,
    ColumnSchema,
    DataDistribution
)
from opensynthetics.data_ops.export_utils import ExportConfig, DataExporter, BatchExporter
from opensynthetics.training_eval.benchmark import (
    SyntheticDatasetBenchmark, 
    BenchmarkConfig,
    BenchmarkSuite
)
from opensynthetics.core.exceptions import GenerationError, ProcessingError, EvaluationError


@click.group()
def synthetic():
    """Synthetic dataset creation and benchmarking commands."""
    pass


@synthetic.command()
@click.option('--template', '-t', 
              type=click.Choice(['customer_data', 'sales_data', 'iot_sensor_data']),
              help='Pre-defined dataset template')
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='JSON configuration file for custom dataset')
@click.option('--num-rows', '-n', default=1000, type=int,
              help='Number of rows to generate')
@click.option('--output-dir', '-o', default='./synthetic_output', type=click.Path(),
              help='Output directory for generated datasets')
@click.option('--format', '-f', 
              type=click.Choice(['json', 'jsonl', 'csv', 'parquet', 'hdf5', 'excel', 'feather']),
              default='parquet', help='Export format')
@click.option('--compression', 
              type=click.Choice(['gzip', 'bz2', 'zip', 'snappy', 'lz4']),
              help='Compression type')
@click.option('--benchmark/--no-benchmark', default=True,
              help='Run quality benchmarking')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--name', help='Dataset name (default: auto-generated)')
def generate(template: Optional[str], config_file: Optional[str], num_rows: int,
             output_dir: str, format: str, compression: Optional[str],
             benchmark: bool, seed: Optional[int], name: Optional[str]):
    """Generate synthetic datasets."""
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create export config
        export_config = ExportConfig(
            format=format,
            compression=compression,
            include_metadata=True,
            create_checksums=True
        )
        
        # Initialize factory
        factory = SyntheticDatasetFactory()
        
        if template:
            # Use pre-defined template
            logger.info(f"Generating dataset using template: {template}")
            result = factory.create_dataset(
                config=template,
                name=name,
                benchmark=benchmark,
                export=True,
                num_rows=num_rows,
                seed=seed,
                export_config=export_config
            )
            
        elif config_file:
            # Load custom configuration
            logger.info(f"Loading configuration from: {config_file}")
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Parse configuration
            config = SyntheticDatasetConfig.parse_obj(config_data)
            if seed is not None:
                config.seed = seed
            config.export_config = export_config
            
            result = factory.create_dataset(
                config=config,
                name=name,
                benchmark=benchmark,
                export=True
            )
            
        else:
            click.echo("Error: Must specify either --template or --config-file", err=True)
            sys.exit(1)
        
        # Save results
        if result.get("error"):
            click.echo(f"Error: {result['error']}", err=True)
            sys.exit(1)
        
        # Export dataset if not already exported
        if not result.get("export_metadata"):
            exporter = DataExporter(export_config)
            dataset_name = name or "synthetic_dataset"
            export_path = output_path / f"{dataset_name}.{format}"
            export_metadata = exporter.export_dataset(result["dataset"], export_path)
            result["export_metadata"] = export_metadata.dict()
        
        # Save metadata
        metadata_path = output_path / f"{name or 'synthetic_dataset'}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "config": result["config"],
                "metadata": result["metadata"],
                "benchmark_metrics": result.get("benchmark_metrics"),
                "export_metadata": result.get("export_metadata")
            }, f, indent=2, default=str)
        
        click.echo(f"[OK] Dataset generated successfully!")
        click.echo(f"[INFO] Rows: {result['metadata']['num_rows']}")
        click.echo(f"[INFO] Columns: {result['metadata']['num_columns']}")
        click.echo(f"[INFO] Size: {result['metadata']['memory_usage_mb']:.2f} MB")
        
        if benchmark and result.get("benchmark_metrics"):
            metrics = result["benchmark_metrics"]
            click.echo(f"[INFO] Quality Score: {metrics.get('overall_quality_score', 0):.3f}")
        
        click.echo(f"[FILE] Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@synthetic.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), required=True,
              help='JSON configuration file with multiple dataset specifications')
@click.option('--output-dir', '-o', default='./synthetic_batch_output', type=click.Path(),
              help='Output directory for generated datasets')
@click.option('--format', '-f', 
              type=click.Choice(['json', 'jsonl', 'csv', 'parquet', 'hdf5', 'excel', 'feather']),
              default='parquet', help='Default export format')
@click.option('--benchmark/--no-benchmark', default=True,
              help='Run quality benchmarking')
@click.option('--parallel/--no-parallel', default=True,
              help='Generate datasets in parallel')
def batch_generate(config_file: str, output_dir: str, format: str, 
                  benchmark: bool, parallel: bool):
    """Generate multiple synthetic datasets in batch."""
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load batch configuration
        with open(config_file, 'r') as f:
            batch_config = json.load(f)
        
        factory = SyntheticDatasetFactory()
        
        # Prepare dataset configs
        dataset_configs = {}
        for name, config_data in batch_config.get("datasets", {}).items():
            if isinstance(config_data, str):
                # Template name
                dataset_configs[name] = config_data
            else:
                # Custom configuration
                dataset_configs[name] = SyntheticDatasetConfig.parse_obj(config_data)
        
        # Generate datasets
        logger.info(f"Generating {len(dataset_configs)} datasets in batch")
        results = factory.create_multiple_datasets(
            configs=dataset_configs,
            benchmark=benchmark,
            export=True
        )
        
        # Export results if needed
        export_config = ExportConfig(format=format, include_metadata=True)
        batch_exporter = BatchExporter()
        
        datasets_to_export = {}
        for name, result in results.items():
            if not result.get("error") and result.get("dataset") is not None:
                datasets_to_export[name] = result["dataset"]
        
        if datasets_to_export:
            export_results = batch_exporter.export_multiple(
                datasets=datasets_to_export,
                output_dir=output_path,
                default_config=export_config
            )
        
        # Save comprehensive metadata
        batch_metadata = {
            "generation_summary": {
                "total_datasets": len(results),
                "successful": sum(1 for r in results.values() if not r.get("error")),
                "failed": sum(1 for r in results.values() if r.get("error")),
                "total_rows": sum(r["metadata"]["num_rows"] for r in results.values() 
                                if not r.get("error") and "metadata" in r)
            },
            "datasets": {}
        }
        
        for name, result in results.items():
            if result.get("error"):
                batch_metadata["datasets"][name] = {"status": "failed", "error": result["error"]}
            else:
                batch_metadata["datasets"][name] = {
                    "status": "success",
                    "metadata": result.get("metadata"),
                    "benchmark_metrics": result.get("benchmark_metrics")
                }
        
        metadata_path = output_path / "batch_generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(batch_metadata, f, indent=2, default=str)
        
        # Print summary
        summary = batch_metadata["generation_summary"]
        click.echo(f"[OK] Batch generation completed!")
        click.echo(f"[INFO] Total datasets: {summary['total_datasets']}")
        click.echo(f"[OK] Successful: {summary['successful']}")
        click.echo(f"[ERROR] Failed: {summary['failed']}")
        click.echo(f"[INFO] Total rows: {summary['total_rows']}")
        click.echo(f"[FILE] Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@synthetic.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--reference-path', '-r', type=click.Path(exists=True),
              help='Reference dataset for comparison')
@click.option('--target-column', '-t', help='Target column for ML evaluation')
@click.option('--output-dir', '-o', default='./benchmark_output', type=click.Path(),
              help='Output directory for benchmark results')
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Benchmark configuration file')
@click.option('--sample-size', type=int, 
              help='Sample size for large datasets')
@click.option('--visualizations/--no-visualizations', default=True,
              help='Create visualization plots')
@click.option('--report/--no-report', default=True,
              help='Generate comprehensive report')
def benchmark(dataset_path: str, reference_path: Optional[str], target_column: Optional[str],
              output_dir: str, config_file: Optional[str], sample_size: Optional[int],
              visualizations: bool, report: bool):
    """Benchmark synthetic dataset quality."""
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        logger.info(f"Loading dataset from: {dataset_path}")
        if dataset_path.endswith('.parquet'):
            synthetic_df = pd.read_parquet(dataset_path)
        elif dataset_path.endswith('.csv'):
            synthetic_df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            synthetic_df = pd.read_json(dataset_path)
        else:
            click.echo(f"Error: Unsupported file format for {dataset_path}", err=True)
            sys.exit(1)
        
        reference_df = None
        if reference_path:
            logger.info(f"Loading reference dataset from: {reference_path}")
            if reference_path.endswith('.parquet'):
                reference_df = pd.read_parquet(reference_path)
            elif reference_path.endswith('.csv'):
                reference_df = pd.read_csv(reference_path)
            elif reference_path.endswith('.json'):
                reference_df = pd.read_json(reference_path)
        
        # Load or create benchmark configuration
        if config_file:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            benchmark_config = BenchmarkConfig.parse_obj(config_data)
        else:
            benchmark_config = BenchmarkConfig(
                create_visualizations=visualizations,
                generate_report=report,
                sample_size=sample_size
            )
        
        # Run benchmark
        benchmarker = SyntheticDatasetBenchmark(benchmark_config)
        
        # Auto-detect target column if not specified
        if not target_column:
            categorical_cols = synthetic_df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = synthetic_df.select_dtypes(include=['number']).columns
            
            if len(categorical_cols) > 0:
                target_column = categorical_cols[0]
                click.echo(f"Auto-detected target column: {target_column} (categorical)")
            elif len(numeric_cols) > 0:
                target_column = numeric_cols[0]
                click.echo(f"Auto-detected target column: {target_column} (numeric)")
        
        logger.info("Running comprehensive dataset benchmark")
        metrics = benchmarker.benchmark_dataset(
            synthetic_data=synthetic_df,
            reference_data=reference_df,
            target_column=target_column
        )
        
        # Save results
        results_path = output_path / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "metrics": metrics.dict(),
                "dataset_info": {
                    "synthetic_shape": synthetic_df.shape,
                    "reference_shape": reference_df.shape if reference_df is not None else None,
                    "target_column": target_column
                },
                "config": benchmark_config.dict()
            }, f, indent=2, default=str)
        
        # Print summary
        click.echo(f"[INFO] Benchmark Results:")
        click.echo(f"   Overall Quality Score: {metrics.overall_quality_score:.3f}")
        click.echo(f"   Completeness: {metrics.completeness_score:.3f}")
        click.echo(f"   Consistency: {metrics.consistency_score:.3f}")
        click.echo(f"   Uniqueness: {metrics.uniqueness_score:.3f}")
        click.echo(f"   Validity: {metrics.validity_score:.3f}")
        click.echo(f"   Statistical Fidelity: {metrics.statistical_fidelity:.3f}")
        click.echo(f"   ML Utility: {metrics.ml_utility_score:.3f}")
        
        if metrics.privacy_score > 0:
            click.echo(f"   Privacy Score: {metrics.privacy_score:.3f}")
        
        click.echo(f"[FILE] Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@synthetic.command()
@click.argument('datasets_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./benchmark_suite_output', type=click.Path(),
              help='Output directory for benchmark results')
@click.option('--reference-path', '-r', type=click.Path(exists=True),
              help='Reference dataset for comparison')
@click.option('--pattern', '-p', default='*.parquet',
              help='File pattern to match datasets')
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Benchmark configuration file')
def benchmark_suite(datasets_dir: str, output_dir: str, reference_path: Optional[str],
                   pattern: str, config_file: Optional[str]):
    """Run benchmark suite on multiple datasets."""
    
    try:
        datasets_path = Path(datasets_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find dataset files
        dataset_files = list(datasets_path.glob(pattern))
        if not dataset_files:
            click.echo(f"No datasets found matching pattern: {pattern}", err=True)
            sys.exit(1)
        
        # Load datasets
        datasets = {}
        for file_path in dataset_files:
            try:
                name = file_path.stem
                if file_path.suffix == '.parquet':
                    datasets[name] = pd.read_parquet(file_path)
                elif file_path.suffix == '.csv':
                    datasets[name] = pd.read_csv(file_path)
                elif file_path.suffix == '.json':
                    datasets[name] = pd.read_json(file_path)
                else:
                    logger.warning(f"Skipping unsupported file: {file_path}")
                    continue
                    
                logger.info(f"Loaded dataset: {name} ({datasets[name].shape})")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not datasets:
            click.echo("No valid datasets found", err=True)
            sys.exit(1)
        
        # Load reference dataset if provided
        reference_df = None
        if reference_path:
            if reference_path.endswith('.parquet'):
                reference_df = pd.read_parquet(reference_path)
            elif reference_path.endswith('.csv'):
                reference_df = pd.read_csv(reference_path)
            elif reference_path.endswith('.json'):
                reference_df = pd.read_json(reference_path)
        
        # Load benchmark configuration
        if config_file:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            benchmark_config = BenchmarkConfig.parse_obj(config_data)
        else:
            benchmark_config = BenchmarkConfig(
                create_visualizations=False,  # Skip individual visualizations for suite
                generate_report=False
            )
        
        # Create benchmark suite
        suite = BenchmarkSuite()
        benchmark = SyntheticDatasetBenchmark(benchmark_config)
        suite.add_benchmark("standard", benchmark)
        
        # Auto-detect target columns
        target_columns = {}
        for name, df in datasets.items():
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(categorical_cols) > 0:
                target_columns[name] = categorical_cols[0]
            elif len(numeric_cols) > 0:
                target_columns[name] = numeric_cols[0]
        
        # Run suite
        logger.info(f"Running benchmark suite on {len(datasets)} datasets")
        results = suite.run_all_benchmarks(
            datasets=datasets,
            reference_data=reference_df,
            target_columns=target_columns
        )
        
        # Save detailed results
        results_path = output_path / "suite_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "suite_results": {name: metrics.dict() for name, metrics in results.items()},
                "summary": {
                    "total_datasets": len(datasets),
                    "benchmarks_run": len(results),
                    "avg_quality_score": sum(m.overall_quality_score for m in results.values()) / len(results) if results else 0
                }
            }, f, indent=2, default=str)
        
        # Print summary
        click.echo(f"[INFO] Benchmark Suite Results:")
        click.echo(f"   Datasets processed: {len(datasets)}")
        click.echo(f"   Benchmarks completed: {len(results)}")
        
        if results:
            avg_quality = sum(m.overall_quality_score for m in results.values()) / len(results)
            best_dataset = max(results.items(), key=lambda x: x[1].overall_quality_score)
            
            click.echo(f"   Average quality score: {avg_quality:.3f}")
            click.echo(f"   Best dataset: {best_dataset[0]} ({best_dataset[1].overall_quality_score:.3f})")
        
        click.echo(f"[FILE] Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@synthetic.command()
@click.option('--template', '-t', 
              type=click.Choice(['customer_data', 'sales_data', 'iot_sensor_data']),
              help='Template to describe')
def describe_template(template: str):
    """Describe available dataset templates."""
    
    templates_info = {
        'customer_data': {
            'description': 'Customer demographic and profile data',
            'columns': [
                'customer_id: Unique customer identifier',
                'age: Customer age (18-80, normal distribution)',
                'income: Annual income (log-normal distribution, correlated with age)',
                'gender: Gender categories (Male/Female/Other)',
                'region: Geographic region (North/South/East/West/Central)',
                'registration_date: Account registration date',
                'is_premium: Premium membership status (correlated with income)'
            ],
            'business_rules': [
                'Customers over 65 automatically get premium status',
                'High income customers (>100k) get premium status'
            ],
            'use_cases': [
                'Customer segmentation analysis',
                'Marketing campaign targeting',
                'Demographic trend analysis'
            ]
        },
        'sales_data': {
            'description': 'E-commerce transaction and sales data',
            'columns': [
                'transaction_id: Unique transaction identifier',
                'customer_id: Customer reference',
                'product_category: Product categories (Electronics/Clothing/etc.)',
                'amount: Transaction amount (gamma distribution)',
                'quantity: Items purchased (correlated with amount)',
                'transaction_date: Transaction timestamp',
                'payment_method: Payment type (Credit/Debit/PayPal/Cash/Transfer)',
                'discount_applied: Whether discount was applied'
            ],
            'business_rules': [
                'Large orders (>$500) automatically get discounts',
                'High quantity orders (>10 items) get discounts'
            ],
            'temporal_patterns': [
                'Monthly seasonality patterns',
                'Increasing trend over time'
            ],
            'use_cases': [
                'Sales forecasting',
                'Customer behavior analysis',
                'Inventory management',
                'Revenue optimization'
            ]
        },
        'iot_sensor_data': {
            'description': 'IoT sensor monitoring and telemetry data',
            'columns': [
                'sensor_id: Sensor device identifier',
                'timestamp: Measurement timestamp',
                'temperature: Temperature readings (normal distribution)',
                'humidity: Humidity percentage (beta distribution, correlated with temperature)',
                'pressure: Atmospheric pressure (normal distribution)',
                'light_level: Light intensity (gamma distribution)',
                'battery_level: Device battery percentage',
                'status: Device status (Online/Offline/Maintenance/Error)'
            ],
            'business_rules': [
                'Low battery (<10%) triggers error status',
                'High temperature (>40°C) triggers error status'
            ],
            'temporal_patterns': [
                'Daily seasonality (light/temperature cycles)',
                'Added sensor noise and outliers'
            ],
            'use_cases': [
                'Environmental monitoring',
                'Predictive maintenance',
                'Anomaly detection',
                'Energy optimization'
            ]
        }
    }
    
    if template:
        info = templates_info.get(template)
        if info:
            click.echo(f"\n[INFO] Template: {template}")
            click.echo(f"[INFO] Description: {info['description']}\n")
            
            click.echo("[INFO] Columns:")
            for col in info['columns']:
                click.echo(f"   • {col}")
            
            if info.get('business_rules'):
                click.echo("\n📏 Business Rules:")
                for rule in info['business_rules']:
                    click.echo(f"   • {rule}")
            
            if info.get('temporal_patterns'):
                click.echo("\n📈 Temporal Patterns:")
                for pattern in info['temporal_patterns']:
                    click.echo(f"   • {pattern}")
            
            click.echo("\n🎯 Use Cases:")
            for use_case in info['use_cases']:
                click.echo(f"   • {use_case}")
        else:
            click.echo(f"Template '{template}' not found", err=True)
    else:
        click.echo("\n📋 Available Templates:\n")
        for name, info in templates_info.items():
            click.echo(f"🔹 {name}: {info['description']}")
        
        click.echo(f"\nUse --template <name> to see detailed information about a specific template.")


@synthetic.command()
@click.option('--output-file', '-o', default='./dataset_config_template.json',
              help='Output file for configuration template')
@click.option('--template', '-t', 
              type=click.Choice(['basic', 'advanced', 'custom']),
              default='basic', help='Template complexity level')
def create_config(output_file: str, template: str):
    """Create configuration file templates."""
    
    if template == 'basic':
        config = {
            "num_rows": 1000,
            "seed": 42,
            "columns": [
                {
                    "name": "id",
                    "data_type": "id",
                    "unique": True,
                    "id_prefix": "ROW",
                    "id_format": "{prefix}_{:06d}"
                },
                {
                    "name": "category",
                    "data_type": "categorical",
                    "distribution": {
                        "categories": ["A", "B", "C"],
                        "weights": [0.5, 0.3, 0.2]
                    }
                },
                {
                    "name": "value",
                    "data_type": "numeric",
                    "distribution": {
                        "distribution_type": "normal",
                        "mean": 100.0,
                        "std": 15.0,
                        "min_value": 0.0
                    }
                }
            ]
        }
    
    elif template == 'advanced':
        config = {
            "num_rows": 5000,
            "seed": 42,
            "consistency_level": 0.95,
            "completeness_level": 0.98,
            "add_outliers": True,
            "outlier_probability": 0.02,
            "add_noise": True,
            "noise_level": 0.05,
            "add_temporal_patterns": True,
            "seasonality": "monthly",
            "trend": "increasing",
            "business_rules": [
                "if age > 65 then status = 'senior'",
                "if income > 100000 then tier = 'premium'"
            ],
            "export_config": {
                "format": "parquet",
                "compression": "snappy",
                "include_metadata": True,
                "create_checksums": True
            },
            "columns": [
                {
                    "name": "user_id",
                    "data_type": "id",
                    "unique": True,
                    "id_prefix": "USER",
                    "id_format": "{prefix}_{:08d}"
                },
                {
                    "name": "age",
                    "data_type": "numeric",
                    "distribution": {
                        "distribution_type": "normal",
                        "mean": 35.0,
                        "std": 12.0,
                        "min_value": 18.0,
                        "max_value": 80.0
                    }
                },
                {
                    "name": "income",
                    "data_type": "numeric",
                    "distribution": {
                        "distribution_type": "lognormal",
                        "mean": 10.5,
                        "std": 0.8,
                        "min_value": 20000.0
                    },
                    "correlation_with": "age",
                    "correlation_strength": 0.3
                },
                {
                    "name": "status",
                    "data_type": "categorical",
                    "distribution": {
                        "categories": ["active", "inactive", "pending", "senior"],
                        "weights": [0.6, 0.2, 0.15, 0.05]
                    }
                },
                {
                    "name": "registration_date",
                    "data_type": "datetime",
                    "datetime_start": "2020-01-01T00:00:00",
                    "datetime_end": "2024-01-01T00:00:00"
                },
                {
                    "name": "description",
                    "data_type": "text",
                    "text_length_range": [10, 100],
                    "nullable": True,
                    "null_probability": 0.1
                }
            ]
        }
    
    else:  # custom
        config = {
            "_description": "Custom dataset configuration template",
            "_instructions": [
                "Modify this template to create your custom synthetic dataset",
                "See documentation for all available options",
                "Run 'opensynthetics synthetic describe-template' for examples"
            ],
            "num_rows": 1000,
            "seed": 42,
            "consistency_level": 0.9,
            "completeness_level": 0.95,
            "add_outliers": False,
            "outlier_probability": 0.05,
            "add_noise": False,
            "noise_level": 0.1,
            "add_temporal_patterns": False,
            "seasonality": None,
            "trend": None,
            "business_rules": [],
            "constraints": [],
            "export_config": {
                "format": "parquet",
                "compression": "snappy",
                "include_metadata": True,
                "preserve_dtypes": True,
                "create_checksums": True
            },
            "columns": [
                {
                    "_example_column": "This is an example - replace with your columns",
                    "name": "example_column",
                    "data_type": "numeric",
                    "distribution": {
                        "distribution_type": "normal",
                        "mean": 0.0,
                        "std": 1.0
                    },
                    "nullable": False,
                    "null_probability": 0.0,
                    "unique": False,
                    "correlation_with": None,
                    "correlation_strength": 0.0
                }
            ],
            "_available_data_types": ["numeric", "categorical", "datetime", "text", "boolean", "id"],
            "_available_distributions": ["normal", "uniform", "exponential", "gamma", "beta", "lognormal"],
            "_available_formats": ["json", "jsonl", "csv", "parquet", "hdf5", "excel", "feather"]
        }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"[OK] Configuration template created: {output_path}")
    click.echo(f"[INFO] Template type: {template}")
    click.echo(f"[INFO] Edit the file to customize your dataset configuration")


if __name__ == '__main__':
    synthetic() 