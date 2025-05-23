"""Typer wrapper for synthetic dataset CLI commands."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
import pandas as pd
from rich.console import Console
from loguru import logger

from opensynthetics.datagen.synthetic_datasets import (
    SyntheticDatasetFactory, 
    DatasetTemplate,
    SyntheticDatasetConfig,
)
from opensynthetics.data_ops.export_utils import ExportConfig, DataExporter, BatchExporter
from opensynthetics.training_eval.benchmark import (
    SyntheticDatasetBenchmark, 
    BenchmarkConfig,
    BenchmarkSuite
)
from opensynthetics.core.exceptions import GenerationError, ProcessingError, EvaluationError

app = typer.Typer(name="synthetic", help="Synthetic dataset creation and benchmarking")
console = Console()


@app.command()
def generate(
    template: Optional[str] = typer.Option(None, "--template", "-t", 
                                          help="Pre-defined dataset template: customer_data, sales_data, iot_sensor_data"),
    config_file: Optional[Path] = typer.Option(None, "--config-file", "-c",
                                              help="JSON configuration file for custom dataset"),
    num_rows: int = typer.Option(1000, "--num-rows", "-n", help="Number of rows to generate"),
    output_dir: Path = typer.Option("./synthetic_output", "--output-dir", "-o", help="Output directory"),
    format: str = typer.Option("parquet", "--format", "-f", 
                              help="Export format: json, jsonl, csv, parquet, hdf5, excel, feather"),
    compression: Optional[str] = typer.Option(None, "--compression",
                                            help="Compression type: gzip, bz2, zip, snappy, lz4"),
    benchmark: bool = typer.Option(True, "--benchmark/--no-benchmark", help="Run quality benchmarking"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    name: Optional[str] = typer.Option(None, "--name", help="Dataset name")
):
    """Generate synthetic datasets."""
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
            console.print(f"[blue]Generating dataset using template: {template}[/blue]")
            
            # Create kwargs for create_dataset, not for the template
            create_kwargs = {
                'num_rows': num_rows
            }
            if seed is not None:
                create_kwargs['seed'] = seed
                
            result = factory.create_dataset(
                config=template,
                name=name,
                benchmark=benchmark,
                export=False,
                **create_kwargs
            )
            
        elif config_file:
            # Load custom configuration
            console.print(f"[blue]Loading configuration from: {config_file}[/blue]")
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Parse configuration
            config = SyntheticDatasetConfig.model_validate(config_data)
            if seed is not None:
                config.seed = seed
            
            result = factory.create_dataset(
                config=config,
                name=name,
                benchmark=benchmark,
                export=False
            )
            
        else:
            console.print("[red]Error: Must specify either --template or --config-file[/red]")
            raise typer.Exit(1)
        
        # Export dataset
        if "dataset" in result and result["dataset"] is not None:
            exporter = DataExporter(export_config)
            dataset_name = name or "synthetic_dataset"
            export_path = output_dir / f"{dataset_name}.{format}"
            export_metadata = exporter.export_dataset(result["dataset"], export_path)
            result["export_metadata"] = export_metadata.model_dump()
        
        # Save metadata
        metadata_path = output_dir / f"{name or 'synthetic_dataset'}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "config": result.get("config"),
                "metadata": result.get("metadata"),
                "benchmark_metrics": result.get("benchmark_metrics"),
                "export_metadata": result.get("export_metadata")
            }, f, indent=2, default=str)
        
        console.print("[green]Dataset generated successfully![/green]")
        console.print(f"Rows: {result['metadata']['num_rows']}")
        console.print(f"Columns: {result['metadata']['num_columns']}")
        console.print(f"Size: {result['metadata']['memory_usage_mb']:.2f} MB")
        
        if benchmark and result.get("benchmark_metrics"):
            metrics = result["benchmark_metrics"]
            console.print(f"Quality Score: {metrics.get('overall_quality_score', 0):.3f}")
        
        console.print(f"Output: {output_dir}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file to benchmark"),
    reference_path: Optional[Path] = typer.Option(None, "--reference", "-r",
                                                  help="Reference dataset for comparison"),
    target_column: Optional[str] = typer.Option(None, "--target", "-t",
                                               help="Target column for ML evaluation"),
    output_dir: Path = typer.Option("./benchmark_output", "--output-dir", "-o",
                                   help="Output directory for results"),
    sample_size: Optional[int] = typer.Option(None, "--sample-size",
                                            help="Sample size for large datasets"),
    visualizations: bool = typer.Option(True, "--visualizations/--no-visualizations",
                                       help="Create visualization plots"),
    report: bool = typer.Option(True, "--report/--no-report",
                               help="Generate comprehensive report")
):
    """Benchmark synthetic dataset quality."""
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        console.print(f"[blue]Loading dataset from: {dataset_path}[/blue]")
        if dataset_path.suffix == '.parquet':
            synthetic_df = pd.read_parquet(dataset_path)
        elif dataset_path.suffix == '.csv':
            synthetic_df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.json':
            synthetic_df = pd.read_json(dataset_path)
        else:
            console.print(f"[red]Error: Unsupported file format: {dataset_path.suffix}[/red]")
            raise typer.Exit(1)
        
        reference_df = None
        if reference_path:
            console.print(f"[blue]Loading reference dataset from: {reference_path}[/blue]")
            if reference_path.suffix == '.parquet':
                reference_df = pd.read_parquet(reference_path)
            elif reference_path.suffix == '.csv':
                reference_df = pd.read_csv(reference_path)
            elif reference_path.suffix == '.json':
                reference_df = pd.read_json(reference_path)
        
        # Create benchmark configuration
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
                console.print(f"[yellow]Auto-detected target column: {target_column} (categorical)[/yellow]")
            elif len(numeric_cols) > 0:
                target_column = numeric_cols[0]
                console.print(f"[yellow]Auto-detected target column: {target_column} (numeric)[/yellow]")
        
        console.print("[blue]Running comprehensive dataset benchmark...[/blue]")
        metrics = benchmarker.benchmark_dataset(
            synthetic_data=synthetic_df,
            reference_data=reference_df,
            target_column=target_column
        )
        
        # Save results
        results_path = output_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "metrics": metrics.model_dump(),
                "dataset_info": {
                    "synthetic_shape": synthetic_df.shape,
                    "reference_shape": reference_df.shape if reference_df is not None else None,
                    "target_column": target_column
                },
                "config": benchmark_config.model_dump()
            }, f, indent=2, default=str)
        
        # Print summary
        console.print("[green]Benchmark Results:[/green]")
        console.print(f"  Overall Quality Score: {metrics.overall_quality_score:.3f}")
        console.print(f"  Completeness: {metrics.completeness_score:.3f}")
        console.print(f"  Consistency: {metrics.consistency_score:.3f}")
        console.print(f"  Uniqueness: {metrics.uniqueness_score:.3f}")
        console.print(f"  Validity: {metrics.validity_score:.3f}")
        console.print(f"  Statistical Fidelity: {metrics.statistical_fidelity:.3f}")
        console.print(f"  ML Utility: {metrics.ml_utility_score:.3f}")
        
        if metrics.privacy_score > 0:
            console.print(f"  Privacy Score: {metrics.privacy_score:.3f}")
        
        console.print(f"[blue]Results saved to: {output_dir}[/blue]")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_templates():
    """List available dataset templates."""
    console.print("[green]Available Dataset Templates:[/green]")
    console.print()
    
    templates = {
        'customer_data': {
            'description': 'Customer demographic and profile data',
            'columns': 7,
            'features': 'Age/income correlation, business rules, realistic distributions'
        },
        'sales_data': {
            'description': 'Sales transaction data with temporal patterns',
            'columns': 8,
            'features': 'Seasonal trends, product categories, payment methods'
        },
        'iot_sensor_data': {
            'description': 'IoT sensor readings with environmental data',
            'columns': 8,
            'features': 'Sensor correlations, anomalies, temporal patterns'
        }
    }
    
    for name, info in templates.items():
        console.print(f"[cyan]{name}[/cyan]")
        console.print(f"  Description: {info['description']}")
        console.print(f"  Columns: {info['columns']}")
        console.print(f"  Features: {info['features']}")
        console.print()


if __name__ == "__main__":
    app() 