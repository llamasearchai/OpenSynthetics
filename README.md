# OpenSynthetics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-95%2F98_passing-brightgreen.svg)](#testing)

**Advanced Synthetic Data Generation Platform** - Create high-quality synthetic datasets with comprehensive benchmarking, export capabilities, and LLM training pipeline integration.

OpenSynthetics is a comprehensive platform for generating, benchmarking, and exporting synthetic datasets with advanced quality assessment, multiple format support, and machine learning utility evaluation.

## Key Features

### Synthetic Data Generation
- **Pre-built Templates**: Customer data, sales transactions, IoT sensor data
- **Custom Configurations**: Define your own column schemas, distributions, and correlations
- **Advanced Patterns**: Temporal trends, seasonality, outliers, and business rules
- **Data Types**: Numeric, categorical, datetime, text, boolean, and ID fields
- **Correlation Modeling**: Cholesky decomposition for realistic data relationships

### Quality Benchmarking
- **Multi-dimensional Quality Metrics**: Completeness, consistency, uniqueness, validity
- **Statistical Fidelity**: Distribution similarity and correlation preservation
- **ML Utility Evaluation**: Classification and regression performance assessment
- **Privacy Assessment**: Disclosure risk and privacy score calculation
- **Visualization**: Radar charts, distribution plots, correlation heatmaps

### Export & Integration
- **Multiple Formats**: JSON, JSONL, CSV, Parquet, HDF5, Excel, Feather
- **Compression Support**: Gzip, Snappy, LZ4, Brotli, and more
- **Metadata Preservation**: Schema information, quality metrics, checksums
- **Batch Processing**: Multi-dataset generation and export
- **API Integration**: FastAPI-based REST API

### LLM Integration
- **Scientific Literature Processing**: arXiv and PubMed integration
- **PDF Processing**: Extract training data from scientific papers
- **Training Pipeline**: QLoRA fine-tuning with Hugging Face Transformers
- **Data Validation**: Comprehensive schema validation and error handling

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OpenSynthetics.git
cd OpenSynthetics

# Install with all dependencies
pip install -e ".[all]"

# Or install specific groups
pip install -e ".[training,benchmarking,export]"
```

### Basic Usage

#### 1. Generate Synthetic Data

```python
from opensynthetics.datagen.synthetic_datasets import SyntheticDatasetFactory

# Create factory
factory = SyntheticDatasetFactory()

# Generate customer data
result = factory.create_dataset(
    config='customer_data',
    num_rows=1000,
    benchmark=True,
    export=True
)

# Access the generated dataset
df = result['dataset']
quality_score = result['benchmark_metrics']['overall_quality_score']
print(f"Generated {len(df)} rows with quality score: {quality_score:.3f}")
```

#### 2. Export to Multiple Formats

```python
from opensynthetics.data_ops.export_utils import DataExporter, ExportConfig

# Export to Parquet with compression
config = ExportConfig(format='parquet', compression='snappy')
exporter = DataExporter(config)
metadata = exporter.export_dataset(df, 'output.parquet')

print(f"Exported {metadata.total_records} records, compression ratio: {metadata.compression_ratio:.2f}")
```

#### 3. Quality Benchmarking

```python
from opensynthetics.training_eval.benchmark import SyntheticDatasetBenchmark, BenchmarkConfig

# Configure benchmarking
config = BenchmarkConfig(
    ml_evaluation=True,
    create_visualizations=True,
    generate_report=True
)

# Run benchmark
benchmarker = SyntheticDatasetBenchmark(config)
metrics = benchmarker.benchmark_dataset(df, target_column='category')

print(f"Overall Quality: {metrics.overall_quality_score:.3f}")
print(f"ML Utility: {metrics.ml_utility_score:.3f}")
```

### Command Line Interface

```bash
# Generate synthetic datasets
opensynthetics synthetic generate --template customer_data --num-rows 5000 --format parquet

# Run quality benchmarking
opensynthetics synthetic benchmark dataset.parquet --visualizations --report

# Generate multiple datasets
opensynthetics synthetic batch-generate --config-file batch_config.json

# Scientific data processing
opensynthetics scientific literature workspace_path --arxiv-categories cs.AI cs.LG

# Start API server
opensynthetics api serve --host 0.0.0.0 --port 8000

# Configuration management
opensynthetics config set api_keys.openai your_api_key
opensynthetics config get api_keys
```

## Available Templates

### Customer Data
- **Columns**: customer_id, age, income, gender, region, registration_date, is_premium
- **Features**: Demographic correlations, business rules, realistic distributions
- **Use Cases**: Customer segmentation, marketing analysis, demographic studies

### Sales Data
- **Columns**: transaction_id, customer_id, product_category, amount, quantity, transaction_date, payment_method, discount_applied
- **Features**: Temporal patterns, seasonal trends, transaction correlations
- **Use Cases**: Sales forecasting, customer behavior analysis, revenue optimization

### IoT Sensor Data
- **Columns**: sensor_id, timestamp, temperature, humidity, pressure, light_level, battery_level, status
- **Features**: Sensor correlations, temporal patterns, anomaly simulation
- **Use Cases**: Environmental monitoring, predictive maintenance, anomaly detection

## Architecture

```
OpenSynthetics/
├── opensynthetics/
│   ├── api/                    # FastAPI REST API
│   ├── cli/                    # Command-line interface
│   ├── core/                   # Core configuration and workspace
│   ├── datagen/                # Data generation engines
│   │   ├── synthetic_datasets.py  # Main synthetic data generator
│   │   └── engine.py          # Generation engine
│   ├── data_ops/              # Data operations
│   │   ├── export_utils.py    # Multi-format export
│   │   └── validation.py      # Data validation
│   ├── training_eval/         # Training and evaluation
│   │   └── benchmark.py       # Quality benchmarking
│   ├── llm_core/             # LLM integration
│   └── web_ui/               # Web interface
├── tests/                     # Comprehensive test suite
└── docs/                      # Documentation
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=opensynthetics --cov-report=html

# Run specific test categories
python -m pytest tests/test_synthetic_datasets.py -v
python -m pytest tests/unit/ -v
```

**Test Status**: 95/98 tests passing (3 expected failures for error condition testing)

## Quality Metrics

The platform provides comprehensive quality assessment:

- **Completeness**: Measures missing values and data coverage
- **Consistency**: Evaluates data format and type consistency
- **Uniqueness**: Assesses duplicate records and unique value ratios
- **Validity**: Validates data constraints and business rules
- **Statistical Fidelity**: Compares distributions and correlations
- **ML Utility**: Evaluates predictive performance
- **Privacy Score**: Assesses disclosure risk

## Configuration

### Environment Configuration

```bash
# Set API keys
opensynthetics config set api_keys.openai sk-your-key-here
opensynthetics config set api_keys.huggingface hf-your-key-here

# Configure storage
opensynthetics config set storage.base_dir /path/to/data
opensynthetics config set export.default_format parquet
```

### Custom Dataset Configuration

```json
{
  "num_rows": 10000,
  "columns": [
    {
      "name": "user_id",
      "data_type": "id",
      "unique": true,
      "id_prefix": "USER"
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
      "name": "category",
      "data_type": "categorical",
      "distribution": {
        "categories": ["A", "B", "C"],
        "weights": [0.5, 0.3, 0.2]
      }
    }
  ],
  "business_rules": [
    "if age > 65 then category = 'senior'"
  ],
  "add_temporal_patterns": true,
  "seasonality": "monthly"
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/OpenSynthetics.git
cd OpenSynthetics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run linting
ruff check opensynthetics/
ruff format opensynthetics/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/OpenSynthetics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/OpenSynthetics/discussions)

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://pydantic-docs.helpmanual.io/), and [Pandas](https://pandas.pydata.org/)
- Inspired by the need for high-quality synthetic data in AI/ML research
- Special thanks to the open-source community for their invaluable contributions

---

**Ready to generate high-quality synthetic data?**

```bash
pip install -e ".[all]"
opensynthetics synthetic generate --template customer_data --num-rows 1000
``` 