# OpenSynthetics

Advanced synthetic data generation platform with comprehensive API, CLI, and web interface capabilities.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

OpenSynthetics is a powerful, enterprise-ready platform for generating high-quality synthetic data. It provides a complete solution for data scientists, engineers, and researchers who need realistic test data while maintaining privacy and compliance standards.

### Key Features

- **Advanced Data Generation**: Create realistic synthetic datasets using multiple generation strategies
- **Web Interface**: Modern, responsive UI with real-time visualizations and analytics
- **RESTful API**: Comprehensive API with OpenAPI documentation
- **CLI Tools**: Command-line interface for automation and scripting
- **Workspace Management**: Organize datasets and projects efficiently
- **Quality Benchmarking**: Built-in tools to measure synthetic data quality
- **Multiple Export Formats**: Support for JSON, CSV, Parquet, and more
- **Extensible Architecture**: Plugin system for custom generation strategies

## Installation

### Requirements

- Python 3.11 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/opensynthetics.git
cd opensynthetics
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Run the setup script:
```bash
python setup_dev.py
```

5. Start the server:
```bash
python start_server.py
```

6. Open your browser and navigate to:
   - Web UI: http://localhost:8000/ui
   - API Documentation: http://localhost:8000/docs

## Usage

### Web Interface

The web interface provides a comprehensive dashboard for:
- Creating and managing workspaces
- Generating synthetic datasets
- Visualizing data with advanced charts
- Monitoring system performance
- Managing API integrations

### Command Line Interface

Generate synthetic data from the command line:

```bash
# Generate customer data
opensynthetics generate customer-data output.json --rows 1000

# Generate with specific template
opensynthetics generate --template sales_data --output sales.csv --format csv

# Benchmark data quality
opensynthetics benchmark dataset.json --reference original.json
```

### API Usage

```python
import requests

# Generate synthetic data via API
response = requests.post(
    "http://localhost:8000/api/v1/generate/jobs",
    headers={"X-API-Key": "your-api-key"},
    json={
        "workspace_path": "/path/to/workspace",
        "strategy": "engineering_problems",
        "parameters": {"count": 100},
        "output_dataset": "problems_dataset"
    }
)

result = response.json()
print(f"Generated {result['count']} items")
```

## Architecture

OpenSynthetics follows a modular architecture:

```
opensynthetics/
├── api/           # FastAPI application and routers
├── cli/           # Command-line interface
├── core/          # Core functionality (workspaces, configuration)
├── data_ops/      # Data operations and validation
├── datagen/       # Generation engines and strategies
├── llm_core/      # LLM integrations for advanced generation
├── training/      # Model training utilities
├── training_eval/ # Evaluation and benchmarking
└── web_ui/        # Web interface assets
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=opensynthetics

# Run specific test file
pytest tests/unit/test_workspace.py
```

### Code Quality

The project uses several tools to maintain code quality:

```bash
# Format code
black opensynthetics tests

# Lint code
ruff check opensynthetics

# Type checking
mypy opensynthetics
```

### Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Configuration

OpenSynthetics uses a configuration file located at `~/.opensynthetics/opensynthetics_config.json`. You can manage settings using the CLI:

```bash
# Set OpenAI API key
opensynthetics config set api_keys.openai "your-api-key"

# View current configuration
opensynthetics config get
```

## Deployment

### Docker

Build and run with Docker:

```bash
docker build -t opensynthetics .
docker run -p 8000:8000 opensynthetics
```

### Docker Compose

For a complete setup with database:

```bash
docker-compose up -d
```

## API Documentation

The API provides the following main endpoints:

- `GET /health` - Health check
- `GET /api/v1/workspaces` - List workspaces
- `POST /api/v1/workspaces` - Create workspace
- `POST /api/v1/generate/jobs` - Create generation job
- `GET /api/v1/strategies` - List available strategies

Full API documentation is available at http://localhost:8000/docs when the server is running.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [docs.opensynthetics.io](https://docs.opensynthetics.io)
- Issues: [GitHub Issues](https://github.com/yourusername/opensynthetics/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/opensynthetics/discussions)

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- UI powered by [Three.js](https://threejs.org/) and [D3.js](https://d3js.org/)
- Data processing with [Pandas](https://pandas.pydata.org/) and [DuckDB](https://duckdb.org/)

## Complete Working Example

### 1. Create a Workspace (API)

You can create a workspace using the API:

```bash
curl -X POST "http://localhost:8000/api/v1/workspaces" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{
    "name": "demo_workspace",
    "description": "Demo workspace for OpenSynthetics",
    "tags": ["demo", "example"]
  }'
```

### 2. Generate a Dataset (API)

After creating a workspace, generate a dataset:

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{
    "workspace": "demo_workspace",
    "strategy": "tabular_random",
    "parameters": {"num_rows": 100, "num_columns": 5},
    "dataset": "demo_dataset"
  }'
```

### 3. Visualize in the Web UI

- Open your browser and go to [http://localhost:8000/ui/](http://localhost:8000/ui/)
- Navigate to the **Workspaces** page to see your new workspace.
- Go to the **Datasets** page to view your generated dataset.
- Use the **Visualize** or **Analytics** tabs to explore the data with interactive charts and 3D visualizations.

### 4. Example Python Usage

```python
from opensynthetics.core.workspace import Workspace

# Create a workspace
ws = Workspace.create(name="demo_workspace", description="Demo workspace")

# Create a dataset
ds = ws.create_dataset(name="demo_dataset", description="Demo dataset")

# Add data
data = [
    {"id": 1, "name": "Alice", "score": 95},
    {"id": 2, "name": "Bob", "score": 88},
    {"id": 3, "name": "Charlie", "score": 92},
]
ds.add_data(data)

# List datasets
print(ws.list_datasets())
```

### 5. UI Demo Walkthrough

1. Click **Create Workspace** on the Dashboard or Workspaces page.
2. Enter a name (e.g., `demo_workspace`) and description, then submit.
3. Go to **Generate** and create a new dataset in your workspace.
4. View and analyze your dataset in the **Datasets** and **Visualize** sections. 