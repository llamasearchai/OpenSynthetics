# OpenSynthetics Documentation

## Overview

OpenSynthetics is a production-ready, enterprise-grade synthetic data generation platform designed for machine learning, testing, and analytics workflows. Built with modern Python technologies and featuring a comprehensive REST API, interactive web interface, and CLI tools.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenSynthetics Platform                 │
├─────────────────────────────────────────────────────────────┤
│  Web UI (React-like SPA)  │  REST API (FastAPI)           │
│  - Dashboard & Analytics  │  - Authentication & Security   │
│  - 3D Visualizations     │  - Workspace Management        │
│  - Real-time Monitoring  │  - Data Generation Engines     │
├─────────────────────────────────────────────────────────────┤
│                    Core Engine Layer                       │
│  - Workspace Management  │  - Generation Strategies       │
│  - Dataset Operations    │  - Quality Validation          │
│  - Configuration System  │  - Security & Access Control   │
├─────────────────────────────────────────────────────────────┤
│                    Storage & Integration                   │
│  - SQLite (Metadata)    │  - File System (Datasets)      │
│  - Cloud Storage APIs   │  - External Integrations       │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and serialization
- **SQLite**: Lightweight database for metadata
- **Loguru**: Advanced logging system

**Frontend:**
- **Three.js**: 3D visualizations and data landscapes
- **D3.js**: Interactive charts and network graphs
- **Chart.js**: Statistical visualizations
- **Modern CSS Grid**: Responsive design system

**Data Processing:**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **DuckDB**: In-memory analytics (planned)

## Key Features

### 1. Advanced Data Generation
- Multiple generation strategies (tabular, time-series, graph data)
- Configurable parameters with JSON schema validation
- Quality metrics and validation pipelines
- Scalable generation for large datasets

### 2. Enterprise Security
- API key authentication with scoped access
- Rate limiting and usage analytics
- Secure credential storage
- Audit logging for compliance

### 3. Interactive Visualizations
- Real-time 3D data landscapes
- Network topology visualizations
- Statistical distribution analysis
- Performance monitoring dashboards

### 4. Workspace Management
- Project isolation and organization
- Version control for datasets
- Collaborative features
- Export/import capabilities

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/opensynthetics.git
cd opensynthetics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Initialize configuration
python setup_dev.py

# Start server
python start_server.py
```

### Basic Usage

```python
from opensynthetics.core.workspace import Workspace

# Create workspace
workspace = Workspace.create(
    name="analytics_project",
    description="Customer analytics dataset"
)

# Generate synthetic data
from opensynthetics.datagen.engine import Engine

engine = Engine(workspace)
result = engine.generate(
    strategy="customer_data",
    parameters={"count": 1000, "include_pii": False},
    output_dataset="customers_2024"
)

print(f"Generated {result['count']} records")
```

### API Usage

```bash
# Create workspace
curl -X POST "http://localhost:8000/api/v1/workspaces" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"name": "demo_workspace", "description": "Demo project"}'

# Generate data
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "workspace": "demo_workspace",
    "strategy": "tabular_random",
    "parameters": {"num_rows": 1000, "num_columns": 10},
    "dataset": "sample_data"
  }'
```

## API Reference

### Authentication
All API endpoints require authentication via `X-API-Key` header.

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/workspaces` | List all workspaces |
| POST | `/api/v1/workspaces` | Create new workspace |
| GET | `/api/v1/workspaces/{id}` | Get workspace details |
| GET | `/api/v1/strategies` | List generation strategies |
| POST | `/api/v1/generate` | Generate synthetic data |
| GET | `/api/v1/config` | Get system configuration |

### Response Format

All API responses follow a consistent format:

```json
{
  "status": "success",
  "data": { ... },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456",
    "version": "1.0.0"
  }
}
```

## Deployment

### Production Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  opensynthetics:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENSYNTHETICS_ENV=production
      - OPENSYNTHETICS_API_KEY=your-secure-key
    volumes:
      - ./data:/app/data
      - ./config:/app/config
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENSYNTHETICS_ENV` | Environment (dev/prod) | `development` |
| `OPENSYNTHETICS_API_KEY` | Default API key | Generated |
| `OPENSYNTHETICS_BASE_DIR` | Data storage path | `~/.opensynthetics` |
| `OPENSYNTHETICS_LOG_LEVEL` | Logging level | `INFO` |

## Performance & Scalability

### Benchmarks

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Workspace Creation | < 50ms | Including metadata setup |
| Data Generation (1K rows) | < 200ms | Tabular strategy |
| Data Generation (100K rows) | < 5s | With quality validation |
| API Response Time | < 100ms | 95th percentile |

### Scaling Considerations

- **Horizontal Scaling**: Stateless API design supports load balancing
- **Data Storage**: Configurable storage backends (local, S3, etc.)
- **Memory Usage**: Streaming generation for large datasets
- **Concurrent Users**: Thread-safe operations with proper locking

## Security

### Best Practices

1. **API Key Management**
   - Regular key rotation
   - Scoped permissions
   - Usage monitoring

2. **Data Protection**
   - Encrypted storage options
   - Access logging
   - PII detection and masking

3. **Network Security**
   - HTTPS enforcement
   - CORS configuration
   - Rate limiting

## Development

### Project Structure

```
opensynthetics/
├── api/                 # FastAPI application
│   ├── routers/        # API route handlers
│   ├── models/         # Pydantic models
│   └── middleware/     # Custom middleware
├── core/               # Core business logic
│   ├── workspace.py    # Workspace management
│   ├── config.py       # Configuration system
│   └── security.py     # Authentication & authorization
├── datagen/            # Data generation engines
├── web_ui/             # Frontend assets
└── tests/              # Test suites
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=opensynthetics --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Code Quality

```bash
# Format code
black opensynthetics tests

# Type checking
mypy opensynthetics

# Linting
ruff check opensynthetics
```

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make changes and add tests
5. Ensure all tests pass and code is formatted
6. Submit a pull request

### Code Standards

- **Type Hints**: All functions must include type annotations
- **Documentation**: Docstrings for all public APIs
- **Testing**: Minimum 90% test coverage
- **Performance**: No regressions in benchmark tests

## Roadmap

### Version 1.1 (Q2 2024)
- Advanced ML-based generation strategies
- Real-time collaboration features
- Cloud deployment templates
- Enhanced visualization capabilities

### Version 1.2 (Q3 2024)
- Multi-tenant architecture
- Advanced data lineage tracking
- Integration with popular ML frameworks
- Performance optimizations

### Version 2.0 (Q4 2024)
- Distributed generation cluster
- Advanced AI-powered data synthesis
- Enterprise SSO integration
- Compliance frameworks (GDPR, HIPAA)

## Support

- **Documentation**: [docs.opensynthetics.io](https://docs.opensynthetics.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/opensynthetics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/opensynthetics/discussions)
- **Email**: support@opensynthetics.io

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details. 