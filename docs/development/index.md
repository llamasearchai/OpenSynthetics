# OpenSynthetics Development Guide

This section provides information for developers who want to contribute to or extend OpenSynthetics.

## Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/llamasearchai/opensynthetics.git
   cd opensynthetics
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Project Structure

```
opensynthetics/
├── opensynthetics/         # Main package directory
│   ├── api/                # API implementation
│   ├── cli/                # Command-line interface
│   ├── core/               # Core functionality
│   ├── data_ops/           # Data operations
│   ├── datagen/            # Data generation strategies
│   ├── llm_core/           # LLM integration
│   ├── training_eval/      # Training and evaluation tools
│   └── web_ui/             # Web UI implementation
├── tests/                  # Test suite
├── docs/                   # Documentation
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # Project overview
```

## Code Style

OpenSynthetics follows the [Black](https://black.readthedocs.io/) code style with additional checks from [Ruff](https://github.com/charliermarsh/ruff). These are enforced by pre-commit hooks.

## Testing

Run the test suite with pytest:

```bash
pytest
```

For more specific tests:

```bash
pytest tests/unit/  # Unit tests only
pytest tests/integration/  # Integration tests only
```

## Building Documentation

Documentation is built using MkDocs:

```bash
mkdocs serve  # Serve documentation locally
mkdocs build  # Build documentation
```

## Contribution Workflow

1. Create a fork of the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit your changes (`git commit -m "Add feature: your feature name"`)
6. Push to your fork (`git push origin feature/your-feature-name`)
7. Create a Pull Request

## Contact

For questions or help, please contact Nik Jois at nikjois@llamasearch.ai 