# Contributing to OpenSynthetics

Thank you for your interest in contributing to OpenSynthetics! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

By participating in this project, you are expected to uphold our code of conduct:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a feature branch
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Setup

```bash
# Clone your fork
git clone https://github.com/your-username/OpenSynthetics.git
cd OpenSynthetics

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Example configuration
OPENSYNTHETICS_CONFIG_PATH=/path/to/your/config.json
OPENAI_API_KEY=your-openai-key-here
```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check if an issue is already assigned before starting work
- Comment on the issue to let others know you're working on it

### 2. Create a Branch

```bash
# Create a new branch for your feature
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 3. Make Changes

- Follow the coding standards outlined below
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Your Changes

Follow our commit message conventions (see below).

### 5. Submit a Pull Request

- Push your branch to your fork
- Create a pull request with a clear description
- Link any related issues
- Request review from maintainers

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black opensynthetics/`
- Use isort for import sorting: `isort opensynthetics/`
- Use ruff for linting: `ruff check opensynthetics/`

### Type Hints

All functions and methods must include type hints:

```python
def process_data(data: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Process input data and return status."""
    # Implementation here
    return True, "Success"
```

### Documentation

- Use Google-style docstrings for all functions and classes
- Include parameter descriptions and return value information
- Add examples for complex functions

```python
def generate_data(strategy: str, count: int) -> List[Dict[str, Any]]:
    """Generate synthetic data using the specified strategy.
    
    Args:
        strategy: The generation strategy to use (e.g., "engineering_problems")
        count: Number of data items to generate
        
    Returns:
        List of generated data items as dictionaries
        
    Raises:
        ValueError: If strategy is not supported
        GenerationError: If data generation fails
        
    Example:
        >>> data = generate_data("engineering_problems", 5)
        >>> len(data)
        5
    """
```

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately

```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise ProcessingError(f"Failed to process data: {e}") from e
```

## Testing Guidelines

### Test Structure

- Place tests in the `tests/` directory
- Mirror the source code structure
- Use descriptive test names

### Writing Tests

```python
import pytest
from opensynthetics.core.config import Config

class TestConfig:
    """Tests for the Config class."""
    
    def test_load_config_with_valid_file(self):
        """Test loading configuration from a valid file."""
        # Arrange
        config_data = {"environment": "test"}
        
        # Act
        config = Config(config_data)
        
        # Assert
        assert config.environment == "test"
        
    def test_load_config_with_missing_file(self):
        """Test loading configuration when file is missing."""
        with pytest.raises(FileNotFoundError):
            Config.load("/nonexistent/path")
```

### Test Coverage

- Aim for 90%+ test coverage
- Test both happy paths and error conditions
- Include edge cases and boundary conditions

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=opensynthetics

# Run specific test file
python -m pytest tests/unit/test_config.py

# Run tests with verbose output
python -m pytest -v
```

## Documentation

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms or business logic

### README Updates

Update the README.md when:
- Adding new features
- Changing installation instructions
- Modifying usage examples

### API Documentation

- Update API documentation for new endpoints
- Include request/response examples
- Document error responses

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

### Examples

```
feat(datagen): Add research paper generation strategy

Implement new strategy for generating synthetic research papers
with configurable parameters for field, complexity, and length.
Includes comprehensive validation and error handling.

Closes #123

fix(config): Handle missing configuration file gracefully

Previously, the application would crash when configuration file
was missing. Now it logs a warning and uses default values.

Fixes #456

docs(readme): Update installation instructions

Add detailed setup instructions for development environment
and clarify dependency requirements.
```

### Guidelines

- Use imperative mood ("Add feature" not "Added feature")
- Keep subject line under 50 characters
- Separate subject from body with blank line
- Wrap body at 72 characters
- Reference issues and pull requests in footer

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Update documentation
3. Add changelog entry if applicable
4. Rebase on latest main branch
5. Squash commits if necessary

### PR Description

Include:
- Clear description of changes
- Motivation for the changes
- Screenshots for UI changes
- Breaking changes (if any)
- Migration instructions (if needed)

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all feedback before merge
4. Maintainer will merge when approved

### After Merge

- Delete your feature branch
- Update your local main branch
- Close related issues if applicable

## Issue Reporting

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages or logs
- Screenshots if applicable

### Feature Requests

Include:
- Clear description of the feature
- Use case or motivation
- Proposed implementation (if any)
- Examples of similar features
- Potential impact on existing functionality

### Issue Templates

Use the provided issue templates when creating new issues.

## Release Process

### Versioning

We follow Semantic Versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Steps

1. Update version number
2. Update CHANGELOG.md
3. Create release branch
4. Test release candidate
5. Create GitHub release
6. Publish to PyPI
7. Update documentation

## Getting Help

- Join our Discord community: [link]
- Check existing issues and discussions
- Read the documentation
- Contact maintainers: nikjois@llamasearch.ai

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor highlights

Thank you for contributing to OpenSynthetics! 