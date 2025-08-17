# Contributing to QuickInsights

Thank you for your interest in contributing to QuickInsights! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/quickinsights.git
   cd quickinsights
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/originalusername/quickinsights.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n quickinsights python=3.9
conda activate quickinsights
```

### 2. Install Dependencies

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install all optional dependencies
pip install -e ".[dev,fast,gpu,cloud,profiling]"
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report bugs and issues
- **Feature Requests**: Suggest new features
- **Code Contributions**: Submit pull requests
- **Documentation**: Improve docs and examples
- **Testing**: Add tests or improve test coverage
- **Code Review**: Review pull requests
- **Community**: Help other users

### Before Contributing

1. **Check existing issues**: Search for similar issues or feature requests
2. **Discuss major changes**: Open an issue to discuss significant changes
3. **Follow the roadmap**: Check our development roadmap for priorities

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use isort
- **Code formatting**: Use Black
- **Type hints**: Use mypy for type checking

### Code Formatting

```bash
# Format code with Black
black src/quickinsights/

# Sort imports with isort
isort src/quickinsights/

# Check types with mypy
mypy src/quickinsights/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quickinsights --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for new features
- Ensure test coverage is maintained
- Use descriptive test names
- Follow pytest best practices

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Update documentation when changing code
- Follow Google-style docstrings

### Docstring Format

```python
def analyze_data(df: pd.DataFrame, save_plots: bool = True) -> dict:
    """
    Analyze the given dataframe and return comprehensive insights.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze
    save_plots : bool, default=True
        Whether to save generated plots
        
    Returns
    -------
    dict
        Dictionary containing analysis results
        
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> results = analyze_data(df)
    """
    pass
```

## Pull Request Process

### Creating a Pull Request

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a pull request** on GitHub

### Pull Request Guidelines

- Provide a clear description of changes
- Include relevant issue numbers
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in setup.py
2. **Update changelog** with new features/fixes
3. **Create release branch** and test thoroughly
4. **Merge to main** and create GitHub release
5. **Build and upload** to PyPI

## Getting Help

If you need help with contributing:

- Check existing documentation
- Search GitHub issues
- Ask questions in discussions
- Contact maintainers directly

## Recognition

Contributors will be recognized in:

- Project README
- Release notes
- Contributor statistics
- Special acknowledgments for significant contributions

Thank you for contributing to QuickInsights! Your contributions help make this library better for everyone.
