# Contributing to QuickInsights

🎉 Thank you for your interest in contributing to QuickInsights! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## 🚀 Getting Started

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

## 🔧 Development Setup

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

## 📝 Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Report bugs and issues
- 🚀 **Feature Requests**: Suggest new features
- 💻 **Code Contributions**: Submit pull requests
- 📚 **Documentation**: Improve docs and examples
- 🧪 **Testing**: Add tests or improve test coverage
- 🔍 **Code Review**: Review pull requests
- 📢 **Community**: Help other users

### Before Contributing

1. **Check existing issues**: Search for similar issues or feature requests
2. **Discuss major changes**: Open an issue to discuss significant changes
3. **Follow the roadmap**: Check our development roadmap for priorities

## 🎨 Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use isort
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with Black
black src/quickinsights/

# Sort imports with isort
isort src/quickinsights/

# Check code style with flake8
flake8 src/quickinsights/ --max-line-length=88 --ignore=E203,W503
```

### Type Checking

We use mypy for static type checking:

```bash
mypy src/quickinsights/ --ignore-missing-imports
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quickinsights --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test function
pytest tests/test_core.py::test_analyze_function
```

### Writing Tests

- **Test coverage**: Aim for >80% coverage
- **Test organization**: Group related tests in classes
- **Test names**: Use descriptive test names
- **Fixtures**: Use pytest fixtures for common setup
- **Mocking**: Mock external dependencies

### Test Structure

```python
import pytest
import quickinsights as qi
import pandas as pd

class TestNewFeature:
    """Test new feature functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = qi.new_feature(self.test_data)
        assert result is not None
        assert len(result) == 3
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            qi.new_feature(empty_df)
```

## 📚 Documentation

### Docstring Standards

Use Google-style docstrings:

```python
def analyze_data(df: pd.DataFrame, method: str = 'auto') -> dict:
    """Analyze data using specified method.
    
    This function provides comprehensive data analysis including
    statistical summaries, visualizations, and insights.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data to analyze
    method : str, default='auto'
        Analysis method to use ('auto', 'basic', 'advanced')
        
    Returns
    -------
    dict
        Analysis results containing summaries and insights
        
    Raises
    ------
    ValueError
        If DataFrame is empty or invalid
    TypeError
        If input is not a DataFrame
        
    Examples
    --------
    >>> import quickinsights as qi
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> results = qi.analyze_data(df)
    >>> print(results['summary'])
    """
    pass
```

### Documentation Files

- **README.md**: Project overview and quick start
- **docs/**: Detailed documentation
- **examples/**: Code examples and tutorials
- **API_REFERENCE.md**: Complete API documentation

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write your code following our style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add new analysis feature

- Add correlation analysis function
- Include statistical significance testing
- Update documentation and examples
- Add comprehensive test coverage"
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

### 5. Pull Request Guidelines

- **Title**: Use conventional commit format
- **Description**: Clearly describe changes and motivation
- **Tests**: Ensure all tests pass
- **Coverage**: Maintain or improve test coverage
- **Documentation**: Update relevant documentation

### Commit Message Format

We use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

## 🚀 Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Steps

1. **Update version** in `setup.py`
2. **Update changelog** with new features/fixes
3. **Create release branch** from main
4. **Run full test suite** and quality checks
5. **Build and test package** locally
6. **Create GitHub release** with changelog
7. **Deploy to PyPI** using deployment script

### Deployment Script

```bash
# Deploy to PyPI
python scripts/deploy_to_pypi.py

# Deploy to Test PyPI first
python scripts/deploy_to_pypi.py --test
```

## 🆘 Getting Help

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Pull Requests**: For code reviews and feedback

### Resources

- **Documentation**: Check our docs first
- **Examples**: Review example code
- **Tests**: Look at test files for usage examples
- **Issues**: Search existing issues for solutions

## 🙏 Recognition

Contributors will be recognized in:

- **README.md**: Contributors list
- **Release notes**: For significant contributions
- **Documentation**: For documentation improvements

## 📄 License

By contributing to QuickInsights, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to QuickInsights! 🎉

Your contributions help make data analysis more accessible and powerful for everyone.
