"""
Pytest configuration and common fixtures
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import warnings

# Suppress ALL warnings globally
warnings.filterwarnings("ignore")

# Also suppress specific warning types
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="session")
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    
    # Numeric data
    numeric_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(5, 2, 100),
        'C': np.random.uniform(0, 10, 100),
        'D': np.random.randint(0, 100, 100)
    })
    
    # Mixed data
    mixed_df = pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'A', 'C', 'B'],
        'text': ['hello', 'world', 'test', 'data', 'analysis'],
        'date': pd.date_range('2023-01-01', periods=5)
    })
    
    # Data with issues
    dirty_df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
        'C': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E']
    })
    
    return {
        'numeric': numeric_df,
        'mixed': mixed_df,
        'dirty': dirty_df
    }


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def sample_files(temp_dir):
    """Create sample files for testing"""
    # Create sample data
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['A', 'B', 'C', 'D', 'E']
    })
    
    # Save to different formats
    csv_path = os.path.join(temp_dir, 'test.csv')
    excel_path = os.path.join(temp_dir, 'test.xlsx')
    json_path = os.path.join(temp_dir, 'test.json')
    
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)
    df.to_json(json_path, orient='records')
    
    return {
        'df': df,
        'csv': csv_path,
        'excel': excel_path,
        'json': json_path
    }


@pytest.fixture(scope="session")
def large_dataset():
    """Create large dataset for performance testing"""
    np.random.seed(42)
    
    large_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 10000),
        'B': np.random.normal(5, 2, 10000),
        'C': np.random.choice(['X', 'Y', 'Z'], 10000),
        'D': np.random.uniform(0, 100, 10000)
    })
    
    return large_df


def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark tests based on file name
        if 'test_core' in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif 'test_quick_insights' in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif 'test_smart_cleaner' in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif 'test_easy_start' in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif 'test_dashboard' in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if 'large_dataset' in item.nodeid or 'performance' in item.nodeid:
            item.add_marker(pytest.mark.slow)
