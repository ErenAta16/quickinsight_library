# QuickInsights API Reference

## Overview

QuickInsights is a Python library that provides creative and innovative analysis tools for large datasets.

## Installation

```bash
pip install quickinsights
```

## Core Functions

### `analyze(df, show_plots=True, save_plots=False, output_dir="./quickinsights_output")`

Performs comprehensive analysis on the dataset.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze
- `show_plots` (bool): Whether to display plots
- `save_plots` (bool): Whether to save plots
- `output_dir` (str): Directory to save plots

**Returns:**
- `dict`: Analysis results

**Example:**
```python
import quickinsights as qi
import pandas as pd

df = pd.read_csv('data.csv')
results = qi.analyze(df, save_plots=True)
```

### `analyze_numeric(df, show_plots=True, save_plots=False, output_dir="./quickinsights_output")`

Performs detailed analysis on numerical variables.

**Parameters:**
- `df` (pd.DataFrame): Dataset containing only numerical variables
- `show_plots` (bool): Whether to display plots
- `save_plots` (bool): Whether to save plots
- `output_dir` (str): Directory to save plots

**Returns:**
- `dict`: Numerical analysis results

### `analyze_categorical(df, show_plots=True, save_plots=False, output_dir="./quickinsights_output")`

Performs detailed analysis on categorical variables.

**Parameters:**
- `df` (pd.DataFrame): Dataset containing only categorical variables
- `show_plots` (bool): Whether to display plots
- `save_plots` (bool): Whether to save plots
- `output_dir` (str): Directory to save plots

**Returns:**
- `dict`: Categorical analysis results

## Visualization Functions

### `correlation_matrix(df, method='pearson', save_plots=False, output_dir="./quickinsights_output")`

Visualizes correlation matrix between numerical variables.

**Parameters:**
- `df` (pd.DataFrame): Dataset containing only numerical variables
- `method` (str): Correlation calculation method ('pearson', 'spearman')
- `save_plots` (bool): Whether to save the plot
- `output_dir` (str): Directory to save the plot

### `distribution_plots(df, save_plots=False, output_dir="./quickinsights_output")`

Creates distribution plots for numerical variables.

**Parameters:**
- `df` (pd.DataFrame): Dataset containing only numerical variables
- `save_plots` (bool): Whether to save plots
- `output_dir` (str): Directory to save plots

### `summary_stats(df)`

Calculates statistical summary of the dataset.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze

**Returns:**
- `dict`: Statistical summary

## Utility Functions

### `get_data_info(df)`

Provides general information about the dataset.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze

**Returns:**
- `dict`: Dataset information

### `detect_outliers(df, method='iqr', threshold=1.5)`

Detects outliers in numerical variables.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze
- `method` (str): Outlier detection method ('iqr', 'zscore')
- `threshold` (float): Threshold for outlier detection

**Returns:**
- `dict`: Outlier information

### `validate_dataframe(df)`

Validates dataframe structure and content.

**Parameters:**
- `df` (pd.DataFrame): Dataset to validate

**Returns:**
- `bool`: Validation result

## Performance Optimization Functions

### `lazy_evaluate(func)`

Decorator for lazy evaluation of functions.

**Parameters:**
- `func`: Function to wrap

**Returns:**
- Wrapped function that executes only when called

**Example:**
```python
@qi.lazy_evaluate
def expensive_function(x):
    return x ** 2

lazy_result = expensive_function(5)
result = lazy_result()  # Now executes
```

### `cache_result(ttl=3600)`

Decorator for caching function results.

**Parameters:**
- `ttl` (int): Time to live in seconds

**Returns:**
- Decorated function with caching

**Example:**
```python
@qi.cache_result(ttl=3600)
def slow_function(x):
    return x ** 3

result1 = slow_function(5)  # Slow
result2 = slow_function(5)  # Fast (from cache)
```

### `parallel_process(func, data, max_workers=None)`

Processes data in parallel using multiple workers.

**Parameters:**
- `func`: Function to apply to each item
- `data`: Data to process
- `max_workers` (int): Maximum number of workers

**Returns:**
- `list`: Processed results

## Big Data Functions

### `memory_optimize(df)`

Optimizes memory usage of the dataframe.

**Parameters:**
- `df` (pd.DataFrame): Dataset to optimize

**Returns:**
- `pd.DataFrame`: Memory-optimized dataset

### `process_in_chunks(df, func, chunk_size=10000)`

Processes large datasets in chunks.

**Parameters:**
- `df` (pd.DataFrame): Dataset to process
- `func`: Function to apply to each chunk
- `chunk_size` (int): Size of each chunk

**Returns:**
- `list`: Results from processing chunks

## Cloud Integration Functions

### `upload_to_cloud(file_path, provider, remote_path, **kwargs)`

Uploads files to cloud storage.

**Parameters:**
- `file_path` (str): Local file path
- `provider` (str): Cloud provider ('aws', 'azure', 'gcs')
- `remote_path` (str): Remote file path
- `**kwargs`: Provider-specific parameters

**Returns:**
- `bool`: Upload success status

### `download_from_cloud(provider, remote_path, **kwargs)`

Downloads files from cloud storage.

**Parameters:**
- `provider` (str): Cloud provider ('aws', 'azure', 'gcs')
- `remote_path` (str): Remote file path
- `**kwargs`: Provider-specific parameters

**Returns:**
- Downloaded data or file path

## AI-Powered Analysis Functions

### `AIInsightEngine(df)`

AI-powered analysis engine for discovering patterns and insights.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze

**Methods:**
- `get_insights()`: Returns comprehensive insights
- `discover_patterns()`: Discovers data patterns
- `predict_trends()`: Predicts future trends
- `get_feature_importance()`: Calculates feature importance

## Real-time Pipeline Functions

### `RealTimePipeline(name)`

Real-time data processing pipeline.

**Parameters:**
- `name` (str): Pipeline name

**Methods:**
- `add_transformation(func)`: Adds data transformation
- `add_filter(func)`: Adds data filter
- `start()`: Starts the pipeline
- `stop()`: Stops the pipeline
- `process_stream(data_stream)`: Processes streaming data

## Data Validation Functions

### `validate_data_types(df, expected_types)`

Validates data types of dataframe columns.

**Parameters:**
- `df` (pd.DataFrame): Dataset to validate
- `expected_types` (dict): Expected column types

**Returns:**
- `dict`: Validation results

### `check_data_quality(df)`

Checks overall data quality.

**Parameters:**
- `df` (pd.DataFrame): Dataset to check

**Returns:**
- `dict`: Quality metrics

## Error Handling

All functions include proper error handling and will raise appropriate exceptions for invalid inputs or processing errors.

## Performance Notes

- Use lazy evaluation for expensive computations
- Apply caching for frequently called functions
- Use parallel processing for large datasets
- Consider memory optimization for big data operations

## Examples

For complete usage examples, see the examples directory and the main README file.

This API reference covers the core functionality of QuickInsights. For advanced usage patterns and best practices, refer to the documentation and examples.
