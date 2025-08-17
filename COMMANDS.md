# QuickInsights Command Reference

This document provides a comprehensive reference for all QuickInsights library commands and usage examples.

## Core Analysis Commands

### Data Information
```python
import quickinsights as qi
import pandas as pd

# Get general information about dataset
df = pd.read_csv('data.csv')
info = qi.get_data_info(df)

# Validate dataframe structure
is_valid = qi.validate_dataframe(df)
```

### Numerical Variable Analysis
```python
# Detailed analysis for numerical variables
numeric_analysis = qi.analyze_numeric(df)

# Statistical summary
summary = qi.summary_stats(df)

# Outlier detection
outliers = qi.detect_outliers(df)
```

### Categorical Variable Analysis
```python
# Analysis for categorical variables
categorical_analysis = qi.analyze_categorical(df)
```

### Comprehensive Analysis
```python
# Perform all analyses with single command
results = qi.analyze(df, save_plots=True, output_dir="./output")
```

## Visualization Commands

### Correlation Analysis
```python
# Correlation matrix
qi.correlation_matrix(df, method='pearson', save_plots=True)

# Interactive correlation matrix
qi.create_interactive_plots(df, save_html=True)
```

### Distribution Plots
```python
# Distribution plots
qi.distribution_plots(df, save_plots=True)

# Box plots
qi.box_plots(df, save_plot=True)
```

## Performance Optimization Commands

### Lazy Evaluation
```python
# Wrap functions with lazy evaluation
@qi.lazy_evaluate
def expensive_function(x):
    # Expensive computation
    return x ** 2

# Function only executes when called
lazy_result = expensive_function(5)
result = lazy_result()  # Now executes
```

### Caching
```python
# Cache results for performance
@qi.cache_result(ttl=3600)  # 1 hour
def slow_function(x):
    # Slow computation
    return x ** 3

# First call slow, subsequent calls fast
result1 = slow_function(5)
result2 = slow_function(5)  # From cache
```

### Parallel Processing
```python
# Parallel processing for multiple items
def process_item(x):
    return x ** 2

# Process 10 items in parallel
results = qi.parallel_process(process_item, range(10))
```

## Big Data Commands

### Memory Optimization
```python
# Optimize memory usage
optimized_df = qi.memory_optimize(df)

# Check memory constraints
memory_info = qi.check_memory_constraints(estimated_mb=1000)
```

### Chunked Processing
```python
# Process large datasets in chunks
def process_chunk(chunk):
    return chunk.mean()

results = qi.process_in_chunks(df, process_chunk, chunk_size=10000)
```

## Cloud Integration Commands

### AWS S3 Operations
```python
# Upload to S3
qi.upload_to_cloud('data.csv', 'aws', 'bucket/data.csv', bucket_name='my-bucket')

# Download from S3
data = qi.download_from_cloud('aws', 'bucket/data.csv', bucket_name='my-bucket')

# Process data directly from cloud
result = qi.process_cloud_data('aws', 'bucket/data.csv', processor_func, bucket_name='my-bucket')
```

### Azure Blob Storage
```python
# Upload to Azure
qi.upload_to_cloud('data.csv', 'azure', 'container/data.csv', container_name='my-container')

# Download from Azure
data = qi.download_from_cloud('azure', 'container/data.csv', container_name='my-container')
```

### Google Cloud Storage
```python
# Upload to GCS
qi.upload_to_cloud('data.csv', 'gcs', 'bucket/data.csv', bucket_name='my-bucket')

# Download from GCS
data = qi.download_from_cloud('gcs', 'bucket/data.csv', bucket_name='my-bucket')
```

## AI-Powered Analysis Commands

### Pattern Discovery
```python
from quickinsights.ai_insights import AIInsightEngine

# Initialize AI engine
ai_engine = AIInsightEngine(df)

# Get comprehensive insights
insights = ai_engine.get_insights()

# Discover patterns
patterns = ai_engine.discover_patterns()

# Predict trends
trends = ai_engine.predict_trends()
```

### Feature Importance
```python
# Get feature importance scores
importance = ai_engine.get_feature_importance()

# Feature selection
selected_features = ai_engine.select_features(top_k=10)
```

## Real-time Pipeline Commands

### Pipeline Setup
```python
from quickinsights.realtime_pipeline import RealTimePipeline

# Create pipeline
pipeline = RealTimePipeline("DataProcessing")

# Add transformations
pipeline.add_transformation(lambda x: x * 2)
pipeline.add_filter(lambda x: x > 10)

# Start processing
pipeline.start()
```

### Data Processing
```python
# Process streaming data
results = pipeline.process_stream(data_stream)

# Get pipeline statistics
stats = pipeline.get_statistics()

# Stop pipeline
pipeline.stop()
```

## Data Validation Commands

### Data Quality Checks
```python
# Validate data types
type_validation = qi.validate_data_types(df)

# Check for missing values
missing_analysis = qi.analyze_missing_values(df)

# Validate data ranges
range_validation = qi.validate_data_ranges(df)
```

### Data Cleaning
```python
# Remove duplicates
clean_df = qi.remove_duplicates(df)

# Fill missing values
filled_df = qi.fill_missing_values(df, strategy='mean')

# Handle outliers
cleaned_df = qi.handle_outliers(df, method='iqr')
```

## Utility Commands

### System Information
```python
# Get system capabilities
system_info = qi.get_system_info()

# Check dependencies
dependencies = qi.check_dependencies()

# Get available features
features = qi.get_available_features()
```

### Performance Monitoring
```python
# Monitor execution time
@qi.measure_time
def slow_function():
    # Function to monitor
    pass

# Get performance metrics
metrics = qi.get_performance_metrics()
```

## Advanced Usage Examples

### Custom Analysis Pipeline
```python
# Create custom analysis workflow
def custom_analysis(df):
    # Data validation
    if not qi.validate_dataframe(df):
        raise ValueError("Invalid dataframe")
    
    # Memory optimization
    df_opt = qi.memory_optimize(df)
    
    # Comprehensive analysis
    results = qi.analyze(df_opt, save_plots=True)
    
    # AI insights
    ai_engine = AIInsightEngine(df_opt)
    insights = ai_engine.get_insights()
    
    return results, insights

# Execute custom analysis
results, insights = custom_analysis(df)
```

### Batch Processing
```python
# Process multiple files
files = ['data1.csv', 'data2.csv', 'data3.csv']
results = []

for file in files:
    df = pd.read_csv(file)
    result = qi.analyze(df, save_plots=True, output_dir=f"./output/{file}")
    results.append(result)

# Combine results
combined_results = qi.combine_analysis_results(results)
```

This command reference covers the essential functionality of QuickInsights. For detailed API documentation, see the individual module documentation files.
