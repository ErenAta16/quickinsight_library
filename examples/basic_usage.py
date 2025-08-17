"""
QuickInsights - Basic Usage Example

This file demonstrates the basic features of the QuickInsights library.
"""

import pandas as pd
import numpy as np
import quickinsights as qi

def create_sample_data():
    """Creates a sample dataset for demonstration"""
    np.random.seed(42)
    
    # Numerical variables
    n_samples = 1000
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'experience': np.random.exponential(5, n_samples),
        'performance': np.random.uniform(0, 100, n_samples),
        'city': np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Bursa'], n_samples),
        'education': np.random.choice(['High School', 'University', 'Masters', 'PhD'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples)
    }
    
    # Add outliers
    data['salary'][:10] = np.random.normal(150000, 20000, 10)  # High salary
    data['age'][:5] = np.random.normal(70, 5, 5)  # Older employees
    
    # Add missing values
    data['performance'][:50] = np.nan
    
    return pd.DataFrame(data)

def basic_analysis_example():
    """Basic analysis example"""
    print("QuickInsights - Basic Analysis Example")
    print("=" * 50)
    
    # Create dataset
    df = create_sample_data()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Perform comprehensive analysis
    print("\nStarting comprehensive analysis...")
    results = qi.analyze(df, show_plots=True, save_plots=False)
    
    return results

def numeric_analysis_example():
    """Numerical variable analysis example"""
    print("\nNumerical Variable Analysis")
    print("-" * 30)
    
    df = create_sample_data()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Analyze numerical variables
    numeric_results = qi.analyze_numeric(df[numeric_cols])
    
    # Detect outliers
    outliers = qi.detect_outliers(df[numeric_cols])
    print(f"Outliers detected: {outliers.sum().sum()}")
    
    return numeric_results

def categorical_analysis_example():
    """Categorical variable analysis example"""
    print("\nCategorical Variable Analysis")
    print("-" * 30)
    
    df = create_sample_data()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Analyze categorical variables
    categorical_results = qi.analyze_categorical(df[categorical_cols])
    
    return categorical_results

def performance_optimization_example():
    """Performance optimization example"""
    print("\nPerformance Optimization Example")
    print("-" * 30)
    
    df = create_sample_data()
    
    # Memory optimization
    print("Optimizing memory usage...")
    optimized_df = qi.memory_optimize(df)
    
    # Lazy evaluation
    @qi.lazy_evaluate
    def expensive_analysis(data):
        return data.groupby('city')['salary'].agg(['mean', 'std', 'count'])
    
    # Analysis only executes when called
    lazy_result = expensive_analysis(optimized_df)
    print("Lazy analysis created (not executed yet)")
    
    # Now execute the analysis
    result = lazy_result()
    print("Analysis executed and results obtained")
    
    return result

def visualization_example():
    """Visualization example"""
    print("\nVisualization Example")
    print("-" * 30)
    
    df = create_sample_data()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create correlation matrix
    print("Creating correlation matrix...")
    qi.correlation_matrix(df[numeric_cols], save_plots=True)
    
    # Create distribution plots
    print("Creating distribution plots...")
    qi.distribution_plots(df[numeric_cols], save_plots=True)
    
    # Create interactive plots
    print("Creating interactive plots...")
    qi.create_interactive_plots(df[numeric_cols], save_html=True)
    
    print("All visualizations created successfully")

def data_validation_example():
    """Data validation example"""
    print("\nData Validation Example")
    print("-" * 30)
    
    df = create_sample_data()
    
    # Validate dataframe
    is_valid = qi.validate_dataframe(df)
    print(f"Dataframe validation: {'Passed' if is_valid else 'Failed'}")
    
    # Get data information
    info = qi.get_data_info(df)
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    print(f"Data types: {info['dtypes']}")
    
    return info

def main():
    """Main function to run all examples"""
    print("QuickInsights Library Examples")
    print("=" * 40)
    
    try:
        # Run basic analysis
        basic_results = basic_analysis_example()
        
        # Run numerical analysis
        numeric_results = numeric_analysis_example()
        
        # Run categorical analysis
        categorical_results = categorical_analysis_example()
        
        # Run performance optimization
        perf_results = performance_optimization_example()
        
        # Run visualization examples
        visualization_example()
        
        # Run data validation
        validation_results = data_validation_example()
        
        print("\nAll examples completed successfully!")
        print("Check the output directory for generated plots and reports.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        return None

if __name__ == "__main__":
    main()
