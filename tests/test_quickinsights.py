"""
QuickInsights Test Suite

This file tests the basic functions of the library.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import warnings
import time

# Add main directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import quickinsights as qi


class TestQuickInsights(unittest.TestCase):
    """Test class for QuickInsights library"""
    
    def setUp(self):
        """Setup before each test"""
        # Create test dataset
        np.random.seed(42)
        n_samples = 100
        
        self.test_data = pd.DataFrame({
            'age': np.random.normal(30, 5, n_samples),
            'salary': np.random.normal(40000, 10000, n_samples),
            'city': np.random.choice(['Istanbul', 'Ankara'], n_samples),
            'education': np.random.choice(['High School', 'University'], n_samples)
        })
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
    
    def tearDown(self):
        """Cleanup after each test"""
        # Clean temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_data_info(self):
        """Test data information retrieval function"""
        info = qi.get_data_info(self.test_data)

        self.assertEqual(info['shape'][0], 100)  # rows
        self.assertEqual(info['shape'][1], 4)    # columns
        self.assertIn('age', info['dtypes'])
        self.assertIn('salary', info['dtypes'])
        self.assertIn('city', info['dtypes'])
        self.assertIn('education', info['dtypes'])
    
    def test_detect_outliers(self):
        """Test outlier detection function"""
        numeric_data = self.test_data[['age', 'salary']]
        outliers = qi.detect_outliers(numeric_data)

        self.assertIsInstance(outliers, pd.DataFrame)
        # Original columns + outlier columns
        expected_columns = len(numeric_data.columns) * 2  # One outlier column for each numeric column
        self.assertEqual(outliers.shape[1], expected_columns)
        self.assertEqual(outliers.shape[0], numeric_data.shape[0])
        
        # Data type check - outlier columns should be boolean
        for col in outliers.columns:
            if col.endswith('_outlier'):
                self.assertEqual(outliers[col].dtype, bool)
            else:
                # Original columns should maintain their original dtypes
                self.assertEqual(outliers[col].dtype, numeric_data[col].dtype)
    
    def test_validate_dataframe(self):
        """Test DataFrame validation function"""
        # Valid DataFrame
        result = qi.validate_dataframe(self.test_data)
        self.assertTrue(result)
        
        # Empty DataFrame
        with self.assertRaises(ValueError):
            qi.validate_dataframe(pd.DataFrame())
        
        # Invalid type
        with self.assertRaises(TypeError):
            qi.validate_dataframe("invalid data")
    
    def test_summary_stats(self):
        """Test statistical summary function"""
        numeric_data = self.test_data[['age', 'salary']]
        summary = qi.summary_stats(numeric_data)
        
        self.assertIn('age', summary)
        self.assertIn('salary', summary)
        self.assertIn('mean', summary['age'])
        self.assertIn('std', summary['age'])
        self.assertIn('min', summary['age'])
        self.assertIn('max', summary['age'])
    
    def test_analyze_numeric(self):
        """Test numerical analysis function"""
        numeric_data = self.test_data[['age', 'salary']]
        results = qi.analyze_numeric(numeric_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('age', results)
        self.assertIn('salary', results)
        
        # Check that each column has required statistics
        for col in ['age', 'salary']:
            col_stats = results[col]
            self.assertIn('mean', col_stats)
            self.assertIn('std', col_stats)
            self.assertIn('min', col_stats)
            self.assertIn('max', col_stats)
    
    def test_analyze_categorical(self):
        """Test categorical analysis function"""
        categorical_data = self.test_data[['city', 'education']]
        results = qi.analyze_categorical(categorical_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('city', results)
        self.assertIn('education', results)
        
        # Check that each column has required statistics
        for col in ['city', 'education']:
            col_stats = results[col]
            self.assertIn('unique_values', col_stats)
            self.assertIn('most_common', col_stats)
            self.assertIn('distribution', col_stats)
    
    def test_correlation_matrix(self):
        """Test correlation matrix function"""
        numeric_data = self.test_data[['age', 'salary']]
        
        # Test without saving plots
        result = qi.correlation_matrix(numeric_data, save_plots=False)
        self.assertIsInstance(result, dict)
        self.assertIn('correlation_matrix', result)
        
        # Test with saving plots
        result = qi.correlation_matrix(numeric_data, save_plots=True, output_dir=self.temp_dir)
        self.assertIsInstance(result, dict)
        
        # Check if plot files were created
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        self.assertGreater(len(plot_files), 0)
    
    def test_distribution_plots(self):
        """Test distribution plots function"""
        numeric_data = self.test_data[['age', 'salary']]
        
        result = qi.distribution_plots(numeric_data, save_plots=True, output_dir=self.temp_dir)
        self.assertIsInstance(result, dict)
        
        # Check if plot files were created
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        self.assertGreater(len(plot_files), 0)
    
    def test_memory_optimization(self):
        """Test memory optimization function"""
        original_memory = self.test_data.memory_usage(deep=True).sum()
        optimized_df = qi.memory_optimize(self.test_data)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Memory should be reduced or at least not increased
        self.assertLessEqual(optimized_memory, original_memory)
        
        # Data should remain the same
        pd.testing.assert_frame_equal(self.test_data, optimized_df)
    
    def test_lazy_evaluation(self):
        """Test lazy evaluation decorator"""
        @qi.lazy_evaluate
        def expensive_function(x):
            return x ** 2
        
        # Function should not execute immediately
        lazy_result = expensive_function(5)
        self.assertIsInstance(lazy_result, type(lambda: None))
        
        # Function should execute when called
        result = lazy_result()
        self.assertEqual(result, 25)
    
    def test_caching(self):
        """Test caching decorator"""
        call_count = 0
        
        @qi.cache_result(ttl=3600)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should increment counter
        result1 = test_function(5)
        self.assertEqual(call_count, 1)
        self.assertEqual(result1, 10)
        
        # Second call should use cache
        result2 = test_function(5)
        self.assertEqual(call_count, 1)  # Should not increment
        self.assertEqual(result2, 10)
    
    def test_parallel_processing(self):
        """Test parallel processing function"""
        def process_item(x):
            return x ** 2
        
        data = list(range(10))
        results = qi.parallel_process(process_item, data)
        
        self.assertEqual(len(results), 10)
        self.assertEqual(results[0], 0)
        self.assertEqual(results[5], 25)
        self.assertEqual(results[9], 81)
    
    def test_data_validation_functions(self):
        """Test data validation functions"""
        # Test data type validation
        expected_types = {
            'age': 'numeric',
            'salary': 'numeric',
            'city': 'object',
            'education': 'object'
        }
        
        validation_result = qi.validate_data_types(self.test_data, expected_types)
        self.assertIsInstance(validation_result, dict)
        
        # Test data quality check
        quality_result = qi.check_data_quality(self.test_data)
        self.assertIsInstance(quality_result, dict)
        self.assertIn('overall_score', quality_result)
    
    def test_error_handling(self):
        """Test error handling in various functions"""
        # Test with None input
        with self.assertRaises((TypeError, ValueError)):
            qi.get_data_info(None)
        
        # Test with empty list
        with self.assertRaises((TypeError, ValueError)):
            qi.analyze_numeric([])
        
        # Test with invalid file path
        with self.assertRaises((OSError, ValueError)):
            qi.correlation_matrix(self.test_data, save_plots=True, output_dir="/invalid/path")
    
    def test_performance_under_load(self):
        """Test performance with larger datasets"""
        # Create larger test dataset
        large_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 10000),
            'col2': np.random.normal(0, 1, 10000),
            'col3': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        # Test that functions can handle larger datasets
        start_time = time.time()
        info = qi.get_data_info(large_data)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 5 seconds)
        self.assertLess(end_time - start_time, 5.0)
        self.assertIsInstance(info, dict)
    
    def test_output_directory_creation(self):
        """Test automatic output directory creation"""
        new_output_dir = os.path.join(self.temp_dir, 'new_output')
        
        # Function should create directory if it doesn't exist
        result = qi.correlation_matrix(
            self.test_data[['age', 'salary']], 
            save_plots=True, 
            output_dir=new_output_dir
        )
        
        self.assertTrue(os.path.exists(new_output_dir))
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
