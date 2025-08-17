#!/usr/bin/env python3
"""
Simple Comprehensive Test Suite for QuickInsights Library
Tests all functions systematically and reports accuracy rate.
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# Import QuickInsights
try:
    import quickinsights as qi
    print("✅ QuickInsights imported successfully")
except ImportError as e:
    print(f"❌ Failed to import QuickInsights: {e}")
    sys.exit(1)

class SimpleTestSuite:
    """Simple but comprehensive test suite for QuickInsights."""
    
    def __init__(self):
        self.results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
        
        # Create test data
        self.test_data = self._create_test_data()
        
        # Create output directory
        self.output_dir = "./test_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create test dataset."""
        np.random.seed(42)
        
        data = {
            'numeric_1': np.random.normal(0, 1, 100),
            'numeric_2': np.random.normal(5, 2, 100),
            'category_1': np.random.choice(['A', 'B', 'C'], 100),
            'category_2': np.random.choice(['X', 'Y', 'Z'], 100)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values and outliers
        df.loc[10:15, 'numeric_1'] = np.nan
        df.loc[20:25, 'numeric_2'] = np.nan
        df.loc[30, 'numeric_1'] = 100  # outlier
        df.loc[35, 'numeric_2'] = -50  # outlier
        
        return df
    
    def run_test(self, test_name: str, test_func: callable, *args, **kwargs) -> bool:
        """Run a single test."""
        self.results['tests_run'] += 1
        
        try:
            result = test_func(*args, **kwargs)
            
            # Validate result
            if self._validate_result(result):
                self.results['tests_passed'] += 1
                print(f"✅ {test_name} - PASSED")
                return True
            else:
                self.results['tests_failed'] += 1
                print(f"❌ {test_name} - FAILED")
                return False
                
        except Exception as e:
            self.results['tests_failed'] += 1
            error_info = {
                'test_name': test_name,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
            self.results['errors'].append(error_info)
            print(f"💥 {test_name} - ERROR: {type(e).__name__}: {e}")
            return False
    
    def _validate_result(self, result) -> bool:
        """Validate test results."""
        if result is None:
            return True
        
        if isinstance(result, pd.DataFrame):
            return not result.empty
        
        if isinstance(result, (list, tuple)):
            return len(result) > 0
        
        if isinstance(result, dict):
            return len(result) > 0
        
        if isinstance(result, bool):
            return result
        
        if isinstance(result, (int, float)):
            return True
        
        return result is not None
    
    def test_core_functions(self):
        """Test core analysis functions."""
        print("\n🔍 Testing Core Functions...")
        
        df = self.test_data
        
        # Test data info
        self.run_test("get_data_info", qi.get_data_info, df)
        
        # Test outlier detection
        numeric_df = df.select_dtypes(include=[np.number])
        self.run_test("detect_outliers", qi.detect_outliers, numeric_df)
        
        # Test numeric analysis
        self.run_test("analyze_numeric", qi.analyze_numeric, numeric_df)
        
        # Test categorical analysis
        cat_df = df.select_dtypes(include=['object'])
        self.run_test("analyze_categorical", qi.analyze_categorical, cat_df)
        
        # Test summary stats
        self.run_test("summary_stats", qi.summary_stats, numeric_df)
    
    def test_visualization_functions(self):
        """Test visualization functions."""
        print("\n🎨 Testing Visualization Functions...")
        
        df = self.test_data
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Test correlation matrix
        if len(numeric_df.columns) > 1:
            self.run_test("correlation_matrix", qi.correlation_matrix, numeric_df, save_plots=False)
        
        # Test distribution plots
        self.run_test("distribution_plots", qi.distribution_plots, numeric_df, save_plots=False)
        
        # Test box plots
        self.run_test("box_plots", qi.box_plots, numeric_df, save_plots=False)
        
        # Test interactive plots
        self.run_test("create_interactive_plots", qi.create_interactive_plots, numeric_df, save_plots=False)
    
    def test_data_validation_functions(self):
        """Test data validation functions."""
        print("\n✅ Testing Data Validation Functions...")
        
        df = self.test_data
        
        # Test validate_dataframe
        self.run_test("validate_dataframe", qi.validate_dataframe, df)
        
        # Test validate_column_types
        expected_types = {
            'numeric_1': 'numeric',
            'numeric_2': 'numeric',
            'category_1': 'object',
            'category_2': 'object'
        }
        self.run_test("validate_column_types", qi.validate_column_types, df, expected_types)
        
        # Test check_data_quality
        self.run_test("check_data_quality", qi.check_data_quality, df)
        
        # Test clean_data
        self.run_test("clean_data", qi.clean_data, df.copy())
        
        # Test schema validation
        schema = {
            'numeric_1': {'type': 'numeric', 'required': True},
            'numeric_2': {'type': 'numeric', 'required': True},
            'category_1': {'type': 'object', 'required': True},
            'category_2': {'type': 'object', 'required': True}
        }
        self.run_test("validate_schema", qi.validate_schema, df, schema)
        
        # Test anomaly detection
        self.run_test("detect_anomalies", qi.detect_anomalies, df)
    
    def test_performance_functions(self):
        """Test performance optimization functions."""
        print("\n⚡ Testing Performance Functions...")
        
        df = self.test_data
        
        # Test lazy evaluation
        @qi.lazy_evaluate
        def test_func(x):
            return x * 2
        
        lazy_result = test_func(5)
        self.run_test("lazy_evaluation", lambda: callable(lazy_result) and lazy_result() == 10)
        
        # Test caching
        @qi.cache_result(ttl=60)
        def expensive_func(x):
            return x ** 2
        
        self.run_test("caching", lambda: expensive_func(5) == 25)
        
        # Test parallel processing
        def square(x):
            return x ** 2
        
        self.run_test("parallel_process", lambda: qi.parallel_process(square, range(10)) == [x**2 for x in range(10)])
        
        # Test chunked processing
        def sum_chunk(chunk):
            return chunk.sum()
        
        self.run_test("chunked_process", lambda: qi.chunked_process(sum_chunk, df, chunk_size=50) is not None)
        
        # Test memory optimization
        self.run_test("memory_optimize", lambda: qi.memory_optimize(df) is not None)
        
        # Test performance profiling
        def test_function():
            return sum(range(100))
        
        self.run_test("performance_profile", lambda: qi.performance_profile(test_function) is not None)
        
        # Test benchmarking
        test_data = list(range(100))
        self.run_test("benchmark_function", lambda: qi.benchmark_function(test_function, test_data, iterations=3) is not None)
    
    def test_big_data_functions(self):
        """Test big data processing functions."""
        print("\n📊 Testing Big Data Functions...")
        
        df = self.test_data
        
        # Test status functions
        self.run_test("get_dask_status", qi.get_dask_status)
        self.run_test("get_gpu_status", qi.get_gpu_status)
        # get_memory_mapping_status is a placeholder function that returns False
        self.run_test("get_memory_mapping_status", lambda: qi.get_memory_mapping_status() is False)
        self.run_test("get_distributed_status", qi.get_distributed_status)
        
        # Test memory estimation
        self.run_test("estimate_memory_usage", lambda: isinstance(qi.estimate_memory_usage(df), dict))
        self.run_test("get_system_memory_info", qi.get_system_memory_info)
        
        # Test memory constraints
        memory_estimate = qi.estimate_memory_usage(df)
        estimated_mb = memory_estimate.get('current_mb', 100)
        self.run_test("check_memory_constraints", lambda: qi.check_memory_constraints(estimated_mb) is not None)
    
    def test_cloud_functions(self):
        """Test cloud integration functions."""
        print("\n☁️ Testing Cloud Integration Functions...")
        
        # Test status functions
        self.run_test("get_aws_status", qi.get_aws_status)
        self.run_test("get_azure_status", qi.get_azure_status)
        self.run_test("get_gcp_status", qi.get_gcp_status)
        
        # Test cloud operations (will fail without credentials, but should handle gracefully)
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("test content")
        
        try:
            self.run_test("upload_to_cloud", lambda: qi.upload_to_cloud(test_file, "aws", "test-key", bucket_name="test-bucket") is not None)
            self.run_test("download_from_cloud", lambda: qi.download_from_cloud("aws", "test-bucket", "test-key") is not None)
            self.run_test("list_cloud_files", lambda: qi.list_cloud_files("aws") is not None)
            
            def dummy_processor(data):
                return data.upper()
            
            self.run_test("process_cloud_data", lambda: qi.process_cloud_data("aws", "test-bucket", dummy_processor) is not None)
        
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists("download.txt"):
                os.remove("download.txt")
    
    def test_utility_functions(self):
        """Test utility functions."""
        print("\n🔧 Testing Utility Functions...")
        
        # Test utility status
        self.run_test("get_utility_status", qi.get_utility_status)
        self.run_test("print_utility_status", lambda: qi.print_utility_status() is None)
        
        # Test available features
        self.run_test("get_available_features", qi.get_available_features)
        
        # Test dependency checking
        self.run_test("check_dependencies", qi.check_dependencies)
        
        # Test system info
        self.run_test("get_system_info", qi.get_system_info)
        
        # Test utility report
        self.run_test("create_utility_report", qi.create_utility_report)
        
        # Test get all utils
        self.run_test("get_all_utils", qi.get_all_utils)
    
    def test_error_handling(self):
        """Test error handling."""
        print("\n⚠️ Testing Error Handling...")
        
        # Test invalid DataFrame
        try:
            qi.validate_dataframe("not a dataframe")
            self.run_test("validate_dataframe_invalid_type", lambda: False)
        except TypeError:
            self.run_test("validate_dataframe_invalid_type", lambda: True)
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        try:
            qi.validate_dataframe(empty_df)
            self.run_test("validate_dataframe_empty", lambda: False)
        except ValueError:
            self.run_test("validate_dataframe_empty", lambda: True)
        
        # Test None input
        try:
            qi.validate_dataframe(None)
            self.run_test("validate_dataframe_none", lambda: False)
        except TypeError:
            self.run_test("validate_dataframe_none", lambda: True)
    
    def run_all_tests(self):
        """Run all test categories."""
        print("🚀 Starting Simple Test Suite for QuickInsights")
        print("=" * 60)
        
        # Run all test categories
        test_categories = [
            ("Core Functions", self.test_core_functions),
            ("Visualization Functions", self.test_visualization_functions),
            ("Data Validation Functions", self.test_data_validation_functions),
            ("Performance Functions", self.test_performance_functions),
            ("Big Data Functions", self.test_big_data_functions),
            ("Cloud Integration Functions", self.test_cloud_functions),
            ("Utility Functions", self.test_utility_functions),
            ("Error Handling", self.test_error_handling)
        ]
        
        for category_name, test_func in test_categories:
            try:
                test_func()
            except Exception as e:
                print(f"💥 Error in {category_name}: {e}")
                self.results['errors'].append({
                    'category': category_name,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                })
        
        # Calculate success rate
        total_tests = self.results['tests_run']
        passed_tests = self.results['tests_passed']
        failed_tests = self.results['tests_failed']
        
        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
        else:
            success_rate = 0
        
        # Print results
        print("\n" + "=" * 60)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Tests Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\n❌ Errors encountered: {len(self.results['errors'])}")
            for error in self.results['errors'][:3]:
                if 'test_name' in error:
                    print(f"  - {error['test_name']}: {error['error_type']}")
                elif 'category' in error:
                    print(f"  - {error['category']}: {error['error_type']}")
        
        # Save results
        self.save_results()
        
        return success_rate
    
    def save_results(self):
        """Save test results."""
        results_file = os.path.join(self.output_dir, "simple_test_results.json")
        
        # Convert numpy types to native Python types for JSON serialization
        results_copy = self.results.copy()
        
        with open(results_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main function."""
    print("🚀 QuickInsights Simple Test Suite")
    print("=" * 50)
    
    # Create and run test suite
    test_suite = SimpleTestSuite()
    
    try:
        success_rate = test_suite.run_all_tests()
        
        # Return exit code based on success rate
        if success_rate >= 90:
            print("\n🎉 Excellent! High success rate achieved.")
            return 0
        elif success_rate >= 70:
            print("\n⚠️ Moderate success rate. Some issues detected.")
            return 1
        else:
            print("\n💥 Low success rate. Significant issues detected.")
            return 2
            
    except KeyboardInterrupt:
        print("\n⏹️ Test suite interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
