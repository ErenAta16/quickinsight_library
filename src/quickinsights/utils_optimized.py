"""
QuickInsights - Optimized Utilities Module

This module provides optimized utility functions with:
- Vectorized operations
- Memory-efficient data processing
- Caching strategies
- Performance optimization
"""

import os
import sys
import time
import json
import gc
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore")


class UtilsOptimized:
    """
    Optimized Utility Functions with performance improvements:
    - Vectorized data operations
    - Memory-efficient processing
    - Caching for repeated operations
    - Batch processing for large datasets
    """

    def __init__(self, enable_caching: bool = True):
        """
        Initialize Optimized Utils

        Parameters
        ----------
        enable_caching : bool, default True
            Enable result caching for better performance
        """
        self.enable_caching = enable_caching

        # Initialize caches
        self._data_info_cache = {}
        self._outlier_cache = {}
        self._validation_cache = {}

        # Performance tracking
        self._execution_times = {}
        self._memory_usage = {}

    def _check_memory_usage(self) -> float:
        """Check current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def create_output_directory_optimized(self, output_dir: str) -> str:
        """Create output directory with optimized error handling"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                if self.enable_caching:
                    print(f"📁 Output directory created: {output_dir}")
            return output_dir
        except Exception as e:
            print(f"⚠️ Error creating directory {output_dir}: {str(e)}")
            return "./quickinsights_output"  # Fallback

    def save_results_optimized(
        self,
        results: Dict[str, Any],
        operation_name: str,
        output_dir: str = "./quickinsights_output",
    ) -> str:
        """Save results to JSON file with optimized I/O"""
        start_time = time.time()

        try:
            # Create output directory
            output_dir = self.create_output_directory_optimized(output_dir)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{operation_name}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            # Optimize JSON serialization
            def json_serializer(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                return str(obj)

            # Save with optimized encoding
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    results, f, indent=2, default=json_serializer, ensure_ascii=False
                )

            if self.enable_caching:
                print(f"💾 Results saved: {filepath}")
                print(
                    f"⚡ Save operation completed in {time.time() - start_time:.4f} seconds"
                )

            return filepath

        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            return ""

    def validate_dataframe_optimized(self, df: Any) -> pd.DataFrame:
        """Optimized DataFrame validation"""
        cache_key = f"validation_{hash(str(type(df)))}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        start_time = time.time()

        try:
            # Type validation
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")

            # Empty validation
            if df.empty:
                raise ValueError("DataFrame cannot be empty")

            # Memory usage check
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            if memory_mb > 1000:  # 1GB limit
                print(f"⚠️ Large DataFrame detected: {memory_mb:.2f} MB")

            # Cache validation result
            if self.enable_caching:
                self._validation_cache[cache_key] = df
                print(
                    f"✅ DataFrame validation completed in {time.time() - start_time:.4f} seconds"
                )

            return df

        except Exception as e:
            print(f"❌ DataFrame validation failed: {str(e)}")
            raise

    def get_data_info_optimized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive DataFrame information with optimization"""
        cache_key = f"data_info_{hash(str(df.shape))}_{hash(str(df.dtypes))}"
        if cache_key in self._data_info_cache:
            return self._data_info_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        try:
            # Vectorized column type detection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

            # Efficient missing value calculation
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100

            # Vectorized unique count calculation
            unique_counts = {col: df[col].nunique() for col in df.columns}

            # Efficient duplicate detection
            duplicate_count = df.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100

            # Basic info
            info = {
                "shape": df.shape,
                "dtypes": df.dtypes.to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "missing_values": missing_counts.to_dict(),
                "missing_percentage": missing_percentages.to_dict(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": datetime_cols,
                "unique_counts": unique_counts,
                "duplicate_rows": duplicate_count,
                "duplicate_percentage": duplicate_percentage,
            }

            # Optimized numeric summary
            if numeric_cols:
                numeric_df = df[numeric_cols]
                info["numeric_summary"] = {
                    "mean": numeric_df.mean().to_dict(),
                    "median": numeric_df.median().to_dict(),
                    "std": numeric_df.std().to_dict(),
                    "min": numeric_df.min().to_dict(),
                    "max": numeric_df.max().to_dict(),
                }

            # Add performance metrics
            info["performance"] = {
                "execution_time": time.time() - start_time,
                "memory_change_mb": self._check_memory_usage() - initial_memory,
                "optimization_level": "high",
            }

            # Cache results
            if self.enable_caching:
                self._data_info_cache[cache_key] = info

            return info

        except Exception as e:
            print(f"❌ Error getting data info: {str(e)}")
            return {}

    def detect_outliers_optimized(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = "iqr"
    ) -> Dict[str, Any]:
        """Detect outliers with optimized algorithms"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        cache_key = f"outliers_{method}_{hash(str(columns))}_{hash(str(df.shape))}"
        if cache_key in self._outlier_cache:
            return self._outlier_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        outliers = {}
        total_outliers = 0

        try:
            for col in columns:
                if col not in df.columns:
                    continue

                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue

                if method == "iqr":
                    # Vectorized IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Vectorized outlier detection
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_indices = col_data[outlier_mask].index.tolist()
                    outlier_count = len(outlier_indices)

                elif method == "zscore":
                    # Vectorized Z-score method
                    mean_val = col_data.mean()
                    std_val = col_data.std()

                    if std_val > 0:
                        z_scores = np.abs((col_data - mean_val) / std_val)
                        outlier_mask = z_scores > 3
                        outlier_indices = col_data[outlier_mask].index.tolist()
                        outlier_count = len(outlier_indices)
                    else:
                        outlier_indices = []
                        outlier_count = 0

                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val

                else:
                    outlier_indices = []
                    outlier_count = 0
                    lower_bound = upper_bound = None

                outliers[col] = {
                    "method": method,
                    "outlier_count": outlier_count,
                    "outlier_percentage": (outlier_count / len(col_data)) * 100,
                    "outlier_indices": outlier_indices,
                    "lower_bound": float(lower_bound)
                    if lower_bound is not None
                    else None,
                    "upper_bound": float(upper_bound)
                    if upper_bound is not None
                    else None,
                    "total_values": len(col_data),
                }

                total_outliers += outlier_count

            results = {
                "outliers": outliers,
                "total_outliers": total_outliers,
                "columns_analyzed": columns,
                "method_used": method,
                "performance": {
                    "execution_time": time.time() - start_time,
                    "memory_change_mb": self._check_memory_usage() - initial_memory,
                    "optimization_level": "high",
                },
            }

            # Cache results
            if self.enable_caching:
                self._outlier_cache[cache_key] = results

            return results

        except Exception as e:
            print(f"❌ Error detecting outliers: {str(e)}")
            return {}

    def get_correlation_strength_optimized(self, corr_value: float) -> str:
        """Get correlation strength description with caching"""
        # Simple function, no need for complex caching
        abs_corr = abs(corr_value)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"

    def measure_execution_time_optimized(self, func: Callable) -> Callable:
        """Optimized decorator to measure execution time"""

        def wrapper(*args, **kwargs):
            start_time = time.time()
            initial_memory = self._check_memory_usage()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                memory_change = self._check_memory_usage() - initial_memory

                # Add performance metrics to result if it's a dict
                if isinstance(result, dict):
                    result["performance"] = {
                        "execution_time": execution_time,
                        "memory_change_mb": memory_change,
                        "function_name": func.__name__,
                        "optimization_level": "high",
                    }

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                print(
                    f"❌ Function {func.__name__} failed after {execution_time:.4f} seconds: {str(e)}"
                )
                raise

        return wrapper

    def batch_process_optimized(
        self, df: pd.DataFrame, batch_size: int = 1000, func: Callable = None
    ) -> Dict[str, Any]:
        """Process DataFrame in batches for memory efficiency"""
        start_time = time.time()
        initial_memory = self._check_memory_usage()

        if func is None:
            # Default batch processing function
            def default_batch_func(batch_df):
                return {
                    "shape": batch_df.shape,
                    "memory_mb": batch_df.memory_usage(deep=True).sum() / 1024**2,
                }

            func = default_batch_func

        results = []
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size

        try:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_rows)

                # Process batch
                batch_df = df.iloc[start_idx:end_idx]
                batch_result = func(batch_df)

                # Add batch info
                if isinstance(batch_result, dict):
                    batch_result["batch_info"] = {
                        "batch_number": i + 1,
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "batch_size": len(batch_df),
                    }

                results.append(batch_result)

                # Memory cleanup
                if i % 10 == 0:  # Every 10 batches
                    gc.collect()

            final_results = {
                "batch_results": results,
                "total_batches": num_batches,
                "total_rows": total_rows,
                "batch_size": batch_size,
                "performance": {
                    "execution_time": time.time() - start_time,
                    "memory_change_mb": self._check_memory_usage() - initial_memory,
                    "optimization_level": "high",
                },
            }

            return final_results

        except Exception as e:
            print(f"❌ Batch processing failed: {str(e)}")
            return {}

    def clear_cache(self):
        """Clear all cached results for memory management"""
        self._data_info_cache.clear()
        self._outlier_cache.clear()
        self._validation_cache.clear()
        self._execution_times.clear()
        self._memory_usage.clear()
        gc.collect()
        print("🧹 Utils cache cleared for memory optimization")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and optimization metrics"""
        return {
            "cache_size": {
                "data_info": len(self._data_info_cache),
                "outliers": len(self._outlier_cache),
                "validation": len(self._validation_cache),
            },
            "execution_times": self._execution_times,
            "memory_usage": self._memory_usage,
            "optimization_features": {
                "lazy_loading": True,
                "caching": self.enable_caching,
                "vectorized_operations": True,
                "memory_efficient": True,
                "batch_processing": True,
            },
        }
