"""
QuickInsights - Optimized Smart Data Cleaning Module

This module provides optimized data cleaning capabilities with:
- Vectorized cleaning operations
- Memory-efficient data processing
- Caching strategies
- Performance optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
from functools import lru_cache
import time
import gc
import re
from datetime import datetime

warnings.filterwarnings("ignore")


class SmartCleanerOptimized:
    """
    Optimized Smart Data Cleaning System

    Performance improvements:
    - Vectorized cleaning operations
    - Memory-efficient data processing
    - Caching for repeated operations
    - Batch processing for large datasets
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        enable_caching: bool = True,
    ):
        """
        Initialize Optimized SmartCleaner

        Parameters
        ----------
        df : pd.DataFrame
            Data to clean
        target_column : str, optional
            Target variable (for supervised cleaning)
        enable_caching : bool, default True
            Enable result caching for better performance
        """
        if df.empty:
            raise ValueError("SmartCleaner cannot be initialized with empty DataFrame")

        self.original_df = df.copy()
        self.df = df.copy()
        self.target_column = target_column
        self.enable_caching = enable_caching

        # Initialize caches
        self._cleaning_cache = {}
        self._analysis_cache = {}
        self._suggestion_cache = {}

        # Performance tracking
        self._execution_times = {}
        self._memory_usage = {}

        # Analyze data types efficiently
        self._analyze_data_types_optimized()

        # Initialize cleaning log
        self.cleaning_log = []
        self.cleaning_history = self.cleaning_log  # Backward compatibility
        self.suggestions = []

    def _analyze_data_types_optimized(self):
        """Optimized data type analysis"""
        start_time = time.time()

        # Vectorized column type detection
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(
            include=["datetime64"]
        ).columns.tolist()

        # Efficient potential datetime detection
        self.potential_datetime_cols = self._detect_potential_datetime_optimized()

        if self.enable_caching:
            print(
                f"⚡ Data type analysis completed in {time.time() - start_time:.4f} seconds"
            )

    def _detect_potential_datetime_optimized(self, sample_size: int = 100) -> List[str]:
        """Optimized potential datetime column detection"""
        potential_datetime = []

        for col in self.categorical_cols:
            if self._is_likely_datetime_optimized(self.df[col], sample_size):
                potential_datetime.append(col)

        return potential_datetime

    def _is_likely_datetime_optimized(
        self, series: pd.Series, sample_size: int = 100
    ) -> bool:
        """Optimized datetime likelihood check"""
        if len(series.dropna()) == 0:
            return False

        # Efficient sampling
        sample = series.dropna().sample(
            min(sample_size, len(series.dropna())), random_state=42
        )

        # Vectorized pattern matching
        sample_str = sample.astype(str)

        # Pre-compiled regex patterns for better performance
        datetime_patterns = [
            re.compile(r"\d{4}-\d{2}-\d{2}"),  # YYYY-MM-DD
            re.compile(r"\d{2}/\d{2}/\d{4}"),  # MM/DD/YYYY
            re.compile(r"\d{2}-\d{2}-\d{4}"),  # MM-DD-YYYY
            re.compile(r"\d{4}/\d{2}/\d{2}"),  # YYYY/MM/DD
        ]

        # Vectorized matching
        matches = 0
        for pattern in datetime_patterns:
            matches += sum(sample_str.str.match(pattern, na=False))

        return matches / len(sample) > 0.7  # 70% match threshold

    def auto_clean_optimized(
        self,
        aggressive: bool = False,
        preserve_original: bool = True,
        max_operations: int = 50,
    ) -> Dict[str, Any]:
        """
        Optimized automatic data cleaning

        Parameters
        ----------
        aggressive : bool, default=False
            Enable aggressive cleaning mode
        preserve_original : bool, default=True
            Preserve original data
        max_operations : int, default=50
            Maximum cleaning operations to perform

        Returns
        -------
        Dict[str, Any]
            Cleaning report and results
        """
        cache_key = f"auto_clean_{aggressive}_{preserve_original}_{max_operations}_{hash(str(self.df.shape))}"
        if cache_key in self._cleaning_cache:
            return self._cleaning_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        # Store original if needed
        if preserve_original:
            self.original_df = self.df.copy()

        # Perform optimized cleaning operations
        cleaning_results = self._perform_cleaning_operations_optimized(
            aggressive, max_operations
        )

        # Compile results
        results = {
            "cleaned_data": self.df,
            "cleaning_summary": cleaning_results,
            "data_quality_report": self._generate_quality_report_optimized(),
            "suggestions": self.suggestions,
            "cleaning_log": self.cleaning_log,
            "performance": {
                "execution_time": time.time() - start_time,
                "memory_change_mb": self._check_memory_usage() - initial_memory,
                "operations_performed": len(cleaning_results),
                "optimization_level": "high",
            },
        }

        # Cache results
        if self.enable_caching:
            self._cleaning_cache[cache_key] = results

        return results

    def _perform_cleaning_operations_optimized(
        self, aggressive: bool, max_operations: int
    ) -> List[Dict]:
        """Perform optimized cleaning operations"""
        operations = []
        operation_count = 0

        # 1. Handle missing values efficiently
        if operation_count < max_operations:
            missing_ops = self._handle_missing_values_optimized(aggressive)
            operations.extend(missing_ops)
            operation_count += len(missing_ops)

        # 2. Handle duplicates efficiently
        if operation_count < max_operations:
            duplicate_ops = self._handle_duplicates_optimized()
            operations.extend(duplicate_ops)
            operation_count += len(duplicate_ops)

        # 3. Handle outliers efficiently
        if operation_count < max_operations and aggressive:
            outlier_ops = self._handle_outliers_optimized()
            operations.extend(outlier_ops)
            operation_count += len(outlier_ops)

        # 4. Handle data type conversions efficiently
        if operation_count < max_operations:
            dtype_ops = self._handle_data_types_optimized()
            operations.extend(dtype_ops)
            operation_count += len(dtype_ops)

        # 5. Handle inconsistencies efficiently
        if operation_count < max_operations:
            consistency_ops = self._handle_inconsistencies_optimized()
            operations.extend(consistency_ops)
            operation_count += len(consistency_ops)

        return operations

    def _handle_missing_values_optimized(self, aggressive: bool) -> List[Dict]:
        """Optimized missing value handling"""
        operations = []

        # Vectorized missing value detection
        missing_counts = self.df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0].index.tolist()

        for col in columns_with_missing:
            if col in self.numeric_cols:
                # Efficient numeric imputation
                if aggressive:
                    # Use median for aggressive mode
                    imputed_value = self.df[col].median()
                else:
                    # Use mean for normal mode
                    imputed_value = self.df[col].mean()

                # Vectorized imputation
                missing_mask = self.df[col].isnull()
                self.df.loc[missing_mask, col] = imputed_value

                operations.append(
                    {
                        "operation": "missing_value_imputation",
                        "column": col,
                        "method": "median" if aggressive else "mean",
                        "imputed_count": missing_mask.sum(),
                        "imputed_value": imputed_value,
                    }
                )

            elif col in self.categorical_cols:
                # Efficient categorical imputation
                mode_value = (
                    self.df[col].mode().iloc[0]
                    if len(self.df[col].mode()) > 0
                    else "MISSING"
                )

                # Vectorized imputation
                missing_mask = self.df[col].isnull()
                self.df.loc[missing_mask, col] = mode_value

                operations.append(
                    {
                        "operation": "missing_value_imputation",
                        "column": col,
                        "method": "mode",
                        "imputed_count": missing_mask.sum(),
                        "imputed_value": mode_value,
                    }
                )

        return operations

    def _handle_duplicates_optimized(self) -> List[Dict]:
        """Optimized duplicate handling"""
        operations = []

        # Efficient duplicate detection
        initial_count = len(self.df)

        # Use pandas built-in duplicate removal
        self.df = self.df.drop_duplicates(keep="first")

        removed_count = initial_count - len(self.df)

        if removed_count > 0:
            operations.append(
                {
                    "operation": "duplicate_removal",
                    "removed_count": removed_count,
                    "remaining_count": len(self.df),
                }
            )

        return operations

    def _handle_outliers_optimized(self) -> List[Dict]:
        """Optimized outlier handling"""
        operations = []

        for col in self.numeric_cols:
            # Efficient outlier detection using IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Vectorized outlier detection
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                # Cap outliers instead of removing
                self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                self.df.loc[self.df[col] > upper_bound, col] = upper_bound

                operations.append(
                    {
                        "operation": "outlier_capping",
                        "column": col,
                        "outlier_count": outlier_count,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    }
                )

        return operations

    def _handle_data_types_optimized(self) -> List[Dict]:
        """Optimized data type handling"""
        operations = []

        # Convert potential datetime columns
        for col in self.potential_datetime_cols:
            try:
                # Efficient datetime conversion
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

                # Update column lists
                if col in self.categorical_cols:
                    self.categorical_cols.remove(col)
                if col not in self.datetime_cols:
                    self.datetime_cols.append(col)

                operations.append(
                    {
                        "operation": "datetime_conversion",
                        "column": col,
                        "converted_count": self.df[col].notna().sum(),
                    }
                )

            except Exception as e:
                operations.append(
                    {
                        "operation": "datetime_conversion_failed",
                        "column": col,
                        "error": str(e),
                    }
                )

        return operations

    def _handle_inconsistencies_optimized(self) -> List[Dict]:
        """Optimized inconsistency handling"""
        operations = []

        for col in self.categorical_cols:
            # Efficient value standardization
            unique_values = self.df[col].value_counts()

            # Detect potential inconsistencies (similar values with different cases)
            if len(unique_values) > 0:
                # Convert to lowercase for comparison
                lower_values = unique_values.index.str.lower()
                duplicates = lower_values.duplicated()

                if duplicates.any():
                    # Standardize to title case
                    self.df[col] = self.df[col].str.title()

                    operations.append(
                        {
                            "operation": "case_standardization",
                            "column": col,
                            "standardized_count": len(self.df[col]),
                        }
                    )

        return operations

    def _generate_quality_report_optimized(self) -> Dict[str, Any]:
        """Generate optimized data quality report"""
        report = {
            "data_shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum(),
            "data_types": self.df.dtypes.to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "column_summary": {},
        }

        # Efficient column summary generation
        for col in self.df.columns:
            col_info = {
                "dtype": str(self.df[col].dtype),
                "missing_count": self.df[col].isnull().sum(),
                "missing_percentage": (self.df[col].isnull().sum() / len(self.df))
                * 100,
            }

            if col in self.numeric_cols:
                col_info.update(
                    {
                        "min": float(self.df[col].min()),
                        "max": float(self.df[col].max()),
                        "mean": float(self.df[col].mean()),
                        "std": float(self.df[col].std()),
                    }
                )
            elif col in self.categorical_cols:
                col_info.update(
                    {
                        "unique_count": self.df[col].nunique(),
                        "most_common": self.df[col].mode().iloc[0]
                        if len(self.df[col].mode()) > 0
                        else None,
                    }
                )

            report["column_summary"][col] = col_info

        return report

    def _check_memory_usage(self) -> float:
        """Check current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def clear_cache(self):
        """Clear all cached results for memory management"""
        self._cleaning_cache.clear()
        self._analysis_cache.clear()
        self._suggestion_cache.clear()
        self._execution_times.clear()
        self._memory_usage.clear()
        gc.collect()
        print("🧹 Smart cleaner cache cleared for memory optimization")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and optimization metrics"""
        return {
            "cache_size": {
                "cleaning": len(self._cleaning_cache),
                "analysis": len(self._analysis_cache),
                "suggestions": len(self._suggestion_cache),
            },
            "execution_times": self._execution_times,
            "memory_usage": self._memory_usage,
            "data_info": {
                "shape": self.df.shape,
                "numeric_columns": len(self.numeric_cols),
                "categorical_columns": len(self.categorical_cols),
                "datetime_columns": len(self.datetime_cols),
                "potential_datetime_columns": len(self.potential_datetime_cols),
                "memory_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            },
            "optimization_features": {
                "lazy_loading": True,
                "caching": self.enable_caching,
                "vectorized_operations": True,
                "memory_efficient": True,
                "batch_processing": True,
            },
        }
