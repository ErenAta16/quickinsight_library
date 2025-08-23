"""
QuickInsights - Optimized Core Module

This module provides optimized core data analysis capabilities with:
- Efficient data processing
- Vectorized operations
- Memory optimization
- Caching strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from functools import lru_cache
import time
import gc
import os

warnings.filterwarnings("ignore")


# Optimized lazy loading with caching
@lru_cache(maxsize=32)
def _get_visualization_libraries_cached():
    """Get visualization libraries with cached lazy loading."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        return plt, sns
    except ImportError:
        return None, None


def _get_visualization_libraries():
    """Get visualization libraries with lazy loading."""
    return _get_visualization_libraries_cached()


class CoreAnalyzerOptimized:
    """
    Optimized core data analyzer with performance improvements:
    - Vectorized operations
    - Memory-efficient processing
    - Lazy evaluation
    - Caching strategies
    """

    def __init__(
        self,
        df: pd.DataFrame,
        enable_caching: bool = True,
        output_dir: str = "analysis_output",
    ):
        """
        Initialize Optimized Core Analyzer

        Parameters
        ----------
        df : pd.DataFrame
            Data to analyze
        enable_caching : bool, default True
            Enable result caching for better performance
        output_dir : str, default "analysis_output"
            Directory to save analysis outputs
        """
        if df.empty:
            raise ValueError("CoreAnalyzer cannot be initialized with empty DataFrame")

        self.df = df
        self.enable_caching = enable_caching
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize caches
        self._analysis_cache = {}
        self._stats_cache = {}
        self._plots_cache = {}

        # Vectorized column type detection
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # Performance tracking
        self._execution_times = {}
        self._memory_usage = {}

    def _check_memory_usage(self) -> bool:
        """Check current memory usage"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            return 0

    def analyze_optimized(
        self, save_plots: bool = True, max_plots: int = 10
    ) -> Dict[str, Any]:
        """
        Optimized comprehensive data analysis

        Parameters
        ----------
        save_plots : bool, default True
            Whether to save plots
        max_plots : int, default 10
            Maximum number of plots to generate

        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis results
        """
        cache_key = f"analysis_{save_plots}_{max_plots}_{hash(str(self.df.shape))}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        # Run optimized analyses
        numeric_analysis = self.analyze_numeric_optimized(save_plots, max_plots)
        categorical_analysis = self.analyze_categorical_optimized(save_plots, max_plots)

        # Compile results
        results = {
            "data_overview": {
                "shape": self.df.shape,
                "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
                "dtypes": self.df.dtypes.to_dict(),
                "missing_values": self.df.isnull().sum().to_dict(),
            },
            "numeric_analysis": numeric_analysis,
            "categorical_analysis": categorical_analysis,
            "performance": {
                "execution_time": time.time() - start_time,
                "memory_change_mb": self._check_memory_usage() - initial_memory,
                "optimization_level": "high",
            },
        }

        # Cache results
        if self.enable_caching:
            self._analysis_cache[cache_key] = results

        return results

    def analyze_numeric_optimized(
        self, save_plots: bool = True, max_plots: int = 5
    ) -> Dict[str, Any]:
        """Optimized numeric column analysis"""
        if not self.numeric_cols:
            return {"message": "No numeric columns found"}

        cache_key = f"numeric_{save_plots}_{max_plots}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        start_time = time.time()

        # Vectorized statistics calculation
        stats = {}
        for col in self.numeric_cols[:max_plots]:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                stats[col] = {
                    "count": len(col_data),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                    "skewness": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis()),
                }

        # Generate plots if requested
        plots_generated = []
        if save_plots and len(self.numeric_cols) > 0:
            plots_generated = self._generate_numeric_plots(max_plots)

        results = {
            "statistics": stats,
            "plots_generated": plots_generated,
            "execution_time": time.time() - start_time,
        }

        # Cache results
        if self.enable_caching:
            self._stats_cache[cache_key] = results

        return results

    def analyze_categorical_optimized(
        self, save_plots: bool = True, max_plots: int = 5
    ) -> Dict[str, Any]:
        """Optimized categorical column analysis"""
        if not self.categorical_cols:
            return {"message": "No categorical columns found"}

        cache_key = f"categorical_{save_plots}_{max_plots}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        start_time = time.time()

        # Vectorized statistics calculation
        stats = {}
        for col in self.categorical_cols[:max_plots]:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                stats[col] = {
                    "count": len(col_data),
                    "unique_values": len(value_counts),
                    "most_common": value_counts.index[0]
                    if len(value_counts) > 0
                    else None,
                    "most_common_count": int(value_counts.iloc[0])
                    if len(value_counts) > 0
                    else 0,
                    "missing_values": int(self.df[col].isnull().sum()),
                }

        # Generate plots if requested
        plots_generated = []
        if save_plots and len(self.categorical_cols) > 0:
            plots_generated = self._generate_categorical_plots(max_plots)

        results = {
            "statistics": stats,
            "plots_generated": plots_generated,
            "execution_time": time.time() - start_time,
        }

        # Cache results
        if self.enable_caching:
            self._stats_cache[cache_key] = results

        return results

    def _generate_numeric_plots(self, max_plots: int) -> List[str]:
        """Generate numeric column plots efficiently"""
        plots_generated = []

        try:
            plt, sns = _get_visualization_libraries()
            if plt is None or sns is None:
                return plots_generated

            # Set style for better performance
            plt.style.use("default")
            sns.set_palette("husl")

            for i, col in enumerate(self.numeric_cols[:max_plots]):
                if i >= max_plots:
                    break

                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f"Analysis of {col}", fontsize=16)

                # Histogram
                axes[0, 0].hist(
                    self.df[col].dropna(), bins=30, alpha=0.7, edgecolor="black"
                )
                axes[0, 0].set_title("Distribution")
                axes[0, 0].set_xlabel(col)
                axes[0, 0].set_ylabel("Frequency")

                # Box plot
                axes[0, 1].boxplot(self.df[col].dropna())
                axes[0, 1].set_title("Box Plot")
                axes[0, 1].set_ylabel(col)

                # Q-Q plot for normality check
                from scipy import stats

                qq_data = self.df[col].dropna()
                if len(qq_data) > 0:
                    stats.probplot(qq_data, dist="norm", plot=axes[1, 0])
                    axes[1, 0].set_title("Q-Q Plot (Normality Check)")

                # Time series if index is datetime
                if (
                    len(self.datetime_cols) > 0
                    and self.df.index.dtype == "datetime64[ns]"
                ):
                    axes[1, 1].plot(self.df.index, self.df[col])
                    axes[1, 1].set_title("Time Series")
                    axes[1, 1].set_xlabel("Time")
                    axes[1, 1].set_ylabel(col)
                    axes[1, 1].tick_params(axis="x", rotation=45)
                else:
                    # Scatter plot with first numeric column
                    if len(self.numeric_cols) > 1:
                        other_col = [c for c in self.numeric_cols if c != col][0]
                        axes[1, 1].scatter(self.df[other_col], self.df[col], alpha=0.6)
                        axes[1, 1].set_title(f"{col} vs {other_col}")
                        axes[1, 1].set_xlabel(other_col)
                        axes[1, 1].set_ylabel(col)

                plt.tight_layout()

                # Save plot
                plot_path = os.path.join(self.output_dir, f"{col}_analysis.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plots_generated.append(plot_path)

                # Close figure to free memory
                plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error generating numeric plots: {str(e)}")

        return plots_generated

    def _generate_categorical_plots(self, max_plots: int) -> List[str]:
        """Generate categorical column plots efficiently"""
        plots_generated = []

        try:
            plt, sns = _get_visualization_libraries()
            if plt is None or sns is None:
                return plots_generated

            # Set style for better performance
            plt.style.use("default")
            sns.set_palette("husl")

            for i, col in enumerate(self.categorical_cols[:max_plots]):
                if i >= max_plots:
                    break

                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f"Analysis of {col}", fontsize=16)

                # Value counts bar plot
                value_counts = self.df[col].value_counts().head(10)
                axes[0].bar(range(len(value_counts)), value_counts.values)
                axes[0].set_title("Top 10 Values")
                axes[0].set_xlabel("Values")
                axes[0].set_ylabel("Count")
                axes[0].set_xticks(range(len(value_counts)))
                axes[0].set_xticklabels(value_counts.index, rotation=45, ha="right")

                # Pie chart for top 5 values
                top_5 = value_counts.head(5)
                axes[1].pie(top_5.values, labels=top_5.index, autopct="%1.1f%%")
                axes[1].set_title("Top 5 Values Distribution")

                plt.tight_layout()

                # Save plot
                plot_path = os.path.join(self.output_dir, f"{col}_analysis.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plots_generated.append(plot_path)

                # Close figure to free memory
                plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error generating categorical plots: {str(e)}")

        return plots_generated

    def compare_performance(
        self, other_analyzer: "CoreAnalyzerOptimized"
    ) -> Dict[str, Any]:
        """Compare performance with another analyzer"""
        comparison = {
            "current_analyzer": {
                "data_shape": self.df.shape,
                "cache_size": len(self._analysis_cache),
                "execution_times": self._execution_times,
            },
            "other_analyzer": {
                "data_shape": other_analyzer.df.shape,
                "cache_size": len(other_analyzer._analysis_cache),
                "execution_times": other_analyzer._execution_times,
            },
            "performance_metrics": {
                "current_memory_mb": self._check_memory_usage(),
                "other_memory_mb": other_analyzer._check_memory_usage(),
            },
        }

        return comparison

    def clear_cache(self):
        """Clear all cached results for memory management"""
        self._analysis_cache.clear()
        self._stats_cache.clear()
        self._plots_cache.clear()
        self._execution_times.clear()
        self._memory_usage.clear()
        gc.collect()
        print("🧹 Core analyzer cache cleared for memory optimization")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and optimization metrics"""
        return {
            "cache_size": {
                "analysis": len(self._analysis_cache),
                "stats": len(self._stats_cache),
                "plots": len(self._plots_cache),
            },
            "execution_times": self._execution_times,
            "memory_usage": self._memory_usage,
            "data_info": {
                "shape": self.df.shape,
                "numeric_columns": len(self.numeric_cols),
                "categorical_columns": len(self.categorical_cols),
                "datetime_columns": len(self.datetime_cols),
                "memory_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            },
            "optimization_features": {
                "lazy_loading": True,
                "caching": self.enable_caching,
                "vectorized_operations": True,
                "memory_efficient": True,
            },
        }
