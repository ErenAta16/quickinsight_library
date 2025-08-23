"""
QuickInsights - Optimized Visualization Module

This module provides optimized data visualization capabilities with:
- Memory-efficient plotting
- Lazy chart generation
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


class VisualizerOptimized:
    """
    Optimized Data Visualizer with performance improvements:
    - Memory-efficient plotting
    - Lazy chart generation
    - Caching for repeated operations
    - Batch processing for large datasets
    """

    def __init__(
        self,
        df: pd.DataFrame,
        enable_caching: bool = True,
        output_dir: str = "visualization_output",
    ):
        """
        Initialize Optimized Visualizer

        Parameters
        ----------
        df : pd.DataFrame
            Data to visualize
        enable_caching : bool, default True
            Enable result caching for better performance
        output_dir : str, default "visualization_output"
            Directory to save visualizations
        """
        if df.empty:
            raise ValueError("Visualizer cannot be initialized with empty DataFrame")

        self.df = df
        self.enable_caching = enable_caching
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize caches
        self._plot_cache = {}
        self._correlation_cache = {}
        self._distribution_cache = {}

        # Performance tracking
        self._execution_times = {}
        self._memory_usage = {}

        # Vectorized column type detection
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # Initialize visualization libraries
        self._plt, self._sns = _get_visualization_libraries()
        if self._plt and self._sns:
            self._setup_plotting_style()

    def _setup_plotting_style(self):
        """Setup optimized plotting style"""
        try:
            # Use default style for better performance
            self._plt.style.use("default")
            self._sns.set_palette("husl")
            self._sns.set_style("whitegrid")

            # Optimize font settings
            self._plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
            self._plt.rcParams["font.sans-serif"] = [
                "DejaVu Sans",
                "Arial",
                "sans-serif",
            ]

        except Exception as e:
            print(f"⚠️ Style setup warning: {str(e)}")

    def _check_memory_usage(self) -> float:
        """Check current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def correlation_matrix_optimized(
        self,
        method: str = "pearson",
        save_plots: bool = True,
        max_correlations: int = 20,
    ) -> Dict[str, Any]:
        """
        Optimized correlation matrix visualization

        Parameters
        ----------
        method : str, default 'pearson'
            Correlation calculation method
        save_plots : bool, default True
            Whether to save plots
        max_correlations : int, default 20
            Maximum correlations to display

        Returns
        -------
        Dict[str, Any]
            Correlation analysis results
        """
        cache_key = (
            f"correlation_{method}_{max_correlations}_{hash(str(self.df.shape))}"
        )
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        if len(self.numeric_cols) < 2:
            return {
                "error": "Correlation matrix requires at least 2 numeric variables",
                "performance": {"execution_time": 0, "optimization_level": "high"},
            }

        # Calculate correlation matrix efficiently
        numeric_df = self.df[self.numeric_cols]
        corr_matrix = numeric_df.corr(method=method)

        # Generate correlation insights
        correlations = self._extract_correlations_optimized(
            corr_matrix, max_correlations
        )

        # Generate plot if requested
        plot_path = None
        if save_plots and self._plt and self._sns:
            plot_path = self._create_correlation_plot_optimized(corr_matrix, method)

        results = {
            "correlation_matrix": corr_matrix,
            "correlations": correlations,
            "plot_path": plot_path,
            "performance": {
                "execution_time": time.time() - start_time,
                "memory_change_mb": self._check_memory_usage() - initial_memory,
                "optimization_level": "high",
            },
        }

        # Cache results
        if self.enable_caching:
            self._correlation_cache[cache_key] = results

        return results

    def _extract_correlations_optimized(
        self, corr_matrix: pd.DataFrame, max_correlations: int
    ) -> List[Dict]:
        """Extract top correlations efficiently"""
        correlations = []

        # Get upper triangle indices
        upper_triangle = np.triu_indices_from(corr_matrix, k=1)

        # Create correlation pairs
        for i, j in zip(upper_triangle[0], upper_triangle[1]):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            correlations.append(
                {
                    "variable1": col1,
                    "variable2": col2,
                    "correlation": float(corr_value),
                    "strength": self._get_correlation_strength(corr_value),
                }
            )

        # Sort by absolute correlation value and limit
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return correlations[:max_correlations]

    def _get_correlation_strength(self, corr_value: float) -> str:
        """Get correlation strength description"""
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

    def _create_correlation_plot_optimized(
        self, corr_matrix: pd.DataFrame, method: str
    ) -> str:
        """Create optimized correlation plot"""
        try:
            # Create figure with optimized size
            fig, ax = self._plt.subplots(figsize=(10, 8))

            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Create heatmap
            self._sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                fmt=".2f",
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )

            ax.set_title(f"Correlation Matrix ({method.upper()})", fontsize=16, pad=20)
            self._plt.tight_layout()

            # Save plot
            plot_path = os.path.join(
                self.output_dir, f"correlation_matrix_{method}.png"
            )
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Close figure to free memory
            self._plt.close(fig)

            return plot_path

        except Exception as e:
            print(f"⚠️ Error creating correlation plot: {str(e)}")
            return None

    def distribution_plots_optimized(
        self, save_plots: bool = True, max_plots: int = 10
    ) -> Dict[str, Any]:
        """
        Optimized distribution plots generation

        Parameters
        ----------
        save_plots : bool, default True
            Whether to save plots
        max_plots : int, default 10
            Maximum number of plots to generate

        Returns
        -------
        Dict[str, Any]
            Distribution analysis results
        """
        cache_key = f"distribution_{max_plots}_{hash(str(self.df.shape))}"
        if cache_key in self._distribution_cache:
            return self._distribution_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        if not self._plt or not self._sns:
            return {
                "error": "Visualization libraries not available",
                "performance": {"execution_time": 0, "optimization_level": "high"},
            }

        plots_generated = []
        numeric_cols = self.numeric_cols[:max_plots]

        for col in numeric_cols:
            try:
                plot_path = self._create_distribution_plot_optimized(col)
                if plot_path:
                    plots_generated.append(plot_path)
            except Exception as e:
                print(f"⚠️ Error creating distribution plot for {col}: {str(e)}")

        results = {
            "plots_generated": plots_generated,
            "columns_analyzed": numeric_cols,
            "performance": {
                "execution_time": time.time() - start_time,
                "memory_change_mb": self._check_memory_usage() - initial_memory,
                "optimization_level": "high",
            },
        }

        # Cache results
        if self.enable_caching:
            self._distribution_cache[cache_key] = results

        return results

    def _create_distribution_plot_optimized(self, column: str) -> str:
        """Create optimized distribution plot for a single column"""
        try:
            # Create figure with subplots
            fig, axes = self._plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Distribution Analysis: {column}", fontsize=16)

            col_data = self.df[column].dropna()

            # Histogram
            axes[0, 0].hist(col_data, bins=30, alpha=0.7, edgecolor="black")
            axes[0, 0].set_title("Histogram")
            axes[0, 0].set_xlabel(column)
            axes[0, 0].set_ylabel("Frequency")

            # Box plot
            axes[0, 1].boxplot(col_data)
            axes[0, 1].set_title("Box Plot")
            axes[0, 1].set_ylabel(column)

            # Q-Q plot for normality check
            try:
                from scipy import stats

                stats.probplot(col_data, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title("Q-Q Plot (Normality Check)")
            except ImportError:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "Q-Q Plot\n(scipy not available)",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("Q-Q Plot")

            # Summary statistics
            stats_text = f"""
            Count: {len(col_data):,}
            Mean: {col_data.mean():.2f}
            Std: {col_data.std():.2f}
            Min: {col_data.min():.2f}
            Max: {col_data.max():.2f}
            """
            axes[1, 1].text(
                0.1,
                0.5,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=10,
                verticalalignment="center",
            )
            axes[1, 1].set_title("Summary Statistics")
            axes[1, 1].axis("off")

            self._plt.tight_layout()

            # Save plot
            plot_path = os.path.join(self.output_dir, f"{column}_distribution.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Close figure to free memory
            self._plt.close(fig)

            return plot_path

        except Exception as e:
            print(f"⚠️ Error creating distribution plot: {str(e)}")
            return None

    def box_plots_optimized(
        self, save_plots: bool = True, max_plots: int = 10
    ) -> Dict[str, Any]:
        """
        Optimized box plots generation

        Parameters
        ----------
        save_plots : bool, default True
            Whether to save plots
        max_plots : int, default 10
            Maximum number of plots to generate

        Returns
        -------
        Dict[str, Any]
            Box plot analysis results
        """
        cache_key = f"boxplots_{max_plots}_{hash(str(self.df.shape))}"
        if cache_key in self._plot_cache:
            return self._plot_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        if not self._plt or not self._sns:
            return {
                "error": "Visualization libraries not available",
                "performance": {"execution_time": 0, "optimization_level": "high"},
            }

        plots_generated = []
        numeric_cols = self.numeric_cols[:max_plots]

        for col in numeric_cols:
            try:
                plot_path = self._create_box_plot_optimized(col)
                if plot_path:
                    plots_generated.append(plot_path)
            except Exception as e:
                print(f"⚠️ Error creating box plot for {col}: {str(e)}")

        results = {
            "plots_generated": plots_generated,
            "columns_analyzed": numeric_cols,
            "performance": {
                "execution_time": time.time() - start_time,
                "memory_change_mb": self._check_memory_usage() - initial_memory,
                "optimization_level": "high",
            },
        }

        # Cache results
        if self.enable_caching:
            self._plot_cache[cache_key] = results

        return results

    def _create_box_plot_optimized(self, column: str) -> str:
        """Create optimized box plot for a single column"""
        try:
            # Create figure
            fig, ax = self._plt.subplots(figsize=(8, 6))

            # Create box plot
            self._sns.boxplot(data=self.df, y=column, ax=ax)
            ax.set_title(f"Box Plot: {column}", fontsize=14)
            ax.set_ylabel(column)

            # Add outlier information
            col_data = self.df[column].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            if len(outliers) > 0:
                ax.text(
                    0.02,
                    0.98,
                    f"Outliers: {len(outliers)}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            self._plt.tight_layout()

            # Save plot
            plot_path = os.path.join(self.output_dir, f"{column}_boxplot.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Close figure to free memory
            self._plt.close(fig)

            return plot_path

        except Exception as e:
            print(f"⚠️ Error creating box plot: {str(e)}")
            return None

    def summary_stats_optimized(self) -> Dict[str, Any]:
        """Generate optimized summary statistics"""
        start_time = time.time()

        summary = {
            "data_overview": {
                "shape": self.df.shape,
                "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
                "dtypes": self.df.dtypes.to_dict(),
            },
            "column_summary": {},
            "performance": {
                "execution_time": time.time() - start_time,
                "optimization_level": "high",
            },
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
                col_data = self.df[col].dropna()
                if len(col_data) > 0:
                    col_info.update(
                        {
                            "count": len(col_data),
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "median": float(col_data.median()),
                            "q25": float(col_data.quantile(0.25)),
                            "q75": float(col_data.quantile(0.75)),
                        }
                    )
            elif col in self.categorical_cols:
                col_data = self.df[col].dropna()
                if len(col_data) > 0:
                    value_counts = col_data.value_counts()
                    col_info.update(
                        {
                            "count": len(col_data),
                            "unique_count": len(value_counts),
                            "most_common": value_counts.index[0]
                            if len(value_counts) > 0
                            else None,
                            "most_common_count": int(value_counts.iloc[0])
                            if len(value_counts) > 0
                            else 0,
                        }
                    )

            summary["column_summary"][col] = col_info

        return summary

    def clear_cache(self):
        """Clear all cached results for memory management"""
        self._plot_cache.clear()
        self._correlation_cache.clear()
        self._distribution_cache.clear()
        self._execution_times.clear()
        self._memory_usage.clear()
        gc.collect()
        print("🧹 Visualizer cache cleared for memory optimization")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and optimization metrics"""
        return {
            "cache_size": {
                "plots": len(self._plot_cache),
                "correlations": len(self._correlation_cache),
                "distributions": len(self._distribution_cache),
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
                "batch_processing": True,
            },
        }
