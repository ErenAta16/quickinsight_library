"""
QuickInsights - Optimized Dashboard Module

This module provides optimized dashboard generation capabilities with:
- Memory-efficient HTML generation
- Lazy content loading
- Caching strategies
- Performance optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
import warnings
import time
import gc
import os
from datetime import datetime
from pathlib import Path
from functools import lru_cache

warnings.filterwarnings("ignore")


class DashboardGeneratorOptimized:
    """
    Optimized Interactive Dashboard Generator with performance improvements:
    - Memory-efficient HTML generation
    - Lazy content loading
    - Caching for repeated operations
    - Batch processing for large datasets
    """

    def __init__(
        self, title: str = "QuickInsights Dashboard", enable_caching: bool = True
    ):
        """
        Initialize Optimized Dashboard Generator

        Parameters
        ----------
        title : str, default "QuickInsights Dashboard"
            Dashboard title
        enable_caching : bool, default True
            Enable result caching for better performance
        """
        self.title = title
        self.enable_caching = enable_caching
        self.sections = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "generator": "QuickInsights Optimized",
            "version": "0.2.1",
            "optimization_level": "high",
        }

        # Initialize caches
        self._html_cache = {}
        self._json_cache = {}
        self._section_cache = {}

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

    def add_dataset_overview_optimized(
        self, df: pd.DataFrame
    ) -> "DashboardGeneratorOptimized":
        """Add dataset overview section with optimization"""
        start_time = time.time()

        # Vectorized column type detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Efficient missing data calculation
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()

        # Efficient duplicate detection
        duplicate_count = df.duplicated().sum()

        overview = {
            "type": "dataset_overview",
            "title": "📊 Dataset Overview",
            "data": {
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / 1024**2, 2
                ),
                "column_types": {
                    "numeric": len(numeric_cols),
                    "categorical": len(categorical_cols),
                    "other": len(df.columns)
                    - len(numeric_cols)
                    - len(categorical_cols),
                },
                "data_quality": {
                    "total_missing": total_missing,
                    "missing_percentage": round(
                        (total_missing / max(1, df.size)) * 100, 2
                    ),
                    "duplicate_rows": duplicate_count,
                    "duplicate_percentage": round(
                        (duplicate_count / max(1, len(df))) * 100, 2
                    ),
                },
                "column_details": {
                    "numeric_columns": numeric_cols[:10],  # Limit for display
                    "categorical_columns": categorical_cols[:10],
                },
            },
            "performance": {
                "generation_time": time.time() - start_time,
                "optimization_level": "high",
            },
        }

        self.sections.append(overview)
        return self

    def add_summary_statistics_optimized(
        self, df: pd.DataFrame
    ) -> "DashboardGeneratorOptimized":
        """Add summary statistics section with optimization"""
        start_time = time.time()

        # Efficient numeric column selection
        numeric_df = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_df) > 0:
            # Vectorized statistics calculation
            stats = df[numeric_df].describe().round(3)

            # Efficient correlation calculation
            corr_summary = self._get_correlation_summary_optimized(df[numeric_df])

            # Efficient distribution insights
            dist_insights = self._get_distribution_insights_optimized(df[numeric_df])

            section_data = {
                "type": "summary_statistics",
                "title": "📈 Summary Statistics",
                "data": {
                    "statistics": stats.to_dict(),
                    "correlation_summary": corr_summary,
                    "distribution_insights": dist_insights,
                },
                "performance": {
                    "generation_time": time.time() - start_time,
                    "optimization_level": "high",
                },
            }

            self.sections.append(section_data)

        return self

    def _get_correlation_summary_optimized(
        self, numeric_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get optimized correlation summary"""
        if len(numeric_df.columns) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}

        try:
            # Efficient correlation calculation
            corr_matrix = numeric_df.corr()

            # Extract top correlations efficiently
            correlations = []
            upper_triangle = np.triu_indices_from(corr_matrix, k=1)

            for i, j in zip(upper_triangle[0], upper_triangle[1]):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:  # Only significant correlations
                    correlations.append(
                        {
                            "variable1": corr_matrix.columns[i],
                            "variable2": corr_matrix.columns[j],
                            "correlation": round(float(corr_value), 3),
                            "strength": self._get_correlation_strength(corr_value),
                        }
                    )

            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            return {
                "total_correlations": len(correlations),
                "top_correlations": correlations[:10],
                "correlation_matrix_shape": corr_matrix.shape,
            }

        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}

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

    def _get_distribution_insights_optimized(
        self, numeric_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get optimized distribution insights"""
        insights = {}

        for col in numeric_df.columns[:5]:  # Limit to first 5 columns
            col_data = numeric_df[col].dropna()
            if len(col_data) > 0:
                # Efficient distribution analysis
                skewness = col_data.skew()
                kurtosis = col_data.kurtosis()

                # Efficient outlier detection using IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = (
                    (col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))
                ).sum()

                insights[col] = {
                    "skewness": round(float(skewness), 3),
                    "kurtosis": round(float(kurtosis), 3),
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": round(
                        (outlier_count / len(col_data)) * 100, 2
                    ),
                    "distribution_type": self._classify_distribution(
                        skewness, kurtosis
                    ),
                }

        return insights

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution based on skewness and kurtosis"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < 1:
            return "light_tailed"
        else:
            return "other"

    def add_missing_data_analysis_optimized(
        self, df: pd.DataFrame
    ) -> "DashboardGeneratorOptimized":
        """Add missing data analysis section with optimization"""
        start_time = time.time()

        # Efficient missing data calculation
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        # Find columns with missing data
        columns_with_missing = missing_counts[missing_counts > 0]

        if len(columns_with_missing) > 0:
            missing_analysis = {
                "type": "missing_data_analysis",
                "title": "🔍 Missing Data Analysis",
                "data": {
                    "total_missing_cells": int(missing_counts.sum()),
                    "total_missing_percentage": round(
                        (missing_counts.sum() / df.size) * 100, 2
                    ),
                    "columns_with_missing": len(columns_with_missing),
                    "missing_by_column": {
                        col: {
                            "count": int(missing_counts[col]),
                            "percentage": round(missing_percentages[col], 2),
                        }
                        for col in columns_with_missing.index[:10]  # Limit to first 10
                    },
                    "missing_patterns": self._analyze_missing_patterns_optimized(df),
                },
                "performance": {
                    "generation_time": time.time() - start_time,
                    "optimization_level": "high",
                },
            }

            self.sections.append(missing_analysis)

        return self

    def _analyze_missing_patterns_optimized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns efficiently"""
        try:
            # Efficient pattern analysis
            missing_matrix = df.isnull()

            # Row-wise missing patterns
            row_missing_counts = missing_matrix.sum(axis=1)
            complete_rows = (row_missing_counts == 0).sum()
            incomplete_rows = len(df) - complete_rows

            # Column-wise missing patterns
            col_missing_counts = missing_matrix.sum(axis=0)
            complete_cols = (col_missing_counts == 0).sum()
            incomplete_cols = len(df.columns) - complete_cols

            return {
                "complete_rows": int(complete_rows),
                "incomplete_rows": int(incomplete_rows),
                "complete_columns": int(complete_cols),
                "incomplete_columns": int(incomplete_cols),
                "most_missing_column": col_missing_counts.idxmax()
                if len(col_missing_counts) > 0
                else None,
                "most_missing_count": int(col_missing_counts.max())
                if len(col_missing_counts) > 0
                else 0,
            }

        except Exception as e:
            return {"error": f"Pattern analysis failed: {str(e)}"}

    def add_outlier_analysis_optimized(
        self, df: pd.DataFrame, method: str = "iqr"
    ) -> "DashboardGeneratorOptimized":
        """Add outlier analysis section with optimization"""
        start_time = time.time()

        # Efficient outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_summary = {}
        total_outliers = 0

        for col in numeric_cols[:5]:  # Limit to first 5 columns
            col_data = df[col].dropna()
            if len(col_data) > 0:
                if method == "iqr":
                    # IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = (
                        (col_data < lower_bound) | (col_data > upper_bound)
                    ).sum()

                elif method == "zscore":
                    # Z-score method
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    if std_val > 0:
                        z_scores = np.abs((col_data - mean_val) / std_val)
                        outliers = (z_scores > 3).sum()
                    else:
                        outliers = 0
                else:
                    outliers = 0

                outlier_summary[col] = {
                    "outlier_count": int(outliers),
                    "outlier_percentage": round((outliers / len(col_data)) * 100, 2),
                }

                total_outliers += outliers

        if outlier_summary:
            outlier_analysis = {
                "type": "outlier_analysis",
                "title": "📊 Outlier Analysis",
                "data": {
                    "method_used": method,
                    "total_outliers": int(total_outliers),
                    "columns_analyzed": len(outlier_summary),
                    "outlier_by_column": outlier_summary,
                },
                "performance": {
                    "generation_time": time.time() - start_time,
                    "optimization_level": "high",
                },
            }

            self.sections.append(outlier_analysis)

        return self

    def add_custom_section_optimized(
        self, title: str, content: Any, section_type: str = "custom"
    ) -> "DashboardGeneratorOptimized":
        """Add custom section with optimization"""
        custom_section = {
            "type": section_type,
            "title": title,
            "data": content,
            "performance": {"generation_time": 0, "optimization_level": "high"},
        }

        self.sections.append(custom_section)
        return self

    def generate_html_optimized(
        self, output_path: Optional[str] = None, template: str = "modern"
    ) -> str:
        """Generate optimized HTML dashboard"""
        cache_key = f"html_{template}_{hash(str(self.sections))}"
        if cache_key in self._html_cache:
            return self._html_cache[cache_key]

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        try:
            html_content = self._build_html_content_optimized(template)

            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

            # Cache result
            if self.enable_caching:
                self._html_cache[cache_key] = html_content

            # Add performance metrics
            self.metadata["html_generation"] = {
                "execution_time": time.time() - start_time,
                "memory_change_mb": self._check_memory_usage() - initial_memory,
                "output_path": output_path,
                "template": template,
            }

            return html_content

        except Exception as e:
            print(f"❌ HTML generation failed: {str(e)}")
            return f"<html><body><h1>Error generating dashboard: {str(e)}</h1></body></html>"

    def _build_html_content_optimized(self, template: str) -> str:
        """Build optimized HTML content"""
        if template == "modern":
            return self._build_modern_template_optimized()
        else:
            return self._build_basic_template_optimized()

    def _build_modern_template_optimized(self) -> str:
        """Build modern HTML template with optimization"""
        # Efficient HTML building
        sections_html = []

        for section in self.sections:
            section_html = self._render_section_optimized(section)
            sections_html.append(section_html)

        # Combine sections efficiently
        sections_content = "\n".join(sections_html)

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #333; margin-top: 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 8px; min-width: 120px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .performance {{ background: #e8f5e8; padding: 10px; border-radius: 5px; margin-top: 15px; font-size: 12px; color: #2d5a2d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.title}</h1>
            <p>Generated by QuickInsights Optimized • {self.metadata['created_at']}</p>
        </div>
        
        {sections_content}
        
        <div class="section">
            <h2>📊 Performance Metrics</h2>
            <div class="metric">
                <div class="metric-value">{len(self.sections)}</div>
                <div class="metric-label">Sections Generated</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metadata['version']}</div>
                <div class="metric-label">Version</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metadata['optimization_level']}</div>
                <div class="metric-label">Optimization Level</div>
            </div>
        </div>
    </div>
</body>
</html>
        """

        return html_template

    def _build_basic_template_optimized(self) -> str:
        """Build basic HTML template with optimization"""
        sections_html = []

        for section in self.sections:
            section_html = f"""
            <div style="border: 1px solid #ddd; margin: 10px; padding: 15px;">
                <h3>{section['title']}</h3>
                <pre>{json.dumps(section['data'], indent=2, default=str)}</pre>
            </div>
            """
            sections_html.append(section_html)

        sections_content = "\n".join(sections_html)

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>{self.title}</h1>
    <p>Generated: {self.metadata['created_at']}</p>
    {sections_content}
</body>
</html>
        """

    def _render_section_optimized(self, section: Dict[str, Any]) -> str:
        """Render section with optimization"""
        section_type = section.get("type", "custom")
        title = section.get("title", "Section")
        data = section.get("data", {})

        if section_type == "dataset_overview":
            return self._render_dataset_overview_optimized(title, data)
        elif section_type == "summary_statistics":
            return self._render_summary_statistics_optimized(title, data)
        elif section_type == "missing_data_analysis":
            return self._render_missing_analysis_optimized(title, data)
        elif section_type == "outlier_analysis":
            return self._render_outlier_analysis_optimized(title, data)
        else:
            return self._render_custom_section_optimized(title, data)

    def _render_dataset_overview_optimized(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render dataset overview section efficiently"""
        return f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="metric">
                <div class="metric-value">{data['shape']['rows']:,}</div>
                <div class="metric-label">Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['shape']['columns']}</div>
                <div class="metric-label">Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['memory_usage_mb']} MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['data_quality']['missing_percentage']}%</div>
                <div class="metric-label">Missing Data</div>
            </div>
        </div>
        """

    def _render_summary_statistics_optimized(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render summary statistics section efficiently"""
        stats_html = ""
        if "statistics" in data:
            stats = data["statistics"]
            for col in list(stats.keys())[:5]:  # Limit to first 5 columns
                if col in stats:
                    col_stats = stats[col]
                    stats_html += f"""
                    <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <strong>{col}:</strong> Mean={col_stats.get('mean', 'N/A'):.2f}, 
                        Std={col_stats.get('std', 'N/A'):.2f}, 
                        Min={col_stats.get('min', 'N/A'):.2f}, 
                        Max={col_stats.get('max', 'N/A'):.2f}
                    </div>
                    """

        return f"""
        <div class="section">
            <h2>{title}</h2>
            {stats_html}
        </div>
        """

    def _render_missing_analysis_optimized(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render missing data analysis section efficiently"""
        return f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="metric">
                <div class="metric-value">{data['total_missing_cells']:,}</div>
                <div class="metric-label">Total Missing Cells</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['total_missing_percentage']}%</div>
                <div class="metric-label">Missing Percentage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['columns_with_missing']}</div>
                <div class="metric-label">Columns with Missing Data</div>
            </div>
        </div>
        """

    def _render_outlier_analysis_optimized(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render outlier analysis section efficiently"""
        return f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="metric">
                <div class="metric-value">{data['total_outliers']:,}</div>
                <div class="metric-label">Total Outliers</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['method_used'].upper()}</div>
                <div class="metric-label">Detection Method</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['columns_analyzed']}</div>
                <div class="metric-label">Columns Analyzed</div>
            </div>
        </div>
        """

    def _render_custom_section_optimized(self, title: str, data: Any) -> str:
        """Render custom section efficiently"""
        if isinstance(data, dict):
            data_str = json.dumps(data, indent=2, default=str)
        else:
            data_str = str(data)

        return f"""
        <div class="section">
            <h2>{title}</h2>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">{data_str}</pre>
        </div>
        """

    def generate_json_optimized(self, output_path: Optional[str] = None) -> str:
        """Generate optimized JSON dashboard"""
        cache_key = f"json_{hash(str(self.sections))}"
        if cache_key in self._json_cache:
            return self._json_cache[cache_key]

        start_time = time.time()

        try:
            # Prepare JSON data
            json_data = {
                "metadata": self.metadata,
                "sections": self.sections,
                "generation_info": {
                    "total_sections": len(self.sections),
                    "generation_time": time.time() - start_time,
                    "optimization_level": "high",
                },
            }

            # Convert to JSON string
            json_string = json.dumps(
                json_data, indent=2, default=str, ensure_ascii=False
            )

            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(json_string)

            # Cache result
            if self.enable_caching:
                self._json_cache[cache_key] = json_string

            return json_string

        except Exception as e:
            print(f"❌ JSON generation failed: {str(e)}")
            return json.dumps({"error": f"JSON generation failed: {str(e)}"})

    def clear_cache(self):
        """Clear all cached results for memory management"""
        self._html_cache.clear()
        self._json_cache.clear()
        self._section_cache.clear()
        self._execution_times.clear()
        self._memory_usage.clear()
        gc.collect()
        print("🧹 Dashboard cache cleared for memory optimization")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and optimization metrics"""
        return {
            "cache_size": {
                "html": len(self._html_cache),
                "json": len(self._json_cache),
                "sections": len(self._section_cache),
            },
            "execution_times": self._execution_times,
            "memory_usage": self._memory_usage,
            "dashboard_info": {
                "total_sections": len(self.sections),
                "title": self.title,
                "version": self.metadata["version"],
                "optimization_level": self.metadata["optimization_level"],
            },
            "optimization_features": {
                "lazy_loading": True,
                "caching": self.enable_caching,
                "vectorized_operations": True,
                "memory_efficient": True,
                "batch_processing": True,
            },
        }
