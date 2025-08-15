"""
QuickInsights - Tek komutla veri seti analizi

Bu kütüphane, veri analizi öğrenenler ve hızlı veri keşfi yapmak isteyenler için
tasarlanmıştır. Tek satır kod ile veri setiniz hakkında kapsamlı analiz ve
görselleştirmeler elde edebilirsiniz.
"""

from .core import analyze, analyze_numeric, analyze_categorical, LazyAnalyzer, parallel_analysis, chunked_analysis
from .visualizer import correlation_matrix, distribution_plots, summary_stats, create_interactive_plots, box_plots
from .utils import get_data_info, detect_outliers, optimize_dtypes, get_data_sample, AnalysisCache, fast_correlation_matrix, fast_outlier_detection, fast_summary_stats, get_numba_status, benchmark_numba_vs_pandas, get_dask_status, create_dask_client, convert_to_dask, dask_analyze_large_dataset, benchmark_dask_vs_pandas, create_large_test_dataset, get_memory_mapping_status, create_memory_mapped_array, benchmark_memory_vs_mmap, get_profiling_status, profile_function, get_memory_usage, start_memory_tracking, get_memory_snapshot, benchmark_suite, get_async_status, async_analyze_dataset, async_data_loading, async_benchmark_async_vs_sync, run_async_example, StreamingAnalyzer, create_streaming_data_generator, benchmark_streaming_vs_batch, get_gpu_status, convert_to_gpu_array, gpu_correlation_matrix, gpu_outlier_detection, gpu_summary_stats, benchmark_gpu_vs_cpu, get_cloud_status, CloudDataManager, benchmark_cloud_vs_local

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "analyze",
    "analyze_numeric",
    "analyze_categorical",
    "LazyAnalyzer",
    "parallel_analysis",
    "chunked_analysis",
    "correlation_matrix",
    "distribution_plots",
    "summary_stats",
    "create_interactive_plots",
    "box_plots",
    "get_data_info",
    "detect_outliers",
    "optimize_dtypes",
    "get_data_sample",
    "AnalysisCache",
    "fast_correlation_matrix",
    "fast_outlier_detection",
    "fast_summary_stats",
    "get_numba_status",
    "benchmark_numba_vs_pandas",
    "get_dask_status",
    "create_dask_client",
    "convert_to_dask",
    "dask_analyze_large_dataset",
    "benchmark_dask_vs_pandas",
    "create_large_test_dataset",
    "get_memory_mapping_status",
    "create_memory_mapped_array",
    "benchmark_memory_vs_mmap",
    "get_profiling_status",
    "profile_function",
    "get_memory_usage",
    "start_memory_tracking",
    "get_memory_snapshot",
    "benchmark_suite",
    "get_async_status",
    "async_analyze_dataset",
    "async_data_loading",
    "async_benchmark_async_vs_sync",
    "run_async_example",
    "StreamingAnalyzer",
    "create_streaming_data_generator",
    "benchmark_streaming_vs_batch",
    "get_gpu_status",
    "convert_to_gpu_array",
    "gpu_correlation_matrix",
    "gpu_outlier_detection",
    "gpu_summary_stats",
    "benchmark_gpu_vs_cpu",
    "get_cloud_status",
    "CloudDataManager",
    "benchmark_cloud_vs_local"
]
