"""
QuickInsights - Performance Baseline Measurement System

This module provides comprehensive performance measurement tools including:
- Performance benchmarking suite
- Current performance metrics measurement
- Performance regression detection
- Baseline documentation and reporting
"""

import time
import psutil
import gc
import logging
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Generator
from dataclasses import dataclass, field
from contextlib import contextmanager
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""

    operation_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Benchmark result for a specific operation"""

    operation_name: str
    iterations: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    std_deviation: float
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_usage_percent: float
    success_rate: float
    timestamp: datetime


@dataclass
class PerformanceBaseline:
    """Complete performance baseline for the system"""

    baseline_date: datetime
    system_info: Dict[str, Any]
    benchmark_results: List[BenchmarkResult]
    overall_score: float
    recommendations: List[str]


class PerformanceProfiler:
    """Comprehensive performance profiling and measurement system"""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.metrics: List[PerformanceMetric] = []
        self.baseline_results: List[BenchmarkResult] = []
        self.logger = logging.getLogger(__name__)

        # Performance thresholds
        self.warning_threshold_ms = 1000  # 1 second
        self.critical_threshold_ms = 5000  # 5 seconds

        # Memory thresholds
        self.memory_warning_mb = 100
        self.memory_critical_mb = 500

    @contextmanager
    def profile(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for profiling operations"""
        # Force garbage collection before measurement
        gc.collect()

        # Get initial memory state
        memory_before = self.process.memory_info().rss / 1024 / 1024
        cpu_before = self.process.cpu_percent()

        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss

        try:
            yield
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            # Get final measurements
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss

            # Calculate metrics
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            memory_after = end_memory / 1024 / 1024
            memory_peak = max(start_memory, end_memory) / 1024 / 1024
            cpu_after = self.process.cpu_percent()
            cpu_percent = (cpu_before + cpu_after) / 2

            # Create metric
            metric = PerformanceMetric(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                cpu_percent=cpu_percent,
                timestamp=time.time(),
                success=success,
                error_message=error_message,
            )

            # Store metric
            self.metrics.append(metric)

            # Log performance information
            if execution_time > self.critical_threshold_ms:
                self.logger.warning(
                    f"Critical performance: {operation_name} took {execution_time:.2f}ms"
                )
            elif execution_time > self.warning_threshold_ms:
                self.logger.info(
                    f"Performance warning: {operation_name} took {execution_time:.2f}ms"
                )
            else:
                self.logger.debug(
                    f"Good performance: {operation_name} took {execution_time:.2f}ms"
                )

    def get_last_metric(self, metric_name: str) -> Optional[Any]:
        """Get the last measured metric value"""
        if not self.metrics:
            return None

        last_metric = self.metrics[-1]

        if hasattr(last_metric, metric_name):
            return getattr(last_metric, metric_name)

        return None

    def get_metrics(self) -> List[PerformanceMetric]:
        """Get all collected metrics"""
        return self.metrics.copy()

    def measure_operation(
        self, operation_name: str, operation_func: Callable, *args: Any, **kwargs: Any
    ) -> PerformanceMetric:
        """Measure performance of a single operation"""
        # Force garbage collection before measurement
        gc.collect()

        # Get initial memory state
        memory_before = self.process.memory_info().rss / 1024 / 1024
        cpu_before = self.process.cpu_percent()

        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss

        try:
            # Execute operation
            result = operation_func(*args, **kwargs)
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            result = None

        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss

        # Calculate metrics
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        memory_after = end_memory / 1024 / 1024
        memory_peak = max(start_memory, end_memory) / 1024 / 1024
        cpu_after = self.process.cpu_percent()
        cpu_avg = (cpu_before + cpu_after) / 2

        # Create metric
        metric = PerformanceMetric(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=memory_peak,
            cpu_percent=cpu_avg,
            timestamp=time.time(),
            success=success,
            error_message=error_message,
        )

        self.metrics.append(metric)

        # Log performance information
        self._log_performance_metric(metric)

        return metric

    def benchmark_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        iterations: int = 10,
        warmup_iterations: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Benchmark an operation with multiple iterations"""
        self.logger.info(
            f"Starting benchmark for {operation_name} with {iterations} iterations"
        )

        # Warmup phase
        for i in range(warmup_iterations):
            try:
                operation_func(*args, **kwargs)
                gc.collect()
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i+1} failed: {e}")

        # Actual benchmark
        execution_times = []
        memory_usage = []
        memory_peaks = []
        cpu_usage = []
        success_count = 0

        for i in range(iterations):
            try:
                # Force GC before each iteration
                gc.collect()

                # Measure single operation
                metric = self.measure_operation(
                    f"{operation_name}_iter_{i}", operation_func, *args, **kwargs
                )

                if metric.success:
                    execution_times.append(metric.execution_time)
                    memory_usage.append(metric.memory_after)
                    memory_peaks.append(metric.memory_peak)
                    cpu_usage.append(metric.cpu_percent)
                    success_count += 1
                else:
                    self.logger.warning(
                        f"Iteration {i+1} failed: {metric.error_message}"
                    )

            except Exception as e:
                self.logger.error(f"Benchmark iteration {i+1} failed: {e}")

        # Calculate statistics
        if execution_times:
            total_time = sum(execution_times)
            average_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_deviation = (
                statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            )

            memory_usage_avg = statistics.mean(memory_usage)
            memory_peak_avg = statistics.mean(memory_peaks)
            cpu_usage_avg = statistics.mean(cpu_usage)
            success_rate = success_count / iterations
        else:
            # No successful iterations
            total_time = average_time = min_time = max_time = std_deviation = 0
            memory_usage_avg = memory_peak_avg = cpu_usage_avg = 0
            success_rate = 0

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            operation_name=operation_name,
            iterations=iterations,
            total_time=total_time,
            average_time=average_time,
            min_time=min_time,
            max_time=max_time,
            std_deviation=std_deviation,
            memory_usage_mb=memory_usage_avg,
            memory_peak_mb=memory_peak_avg,
            cpu_usage_percent=cpu_usage_avg,
            success_rate=success_rate,
            timestamp=datetime.now(),
        )

        self.baseline_results.append(benchmark_result)

        # Log benchmark results
        self._log_benchmark_result(benchmark_result)

        return benchmark_result

    def _log_performance_metric(self, metric: PerformanceMetric) -> None:
        """Log performance metric with appropriate level"""
        if metric.execution_time > self.critical_threshold_ms:
            level = "ERROR"
            icon = "🚨"
        elif metric.execution_time > self.warning_threshold_ms:
            level = "WARNING"
            icon = "⚠️"
        else:
            level = "INFO"
            icon = "✅"

        message = (
            f"{icon} {metric.operation_name}: "
            f"{metric.execution_time:.2f}ms, "
            f"Memory: {metric.memory_peak:.1f}MB, "
            f"CPU: {metric.cpu_percent:.1f}%"
        )

        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def _log_benchmark_result(self, result: BenchmarkResult) -> None:
        """Log benchmark results"""
        message = (
            f"📊 Benchmark {result.operation_name}: "
            f"Avg: {result.average_time:.2f}ms, "
            f"Min: {result.min_time:.2f}ms, "
            f"Max: {result.max_time:.2f}ms, "
            f"Std: {result.std_deviation:.2f}ms, "
            f"Success: {result.success_rate:.1%}"
        )

        self.logger.info(message)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics:
            return {"error": "No performance metrics available"}

        successful_metrics = [m for m in self.metrics if m.success]

        if not successful_metrics:
            return {"error": "No successful operations recorded"}

        execution_times = [m.execution_time for m in successful_metrics]
        memory_usage = [m.memory_peak for m in successful_metrics]
        cpu_usage = [m.cpu_percent for m in successful_metrics]

        summary = {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(self.metrics),
            "execution_time": {
                "average_ms": statistics.mean(execution_times),
                "median_ms": statistics.median(execution_times),
                "min_ms": min(execution_times),
                "max_ms": max(execution_times),
                "std_deviation_ms": statistics.stdev(execution_times)
                if len(execution_times) > 1
                else 0,
            },
            "memory_usage": {
                "average_mb": statistics.mean(memory_usage),
                "median_mb": statistics.median(memory_usage),
                "min_mb": min(memory_usage),
                "max_mb": max(memory_usage),
                "peak_mb": max(memory_usage),
            },
            "cpu_usage": {
                "average_percent": statistics.mean(cpu_usage),
                "median_percent": statistics.median(cpu_usage),
                "min_percent": min(cpu_usage),
                "max_percent": max(cpu_usage),
            },
            "performance_score": self._calculate_performance_score(successful_metrics),
        }

        return summary

    def _calculate_performance_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics:
            return 0.0

        # Score based on execution time, memory usage, and success rate
        avg_execution_time = statistics.mean([m.execution_time for m in metrics])
        avg_memory_usage = statistics.mean([m.memory_peak for m in metrics])
        success_rate = len([m for m in metrics if m.success]) / len(metrics)

        # Normalize scores (lower is better for time and memory)
        time_score = max(0, 100 - (avg_execution_time / 100))  # 100ms = 0 points
        memory_score = max(0, 100 - (avg_memory_usage / 10))  # 1000MB = 0 points
        success_score = success_rate * 100

        # Weighted average
        overall_score = time_score * 0.4 + memory_score * 0.3 + success_score * 0.3

        return min(100, max(0, overall_score))

    def export_performance_report(self, filepath: Optional[str] = None) -> str:
        """Export detailed performance report"""
        if not filepath:
            filepath = f"performance_report_{int(time.time())}.json"

        report = {
            "report_date": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "metrics": [
                {
                    "operation_name": m.operation_name,
                    "execution_time_ms": m.execution_time,
                    "memory_before_mb": m.memory_before,
                    "memory_after_mb": m.memory_after,
                    "memory_peak_mb": m.memory_peak,
                    "cpu_percent": m.cpu_percent,
                    "timestamp": m.timestamp,
                    "success": m.success,
                    "error_message": m.error_message,
                }
                for m in self.metrics
            ],
            "benchmarks": [
                {
                    "operation_name": b.operation_name,
                    "iterations": b.iterations,
                    "total_time_ms": b.total_time,
                    "average_time_ms": b.average_time,
                    "min_time_ms": b.min_time,
                    "max_time_ms": b.max_time,
                    "std_deviation_ms": b.std_deviation,
                    "memory_usage_mb": b.memory_usage_mb,
                    "memory_peak_mb": b.memory_peak_mb,
                    "cpu_usage_percent": b.cpu_usage_percent,
                    "success_rate": b.success_rate,
                    "timestamp": b.timestamp.isoformat(),
                }
                for b in self.baseline_results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Performance report exported to {filepath}")
        return filepath


class PerformanceBaselineManager:
    """Manager for creating and maintaining performance baselines"""

    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.current_baseline: Optional[PerformanceBaseline] = None
        self.logger = logging.getLogger(__name__)

    def create_baseline(self, profiler: PerformanceProfiler) -> PerformanceBaseline:
        """Create a new performance baseline"""
        # Get system information
        system_info = self._get_system_info()

        # Get current performance summary
        performance_summary = profiler.get_performance_summary()

        # Create baseline
        baseline = PerformanceBaseline(
            baseline_date=datetime.now(),
            system_info=system_info,
            benchmark_results=profiler.baseline_results.copy(),
            overall_score=performance_summary.get("performance_score", 0),
            recommendations=self._generate_recommendations(performance_summary),
        )

        self.current_baseline = baseline
        self._save_baseline(baseline)

        self.logger.info(
            f"Performance baseline created with score: {baseline.overall_score:.1f}/100"
        )
        return baseline

    def load_baseline(self) -> Optional[PerformanceBaseline]:
        """Load existing performance baseline"""
        if not self.baseline_file.exists():
            self.logger.warning(f"Baseline file not found: {self.baseline_file}")
            return None

        try:
            with open(self.baseline_file, "r") as f:
                data = json.load(f)

            # Convert back to PerformanceBaseline object
            baseline = PerformanceBaseline(
                baseline_date=datetime.fromisoformat(data["baseline_date"]),
                system_info=data["system_info"],
                benchmark_results=[],  # Simplified for now
                overall_score=data["overall_score"],
                recommendations=data["recommendations"],
            )

            self.current_baseline = baseline
            self.logger.info(f"Baseline loaded from {self.baseline_file}")
            return baseline

        except Exception as e:
            self.logger.error(f"Error loading baseline: {e}")
            return None

    def compare_with_baseline(
        self, current_profiler: PerformanceProfiler
    ) -> Dict[str, Any]:
        """Compare current performance with baseline"""
        if not self.current_baseline:
            return {"error": "No baseline available for comparison"}

        current_summary = current_profiler.get_performance_summary()
        baseline_score = self.current_baseline.overall_score
        current_score = current_summary.get("performance_score", 0)

        # Calculate performance change
        score_change = current_score - baseline_score
        score_change_percent = (
            (score_change / baseline_score * 100) if baseline_score > 0 else 0
        )

        # Determine regression status
        if score_change < -10:
            status = "REGRESSION"
            severity = "HIGH"
        elif score_change < -5:
            status = "REGRESSION"
            severity = "MEDIUM"
        elif score_change < 0:
            status = "MINOR_REGRESSION"
            severity = "LOW"
        elif score_change > 10:
            status = "IMPROVEMENT"
            severity = "HIGH"
        elif score_change > 5:
            status = "IMPROVEMENT"
            severity = "MEDIUM"
        else:
            status = "STABLE"
            severity = "NONE"

        comparison = {
            "baseline_score": baseline_score,
            "current_score": current_score,
            "score_change": score_change,
            "score_change_percent": score_change_percent,
            "status": status,
            "severity": severity,
            "baseline_date": self.current_baseline.baseline_date.isoformat(),
            "comparison_date": datetime.now().isoformat(),
            "recommendations": self._generate_regression_recommendations(score_change),
        }

        return comparison

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            import platform

            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "node": platform.node(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "disk_usage": {
                    "total_gb": psutil.disk_usage("/").total / 1024**3,
                    "free_gb": psutil.disk_usage("/").free / 1024**3,
                },
            }

            return system_info

        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self, performance_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        avg_execution_time = performance_summary.get("execution_time", {}).get(
            "average_ms", 0
        )
        avg_memory_usage = performance_summary.get("memory_usage", {}).get(
            "average_mb", 0
        )
        performance_score = performance_summary.get("performance_score", 0)

        if avg_execution_time > 1000:
            recommendations.append(
                "Consider optimizing slow operations (>1s execution time)"
            )

        if avg_memory_usage > 100:
            recommendations.append(
                "Review memory usage patterns and implement memory optimization"
            )

        if performance_score < 70:
            recommendations.append(
                "Overall performance needs improvement - review all operations"
            )
        elif performance_score < 85:
            recommendations.append("Performance is acceptable but could be improved")
        else:
            recommendations.append(
                "Performance is excellent - maintain current standards"
            )

        return recommendations

    def _generate_regression_recommendations(self, score_change: float) -> List[str]:
        """Generate recommendations for performance regression"""
        recommendations = []

        if score_change < -10:
            recommendations.append(
                "Critical performance regression detected - immediate investigation required"
            )
            recommendations.append("Review recent code changes for performance impact")
            recommendations.append("Consider rolling back recent changes if necessary")
        elif score_change < -5:
            recommendations.append(
                "Significant performance regression - investigate root cause"
            )
            recommendations.append("Profile slow operations and optimize bottlenecks")
        elif score_change < 0:
            recommendations.append(
                "Minor performance regression - monitor and optimize"
            )

        return recommendations

    def _save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to file"""
        try:
            baseline_data = {
                "baseline_date": baseline.baseline_date.isoformat(),
                "system_info": baseline.system_info,
                "overall_score": baseline.overall_score,
                "recommendations": baseline.recommendations,
            }

            with open(self.baseline_file, "w") as f:
                json.dump(baseline_data, f, indent=2)

            self.logger.info(f"Baseline saved to {self.baseline_file}")

        except Exception as e:
            self.logger.error(f"Error saving baseline: {e}")


@contextmanager
def performance_profile(
    operation_name: str, profiler: PerformanceProfiler
) -> Generator[None, None, None]:
    """Context manager for performance profiling"""
    start_time = time.perf_counter()
    start_memory = profiler.process.memory_info().rss / 1024 / 1024

    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = profiler.process.memory_info().rss / 1024 / 1024

        execution_time = (end_time - start_time) * 1000
        memory_change = end_memory - start_memory

        profiler.logger.info(
            f"Performance profile for {operation_name}: "
            f"{execution_time:.2f}ms, "
            f"Memory change: {memory_change:+.1f}MB "
            f"({start_memory:.1f}MB -> {end_memory:.1f}MB)"
        )


# Convenience functions
def create_performance_profiler() -> PerformanceProfiler:
    """Create and configure a performance profiler"""
    return PerformanceProfiler()


def create_baseline_manager(
    baseline_file: str = "performance_baseline.json",
) -> PerformanceBaselineManager:
    """Create and configure a baseline manager"""
    return PerformanceBaselineManager(baseline_file)


def quick_performance_test(
    operation_func: Callable, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Quick performance test for a single operation"""
    profiler = PerformanceProfiler()
    metric = profiler.measure_operation("quick_test", operation_func, *args, **kwargs)
    return profiler.get_performance_summary()
