"""
Integration tests for security and performance modules
"""
import pytest
import tempfile
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Test the actual imports
def test_module_imports():
    """Test that all security and performance modules can be imported"""
    try:
        from quickinsights.security_utils import (
            OWASPSecurityAuditor,
            InputValidator,
            SecurityTestSuite,
            run_security_assessment,
            validate_and_sanitize_input,
            run_security_tests
        )
        assert True, "Security modules imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import security modules: {e}")
    
    try:
        from quickinsights.memory_manager_v2 import (
            MemoryProfiler,
            IntelligentCache,
            create_memory_profiler,
            create_intelligent_cache,
            get_memory_usage
        )
        assert True, "Memory management modules imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import memory management modules: {e}")
    
    try:
        from quickinsights.performance_baseline import (
            PerformanceProfiler,
            PerformanceBaselineManager,
            create_performance_profiler,
            create_baseline_manager,
            quick_performance_test
        )
        assert True, "Performance modules imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import performance modules: {e}")

class TestSecurityPerformanceIntegration:
    """Integration tests for security and performance features"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'numeric': np.random.randn(1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000),
            'text': [f"text_{i}" for i in range(1000)]
        })
        
        # Create test Python file with vulnerabilities
        self.test_file = os.path.join(self.temp_dir, "test_code.py")
        with open(self.test_file, 'w') as f:
            f.write("""
import os
import subprocess

def dangerous_function():
    user_input = input("Enter command: ")
    os.system(user_input)  # A01:2021-Injection
    
    password = "hardcoded_password"  # A02:2021-Cryptographic Failures
    
    if user_input == "admin":
        is_admin = True  # A07:2021-Identification and Authentication Failures
""")
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_security_assessment_with_performance_monitoring(self):
        """Test security assessment while monitoring performance"""
        from quickinsights.security_utils import OWASPSecurityAuditor
        from quickinsights.performance_baseline import PerformanceProfiler
        
        # Create profiler
        profiler = PerformanceProfiler()
        
        # Measure security assessment performance
        with profiler.profile("security_assessment"):
            auditor = OWASPSecurityAuditor(self.temp_dir)
            report = auditor.run_comprehensive_assessment()
        
        # Verify security assessment results
        assert 'total_vulnerabilities' in report
        assert 'security_score' in report
        assert report['total_vulnerabilities'] > 0
        
        # Verify performance metrics were collected
        summary = profiler.get_performance_summary()
        assert 'total_operations' in summary
        assert summary['total_operations'] > 0
    
    def test_memory_profiling_with_security_validation(self):
        """Test memory profiling with input validation"""
        from quickinsights.memory_manager_v2 import MemoryProfiler, IntelligentCache
        from quickinsights.security_utils import InputValidator
        
        # Create memory profiler
        memory_profiler = MemoryProfiler(alert_threshold_mb=50)
        
        # Create cache
        cache = IntelligentCache(max_size=100, max_memory_mb=50)
        
        # Create input validator
        validator = InputValidator()
        
        # Start memory monitoring
        memory_profiler.start_monitoring(interval=0.1)
        
        try:
            # Perform operations with security validation
            for i in range(100):
                # Validate input
                safe_input = validator.sanitize_html_input(f"<p>Data {i}</p>")
                
                # Cache data
                cache.set(f"key_{i}", safe_input)
                
                # Small delay to allow monitoring
                time.sleep(0.01)
            
            # Stop monitoring
            memory_profiler.stop_monitoring()
            
            # Get memory summary
            memory_summary = memory_profiler.get_memory_summary()
            
            # Verify monitoring worked
            assert 'current_memory_mb' in memory_summary
            assert 'snapshots_count' in memory_summary
            assert memory_summary['snapshots_count'] > 0
            
            # Verify cache worked
            cache_stats = cache.get_stats()
            assert cache_stats['size'] > 0
            assert cache_stats['hit_rate'] >= 0
            
        finally:
            memory_profiler.stop_monitoring()
    
    def test_performance_baseline_with_security_checks(self):
        """Test performance baseline creation with security validation"""
        from quickinsights.performance_baseline import PerformanceProfiler, PerformanceBaselineManager
        from quickinsights.security_utils import InputValidator
        
        # Create profiler and baseline manager
        profiler = PerformanceProfiler()
        baseline_manager = PerformanceBaselineManager("test_baseline.json")
        
        # Create input validator
        validator = InputValidator()
        
        # Define test operations
        def safe_operation(data):
            """Operation with security validation"""
            validated_data = validator.sanitize_html_input(str(data))
            return len(validated_data)
        
        def unsafe_operation(data):
            """Operation without security validation"""
            return len(str(data))
        
        # Benchmark safe operation
        safe_result = profiler.benchmark_operation(
            "safe_operation", 
            safe_operation, 
            iterations=5,
            data="<script>alert('xss')</script>Hello World"
        )
        
        # Benchmark unsafe operation
        unsafe_result = profiler.benchmark_operation(
            "unsafe_operation", 
            unsafe_operation, 
            iterations=5,
            data="<script>alert('xss')</script>Hello World"
        )
        
        # Create baseline
        baseline = baseline_manager.create_baseline(profiler)
        
        # Verify baseline
        assert baseline.overall_score > 0
        assert len(baseline.recommendations) > 0
        assert 'platform' in baseline.system_info  # Check actual content
        
        # Cleanup
        if os.path.exists("test_baseline.json"):
            os.remove("test_baseline.json")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        from quickinsights.security_utils import run_security_assessment
        from quickinsights.memory_manager_v2 import create_memory_profiler, create_intelligent_cache
        from quickinsights.performance_baseline import create_performance_profiler, create_baseline_manager
        
        # 1. Security Assessment
        security_report = run_security_assessment(self.temp_dir)
        assert 'security_score' in security_report
        
        # 2. Memory Profiling
        memory_profiler = create_memory_profiler(alert_threshold_mb=100)
        memory_profiler.start_monitoring(interval=0.1)
        
        # 3. Performance Profiling
        perf_profiler = create_performance_profiler()
        
        # 4. Cache Operations
        cache = create_intelligent_cache(max_size=50, max_memory_mb=25)
        
        # Perform operations
        for i in range(50):
            # Cache data
            cache.set(f"key_{i}", f"value_{i}")
            
            # Measure performance
            perf_profiler.measure_operation(
                f"cache_operation_{i}",
                lambda: cache.get(f"key_{i}")
            )
        
        # Stop monitoring
        memory_profiler.stop_monitoring()
        
        # 5. Create Performance Baseline
        baseline_manager = create_baseline_manager("e2e_baseline.json")
        baseline = baseline_manager.create_baseline(perf_profiler)
        
        # 6. Compare with baseline
        comparison = baseline_manager.compare_with_baseline(perf_profiler)
        
        # Verify results
        assert 'status' in comparison
        assert 'current_score' in comparison
        assert 'baseline_score' in comparison
        
        # Cleanup
        if os.path.exists("e2e_baseline.json"):
            os.remove("e2e_baseline.json")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in integrated system"""
        from quickinsights.security_utils import InputValidator
        from quickinsights.memory_manager_v2 import MemoryProfiler
        from quickinsights.performance_baseline import PerformanceProfiler
        
        # Create components
        validator = InputValidator()
        memory_profiler = MemoryProfiler()
        perf_profiler = PerformanceProfiler()
        
        # Test with invalid inputs - validate_file_path returns False, doesn't raise
        result = validator.validate_file_path("../../../etc/passwd")
        assert result is False
        
        # Test with dangerous inputs
        dangerous_input = '<script>alert("xss")</script>'
        sanitized = validator.sanitize_html_input(dangerous_input)
        assert '<script>' not in sanitized
        
        # Test memory profiler with invalid operations
        try:
            memory_profiler.start_monitoring(interval=-1)  # Invalid interval
        except Exception:
            pass  # Should handle gracefully
        
        # Test performance profiler with failing operations
        def failing_operation():
            raise Exception("Test error")
        
        metric = perf_profiler.measure_operation("failing_op", failing_operation)
        assert not metric.success
        assert metric.error_message is not None
    
    def test_concurrent_operations(self):
        """Test concurrent operations across all modules"""
        import threading
        import concurrent.futures
        
        from quickinsights.security_utils import InputValidator
        from quickinsights.memory_manager_v2 import IntelligentCache
        from quickinsights.performance_baseline import PerformanceProfiler
        
        # Create shared components
        validator = InputValidator()
        cache = IntelligentCache(max_size=1000, max_memory_mb=100)
        profiler = PerformanceProfiler()
        
        # Define worker function
        def worker(worker_id):
            """Worker function for concurrent operations"""
            try:
                # Validate input
                safe_input = validator.sanitize_html_input(f"<p>Worker {worker_id} data</p>")
                
                # Cache data
                cache.set(f"worker_{worker_id}", safe_input)
                
                # Measure performance
                profiler.measure_operation(
                    f"worker_{worker_id}_operation",
                    lambda: len(safe_input)
                )
                
                return True
            except Exception as e:
                return False
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # Verify all operations completed
        assert all(results), "All concurrent operations should complete successfully"
        
        # Verify cache state
        cache_stats = cache.get_stats()
        assert cache_stats['size'] > 0
        
        # Verify performance metrics
        perf_summary = profiler.get_performance_summary()
        assert perf_summary['total_operations'] >= 10

if __name__ == "__main__":
    pytest.main([__file__])
