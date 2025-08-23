"""
Tests for security utilities module
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from quickinsights.security_utils import (
    OWASPSecurityAuditor,
    InputValidator,
    SecurityTestSuite,
    run_security_assessment,
    validate_and_sanitize_input,
    run_security_tests
)

class TestOWASPSecurityAuditor:
    """Test cases for OWASP Security Auditor"""
    
    def setup_method(self):
        """Setup test data"""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_file.py")
        
        # Create test file with potential vulnerabilities
        with open(self.test_file, 'w') as f:
            f.write("""
import os
import subprocess

# Potential injection vulnerability
def dangerous_function():
    user_input = input("Enter command: ")
    os.system(user_input)  # A01:2021-Injection
    
    # Potential cryptographic failure
    password = "hardcoded_password"  # A02:2021-Cryptographic Failures
    
    # Weak authentication
    if user_input == "admin":
        is_admin = True  # A07:2021-Identification and Authentication Failures
""")
        
        self.auditor = OWASPSecurityAuditor(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test data"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_assess_injection_vulnerabilities(self):
        """Test injection vulnerability assessment"""
        # Create a test file with vulnerabilities
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
import os
import subprocess

def vulnerable_function(user_input):
    os.system(user_input)  # Command injection
    exec(user_input)       # Code injection
    eval(user_input)       # Code injection
    subprocess.call(user_input, shell=True)
''')
            temp_file = f.name
        
        try:
            # Create auditor for the temp file directory
            from quickinsights.security_utils import OWASPSecurityAuditor
            auditor = OWASPSecurityAuditor(os.path.dirname(temp_file))
            vulnerabilities = auditor.assess_injection_vulnerabilities()
            
            assert len(vulnerabilities) > 0
            assert any(v.type == 'A01:2021-Injection' for v in vulnerabilities)
            
            # Check specific vulnerability
            injection_vuln = next(v for v in vulnerabilities if v.type == 'A01:2021-Injection')
            assert injection_vuln.severity == 'HIGH'
            assert any(pattern in injection_vuln.code for pattern in ['os.system(', 'exec(', 'eval('])
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        assert injection_vuln.cwe_id == 'CWE-78'
    
    def test_assess_cryptographic_failures(self):
        """Test cryptographic failure assessment"""
        vulnerabilities = self.auditor.assess_cryptographic_failures()
        
        assert len(vulnerabilities) > 0
        assert any(v.type == 'A02:2021-Cryptographic Failures' for v in vulnerabilities)
        
        # Check specific vulnerability
        crypto_vuln = next(v for v in vulnerabilities if v.type == 'A02:2021-Cryptographic Failures')
        assert crypto_vuln.severity == 'HIGH'
        assert 'hardcoded_password' in crypto_vuln.code
        assert crypto_vuln.cwe_id == 'CWE-259'
    
    def test_assess_authentication_failures(self):
        """Test authentication failure assessment"""
        vulnerabilities = self.auditor.assess_authentication_failures()
        
        assert len(vulnerabilities) > 0
        assert any(v.type == 'A07:2021-Identification and Authentication Failures' for v in vulnerabilities)
        
        # Check specific vulnerability
        auth_vuln = next(v for v in vulnerabilities if v.type == 'A07:2021-Identification and Authentication Failures')
        assert auth_vuln.severity == 'HIGH'
        assert 'is_admin = True' in auth_vuln.code
        assert auth_vuln.cwe_id == 'CWE-287'
    
    def test_comprehensive_assessment(self):
        """Test comprehensive security assessment"""
        report = self.auditor.run_comprehensive_assessment()
        
        assert 'assessment_date' in report
        assert 'project_path' in report
        assert 'total_vulnerabilities' in report
        assert 'security_score' in report
        assert 'compliance_status' in report
        assert 'vulnerabilities' in report
        assert 'recommendations' in report
        
        # Check that vulnerabilities were found
        assert report['total_vulnerabilities'] > 0
        assert report['security_score'] < 100  # Should have deductions for vulnerabilities
        
        # Check recommendations
        assert len(report['recommendations']) > 0
        # Check for security-related recommendations (adjust based on actual recommendation content)
        rec_text = ' '.join(report['recommendations']).lower()
        assert any(keyword in rec_text for keyword in ['input', 'validation', 'sanitization', 'security', 'cryptographic'])

class TestInputValidator:
    """Test cases for Input Validator"""
    
    def setup_method(self):
        """Setup test data"""
        self.validator = InputValidator()
    
    def test_sanitize_html_input(self):
        """Test HTML input sanitization"""
        # Test dangerous HTML tags - script tags should be completely removed
        dangerous_input = '<script>alert("xss")</script>Hello World'
        sanitized = self.validator.sanitize_html_input(dangerous_input)
        
        assert '<script>' not in sanitized
        assert 'Hello World' in sanitized
        # Script tags are removed completely, not escaped
        assert 'script' not in sanitized.lower()
        
        # Test HTML escaping for other content
        html_input = '<p>Hello</p>'
        escaped = self.validator.sanitize_html_input(html_input)
        assert '&lt;p&gt;Hello&lt;/p&gt;' in escaped
    
    def test_validate_file_path(self):
        """Test file path validation"""
        import os
        
        # Create a temporary file in current directory
        temp_file = "test_file.txt"
        with open(temp_file, 'w') as f:
            f.write("test")
        
        try:
            # Test valid paths within current directory
            assert self.validator.validate_file_path(temp_file)
            assert self.validator.validate_file_path(f"./{temp_file}")
            
            # Test path traversal attempts
            assert not self.validator.validate_file_path("../../../etc/passwd")
            assert not self.validator.validate_file_path("..\\..\\..\\windows\\system32\\config")
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_validate_sql_input(self):
        """Test SQL input validation"""
        # Test safe inputs
        assert self.validator.validate_sql_input("normal text")
        assert self.validator.validate_sql_input("user123")
        
        # Test SQL injection attempts
        assert not self.validator.validate_sql_input("'; DROP TABLE users; --")
        assert not self.validator.validate_sql_input("admin' OR '1'='1")
        assert not self.validator.validate_sql_input("UNION SELECT * FROM users")
    
    def test_validate_email(self):
        """Test email validation"""
        # Test valid emails
        assert self.validator.validate_email("test@example.com")
        assert self.validator.validate_email("user.name+tag@domain.co.uk")
        
        # Test invalid emails
        assert not self.validator.validate_email("invalid-email")
        assert not self.validator.validate_email("@domain.com")
        assert not self.validator.validate_email("user@")
    
    def test_validate_url(self):
        """Test URL validation"""
        # Test valid URLs
        assert self.validator.validate_url("https://example.com")
        assert self.validator.validate_url("http://subdomain.example.com/path?param=value")
        
        # Test invalid URLs
        assert not self.validator.validate_url("not-a-url")
        assert not self.validator.validate_url("ftp://invalid")
        assert not self.validator.validate_url("javascript:alert('xss')")

class TestSecurityTestSuite:
    """Test cases for Security Test Suite"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_suite = SecurityTestSuite()
    
    @patch('subprocess.run')
    def test_run_bandit_scan_success(self, mock_run):
        """Test successful Bandit scan"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"results": []}',
            stderr=''
        )
        
        result = self.test_suite.run_bandit_scan()
        
        assert result['status'] == 'success'
        assert 'output' in result
        assert 'vulnerabilities' in result
    
    @patch('subprocess.run')
    def test_run_bandit_scan_error(self, mock_run):
        """Test Bandit scan with errors"""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='',
            stderr='Error: No files found'
        )
        
        result = self.test_suite.run_bandit_scan()
        
        assert result['status'] == 'error'
        assert 'error' in result
    
    @patch('subprocess.run')
    def test_run_bandit_scan_timeout(self, mock_run):
        """Test Bandit scan timeout"""
        mock_run.side_effect = Exception("timeout")
        
        result = self.test_suite.run_bandit_scan()
        
        assert result['status'] == 'error'
    
    @patch('subprocess.run')
    def test_run_safety_check_success(self, mock_run):
        """Test successful Safety check"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"vulnerabilities": []}',
            stderr=''
        )
        
        result = self.test_suite.run_safety_check()
        
        assert result['status'] == 'success'
        assert 'output' in result
    
    @patch('subprocess.run')
    def test_run_safety_check_vulnerabilities(self, mock_run):
        """Test Safety check with vulnerabilities"""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='{"vulnerabilities": [{"package": "django", "severity": "high"}]}',
            stderr=''
        )
        
        result = self.test_suite.run_safety_check()
        
        assert result['status'] == 'vulnerabilities_found'
        assert 'output' in result
    
    def test_comprehensive_security_scan(self):
        """Test comprehensive security scan"""
        with patch.object(self.test_suite, 'run_bandit_scan') as mock_bandit, \
             patch.object(self.test_suite, 'run_safety_check') as mock_safety:
            
            mock_bandit.return_value = {
                'status': 'success',
                'vulnerabilities': []
            }
            mock_safety.return_value = {
                'status': 'success',
                'vulnerabilities': []
            }
            
            result = self.test_suite.run_comprehensive_security_scan()
            
            assert 'scan_timestamp' in result
            assert 'tools_executed' in result
            assert 'overall_security_score' in result
            assert len(result['tools_executed']) == 2

class TestSecurityDecorators:
    """Test cases for security decorators"""
    
    def test_secure_input_html(self):
        """Test secure_input decorator with HTML validation"""
        from quickinsights.security_utils import secure_input
        
        @secure_input('html')
        def test_function(text):
            return f"Processed: {text}"
        
        # Test safe input
        result = test_function("Hello World")
        assert "Hello World" in result
        
        # Test dangerous input (should be sanitized - script tags removed completely)
        result = test_function("<script>alert('xss')</script>Hello")
        assert "<script>" not in result
        assert "alert" not in result
        assert "Hello" in result  # Safe content should remain
    
    def test_secure_input_path(self):
        """Test secure_input decorator with path validation"""
        from quickinsights.security_utils import secure_input
        import os
        
        @secure_input('path')
        def test_function(filepath):
            return f"File: {filepath}"
        
        # Create a temporary file for testing
        temp_file = "test_decorator.txt"
        with open(temp_file, 'w') as f:
            f.write("test")
        
        try:
            # Test safe path
            result = test_function(temp_file)
            assert temp_file in result
            
            # Test dangerous path (should raise error)
            with pytest.raises(ValueError, match="Invalid file path"):
                test_function("../../../etc/passwd")
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class TestSecurityIntegration:
    """Integration tests for security features"""
    
    def test_security_assessment_integration(self):
        """Test end-to-end security assessment"""
        # Create temporary test directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.py")
            
            # Create test file with vulnerabilities
            with open(test_file, 'w') as f:
                f.write('eval("print(\'hello\')")')
            
            # Run security assessment
            report = run_security_assessment(temp_dir)
            
            # Verify report structure
            assert 'assessment_date' in report
            assert 'total_vulnerabilities' in report
            assert 'security_score' in report
            assert 'recommendations' in report
    
    def test_input_validation_integration(self):
        """Test end-to-end input validation"""
        # Test HTML sanitization
        dangerous_input = '<script>alert("xss")</script><p>Hello</p>'
        sanitized = validate_and_sanitize_input(dangerous_input, 'html')
        
        assert '<script>' not in sanitized
        assert '&lt;p&gt;Hello&lt;/p&gt;' in sanitized  # HTML should be escaped
        
        # Test SQL validation
        safe_input = "normal_user_input"
        validated = validate_and_sanitize_input(safe_input, 'sql')
        
        assert validated == safe_input
        
        # Test path validation - create a real file for testing
        import os
        safe_path = "test_integration_file.txt"
        with open(safe_path, 'w') as f:
            f.write("test")
        
        try:
            validated_path = validate_and_sanitize_input(safe_path, 'path')
            assert validated_path == safe_path
        finally:
            if os.path.exists(safe_path):
                os.unlink(safe_path)
        
        assert validated_path == safe_path

# Performance tests for security functions
class TestSecurityPerformance:
    """Performance tests for security functions"""
    
    def test_owasp_auditor_performance(self):
        """Test OWASP auditor performance with large codebase"""
        import time
        
        # Create large test file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "large_test.py")
            
            # Generate large file with many lines
            with open(test_file, 'w') as f:
                for i in range(1000):
                    f.write(f"line_{i} = 'safe_content'\n")
            
            # Add some vulnerabilities
            with open(test_file, 'a') as f:
                f.write("os.system('echo hello')\n")
                f.write("password = 'secret123'\n")
            
            auditor = OWASPSecurityAuditor(temp_dir)
            
            start_time = time.perf_counter()
            report = auditor.run_comprehensive_assessment()
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000
            
            # Should complete within reasonable time
            assert execution_time < 5000  # 5 seconds
            assert 'total_vulnerabilities' in report
    
    def test_input_validator_performance(self):
        """Test input validator performance with large inputs"""
        import time
        
        validator = InputValidator()
        
        # Test with large input
        large_input = "x" * 10000 + "<script>alert('xss')</script>" + "y" * 10000
        
        start_time = time.perf_counter()
        sanitized = validator.sanitize_html_input(large_input)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000
        
        # Should complete within reasonable time
        assert execution_time < 1000  # 1 second
        assert '<script>' not in sanitized
        assert len(sanitized) > 0

if __name__ == "__main__":
    pytest.main([__file__])
