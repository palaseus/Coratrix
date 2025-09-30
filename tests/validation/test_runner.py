#!/usr/bin/env python3
"""
Comprehensive Test Runner for Coratrix 4.0

This script runs all tests for Coratrix 4.0 with proper organization
and detailed reporting.
"""

import os
import sys
import time
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class CoratrixTestRunner:
    """Comprehensive test runner for Coratrix 4.0."""
    
    def __init__(self, test_dir: str = "tests", output_dir: str = "test_output"):
        """
        Initialize test runner.
        
        Args:
            test_dir: Directory containing test files
            output_dir: Directory for test output
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results = {}
        self.start_time = time.time()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.errors = []
        self.warnings = []
        
        # Test categories with proper organization
        self.test_categories = {
            'performance': [
                'tests/performance/test_gpu_acceleration.py',
                'tests/performance/test_quantum_performance.py'
            ],
            'integration': [
                'tests/integration/test_end_to_end_workflows.py'
            ],
            'benchmarks': [
                'tests/benchmarks/test_performance_benchmarks.py'
            ],
            'validation': [
                'tests/validation/test_system_validation.py'
            ]
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting Coratrix 4.0 Test Suite")
        logger.info("=" * 60)
        
        # Run each test category
        for category, test_files in self.test_categories.items():
            logger.info(f"Running {category} tests...")
            self._run_test_category(category, test_files)
        
        # Generate comprehensive report
        report = self._generate_report()
        
        # Save results
        self._save_results(report)
        
        return report
    
    def _run_test_category(self, category: str, test_files: List[str]):
        """Run tests in a specific category."""
        category_results = {
            'category': category,
            'tests': [],
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'warnings': []
        }
        
        for test_file in test_files:
            test_path = Path(test_file)
            if test_path.exists():
                result = self._run_single_test(test_path)
                category_results['tests'].append(result)
                
                if result['status'] == 'passed':
                    category_results['passed'] += 1
                    self.passed_tests += 1
                elif result['status'] == 'failed':
                    category_results['failed'] += 1
                    self.failed_tests += 1
                elif result['status'] == 'skipped':
                    category_results['skipped'] += 1
                    self.skipped_tests += 1
                
                self.total_tests += 1
                
                # Collect errors and warnings
                if result['errors']:
                    category_results['errors'].extend(result['errors'])
                    self.errors.extend(result['errors'])
                
                if result['warnings']:
                    category_results['warnings'].extend(result['warnings'])
                    self.warnings.extend(result['warnings'])
            else:
                logger.warning(f"Test file not found: {test_path}")
                category_results['errors'].append(f"Test file not found: {test_path}")
        
        self.test_results[category] = category_results
    
    def _run_single_test(self, test_path: Path) -> Dict[str, Any]:
        """Run a single test file."""
        logger.info(f"Running test: {test_path.name}")
        
        start_time = time.time()
        
        try:
            # Run pytest on the test file
            cmd = [
                sys.executable, '-m', 'pytest', str(test_path),
                '-v', '--tb=short', '--durations=10',
                '--junitxml', str(self.output_dir / f"{test_path.stem}.xml")
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                status = 'passed'
                message = 'Test passed successfully'
            elif result.returncode == 1:
                status = 'failed'
                message = 'Test failed'
            elif result.returncode == 2:
                status = 'skipped'
                message = 'Test skipped'
            else:
                status = 'error'
                message = f'Test error (return code: {result.returncode})'
            
            return {
                'test_file': test_path.name,
                'status': status,
                'message': message,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'errors': [],
                'warnings': []
            }
            
        except subprocess.TimeoutExpired:
            return {
                'test_file': test_path.name,
                'status': 'timeout',
                'message': 'Test timed out after 5 minutes',
                'execution_time': 300,
                'stdout': '',
                'stderr': 'Test timed out',
                'return_code': -1,
                'errors': ['Test timed out'],
                'warnings': []
            }
        except Exception as e:
            return {
                'test_file': test_path.name,
                'status': 'error',
                'message': f'Test runner error: {e}',
                'execution_time': time.time() - start_time,
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'errors': [str(e)],
                'warnings': []
            }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time
        
        # Calculate success rate
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # Generate summary
        summary = {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'success_rate': success_rate,
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'errors': len(getattr(self, 'errors', [])),
            'warnings': len(getattr(self, 'warnings', []))
        }
        
        # Generate detailed results
        detailed_results = {}
        for category, results in self.test_results.items():
            detailed_results[category] = {
                'summary': {
                    'total': len(results['tests']),
                    'passed': results['passed'],
                    'failed': results['failed'],
                    'skipped': results['skipped']
                },
                'tests': results['tests'],
                'errors': results.get('errors', []),
                'warnings': results.get('warnings', [])
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            'summary': summary,
            'detailed_results': detailed_results,
            'recommendations': recommendations,
            'test_results': self.test_results
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if self.failed_tests > 0:
            recommendations.append(f"Fix {self.failed_tests} failed tests")
        
        if self.skipped_tests > 0:
            recommendations.append(f"Investigate {self.skipped_tests} skipped tests")
        
        if self.passed_tests / self.total_tests < 0.8:
            recommendations.append("Improve test coverage and reliability")
        
        # Check for specific issues
        for category, results in self.test_results.items():
            if results['failed'] > 0:
                recommendations.append(f"Focus on {category} test failures")
            
            if results['skipped'] > 0:
                recommendations.append(f"Address {category} test skips")
        
        if not recommendations:
            recommendations.append("🎉 All tests passing - Coratrix 4.0 is ready for production!")
        
        return recommendations
    
    def _save_results(self, report: Dict[str, Any]):
        """Save test results to files."""
        # Save JSON report
        json_path = self.output_dir / 'test_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save HTML report
        html_path = self.output_dir / 'test_report.html'
        self._generate_html_report(report, html_path)
        
        # Save summary
        summary_path = self.output_dir / 'test_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_text(report))
        
        logger.info(f"Test results saved to {self.output_dir}")
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: Path):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Coratrix 4.0 Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .skipped {{ color: orange; }}
        .error {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Coratrix 4.0 Test Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {report['summary']['total_tests']}</p>
        <p><strong>Passed:</strong> <span class="passed">{report['summary']['passed_tests']}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{report['summary']['failed_tests']}</span></p>
        <p><strong>Skipped:</strong> <span class="skipped">{report['summary']['skipped_tests']}</span></p>
        <p><strong>Success Rate:</strong> {report['summary']['success_rate']:.1f}%</p>
        <p><strong>Total Time:</strong> {report['summary']['total_time']:.2f} seconds</p>
        <p><strong>Timestamp:</strong> {report['summary']['timestamp']}</p>
    </div>
    
    <h2>Detailed Results</h2>
"""
        
        for category, results in report['detailed_results'].items():
            html_content += f"""
    <div class="category">
        <h3>{category.replace('_', ' ').title()}</h3>
        <p><strong>Total:</strong> {results['summary']['total']} | 
           <strong>Passed:</strong> <span class="passed">{results['summary']['passed']}</span> | 
           <strong>Failed:</strong> <span class="failed">{results['summary']['failed']}</span> | 
           <strong>Skipped:</strong> <span class="skipped">{results['summary']['skipped']}</span></p>
        
        <table>
            <tr>
                <th>Test File</th>
                <th>Status</th>
                <th>Execution Time</th>
                <th>Message</th>
            </tr>
"""
            
            for test in results['tests']:
                status_class = test['status']
                html_content += f"""
            <tr>
                <td>{test['test_file']}</td>
                <td class="{status_class}">{test['status']}</td>
                <td>{test['execution_time']:.2f}s</td>
                <td>{test['message']}</td>
            </tr>
"""
            
            html_content += "        </table>"
            
            if results['errors']:
                html_content += "<h4>Errors:</h4><ul>"
                for error in results['errors']:
                    html_content += f"<li>{error}</li>"
                html_content += "</ul>"
            
            html_content += "    </div>"
        
        # Add recommendations
        if report['recommendations']:
            html_content += """
    <h2>Recommendations</h2>
    <ul>
"""
            for recommendation in report['recommendations']:
                html_content += f"        <li>{recommendation}</li>"
            html_content += "    </ul>"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_summary_text(self, report: Dict[str, Any]) -> str:
        """Generate text summary of test results."""
        summary = f"""
Coratrix 4.0 Test Report
========================

Summary:
--------
Total Tests: {report['summary']['total_tests']}
Passed: {report['summary']['passed_tests']}
Failed: {report['summary']['failed_tests']}
Skipped: {report['summary']['skipped_tests']}
Success Rate: {report['summary']['success_rate']:.1f}%
Total Time: {report['summary']['total_time']:.2f} seconds
Timestamp: {report['summary']['timestamp']}

Detailed Results:
----------------
"""
        
        for category, results in report['detailed_results'].items():
            summary += f"""
{category.replace('_', ' ').title()}:
  Total: {results['summary']['total']}
  Passed: {results['summary']['passed']}
  Failed: {results['summary']['failed']}
  Skipped: {results['summary']['skipped']}
  
  Tests:
"""
            for test in results['tests']:
                summary += f"    {test['test_file']}: {test['status']} ({test['execution_time']:.2f}s)\n"
            
            if results['errors']:
                summary += "  Errors:\n"
                for error in results['errors']:
                    summary += f"    - {error}\n"
        
        if report['recommendations']:
            summary += "\nRecommendations:\n"
            for recommendation in report['recommendations']:
                summary += f"  - {recommendation}\n"
        
        return summary


def main():
    """Main function to run comprehensive tests."""
    print("🚀 Coratrix 4.0 Test Suite")
    print("=" * 50)
    
    # Create test runner
    runner = CoratrixTestRunner()
    
    # Run all tests
    try:
        report = runner.run_all_tests()
        
        # Print summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Skipped: {report['summary']['skipped_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total Time: {report['summary']['total_time']:.2f} seconds")
        
        # Print recommendations
        if report['recommendations']:
            print("\nRecommendations:")
            for recommendation in report['recommendations']:
                print(f"  - {recommendation}")
        
        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0:
            print("\n❌ Some tests failed!")
            sys.exit(1)
        else:
            print("\n🎉 All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()