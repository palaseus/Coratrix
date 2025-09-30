#!/usr/bin/env python3
"""
Comprehensive correctness test suite for Coratrix.

This module runs all correctness tests including unitary consistency,
property-based testing, and circuit fidelity tests.

NOTE: This module is designed to be run as a standalone script, not as a pytest test.
It should not be discovered by pytest as it would cause duplicate test execution.
"""

import unittest
import sys
import os
import time
import json
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from .test_unitary_consistency import TestUnitaryConsistency
from .test_property_based import TestPropertyBased
from .test_circuit_fidelity import TestCircuitFidelity
from .test_quantum_state import TestQuantumState
from .test_quantum_gates import TestQuantumGates
from .test_entanglement import TestEntanglement

# Prevent pytest from discovering this module as a test
# This module is designed to be run as a standalone script
__test__ = False


class CorrectnessTestSuite:
    """Comprehensive test suite for Coratrix correctness."""
    
    def __init__(self, output_file: str = "correctness_test_results.json"):
        self.output_file = output_file
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_suites': {},
            'summary': {}
        }
    
    def run_test_suite(self, test_class, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite and return results."""
        print(f"\nRunning {suite_name}...")
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Compile results
        suite_results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'runtime_seconds': end_time - start_time,
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
            'failures': [{'test': str(f[0]), 'error': str(f[1])} for f in result.failures],
            'errors': [{'test': str(e[0]), 'error': str(e[1])} for e in result.errors]
        }
        
        return suite_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all correctness tests."""
        print("Starting Coratrix correctness test suite...")
        print("=" * 60)
        
        # Define test suites
        test_suites = [
            (TestQuantumState, "Basic Quantum State Tests"),
            (TestQuantumGates, "Quantum Gates Tests"),
            (TestEntanglement, "Entanglement Analysis Tests"),
            (TestUnitaryConsistency, "Unitary Consistency Tests"),
            (TestPropertyBased, "Property-Based Tests"),
            (TestCircuitFidelity, "Circuit Fidelity Tests")
        ]
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_runtime = 0
        
        # Run each test suite
        for test_class, suite_name in test_suites:
            try:
                suite_results = self.run_test_suite(test_class, suite_name)
                self.results['test_suites'][suite_name] = suite_results
                
                total_tests += suite_results['tests_run']
                total_failures += suite_results['failures']
                total_errors += suite_results['errors']
                total_runtime += suite_results['runtime_seconds']
                
                print(f"{suite_name}: {suite_results['tests_run']} tests, "
                      f"{suite_results['failures']} failures, "
                      f"{suite_results['errors']} errors, "
                      f"{suite_results['runtime_seconds']:.2f}s")
                
            except Exception as e:
                print(f"Error running {suite_name}: {e}")
                self.results['test_suites'][suite_name] = {
                    'error': str(e),
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'runtime_seconds': 0,
                    'success_rate': 0
                }
        
        # Compile summary
        self.results['summary'] = {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_runtime_seconds': total_runtime,
            'overall_success_rate': (total_tests - total_failures - total_errors) / max(total_tests, 1),
            'all_tests_passed': total_failures == 0 and total_errors == 0
        }
        
        # Save results
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 60)
        print("CORRECTNESS TEST SUMMARY")
        print("=" * 60)
        
        summary = self.results['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Failures: {summary['total_failures']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Runtime: {summary['total_runtime_seconds']:.2f} seconds")
        print(f"Success Rate: {summary['overall_success_rate']:.2%}")
        print(f"All Tests Passed: {'YES' if summary['all_tests_passed'] else 'NO'}")
        
        if summary['total_failures'] > 0 or summary['total_errors'] > 0:
            print("\nFAILURES AND ERRORS:")
            print("-" * 40)
            
            for suite_name, suite_results in self.results['test_suites'].items():
                if suite_results.get('failures', 0) > 0 or suite_results.get('errors', 0) > 0:
                    print(f"\n{suite_name}:")
                    for failure in suite_results.get('failures', []):
                        print(f"  FAILURE: {failure['test']}")
                        print(f"    {failure['error']}")
                    for error in suite_results.get('errors', []):
                        print(f"  ERROR: {error['test']}")
                        print(f"    {error['error']}")


def main():
    """Main function to run the correctness test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Coratrix correctness tests')
    parser.add_argument('--output', type=str, default='correctness_test_results.json',
                       help='Output file for test results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Run test suite
    suite = CorrectnessTestSuite(args.output)
    results = suite.run_all_tests()
    
    # Exit with appropriate code
    if results['summary']['all_tests_passed']:
        print("\n✅ All correctness tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some correctness tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
