#!/usr/bin/env python3
"""
Main Test Runner for Coratrix 4.0

This script provides a clean interface to run all Coratrix 4.0 tests
with proper organization and reporting.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function to run Coratrix 4.0 tests."""
    print("🚀 Coratrix 4.0 Test Suite")
    print("=" * 50)
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run the comprehensive test runner
    try:
        result = subprocess.run([
            sys.executable, 'tests/validation/test_runner.py'
        ], check=True)
        
        print("\n🎉 All tests completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())