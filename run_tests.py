#!/usr/bin/env python3
"""
Custom test runner that suppresses all warnings for clean output.
"""
import warnings
import sys
import subprocess

def main():
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Run pytest with warning suppression
    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        "--tb=no", "-q",
        "-W", "ignore::DeprecationWarning",
        "-W", "ignore::UserWarning", 
        "-W", "ignore:scipy.sparse._index.SparseEfficiencyWarning",
        "-W", "ignore:numpy.ComplexWarning"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
