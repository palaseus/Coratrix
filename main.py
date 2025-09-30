#!/usr/bin/env python3
"""
Main entry point for Coratrix quantum computer.

This module provides the main entry point for running Coratrix
from the command line or as a Python module.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.enhanced_cli import main

if __name__ == "__main__":
    main()
