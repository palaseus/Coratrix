"""
Command-line interface for Coratrix quantum computer.

This module provides the CLI interface for running quantum scripts
and interacting with the quantum virtual machine.
"""

from cli.cli import main, run_script, interactive_mode

__all__ = ['main', 'run_script', 'interactive_mode']
