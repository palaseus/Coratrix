"""
Coratrix CLI Interface

This module provides command-line interfaces for Coratrix:
- coratrixc: Quantum compiler CLI
- coratrix: Interactive quantum computing shell
"""

from typing import List
from .compiler_cli import CoratrixCompilerCLI
from .interactive_cli import CoratrixInteractiveCLI

# Main CLI class
class CoratrixCLI:
    """Main CLI interface for Coratrix."""
    
    def __init__(self):
        self.compiler_cli = CoratrixCompilerCLI()
        self.interactive_cli = CoratrixInteractiveCLI()
    
    def run_compiler(self, args: List[str] = None):
        """Run the compiler CLI."""
        self.compiler_cli.run(args)
    
    def run_interactive(self):
        """Run the interactive CLI."""
        self.interactive_cli.start()

__all__ = [
    'CoratrixCLI',
    'CoratrixCompilerCLI',
    'CoratrixInteractiveCLI'
]
