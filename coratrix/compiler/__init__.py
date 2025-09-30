"""
Coratrix Compiler Stack

This module provides the complete compilation pipeline:
- DSL Parser: High-level quantum domain-specific language
- Coratrix IR: Platform-agnostic intermediate representation
- Compiler Passes: Optimization and transformation passes
- Target Generators: Code generation for various frameworks
"""

from .dsl import DSLParser, QuantumDSL
from .ir import CoratrixIR, IRBuilder, IROptimizer
from .passes import CompilerPass, PassManager, PassRegistry
from .targets import TargetGenerator, TargetRegistry
from .compiler import CoratrixCompiler, CompilerOptions, CompilerMode, CompilerResult

__all__ = [
    'DSLParser',
    'QuantumDSL',
    'CoratrixIR',
    'IRBuilder', 
    'IROptimizer',
    'CompilerPass',
    'PassManager',
    'PassRegistry',
    'TargetGenerator',
    'TargetRegistry',
    'CoratrixCompiler',
    'CompilerOptions',
    'CompilerMode',
    'CompilerResult'
]
