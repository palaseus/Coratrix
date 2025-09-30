"""
Coratrix Compiler Infrastructure.

This module provides a complete compilation pipeline from high-level DSL
to Coratrix IR to various target formats (QASM, Qiskit, PennyLane, etc.).
"""

from .dsl import QuantumDSL, DSLParser
from .ir import CoratrixIR, IRBuilder, IROptimizer
from .passes import CompilerPass, PassManager
from .targets import QASMTarget, QiskitTarget, PennyLaneTarget
from .backend import BackendInterface, BackendManager

__all__ = [
    'QuantumDSL', 'DSLParser',
    'CoratrixIR', 'IRBuilder', 'IROptimizer', 
    'CompilerPass', 'PassManager',
    'QASMTarget', 'QiskitTarget', 'PennyLaneTarget',
    'BackendInterface', 'BackendManager'
]
