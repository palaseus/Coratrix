"""
Virtual Machine layer for quantum instruction execution.

This module provides the VM components for parsing and executing
quantum instructions, bridging the gap between quantum programming
and the gate simulator backend.
"""

from vm.parser import QuantumParser
from vm.executor import QuantumExecutor
from vm.instructions import QuantumInstruction, GateInstruction, MeasureInstruction

__all__ = [
    'QuantumParser',
    'QuantumExecutor', 
    'QuantumInstruction',
    'GateInstruction',
    'MeasureInstruction'
]
