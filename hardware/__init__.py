"""
Hardware interface module for Coratrix.

This module provides interfaces for hardware backends, OpenQASM import/export,
and interoperability with other quantum computing frameworks.
"""

from .openqasm_interface import OpenQASMInterface, OpenQASMParser, OpenQASMExporter
from .backend_interface import QuantumBackend, CoratrixSimulatorBackend, NoisySimulatorBackend, IBMQStubBackend, BackendManager, BackendResult, BackendCapabilities

__all__ = [
    'OpenQASMInterface',
    'OpenQASMParser',
    'OpenQASMExporter',
    'QuantumBackend',
    'CoratrixSimulatorBackend',
    'NoisySimulatorBackend',
    'IBMQStubBackend',
    'BackendManager',
    'BackendResult',
    'BackendCapabilities'
]
