"""
Coratrix Core Simulation Module

This module provides the fundamental quantum simulation capabilities:
- Quantum state representation and manipulation
- Quantum circuit construction and execution
- Quantum gate operations
- Quantum algorithms and analysis
"""

from .quantum_state import ScalableQuantumState
from .quantum_circuit import QuantumCircuit
from .quantum_circuit import QuantumGate, GateFactory
from .quantum_algorithms import QuantumAlgorithm, AlgorithmRegistry
from .entanglement import EntanglementAnalyzer
from .noise import NoiseModel, NoiseChannel

__all__ = [
    'ScalableQuantumState',
    'QuantumCircuit', 
    'QuantumGate',
    'GateFactory',
    'QuantumAlgorithm',
    'AlgorithmRegistry',
    'EntanglementAnalyzer',
    'NoiseModel',
    'NoiseChannel'
]
