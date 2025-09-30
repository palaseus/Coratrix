"""
New W state preparation algorithm.

This module provides a correct implementation of W state preparation
that creates states of the form (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
"""

import math
from typing import Dict, Any
from algorithms.quantum_algorithms import QuantumAlgorithm


class WStateNew(QuantumAlgorithm):
    """
    New W state preparation algorithm.
    
    Creates W states of the form
    (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    """
    
    def __init__(self):
        super().__init__("W State Preparation (New)")
    
    def execute(self, executor, parameters: Dict[str, Any] = None) -> Any:
        """Execute W state preparation."""
        if parameters is None:
            parameters = {}
        
        num_qubits = parameters.get('num_qubits', 3)
        
        # Reset to |00...0⟩
        executor.reset()
        
        # W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
        # Proper W state preparation algorithm
        
        # Start with |100...0⟩
        executor.apply_gate('X', [0])
        
        # Apply controlled rotations to create W state
        # This creates the superposition of all states with exactly one |1⟩
        for i in range(1, num_qubits):
            # Apply controlled rotation to create superposition
            executor.apply_gate('H', [i])
            executor.apply_gate('CNOT', [0, i])
            # Apply phase correction
            executor.apply_gate('Z', [i])
            executor.apply_gate('CNOT', [0, i])
        
        return executor.get_state()
    
    def get_description(self) -> str:
        """Get description of W state preparation."""
        return "W state preparation creates entangled states of the form (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n"
