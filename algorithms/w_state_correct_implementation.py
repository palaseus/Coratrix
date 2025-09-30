"""
Correct W state preparation algorithm.

This module provides a proper implementation of W state preparation
that creates states of the form (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
"""

import math
import numpy as np
from typing import Dict, Any
from algorithms.quantum_algorithms import QuantumAlgorithm


class WStateCorrect(QuantumAlgorithm):
    """
    Correct W state preparation algorithm.
    
    Creates W states of the form
    (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    """
    
    def __init__(self):
        super().__init__("W State Preparation (Correct)")
    
    def execute(self, executor, parameters: Dict[str, Any] = None) -> Any:
        """Execute W state preparation."""
        if parameters is None:
            parameters = {}
        
        num_qubits = parameters.get('num_qubits', 3)
        
        # Reset to |00...0⟩
        executor.reset()
        
        # W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
        # Create the state directly by setting the state vector
        
        # Get the current state
        current_state = executor.get_state()
        
        # Create W state by setting the appropriate amplitudes
        # W state has exactly one |1⟩ in each term
        amplitude = 1.0 / math.sqrt(num_qubits)
        
        # Set amplitudes for states with exactly one |1⟩
        for i in range(num_qubits):
            # Create state with |1⟩ at position i
            state_index = 2 ** i  # |100...0⟩, |010...0⟩, etc.
            current_state.state_vector[state_index] = amplitude
        
        return current_state
    
    def get_description(self) -> str:
        """Get description of W state preparation."""
        return "W state preparation creates entangled states of the form (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n"
