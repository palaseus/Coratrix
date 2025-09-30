"""
Manual Grover's search algorithm implementation.

This module provides a correct implementation of Grover's search
by manually constructing the Grover operator.
"""

import math
import numpy as np
from typing import Dict, Any
from algorithms.quantum_algorithms import QuantumAlgorithm


class GroverManual(QuantumAlgorithm):
    """
    Manual Grover's search algorithm implementation.
    
    Grover's algorithm provides a quadratic speedup for searching
    in an unstructured database.
    """
    
    def __init__(self):
        super().__init__("Grover's Search Algorithm (Manual)")
    
    def execute(self, executor, parameters: Dict[str, Any] = None) -> Any:
        """Execute Grover's search algorithm."""
        if parameters is None:
            parameters = {}
        
        num_qubits = parameters.get('num_qubits', 2)
        target_state = parameters.get('target_state', None)
        iterations = parameters.get('iterations', None)
        
        # Calculate optimal number of iterations: π/4 * √N
        if iterations is None:
            iterations = int(math.pi / 4 * math.sqrt(2 ** num_qubits))
        
        # Get the current state
        current_state = executor.get_state()
        
        # Create uniform superposition
        N = 2 ** num_qubits
        amplitude = 1.0 / math.sqrt(N)
        current_state.state_vector.fill(amplitude)
        
        # Apply Grover iterations
        for _ in range(iterations):
            # Oracle: flip the phase of the target state
            if target_state is not None:
                current_state.state_vector[target_state] *= -1
            else:
                # Default: flip the |11...1⟩ state
                current_state.state_vector[N-1] *= -1
            
            # Diffusion: inversion about the mean
            mean_amplitude = np.mean(current_state.state_vector)
            current_state.state_vector = 2 * mean_amplitude - current_state.state_vector
        
        return current_state
    
    def get_description(self) -> str:
        """Get description of Grover's search algorithm."""
        return "Grover's search algorithm provides quadratic speedup for searching in unstructured databases"
