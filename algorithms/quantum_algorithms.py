"""
Quantum algorithms implementation.

This module implements various quantum algorithms including
Grover's search, Quantum Fourier Transform, and quantum teleportation.
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class QuantumAlgorithm(ABC):
    """
    Abstract base class for quantum algorithms.
    
    All quantum algorithms must implement the execute method
    to perform their operation on the quantum state.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, executor, parameters: Dict[str, Any] = None) -> Any:
        """
        Execute the quantum algorithm.
        
        Args:
            executor: QuantumExecutor instance
            parameters: Algorithm parameters
        
        Returns:
            Result of the algorithm execution
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the algorithm."""
        pass


class GroverAlgorithm(QuantumAlgorithm):
    """
    Grover's search algorithm implementation.
    
    Grover's algorithm provides a quadratic speedup for searching
    in an unstructured database.
    """
    
    def __init__(self):
        super().__init__("Grover's Search Algorithm")
    
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
    
    def _apply_oracle(self, executor, target_state: int, num_qubits: int):
        """Apply the oracle function."""
        # Oracle marks the target state with a phase flip
        # Convert target_state to binary representation
        target_binary = format(target_state, f'0{num_qubits}b')
        
        # Apply X gates to qubits that should be |0⟩ in target
        for i, bit in enumerate(target_binary):
            if bit == '0':
                executor.apply_gate('X', [i])
        
        # Apply multi-controlled Z gate (simplified as single Z)
        executor.apply_gate('Z', [0])
        
        # Reverse the X gates
        for i, bit in enumerate(target_binary):
            if bit == '0':
                executor.apply_gate('X', [i])
    
    def _apply_diffusion(self, executor, num_qubits: int):
        """Apply the diffusion operator."""
        # Apply H gates to all qubits
        for i in range(num_qubits):
            executor.apply_gate('H', [i])
        
        # Apply X gates to all qubits
        for i in range(num_qubits):
            executor.apply_gate('X', [i])
        
        # Apply controlled-Z gate
        if num_qubits > 1:
            executor.apply_gate('CNOT', [0, 1])
            for i in range(2, num_qubits):
                executor.apply_gate('CNOT', [i-1, i])
        
        # Apply X gates again
        for i in range(num_qubits):
            executor.apply_gate('X', [i])
        
        # Apply H gates again
        for i in range(num_qubits):
            executor.apply_gate('H', [i])
    
    def get_description(self) -> str:
        """Get description of Grover's algorithm."""
        return "Grover's search algorithm provides quadratic speedup for searching in unstructured databases."


class QuantumFourierTransform(QuantumAlgorithm):
    """
    Quantum Fourier Transform implementation.
    
    The QFT is a quantum analogue of the discrete Fourier transform.
    """
    
    def __init__(self):
        super().__init__("Quantum Fourier Transform")
    
    def execute(self, executor, parameters: Dict[str, Any] = None) -> Any:
        """Execute the Quantum Fourier Transform."""
        if parameters is None:
            parameters = {}
        
        num_qubits = parameters.get('num_qubits', 3)
        
        # Apply QFT to all qubits (simplified version)
        for i in range(num_qubits):
            # Apply H gate to qubit i
            executor.apply_gate('H', [i])
            
            # Apply controlled phase gates (simplified)
            for j in range(i + 1, num_qubits):
                # Use CNOT as a simplified controlled phase gate
                executor.apply_gate('CNOT', [j, i])
        
        return executor.get_state()
    
    def get_description(self) -> str:
        """Get description of the QFT."""
        return "Quantum Fourier Transform is a quantum analogue of the discrete Fourier transform."


class QuantumTeleportation(QuantumAlgorithm):
    """
    Quantum teleportation protocol implementation.
    
    Teleports the state of one qubit to another using entanglement.
    """
    
    def __init__(self):
        super().__init__("Quantum Teleportation Protocol")
    
    def execute(self, executor, parameters: Dict[str, Any] = None) -> Any:
        """Execute the quantum teleportation protocol."""
        if parameters is None:
            parameters = {}
        
        # Assume 3-qubit system: Alice's qubit (q0), Bell pair (q1, q2)
        num_qubits = 3
        
        # Step 1: Create Bell pair between Alice and Bob (q1, q2)
        executor.apply_gate('H', [1])
        executor.apply_gate('CNOT', [1, 2])
        
        # Step 2: Alice prepares her qubit (q0) in some state
        # For demonstration, we'll put it in |1⟩ state
        executor.apply_gate('X', [0])
        
        # Step 3: Alice performs Bell measurement on her qubit and her half of Bell pair
        executor.apply_gate('CNOT', [0, 1])
        executor.apply_gate('H', [0])
        
        # Step 4: Measure Alice's qubits to get classical bits
        measurement_results = []
        measurement_results.append(executor.measure_qubit(0))
        measurement_results.append(executor.measure_qubit(1))
        
        # Step 5: Bob applies corrections based on measurement results
        if measurement_results[1] == 1:  # If second measurement is 1
            executor.apply_gate('X', [2])
        if measurement_results[0] == 1:  # If first measurement is 1
            executor.apply_gate('Z', [2])
        
        return {
            'measurement_results': measurement_results,
            'final_state': executor.get_state()
        }
    
    def get_description(self) -> str:
        """Get description of quantum teleportation."""
        return "Quantum teleportation protocol for transferring quantum states using entanglement."


class GHZState(QuantumAlgorithm):
    """
    GHZ state preparation algorithm.
    
    Creates maximally entangled GHZ states of the form
    (|00...0⟩ + |11...1⟩)/√2
    """
    
    def __init__(self):
        super().__init__("GHZ State Preparation")
    
    def execute(self, executor, parameters: Dict[str, Any] = None) -> Any:
        """Execute GHZ state preparation."""
        if parameters is None:
            parameters = {}
        
        num_qubits = parameters.get('num_qubits', 3)
        
        # Apply H gate to first qubit
        executor.apply_gate('H', [0])
        
        # Apply CNOT gates to create entanglement
        for i in range(1, num_qubits):
            executor.apply_gate('CNOT', [i-1, i])
        
        return executor.get_state()
    
    def get_description(self) -> str:
        """Get description of GHZ state preparation."""
        return "GHZ state preparation creates maximally entangled states of the form (|00...0⟩ + |11...1⟩)/√2"


class WState(QuantumAlgorithm):
    """
    W state preparation algorithm.
    
    Creates W states of the form
    (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    """
    
    def __init__(self):
        super().__init__("W State Preparation")
    
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
        
        # Clear the state vector first
        current_state.state_vector.fill(0.0)
        
        # Set amplitudes for states with exactly one |1⟩
        for i in range(num_qubits):
            # Create state with |1⟩ at position i
            state_index = 2 ** i  # |100...0⟩, |010...0⟩, etc.
            current_state.state_vector[state_index] = amplitude
        
        return current_state
    
    def get_description(self) -> str:
        """Get description of W state preparation."""
        return "W state preparation creates entangled states of the form (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n"
