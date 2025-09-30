"""
Qubit representation and quantum state management.

This module provides the fundamental quantum state representation using
complex state vectors and supports n-qubit systems.
"""

import numpy as np
from typing import List, Tuple, Union
import math


class Qubit:
    """
    Represents a single qubit with its quantum state.
    
    A qubit can exist in a superposition of |0⟩ and |1⟩ states:
    |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1
    
    The state is represented as a complex vector [α, β].
    """
    
    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0):
        """
        Initialize a qubit with given amplitudes.
        
        Args:
            alpha: Amplitude for |0⟩ state (default: 1.0 for |0⟩)
            beta: Amplitude for |1⟩ state (default: 0.0 for |0⟩)
        """
        # Normalize the state to ensure |α|² + |β|² = 1
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm > 0:
            self.state = np.array([alpha/norm, beta/norm], dtype=complex)
        else:
            self.state = np.array([1.0, 0.0], dtype=complex)
    
    def __str__(self) -> str:
        """String representation of the qubit state."""
        alpha, beta = self.state
        return f"|ψ⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩"
    
    def probability_zero(self) -> float:
        """Calculate probability of measuring |0⟩."""
        return float(abs(self.state[0])**2)
    
    def probability_one(self) -> float:
        """Calculate probability of measuring |1⟩."""
        return float(abs(self.state[1])**2)


class QuantumState:
    """
    Represents the quantum state of an n-qubit system.
    
    For n qubits, the state vector has 2^n complex amplitudes.
    Each amplitude corresponds to a computational basis state.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize an n-qubit quantum state.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        
        # Initialize to |00...0⟩ state (all qubits in |0⟩)
        self.state_vector = np.zeros(self.dimension, dtype=complex)
        self.state_vector[0] = 1.0  # |00...0⟩ state
    
    def __str__(self) -> str:
        """String representation of the quantum state."""
        result = []
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:  # Only show non-zero amplitudes
                # Convert index to binary representation
                binary = format(i, f'0{self.num_qubits}b')
                result.append(f"{amplitude:.3f}|{binary}⟩")
        
        return " + ".join(result) if result else "|0⟩"
    
    def get_amplitude(self, state_index: int) -> complex:
        """
        Get the amplitude for a specific computational basis state.
        
        Args:
            state_index: Index of the basis state (0 to 2^n-1)
        
        Returns:
            Complex amplitude for the state
        """
        if 0 <= state_index < self.dimension:
            return self.state_vector[state_index]
        else:
            raise IndexError(f"State index {state_index} out of range [0, {self.dimension-1}]")
    
    def set_amplitude(self, state_index: int, amplitude: complex):
        """
        Set the amplitude for a specific computational basis state.
        
        Args:
            state_index: Index of the basis state
            amplitude: Complex amplitude to set
        """
        if 0 <= state_index < self.dimension:
            self.state_vector[state_index] = amplitude
        else:
            raise IndexError(f"State index {state_index} out of range [0, {self.dimension-1}]")
    
    def normalize(self):
        """Normalize the state vector to ensure probabilities sum to 1."""
        norm = np.sqrt(np.sum(np.abs(self.state_vector)**2))
        if norm > 0:
            self.state_vector /= norm
    
    def get_probabilities(self) -> np.ndarray:
        """
        Get the probability distribution over all basis states.
        
        Returns:
            Array of probabilities for each basis state
        """
        return np.abs(self.state_vector)**2
    
    def get_state_index(self, qubit_states: List[int]) -> int:
        """
        Convert qubit states to state vector index.
        
        Args:
            qubit_states: List of 0s and 1s representing qubit states
        
        Returns:
            Index in the state vector
        """
        if len(qubit_states) != self.num_qubits:
            raise ValueError(f"Expected {self.num_qubits} qubit states, got {len(qubit_states)}")
        
        index = 0
        for i, state in enumerate(qubit_states):
            if state not in [0, 1]:
                raise ValueError(f"Qubit state must be 0 or 1, got {state}")
            index += state * (2 ** (self.num_qubits - 1 - i))
        
        return index
    
    def get_qubit_states(self, state_index: int) -> List[int]:
        """
        Convert state vector index to qubit states.
        
        Args:
            state_index: Index in the state vector
        
        Returns:
            List of 0s and 1s representing qubit states
        """
        if not (0 <= state_index < self.dimension):
            raise IndexError(f"State index {state_index} out of range [0, {self.dimension-1}]")
        
        binary = format(state_index, f'0{self.num_qubits}b')
        return [int(bit) for bit in binary]
