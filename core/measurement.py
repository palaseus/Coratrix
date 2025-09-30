"""
Quantum measurement implementation with probabilistic collapse.

This module provides the Measurement class that handles the probabilistic
collapse of quantum states when measurements are performed.
"""

import numpy as np
import random
from typing import List, Tuple, Union, Dict
from core.qubit import QuantumState


class Measurement:
    """
    Handles quantum measurement with probabilistic state collapse.
    
    When a quantum state is measured, it collapses to one of the
    computational basis states with probability given by the Born rule:
    P(|i⟩) = |⟨i|ψ⟩|² = |αᵢ|²
    """
    
    def __init__(self, quantum_state: QuantumState):
        """
        Initialize measurement for a quantum state.
        
        Args:
            quantum_state: The quantum state to measure
        """
        self.quantum_state = quantum_state
        self.measurement_history = []
    
    def measure_all(self) -> List[int]:
        """
        Measure all qubits and return the result.
        
        Returns:
            List of measurement results (0s and 1s)
        """
        probabilities = self.quantum_state.get_probabilities()
        
        # Choose a basis state according to the probability distribution
        outcome = self._sample_from_distribution(probabilities)
        
        # Collapse the state to the measured outcome
        self._collapse_state(outcome)
        
        # Record the measurement
        self.measurement_history.append(outcome)
        
        # Convert outcome to qubit states
        qubit_states = self.quantum_state.get_qubit_states(outcome)
        return qubit_states
    
    def measure_qubit(self, qubit_index: int) -> int:
        """
        Measure a specific qubit.
        
        Args:
            qubit_index: Index of the qubit to measure
        
        Returns:
            Measurement result (0 or 1)
        """
        if not (0 <= qubit_index < self.quantum_state.num_qubits):
            raise ValueError(f"Qubit index {qubit_index} out of range [0, {self.quantum_state.num_qubits-1}]")
        
        # Calculate probabilities for the qubit being |0⟩ or |1⟩
        prob_zero = 0.0
        prob_one = 0.0
        
        for i, amplitude in enumerate(self.quantum_state.state_vector):
            probability = abs(amplitude)**2
            qubit_state = (i >> (self.quantum_state.num_qubits - 1 - qubit_index)) & 1
            
            if qubit_state == 0:
                prob_zero += probability
            else:
                prob_one += probability
        
        # Normalize probabilities
        total_prob = prob_zero + prob_one
        if total_prob > 0:
            prob_zero /= total_prob
            prob_one /= total_prob
        
        # Sample the measurement result
        if random.random() < prob_zero:
            result = 0
        else:
            result = 1
        
        # Collapse the state based on the measurement
        self._collapse_qubit_state(qubit_index, result)
        
        # Record the measurement
        self.measurement_history.append((qubit_index, result))
        
        return result
    
    def _sample_from_distribution(self, probabilities: np.ndarray) -> int:
        """
        Sample an outcome from a probability distribution.
        
        Args:
            probabilities: Array of probabilities (must sum to 1)
        
        Returns:
            Index of the sampled outcome
        """
        # Use cumulative distribution function
        cumulative = np.cumsum(probabilities)
        random_value = random.random()
        
        for i, cum_prob in enumerate(cumulative):
            if random_value <= cum_prob:
                return i
        
        # Fallback to last index (should not happen with proper probabilities)
        return len(probabilities) - 1
    
    def _collapse_state(self, outcome: int):
        """
        Collapse the quantum state to a specific basis state.
        
        Args:
            outcome: Index of the basis state to collapse to
        """
        # Set all amplitudes to zero except the measured outcome
        self.quantum_state.state_vector.fill(0.0)
        self.quantum_state.state_vector[outcome] = 1.0
    
    def _collapse_qubit_state(self, qubit_index: int, result: int):
        """
        Collapse the state based on measuring a specific qubit.
        
        Args:
            qubit_index: Index of the measured qubit
            result: Measurement result (0 or 1)
        """
        # Zero out amplitudes for states that don't match the measurement
        for i, amplitude in enumerate(self.quantum_state.state_vector):
            qubit_state = (i >> (self.quantum_state.num_qubits - 1 - qubit_index)) & 1
            if qubit_state != result:
                self.quantum_state.state_vector[i] = 0.0
        
        # Renormalize the state
        self.quantum_state.normalize()
    
    def get_measurement_history(self) -> List[Union[int, Tuple[int, int]]]:
        """
        Get the history of all measurements performed.
        
        Returns:
            List of measurement results
        """
        return self.measurement_history.copy()
    
    def measure_multiple(self, quantum_state: QuantumState, shots: int) -> Dict[str, int]:
        """
        Perform multiple measurements and return counts.
        
        Args:
            quantum_state: The quantum state to measure
            shots: Number of measurements to perform
            
        Returns:
            Dictionary mapping bitstrings to counts
        """
        counts = {}
        probabilities = quantum_state.get_probabilities()
        
        for _ in range(shots):
            # Sample from the probability distribution
            outcome = self._sample_from_distribution(probabilities)
            
            # Convert to bitstring
            bitstring = format(outcome, f'0{quantum_state.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def get_expected_value(self, observable: np.ndarray) -> float:
        """
        Calculate the expected value of an observable.
        
        Args:
            observable: Hermitian matrix representing the observable
        
        Returns:
            Expected value ⟨ψ|O|ψ⟩
        """
        if observable.shape != (self.quantum_state.dimension, self.quantum_state.dimension):
            raise ValueError("Observable matrix size doesn't match state dimension")
        
        # Calculate ⟨ψ|O|ψ⟩
        state_vector = self.quantum_state.state_vector
        return float(np.real(state_vector.conj() @ observable @ state_vector))
    
    def get_variance(self, observable: np.ndarray) -> float:
        """
        Calculate the variance of an observable.
        
        Args:
            observable: Hermitian matrix representing the observable
        
        Returns:
            Variance Var(O) = ⟨O²⟩ - ⟨O⟩²
        """
        if observable.shape != (self.quantum_state.dimension, self.quantum_state.dimension):
            raise ValueError("Observable matrix size doesn't match state dimension")
        
        # Calculate ⟨O²⟩
        observable_squared = observable @ observable
        expected_value_squared = self.get_expected_value(observable_squared)
        
        # Calculate ⟨O⟩²
        expected_value = self.get_expected_value(observable)
        expected_value_squared_direct = expected_value**2
        
        return expected_value_squared - expected_value_squared_direct
