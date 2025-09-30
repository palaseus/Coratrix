"""
Entanglement analysis and metrics for quantum states.

This module provides comprehensive entanglement analysis including
entanglement entropy, Bell state detection, and separability tests.
"""

import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional, Union
from .qubit import QuantumState
from .scalable_quantum_state import ScalableQuantumState


class EntanglementAnalyzer:
    """
    Comprehensive entanglement analysis for quantum states.
    
    Provides various entanglement measures and detection methods
    for analyzing quantum states and their entanglement properties.
    """
    
    def __init__(self):
        """Initialize the entanglement analyzer."""
        self.bell_states = {
            '|Φ⁺⟩': np.array([1.0/np.sqrt(2), 0.0, 0.0, 1.0/np.sqrt(2)], dtype=complex),
            '|Φ⁻⟩': np.array([1.0/np.sqrt(2), 0.0, 0.0, -1.0/np.sqrt(2)], dtype=complex),
            '|Ψ⁺⟩': np.array([0.0, 1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], dtype=complex),
            '|Ψ⁻⟩': np.array([0.0, 1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0], dtype=complex)
        }
    
    def analyze_entanglement(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> Dict[str, Any]:
        """
        Perform comprehensive entanglement analysis.
        
        Args:
            quantum_state: Quantum state to analyze
        
        Returns:
            Dictionary containing entanglement analysis results
        """
        analysis = {
            'is_entangled': False,
            'entanglement_entropy': 0.0,
            'is_bell_state': False,
            'bell_state_type': None,
            'is_separable': True,
            'concurrence': 0.0,
            'negativity': 0.0,
            'entanglement_rank': 0,
            'entanglement_measures': {}
        }
        
        if quantum_state.num_qubits < 2:
            return analysis
        
        # Check for Bell states
        bell_analysis = self._detect_bell_state(quantum_state)
        analysis.update(bell_analysis)
        
        # Calculate entanglement entropy
        analysis['entanglement_entropy'] = self._calculate_entanglement_entropy(quantum_state)
        
        # Check separability
        analysis['is_separable'] = self._is_separable(quantum_state)
        analysis['is_entangled'] = not analysis['is_separable']
        
        # Calculate concurrence (for 2-qubit systems)
        if quantum_state.num_qubits == 2:
            analysis['concurrence'] = self._calculate_concurrence(quantum_state)
        
        # Calculate negativity
        analysis['negativity'] = self._calculate_negativity(quantum_state)
        
        # Calculate entanglement rank
        analysis['entanglement_rank'] = self._calculate_entanglement_rank(quantum_state)
        
        # Additional entanglement measures
        analysis['entanglement_measures'] = self._calculate_entanglement_measures(quantum_state)
        
        return analysis
    
    def _detect_bell_state(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> Dict[str, Any]:
        """
        Detect if the quantum state is a Bell state.
        
        Args:
            quantum_state: Quantum state to analyze
        
        Returns:
            Dictionary with Bell state detection results
        """
        if quantum_state.num_qubits != 2:
            return {'is_bell_state': False, 'bell_state_type': None}
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # Check against known Bell states
        for bell_name, bell_state in self.bell_states.items():
            if self._states_equivalent(state_vector, bell_state):
                return {'is_bell_state': True, 'bell_state_type': bell_name}
        
        return {'is_bell_state': False, 'bell_state_type': None}
    
    def _states_equivalent(self, state1: np.ndarray, state2: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if two quantum states are equivalent up to a global phase.
        
        Args:
            state1: First state vector
            state2: Second state vector
            tolerance: Numerical tolerance for comparison
        
        Returns:
            True if states are equivalent
        """
        if len(state1) != len(state2):
            return False
        
        # Check if states are proportional (up to global phase)
        # Find non-zero elements
        non_zero_indices = np.where(np.abs(state1) > tolerance)[0]
        
        if len(non_zero_indices) == 0:
            return np.all(np.abs(state2) < tolerance)
        
        # Calculate the ratio of the first non-zero element
        ratio = state2[non_zero_indices[0]] / state1[non_zero_indices[0]]
        
        # Check if all elements are proportional
        return np.allclose(state1 * ratio, state2, atol=tolerance)
    
    def _calculate_entanglement_entropy(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> float:
        """
        Calculate the entanglement entropy of the quantum state.
        
        For a pure state, this measures the entanglement between subsystems.
        """
        if quantum_state.num_qubits < 2:
            return 0.0
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # For 2-qubit systems, calculate entropy of the first qubit
        if quantum_state.num_qubits == 2:
            # Calculate reduced density matrix for first qubit
            prob_0 = 0.0
            prob_1 = 0.0
            
            for i, amplitude in enumerate(state_vector):
                probability = abs(amplitude)**2
                if i < 2:  # First qubit is 0
                    prob_0 += probability
                else:  # First qubit is 1
                    prob_1 += probability
            
            # Calculate von Neumann entropy
            if prob_0 > 0 and prob_1 > 0:
                entropy = -prob_0 * math.log2(prob_0) - prob_1 * math.log2(prob_1)
                return float(entropy)
            else:
                return 0.0
        
        # For larger systems, use a more general approach
        return self._calculate_general_entanglement_entropy(quantum_state)
    
    def _calculate_general_entanglement_entropy(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> float:
        """
        Calculate entanglement entropy for general n-qubit systems.
        
        This is a simplified implementation for demonstration.
        """
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # Calculate entropy of the first qubit by tracing out others
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, amplitude in enumerate(state_vector):
            probability = abs(amplitude)**2
            # Check if first qubit is 0 or 1
            if i < len(state_vector) // 2:  # First qubit is 0
                prob_0 += probability
            else:  # First qubit is 1
                prob_1 += probability
        
        # Calculate von Neumann entropy
        if prob_0 > 0 and prob_1 > 0:
            entropy = -prob_0 * math.log2(prob_0) - prob_1 * math.log2(prob_1)
            return float(entropy)
        else:
            return 0.0
    
    def _is_separable(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> bool:
        """
        Check if the quantum state is separable (not entangled).
        
        This is a simplified check for demonstration.
        """
        if quantum_state.num_qubits < 2:
            return True
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # For 2-qubit systems, check if the state can be written as |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩
        if quantum_state.num_qubits == 2:
            # Check if the state is a product state
            # For a product state |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩, we have:
            # |ψ⟩ = [a₁b₁, a₁b₂, a₂b₁, a₂b₂]
            # This means: ψ[0]*ψ[3] = ψ[1]*ψ[2]
            
            if abs(state_vector[0]) < 1e-10 or abs(state_vector[3]) < 1e-10:
                # If either corner is zero, check if it's a product state
                if abs(state_vector[1]) < 1e-10 and abs(state_vector[2]) < 1e-10:
                    return True  # |00⟩ or |11⟩ state
                return False  # Likely entangled
            
            # Check the product state condition
            product_condition = abs(state_vector[0] * state_vector[3] - state_vector[1] * state_vector[2])
            return product_condition < 1e-10
        
        # For larger systems, use a more general approach
        return self._is_separable_general(quantum_state)
    
    def _is_separable_general(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> bool:
        """
        Check separability for general n-qubit systems.
        
        This is a simplified implementation for demonstration.
        """
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # For GHZ states and other maximally entangled states,
        # check if the state has the characteristic pattern
        num_qubits = quantum_state.num_qubits
        
        # Check for GHZ state pattern: |00...0⟩ + |11...1⟩
        if num_qubits >= 2:
            # Check if only |00...0⟩ and |11...1⟩ have non-zero amplitudes
            all_zero_indices = [0]  # |00...0⟩
            all_one_indices = [2**num_qubits - 1]  # |11...1⟩
            
            # Check if only these two states have non-zero amplitudes
            non_zero_indices = np.where(np.abs(state_vector) > 1e-10)[0]
            
            if len(non_zero_indices) == 2 and set(non_zero_indices) == set(all_zero_indices + all_one_indices):
                # Check if amplitudes are equal (GHZ state)
                amp_0 = abs(state_vector[0])
                amp_1 = abs(state_vector[2**num_qubits - 1])
                if abs(amp_0 - amp_1) < 1e-10:
                    return False  # GHZ state is entangled
        
        # Check for W state pattern: exactly one |1⟩ per term
        # This is a simplified check
        non_zero_count = np.sum(np.abs(state_vector) > 1e-10)
        if non_zero_count == num_qubits:
            # Check if it's a W state pattern
            w_state_indices = [2**i for i in range(num_qubits)]  # |100...0⟩, |010...0⟩, etc.
            non_zero_indices = np.where(np.abs(state_vector) > 1e-10)[0]
            if set(non_zero_indices) == set(w_state_indices):
                return False  # W state is entangled
        
        # For other cases, use a simple heuristic
        # If there are too many non-zero amplitudes, it's likely entangled
        if non_zero_count > 2 ** (num_qubits - 1):
            return False
        
        return True
    
    def _calculate_concurrence(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> float:
        """
        Calculate the concurrence for 2-qubit systems.
        
        Concurrence is a measure of entanglement for 2-qubit systems.
        """
        if quantum_state.num_qubits != 2:
            return 0.0
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # Calculate concurrence for 2-qubit systems
        # C = 2|αδ - βγ| where |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
        alpha = state_vector[0]  # |00⟩
        beta = state_vector[1]   # |01⟩
        gamma = state_vector[2]  # |10⟩
        delta = state_vector[3]  # |11⟩
        
        concurrence = 2 * abs(alpha * delta - beta * gamma)
        return float(concurrence)
    
    def _calculate_negativity(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> float:
        """
        Calculate the negativity of the quantum state.
        
        Negativity is a measure of entanglement based on the partial transpose.
        """
        if quantum_state.num_qubits < 2:
            return 0.0
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # Calculate negativity for 2-qubit systems
        if quantum_state.num_qubits == 2:
            # For 2-qubit systems, negativity can be calculated from concurrence
            concurrence = self._calculate_concurrence(quantum_state)
            if concurrence <= 1.0:
                negativity = (1 - math.sqrt(1 - concurrence**2)) / 2
            else:
                negativity = 0.5  # Maximum negativity
            return float(negativity)
        
        # For larger systems, use a more general approach
        return self._calculate_general_negativity(quantum_state)
    
    def _calculate_general_negativity(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> float:
        """
        Calculate negativity for general n-qubit systems.
        
        This is a simplified implementation for demonstration.
        """
        # For demonstration, return a simplified measure
        entanglement_entropy = self._calculate_entanglement_entropy(quantum_state)
        return min(entanglement_entropy, 1.0)
    
    def _calculate_entanglement_rank(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> int:
        """
        Calculate the entanglement rank of the quantum state.
        
        Entanglement rank is the minimum number of product states needed
        to represent the state.
        """
        if quantum_state.num_qubits < 2:
            return 1
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # Count non-zero amplitudes
        non_zero_count = np.sum(np.abs(state_vector) > 1e-10)
        
        # The entanglement rank is at least the number of non-zero amplitudes
        return int(non_zero_count)
    
    def _calculate_entanglement_measures(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> Dict[str, float]:
        """
        Calculate additional entanglement measures.
        
        Returns a dictionary of various entanglement measures.
        """
        measures = {}
        
        # Entanglement entropy
        measures['entanglement_entropy'] = self._calculate_entanglement_entropy(quantum_state)
        
        # Concurrence (for 2-qubit systems)
        if quantum_state.num_qubits == 2:
            measures['concurrence'] = self._calculate_concurrence(quantum_state)
        
        # Negativity
        measures['negativity'] = self._calculate_negativity(quantum_state)
        
        # Entanglement rank
        measures['entanglement_rank'] = self._calculate_entanglement_rank(quantum_state)
        
        # Purity (for mixed states, this would be more complex)
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        purity = np.sum(np.abs(state_vector)**4)
        measures['purity'] = float(purity)
        
        return measures
    
    def detect_ghz_state(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> Dict[str, Any]:
        """
        Detect if the quantum state is a GHZ state.
        
        GHZ states are maximally entangled states of the form
        (|00...0⟩ + |11...1⟩)/√2
        """
        if quantum_state.num_qubits < 3:
            return {'is_ghz_state': False, 'ghz_type': None}
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # Check for GHZ state pattern
        # GHZ state has non-zero amplitudes only for |00...0⟩ and |11...1⟩
        expected_amplitude = 1.0 / np.sqrt(2)
        
        # Check |00...0⟩ state
        if abs(state_vector[0] - expected_amplitude) < 1e-10:
            # Check |11...1⟩ state
            if abs(state_vector[-1] - expected_amplitude) < 1e-10:
                # Check that all other amplitudes are zero
                other_amplitudes = state_vector[1:-1]
                if np.all(np.abs(other_amplitudes) < 1e-10):
                    return {'is_ghz_state': True, 'ghz_type': '|GHZ⁺⟩'}
        
        # Check for |GHZ⁻⟩ state
        if abs(state_vector[0] - expected_amplitude) < 1e-10:
            if abs(state_vector[-1] + expected_amplitude) < 1e-10:
                other_amplitudes = state_vector[1:-1]
                if np.all(np.abs(other_amplitudes) < 1e-10):
                    return {'is_ghz_state': True, 'ghz_type': '|GHZ⁻⟩'}
        
        return {'is_ghz_state': False, 'ghz_type': None}
    
    def detect_w_state(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> Dict[str, Any]:
        """
        Detect if the quantum state is a W state.
        
        W states are entangled states of the form
        (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
        """
        if quantum_state.num_qubits < 3:
            return {'is_w_state': False, 'w_type': None}
        
        # Get state vector
        if hasattr(quantum_state, 'to_dense'):
            state_vector = quantum_state.to_dense()
        else:
            state_vector = quantum_state.state_vector
        
        # Check for W state pattern
        # W state has non-zero amplitudes only for states with exactly one |1⟩
        expected_amplitude = 1.0 / np.sqrt(quantum_state.num_qubits)
        
        # Check that only states with exactly one |1⟩ have non-zero amplitudes
        non_zero_indices = np.where(np.abs(state_vector) > 1e-10)[0]
        
        if len(non_zero_indices) == quantum_state.num_qubits:
            # Check that all non-zero amplitudes have the same magnitude
            non_zero_amplitudes = state_vector[non_zero_indices]
            if np.allclose(np.abs(non_zero_amplitudes), expected_amplitude):
                # Check that the states correspond to exactly one |1⟩
                for idx in non_zero_indices:
                    binary = format(idx, f'0{quantum_state.num_qubits}b')
                    if binary.count('1') != 1:
                        return {'is_w_state': False, 'w_type': None}
                
                return {'is_w_state': True, 'w_type': '|W⟩'}
        
        return {'is_w_state': False, 'w_type': None}
    
    def get_entanglement_summary(self, quantum_state: Union[QuantumState, ScalableQuantumState]) -> str:
        """
        Get a human-readable summary of the entanglement analysis.
        
        Args:
            quantum_state: Quantum state to analyze
        
        Returns:
            String summary of entanglement properties
        """
        analysis = self.analyze_entanglement(quantum_state)
        
        summary = []
        summary.append(f"Entanglement Analysis for {quantum_state.num_qubits}-qubit system:")
        summary.append(f"  Entangled: {analysis['is_entangled']}")
        summary.append(f"  Separable: {analysis['is_separable']}")
        summary.append(f"  Entanglement Entropy: {analysis['entanglement_entropy']:.4f}")
        
        if analysis['is_bell_state']:
            summary.append(f"  Bell State: {analysis['bell_state_type']}")
        
        if analysis['concurrence'] > 0:
            summary.append(f"  Concurrence: {analysis['concurrence']:.4f}")
        
        if analysis['negativity'] > 0:
            summary.append(f"  Negativity: {analysis['negativity']:.4f}")
        
        summary.append(f"  Entanglement Rank: {analysis['entanglement_rank']}")
        
        return "\n".join(summary)
