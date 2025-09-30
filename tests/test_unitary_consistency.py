"""
Unitary consistency tests for quantum gates and circuits.

This module tests that quantum gates preserve unitarity and that
circuits maintain proper quantum mechanical properties.
"""

import unittest
import numpy as np
import sys
import os
from typing import List, Tuple, Dict, Any
import math

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qubit import QuantumState
from core.scalable_quantum_state import ScalableQuantumState
from core.gates import XGate, YGate, ZGate, HGate, CNOTGate
from core.entanglement_analysis import EntanglementAnalyzer
from core.circuit import QuantumCircuit


class TestUnitaryConsistency(unittest.TestCase):
    """Test cases for unitary consistency of quantum operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-10
        self.test_qubit_counts = [2, 3, 4, 5]
    
    def test_gate_unitarity(self):
        """Test that all gates are unitary (U†U = I)."""
        gates = [XGate(), YGate(), ZGate(), HGate()]
        
        for num_qubits in self.test_qubit_counts:
            for gate in gates:
                with self.subTest(gate=gate.name, qubits=num_qubits):
                    # Get gate matrix
                    matrix = gate.get_matrix(num_qubits, [0])
                    
                    # Check unitarity: U†U = I
                    unitary_check = matrix.conj().T @ matrix
                    identity = np.eye(matrix.shape[0], dtype=complex)
                    
                    np.testing.assert_allclose(
                        unitary_check, identity, 
                        atol=self.tolerance,
                        err_msg=f"{gate.name} gate is not unitary for {num_qubits} qubits"
                    )
    
    def test_cnot_unitarity(self):
        """Test that CNOT gate is unitary."""
        for num_qubits in [2, 3, 4]:
            cnot = CNOTGate()
            matrix = cnot.get_matrix(num_qubits, [0, 1])
            
            # Check unitarity: U†U = I
            unitary_check = matrix.conj().T @ matrix
            identity = np.eye(matrix.shape[0], dtype=complex)
            
            np.testing.assert_allclose(
                unitary_check, identity,
                atol=self.tolerance,
                err_msg=f"CNOT gate is not unitary for {num_qubits} qubits"
            )
    
    def test_state_normalization_preservation(self):
        """Test that gates preserve state normalization."""
        gates = [XGate(), YGate(), ZGate(), HGate(), CNOTGate()]
        
        for num_qubits in self.test_qubit_counts:
            for gate in gates:
                with self.subTest(gate=gate.name, qubits=num_qubits):
                    # Create random normalized state
                    state = QuantumState(num_qubits)
                    random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
                    random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
                    state.state_vector = random_amplitudes
                    
                    # Check initial normalization
                    initial_norm = np.sum(np.abs(state.state_vector)**2)
                    self.assertAlmostEqual(initial_norm, 1.0, places=10)
                    
                    # Apply gate
                    if gate.name == "CNOT" and num_qubits >= 2:
                        gate.apply(state, [0, 1])
                    else:
                        gate.apply(state, [0])
                    
                    # Check normalization is preserved
                    final_norm = np.sum(np.abs(state.state_vector)**2)
                    self.assertAlmostEqual(final_norm, 1.0, places=10)
    
    def test_reversibility(self):
        """Test that gates are reversible (U†U = I)."""
        gates = [XGate(), YGate(), ZGate(), HGate()]
        
        for num_qubits in self.test_qubit_counts:
            for gate in gates:
                with self.subTest(gate=gate.name, qubits=num_qubits):
                    # Create random state
                    state = QuantumState(num_qubits)
                    random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
                    random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
                    state.state_vector = random_amplitudes
                    
                    # Store original state
                    original_state = state.state_vector.copy()
                    
                    # Apply gate
                    gate.apply(state, [0])
                    
                    # Apply inverse gate (conjugate transpose)
                    gate_matrix = gate.get_matrix(num_qubits, [0])
                    inverse_matrix = gate_matrix.conj().T
                    state.state_vector = inverse_matrix @ state.state_vector
                    
                    # Check reversibility
                    np.testing.assert_allclose(
                        state.state_vector, original_state,
                        atol=self.tolerance,
                        err_msg=f"{gate.name} gate is not reversible for {num_qubits} qubits"
                    )
    
    def test_circuit_unitarity(self):
        """Test that circuits composed of unitary gates are unitary."""
        for num_qubits in self.test_qubit_counts:
            with self.subTest(qubits=num_qubits):
                circuit = QuantumCircuit(num_qubits)
                
                # Add multiple gates
                circuit.add_gate(HGate(), [0])
                circuit.add_gate(XGate(), [0])
                circuit.add_gate(HGate(), [0])
                
                if num_qubits >= 2:
                    circuit.add_gate(CNOTGate(), [0, 1])
                
                # Execute circuit
                circuit.execute()
                
                # Check that final state is normalized
                final_norm = np.sum(np.abs(circuit.get_state().state_vector)**2)
                self.assertAlmostEqual(final_norm, 1.0, places=10)
    
    def test_scalable_state_consistency(self):
        """Test that ScalableQuantumState maintains consistency with QuantumState."""
        for num_qubits in [2, 3, 4]:
            with self.subTest(qubits=num_qubits):
                # Create states with same initial conditions
                state1 = QuantumState(num_qubits)
                state2 = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
                
                # Apply same operations
                h_gate = HGate()
                x_gate = XGate()
                
                h_gate.apply(state1, [0])
                h_gate.apply(state2, [0])
                
                x_gate.apply(state1, [0])
                x_gate.apply(state2, [0])
                
                # Check states are equivalent
                np.testing.assert_allclose(
                    state1.state_vector, state2.to_dense(),
                    atol=self.tolerance,
                    err_msg=f"ScalableQuantumState inconsistent with QuantumState for {num_qubits} qubits"
                )
    
    def test_measurement_statistics(self):
        """Test that measurement statistics follow Born rule."""
        for num_qubits in [2, 3]:
            with self.subTest(qubits=num_qubits):
                # Create Bell state
                state = QuantumState(num_qubits)
                if num_qubits >= 2:
                    # Create |00⟩ + |11⟩ state
                    state.set_amplitude(0, 1.0/np.sqrt(2))
                    state.set_amplitude(2**num_qubits - 1, 1.0/np.sqrt(2))
                    state.normalize()
                
                # Get probabilities
                probabilities = state.get_probabilities()
                
                # Check probabilities sum to 1
                self.assertAlmostEqual(np.sum(probabilities), 1.0, places=10)
                
                # Check specific probabilities for Bell state
                if num_qubits >= 2:
                    self.assertAlmostEqual(probabilities[0], 0.5, places=10)
                    self.assertAlmostEqual(probabilities[-1], 0.5, places=10)
                    for i in range(1, 2**num_qubits - 1):
                        self.assertAlmostEqual(probabilities[i], 0.0, places=10)
    
    def test_entanglement_entropy_bounds(self):
        """Test that entanglement entropy is within valid bounds."""
        analyzer = EntanglementAnalyzer()
        
        for num_qubits in [2, 3, 4]:
            with self.subTest(qubits=num_qubits):
                state = QuantumState(num_qubits)
                
                # Test separable state (should have entropy = 0)
                state.set_amplitude(0, 1.0)
                analysis = analyzer.analyze_entanglement(state)
                entropy = analysis['entanglement_entropy']
                self.assertAlmostEqual(entropy, 0.0, places=10)
                
                # Test maximally entangled state
                if num_qubits >= 2:
                    state.set_amplitude(0, 1.0/np.sqrt(2))
                    state.set_amplitude(2**num_qubits - 1, 1.0/np.sqrt(2))
                    state.normalize()
                    analysis = analyzer.analyze_entanglement(state)
                    entropy = analysis['entanglement_entropy']
                    
                    # Entropy should be between 0 and log2(2) = 1
                    self.assertGreaterEqual(entropy, 0.0)
                    self.assertLessEqual(entropy, 1.0)
    
    def test_gate_composition(self):
        """Test that gate composition equals matrix multiplication."""
        for num_qubits in [2, 3]:
            with self.subTest(qubits=num_qubits):
                # Create circuit with multiple gates
                circuit = QuantumCircuit(num_qubits)
                circuit.add_gate(HGate(), [0])
                circuit.add_gate(XGate(), [0])
                circuit.add_gate(HGate(), [0])
                
                # Get individual gate matrices
                h_matrix = HGate().get_matrix(num_qubits, [0])
                x_matrix = XGate().get_matrix(num_qubits, [0])
                
                # Compute composition matrix
                composition_matrix = h_matrix @ x_matrix @ h_matrix
                
                # Apply gates to state
                state = QuantumState(num_qubits)
                circuit.execute()
                
                # Apply composition matrix to same initial state
                initial_state = np.zeros(2**num_qubits, dtype=complex)
                initial_state[0] = 1.0
                composed_state = composition_matrix @ initial_state
                
                # Check states are equivalent
                np.testing.assert_allclose(
                    circuit.get_state().state_vector, composed_state,
                    atol=self.tolerance,
                    err_msg=f"Gate composition inconsistent for {num_qubits} qubits"
                )


if __name__ == '__main__':
    unittest.main()
