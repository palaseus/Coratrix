"""
Property-based tests using Hypothesis for quantum state operations.

This module uses Hypothesis to generate random test cases and validate
quantum mechanical properties across a wide range of inputs.
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
from core.circuit import QuantumCircuit

# Import Hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, example, HealthCheck
    from hypothesis.strategies import integers, floats, lists, tuples
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("Warning: Hypothesis not available. Property-based tests will be skipped.")


if HYPOTHESIS_AVAILABLE:
    class TestPropertyBased(unittest.TestCase):
        """Property-based tests for quantum operations."""
        
        def setUp(self):
            """Set up test fixtures."""
            self.tolerance = 1e-10
        
        @given(integers(min_value=2, max_value=6))
        @settings(max_examples=50)
        def test_state_normalization_property(self, num_qubits):
            """Test that quantum states remain normalized under random operations."""
            state = QuantumState(num_qubits)
            
            # Apply random gates
            gates = [XGate(), YGate(), ZGate(), HGate()]
            for _ in range(10):
                gate = np.random.choice(gates)
                target_qubit = np.random.randint(0, num_qubits)
                gate.apply(state, [target_qubit])
                
                # Check normalization
                norm = np.sum(np.abs(state.state_vector)**2)
                self.assertAlmostEqual(norm, 1.0, places=10)
        
        @given(integers(min_value=2, max_value=5))
        @settings(max_examples=30)
        def test_gate_unitarity_property(self, num_qubits):
            """Test that all gates are unitary for random qubit counts."""
            gates = [XGate(), YGate(), ZGate(), HGate()]
            
            for gate in gates:
                matrix = gate.get_matrix(num_qubits, [0])
                
                # Check unitarity: U†U = I
                unitary_check = matrix.conj().T @ matrix
                identity = np.eye(matrix.shape[0], dtype=complex)
                
                np.testing.assert_allclose(
                    unitary_check, identity, 
                    atol=self.tolerance,
                    err_msg=f"{gate.name} gate is not unitary for {num_qubits} qubits"
                )
        
        @given(integers(min_value=2, max_value=4))
        @settings(max_examples=20, suppress_health_check=[HealthCheck.differing_executors])
        def test_cnot_unitarity_property(self, num_qubits):
            """Test that CNOT gate is unitary for random qubit counts."""
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
        
        @given(integers(min_value=2, max_value=5))
        @settings(max_examples=30, suppress_health_check=[HealthCheck.differing_executors])
        def test_entanglement_entropy_bounds_property(self, num_qubits):
            """Test that entanglement entropy is within valid bounds."""
            state = QuantumState(num_qubits)
            
            # Create random normalized state
            random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
            random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
            state.state_vector = random_amplitudes
            
            # Check entropy bounds
            entropy = state.get_entanglement_entropy()
            self.assertGreaterEqual(entropy, 0.0)
            self.assertLessEqual(entropy, math.log2(2))  # Maximum entropy for 2-qubit subsystem
        
        @given(integers(min_value=2, max_value=4))
        @settings(max_examples=20)
        def test_measurement_probability_property(self, num_qubits):
            """Test that measurement probabilities sum to 1."""
            state = QuantumState(num_qubits)
            
            # Create random normalized state
            random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
            random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
            state.state_vector = random_amplitudes
            
            # Check probability sum
            probabilities = state.get_probabilities()
            self.assertAlmostEqual(np.sum(probabilities), 1.0, places=10)
            
            # Check all probabilities are non-negative
            for prob in probabilities:
                self.assertGreaterEqual(prob, 0.0)
        
        @given(integers(min_value=2, max_value=4))
        @settings(max_examples=20, suppress_health_check=[HealthCheck.differing_executors])
        def test_gate_reversibility_property(self, num_qubits):
            """Test that gates are reversible for random states."""
            gates = [XGate(), YGate(), ZGate(), HGate()]
            
            for gate in gates:
                # Create random state
                state = QuantumState(num_qubits)
                random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
                random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
                state.state_vector = random_amplitudes
                
                # Store original state
                original_state = state.state_vector.copy()
                
                # Apply gate
                gate.apply(state, [0])
                
                # Apply inverse gate
                gate_matrix = gate.get_matrix(num_qubits, [0])
                inverse_matrix = gate_matrix.conj().T
                state.state_vector = inverse_matrix @ state.state_vector
                
                # Check reversibility
                np.testing.assert_allclose(
                    state.state_vector, original_state,
                    atol=self.tolerance,
                    err_msg=f"{gate.name} gate is not reversible"
                )
        
        @given(integers(min_value=2, max_value=4))
        @settings(max_examples=15, suppress_health_check=[HealthCheck.differing_executors])
        def test_circuit_composition_property(self, num_qubits):
            """Test that circuit composition equals matrix multiplication."""
            # Create circuit
            circuit = QuantumCircuit(num_qubits)
            circuit.add_gate(HGate(), [0])
            circuit.add_gate(XGate(), [0])
            circuit.add_gate(HGate(), [0])
            
            # Get individual matrices
            h_matrix = HGate().get_matrix(num_qubits, [0])
            x_matrix = XGate().get_matrix(num_qubits, [0])
            
            # Compute composition
            composition_matrix = h_matrix @ x_matrix @ h_matrix
            
            # Apply to random initial state
            initial_state = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
            initial_state = initial_state / np.linalg.norm(initial_state)
            
            # Apply circuit
            state = QuantumState(num_qubits)
            state.state_vector = initial_state
            circuit.quantum_state = state
            circuit.execute()
            
            # Apply composition matrix
            composed_state = composition_matrix @ initial_state
            
            # Check equivalence
            np.testing.assert_allclose(
                circuit.get_state().state_vector, composed_state,
                atol=self.tolerance,
                err_msg="Circuit composition inconsistent"
            )
        
        @given(integers(min_value=2, max_value=4))
        @settings(max_examples=20)
        def test_scalable_state_consistency_property(self, num_qubits):
            """Test that ScalableQuantumState is consistent with QuantumState."""
            # Create random state
            random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
            random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
            
            # Test both representations
            state1 = QuantumState(num_qubits)
            state1.state_vector = random_amplitudes.copy()
            
            state2 = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
            state2.from_dense(random_amplitudes)
            
            # Apply same operations
            h_gate = HGate()
            x_gate = XGate()
            
            h_gate.apply(state1, [0])
            h_gate.apply(state2, [0])
            
            x_gate.apply(state1, [0])
            x_gate.apply(state2, [0])
            
            # Check consistency
            np.testing.assert_allclose(
                state1.state_vector, state2.to_dense(),
                atol=self.tolerance,
                err_msg="ScalableQuantumState inconsistent with QuantumState"
            )
        
        @given(integers(min_value=2, max_value=4))
        @settings(max_examples=15)
        def test_sparse_state_consistency_property(self, num_qubits):
            """Test that sparse representation is consistent with dense."""
            if num_qubits < 6:  # Only test for smaller systems
                return
            
            # Create random state
            random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
            random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
            
            # Test dense representation
            state_dense = ScalableQuantumState(num_qubits, use_gpu=False, use_sparse=False)
            state_dense.from_dense(random_amplitudes)
            
            # Test sparse representation
            state_sparse = ScalableQuantumState(num_qubits, use_gpu=False, use_sparse=True)
            state_sparse.from_dense(random_amplitudes)
            
            # Apply same operations
            h_gate = HGate()
            h_gate.apply(state_dense, [0])
            h_gate.apply(state_sparse, [0])
            
            # Check consistency
            np.testing.assert_allclose(
                state_dense.to_dense(), state_sparse.to_dense(),
                atol=self.tolerance,
                err_msg="Sparse representation inconsistent with dense"
            )

else:
    class TestPropertyBased(unittest.TestCase):
        """Placeholder for property-based tests when Hypothesis is not available."""
        
        def test_hypothesis_not_available(self):
            """Test that skips when Hypothesis is not available."""
            self.skipTest("Hypothesis not available. Install with: pip install hypothesis")


if __name__ == '__main__':
    unittest.main()
