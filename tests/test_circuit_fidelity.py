"""
Randomized circuit fidelity tests for quantum operations.

This module tests the fidelity of quantum operations against high-precision
reference implementations to ensure numerical accuracy.
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
from core.entanglement_analysis import EntanglementAnalyzer


class TestCircuitFidelity(unittest.TestCase):
    """Test cases for circuit fidelity and numerical accuracy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-12  # High precision tolerance
        self.test_qubit_counts = [2, 3, 4, 5]
    
    def _create_reference_state(self, num_qubits: int) -> np.ndarray:
        """Create a reference state using high-precision numpy operations."""
        # Use double precision for reference
        state = np.zeros(2**num_qubits, dtype=np.complex128)
        state[0] = 1.0
        return state
    
    def _apply_reference_gate(self, state: np.ndarray, gate_matrix: np.ndarray) -> np.ndarray:
        """Apply gate using high-precision matrix multiplication."""
        return gate_matrix.astype(np.complex128) @ state.astype(np.complex128)
    
    def _calculate_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate fidelity between two quantum states."""
        # Ensure states are normalized
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)
        
        # Fidelity = |⟨ψ₁|ψ₂⟩|²
        overlap = np.vdot(state1, state2)
        fidelity = np.abs(overlap)**2
        return float(fidelity)
    
    def test_single_gate_fidelity(self):
        """Test fidelity of single gate operations."""
        gates = [XGate(), YGate(), ZGate(), HGate()]
        
        for num_qubits in self.test_qubit_counts:
            for gate in gates:
                with self.subTest(gate=gate.name, qubits=num_qubits):
                    # Create reference state
                    ref_state = self._create_reference_state(num_qubits)
                    
                    # Apply gate using reference implementation
                    gate_matrix = gate.get_matrix(num_qubits, [0]).astype(np.complex128)
                    ref_result = self._apply_reference_gate(ref_state, gate_matrix)
                    
                    # Apply gate using Coratrix implementation
                    coratrix_state = QuantumState(num_qubits)
                    gate.apply(coratrix_state, [0])
                    
                    # Calculate fidelity
                    fidelity = self._calculate_fidelity(ref_result, coratrix_state.state_vector)
                    
                    # Check high fidelity
                    self.assertGreater(fidelity, 1.0 - self.tolerance,
                                     f"{gate.name} gate fidelity too low: {fidelity}")
    
    def test_cnot_gate_fidelity(self):
        """Test fidelity of CNOT gate operations."""
        for num_qubits in [2, 3, 4]:
            with self.subTest(qubits=num_qubits):
                # Create reference state
                ref_state = self._create_reference_state(num_qubits)
                
                # Apply CNOT using reference implementation
                cnot_matrix = CNOTGate().get_matrix(num_qubits, [0, 1]).astype(np.complex128)
                ref_result = self._apply_reference_gate(ref_state, cnot_matrix)
                
                # Apply CNOT using Coratrix implementation
                coratrix_state = QuantumState(num_qubits)
                CNOTGate().apply(coratrix_state, [0, 1])
                
                # Calculate fidelity
                fidelity = self._calculate_fidelity(ref_result, coratrix_state.state_vector)
                
                # Check high fidelity
                self.assertGreater(fidelity, 1.0 - self.tolerance,
                                 f"CNOT gate fidelity too low: {fidelity}")
    
    def test_circuit_fidelity(self):
        """Test fidelity of multi-gate circuits."""
        for num_qubits in self.test_qubit_counts:
            with self.subTest(qubits=num_qubits):
                # Create reference state
                ref_state = self._create_reference_state(num_qubits)
                
                # Apply gates using reference implementation
                h_matrix = HGate().get_matrix(num_qubits, [0]).astype(np.complex128)
                x_matrix = XGate().get_matrix(num_qubits, [0]).astype(np.complex128)
                
                ref_result = self._apply_reference_gate(ref_state, h_matrix)
                ref_result = self._apply_reference_gate(ref_result, x_matrix)
                ref_result = self._apply_reference_gate(ref_result, h_matrix)
                
                # Apply gates using Coratrix implementation
                coratrix_state = QuantumState(num_qubits)
                h_gate = HGate()
                x_gate = XGate()
                
                h_gate.apply(coratrix_state, [0])
                x_gate.apply(coratrix_state, [0])
                h_gate.apply(coratrix_state, [0])
                
                # Calculate fidelity
                fidelity = self._calculate_fidelity(ref_result, coratrix_state.state_vector)
                
                # Check high fidelity
                self.assertGreater(fidelity, 1.0 - self.tolerance,
                                 f"Circuit fidelity too low: {fidelity}")
    
    def test_scalable_state_fidelity(self):
        """Test fidelity of ScalableQuantumState operations."""
        for num_qubits in [2, 3, 4]:
            with self.subTest(qubits=num_qubits):
                # Create reference state
                ref_state = self._create_reference_state(num_qubits)
                
                # Apply gate using reference implementation
                h_matrix = HGate().get_matrix(num_qubits, [0]).astype(np.complex128)
                ref_result = self._apply_reference_gate(ref_state, h_matrix)
                
                # Apply gate using ScalableQuantumState
                scalable_state = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
                scalable_state.apply_gate(HGate(), [0])
                
                # Calculate fidelity
                fidelity = self._calculate_fidelity(ref_result, scalable_state.to_dense())
                
                # Check high fidelity
                self.assertGreater(fidelity, 1.0 - self.tolerance,
                                 f"ScalableQuantumState fidelity too low: {fidelity}")
    
    def test_sparse_state_fidelity(self):
        """Test fidelity of sparse state operations."""
        for num_qubits in [6, 7, 8]:  # Test sparse representation
            with self.subTest(qubits=num_qubits):
                # Create reference state
                ref_state = self._create_reference_state(num_qubits)
                
                # Apply gate using reference implementation
                h_matrix = HGate().get_matrix(num_qubits, [0]).astype(np.complex128)
                ref_result = self._apply_reference_gate(ref_state, h_matrix)
                
                # Apply gate using sparse ScalableQuantumState
                sparse_state = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=6)
                sparse_state.from_dense(ref_state)
                sparse_state.apply_gate(HGate(), [0])
                
                # Calculate fidelity
                fidelity = self._calculate_fidelity(ref_result, sparse_state.to_dense())
                
                # Check high fidelity
                self.assertGreater(fidelity, 1.0 - self.tolerance,
                                 f"Sparse state fidelity too low: {fidelity}")
    
    def test_random_circuit_fidelity(self):
        """Test fidelity of random quantum circuits."""
        np.random.seed(42)  # For reproducibility
        
        for num_qubits in [2, 3, 4]:
            with self.subTest(qubits=num_qubits):
                # Create random circuit
                circuit = QuantumCircuit(num_qubits)
                gates = [XGate(), YGate(), ZGate(), HGate()]
                
                # Add random gates
                for _ in range(10):
                    gate = np.random.choice(gates)
                    target_qubit = np.random.randint(0, num_qubits)
                    circuit.add_gate(gate, [target_qubit])
                
                # Create reference state
                ref_state = self._create_reference_state(num_qubits)
                
                # Apply circuit using reference implementation
                ref_result = ref_state.copy()
                for gate, target_qubits in circuit.gates:
                    gate_matrix = gate.get_matrix(num_qubits, target_qubits).astype(np.complex128)
                    ref_result = self._apply_reference_gate(ref_result, gate_matrix)
                
                # Apply circuit using Coratrix implementation
                circuit.execute()
                
                # Calculate fidelity
                fidelity = self._calculate_fidelity(ref_result, circuit.get_state().state_vector)
                
                # Check high fidelity
                self.assertGreater(fidelity, 1.0 - self.tolerance,
                                 f"Random circuit fidelity too low: {fidelity}")
    
    def test_entanglement_fidelity(self):
        """Test fidelity of entanglement operations."""
        for num_qubits in [2, 3, 4]:
            with self.subTest(qubits=num_qubits):
                if num_qubits < 2:
                    continue
                
                # Create Bell state using reference implementation
                ref_state = np.zeros(2**num_qubits, dtype=np.complex128)
                ref_state[0] = 1.0/np.sqrt(2)
                ref_state[2**num_qubits - 1] = 1.0/np.sqrt(2)
                
                # Create Bell state using Coratrix implementation
                coratrix_state = QuantumState(num_qubits)
                coratrix_state.set_amplitude(0, 1.0/np.sqrt(2))
                coratrix_state.set_amplitude(2**num_qubits - 1, 1.0/np.sqrt(2))
                coratrix_state.normalize()
                
                # Calculate fidelity
                fidelity = self._calculate_fidelity(ref_state, coratrix_state.state_vector)
                
                # Check high fidelity
                self.assertGreater(fidelity, 1.0 - self.tolerance,
                                 f"Bell state fidelity too low: {fidelity}")
                
                # Check entanglement properties
                analyzer = EntanglementAnalyzer()
                analysis = analyzer.analyze_entanglement(coratrix_state)
                entropy = analysis['entanglement_entropy']
                self.assertGreater(entropy, 0.9, "Bell state should be highly entangled")
    
    def test_measurement_fidelity(self):
        """Test fidelity of measurement operations."""
        for num_qubits in [2, 3, 4]:
            with self.subTest(qubits=num_qubits):
                # Create superposition state
                state = QuantumState(num_qubits)
                h_gate = HGate()
                h_gate.apply(state, [0])
                
                # Get probabilities
                probabilities = state.get_probabilities()
                
                # Check probability sum
                self.assertAlmostEqual(np.sum(probabilities), 1.0, places=12)
                
                # Check specific probabilities for H|0⟩
                expected_probs = np.zeros(2**num_qubits)
                expected_probs[0] = 0.5  # |00...0⟩
                expected_probs[2**(num_qubits-1)] = 0.5  # |10...0⟩
                
                # Check probabilities match expected values
                for i, (actual, expected) in enumerate(zip(probabilities, expected_probs)):
                    if expected > 0:
                        self.assertAlmostEqual(actual, expected, places=10,
                                             msg=f"Probability mismatch at index {i}")
                    else:
                        self.assertAlmostEqual(actual, 0.0, places=10,
                                             msg=f"Non-zero probability at index {i}")


if __name__ == '__main__':
    unittest.main()
