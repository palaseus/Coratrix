"""
Unit tests for quantum gates.

This module tests the implementation of quantum gates including
X, Y, Z, H, and CNOT gates.
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qubit import QuantumState
from core.gates import XGate, YGate, ZGate, HGate, CNOTGate


class TestQuantumGates(unittest.TestCase):
    """Test cases for quantum gates."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_2q = QuantumState(2)
        self.x_gate = XGate()
        self.y_gate = YGate()
        self.z_gate = ZGate()
        self.h_gate = HGate()
        self.cnot_gate = CNOTGate()
    
    def test_x_gate_single_qubit(self):
        """Test X gate on single qubit."""
        # X|0⟩ = |1⟩ (in 2-qubit system: |00⟩ -> |10⟩)
        self.x_gate.apply(self.state_2q, [0])
        expected_state = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # Reset and test X|1⟩ = |0⟩ (in 2-qubit system: |10⟩ -> |00⟩)
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(2, 1.0)
        self.x_gate.apply(self.state_2q, [0])
        expected_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
    
    def test_y_gate_single_qubit(self):
        """Test Y gate on single qubit."""
        # Y|0⟩ = i|1⟩ (in 2-qubit system: |00⟩ -> i|10⟩)
        self.y_gate.apply(self.state_2q, [0])
        expected_state = np.array([0.0, 0.0, 1j, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # Reset and test Y|1⟩ = -i|0⟩ (in 2-qubit system: |10⟩ -> -i|00⟩)
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(2, 1.0)
        self.y_gate.apply(self.state_2q, [0])
        expected_state = np.array([-1j, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
    
    def test_z_gate_single_qubit(self):
        """Test Z gate on single qubit."""
        # Z|0⟩ = |0⟩ (in 2-qubit system: |00⟩ -> |00⟩)
        self.z_gate.apply(self.state_2q, [0])
        expected_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # Reset and test Z|1⟩ = -|1⟩ (in 2-qubit system: |10⟩ -> -|10⟩)
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(2, 1.0)
        self.z_gate.apply(self.state_2q, [0])
        expected_state = np.array([0.0, 0.0, -1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
    
    def test_h_gate_single_qubit(self):
        """Test H gate on single qubit."""
        # H|0⟩ = (|0⟩ + |1⟩)/√2 (in 2-qubit system: |00⟩ -> (|00⟩ + |10⟩)/√2)
        self.h_gate.apply(self.state_2q, [0])
        expected_state = np.array([1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2), 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # Reset and test H|1⟩ = (|0⟩ - |1⟩)/√2 (in 2-qubit system: |10⟩ -> (|00⟩ - |10⟩)/√2)
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(2, 1.0)
        self.h_gate.apply(self.state_2q, [0])
        expected_state = np.array([1.0/np.sqrt(2), 0.0, -1.0/np.sqrt(2), 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
    
    def test_cnot_gate(self):
        """Test CNOT gate on 2-qubit system."""
        # CNOT|00⟩ = |00⟩
        self.cnot_gate.apply(self.state_2q, [0, 1])
        expected_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # CNOT|01⟩ = |01⟩
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(1, 1.0)
        self.cnot_gate.apply(self.state_2q, [0, 1])
        expected_state = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # CNOT|10⟩ = |11⟩
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(1, 0.0)
        self.state_2q.set_amplitude(2, 1.0)
        self.cnot_gate.apply(self.state_2q, [0, 1])
        expected_state = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # CNOT|11⟩ = |10⟩
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(1, 0.0)
        self.state_2q.set_amplitude(2, 0.0)
        self.state_2q.set_amplitude(3, 1.0)
        self.cnot_gate.apply(self.state_2q, [0, 1])
        expected_state = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
    
    def test_bell_state_creation(self):
        """Test creating Bell state using H and CNOT gates."""
        # Start with |00⟩
        # Apply H to qubit 0: (|00⟩ + |10⟩)/√2
        self.h_gate.apply(self.state_2q, [0])
        
        # Apply CNOT: (|00⟩ + |11⟩)/√2 (Bell state)
        self.cnot_gate.apply(self.state_2q, [0, 1])
        
        # Check Bell state
        expected_state = np.array([1.0/np.sqrt(2), 0.0, 0.0, 1.0/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # Check probabilities
        probabilities = self.state_2q.get_probabilities()
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        np.testing.assert_array_almost_equal(probabilities, expected_probs)
    
    def test_gate_matrix_properties(self):
        """Test that gate matrices are unitary."""
        # Test X gate matrix
        x_matrix = self.x_gate.get_matrix(1, [0])
        self.assertTrue(np.allclose(x_matrix @ x_matrix.conj().T, np.eye(2)))
        
        # Test Y gate matrix
        y_matrix = self.y_gate.get_matrix(1, [0])
        self.assertTrue(np.allclose(y_matrix @ y_matrix.conj().T, np.eye(2)))
        
        # Test Z gate matrix
        z_matrix = self.z_gate.get_matrix(1, [0])
        self.assertTrue(np.allclose(z_matrix @ z_matrix.conj().T, np.eye(2)))
        
        # Test H gate matrix
        h_matrix = self.h_gate.get_matrix(1, [0])
        self.assertTrue(np.allclose(h_matrix @ h_matrix.conj().T, np.eye(2)))
        
        # Test CNOT gate matrix
        cnot_matrix = self.cnot_gate.get_matrix(2, [0, 1])
        self.assertTrue(np.allclose(cnot_matrix @ cnot_matrix.conj().T, np.eye(4)))
    
    def test_invalid_gate_parameters(self):
        """Test error handling for invalid gate parameters."""
        # Test X gate with wrong number of qubits
        with self.assertRaises(ValueError):
            self.x_gate.get_matrix(2, [0, 1])
        
        # Test CNOT gate with wrong number of qubits
        with self.assertRaises(ValueError):
            self.cnot_gate.get_matrix(2, [0])
        
        # Test CNOT gate with wrong number of qubits
        with self.assertRaises(ValueError):
            self.cnot_gate.get_matrix(2, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
