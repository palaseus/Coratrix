"""
Unit tests for quantum state representation.

This module tests the QuantumState class and its methods for
representing and manipulating quantum states.
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qubit import QuantumState


class TestQuantumState(unittest.TestCase):
    """Test cases for QuantumState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_2q = QuantumState(2)  # 2-qubit system
        self.state_3q = QuantumState(3)  # 3-qubit system
    
    def test_initialization(self):
        """Test quantum state initialization."""
        # Test 2-qubit system
        self.assertEqual(self.state_2q.num_qubits, 2)
        self.assertEqual(self.state_2q.dimension, 4)
        
        # Initial state should be |00⟩
        expected_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_2q.state_vector, expected_state)
        
        # Test 3-qubit system
        self.assertEqual(self.state_3q.num_qubits, 3)
        self.assertEqual(self.state_3q.dimension, 8)
        
        # Initial state should be |000⟩
        expected_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(self.state_3q.state_vector, expected_state)
    
    def test_get_amplitude(self):
        """Test getting amplitudes for specific states."""
        # Test |00⟩ state
        self.assertEqual(self.state_2q.get_amplitude(0), 1.0)
        self.assertEqual(self.state_2q.get_amplitude(1), 0.0)
        self.assertEqual(self.state_2q.get_amplitude(2), 0.0)
        self.assertEqual(self.state_2q.get_amplitude(3), 0.0)
    
    def test_set_amplitude(self):
        """Test setting amplitudes for specific states."""
        # Set |01⟩ state
        self.state_2q.set_amplitude(1, 1.0)
        self.assertEqual(self.state_2q.get_amplitude(1), 1.0)
        
        # Set |10⟩ state
        self.state_2q.set_amplitude(2, 1.0)
        self.assertEqual(self.state_2q.get_amplitude(2), 1.0)
        
        # Set |11⟩ state
        self.state_2q.set_amplitude(3, 1.0)
        self.assertEqual(self.state_2q.get_amplitude(3), 1.0)
    
    def test_normalize(self):
        """Test state normalization."""
        # Set unnormalized state
        self.state_2q.set_amplitude(0, 2.0)
        self.state_2q.set_amplitude(1, 2.0)
        
        # Normalize
        self.state_2q.normalize()
        
        # Check normalization
        probabilities = self.state_2q.get_probabilities()
        self.assertAlmostEqual(sum(probabilities), 1.0, places=10)
    
    def test_get_probabilities(self):
        """Test probability calculation."""
        # Initial state |00⟩
        probabilities = self.state_2q.get_probabilities()
        expected = [1.0, 0.0, 0.0, 0.0]
        np.testing.assert_array_almost_equal(probabilities, expected)
        
        # Set |01⟩ state
        self.state_2q.set_amplitude(0, 0.0)
        self.state_2q.set_amplitude(1, 1.0)
        probabilities = self.state_2q.get_probabilities()
        expected = [0.0, 1.0, 0.0, 0.0]
        np.testing.assert_array_almost_equal(probabilities, expected)
    
    def test_get_state_index(self):
        """Test converting qubit states to state vector index."""
        # |00⟩ -> index 0
        self.assertEqual(self.state_2q.get_state_index([0, 0]), 0)
        
        # |01⟩ -> index 1
        self.assertEqual(self.state_2q.get_state_index([0, 1]), 1)
        
        # |10⟩ -> index 2
        self.assertEqual(self.state_2q.get_state_index([1, 0]), 2)
        
        # |11⟩ -> index 3
        self.assertEqual(self.state_2q.get_state_index([1, 1]), 3)
    
    def test_get_qubit_states(self):
        """Test converting state vector index to qubit states."""
        # Index 0 -> |00⟩
        self.assertEqual(self.state_2q.get_qubit_states(0), [0, 0])
        
        # Index 1 -> |01⟩
        self.assertEqual(self.state_2q.get_qubit_states(1), [0, 1])
        
        # Index 2 -> |10⟩
        self.assertEqual(self.state_2q.get_qubit_states(2), [1, 0])
        
        # Index 3 -> |11⟩
        self.assertEqual(self.state_2q.get_qubit_states(3), [1, 1])
    
    def test_bell_state_creation(self):
        """Test creating a Bell state (|00⟩ + |11⟩)/√2."""
        # Set up Bell state
        self.state_2q.set_amplitude(0, 1.0/np.sqrt(2))  # |00⟩
        self.state_2q.set_amplitude(3, 1.0/np.sqrt(2))  # |11⟩
        
        # Check normalization
        probabilities = self.state_2q.get_probabilities()
        self.assertAlmostEqual(sum(probabilities), 1.0, places=10)
        
        # Check that only |00⟩ and |11⟩ have non-zero probability
        self.assertAlmostEqual(probabilities[0], 0.5, places=10)
        self.assertAlmostEqual(probabilities[1], 0.0, places=10)
        self.assertAlmostEqual(probabilities[2], 0.0, places=10)
        self.assertAlmostEqual(probabilities[3], 0.5, places=10)
    
    def test_invalid_qubit_count(self):
        """Test error handling for invalid qubit counts."""
        with self.assertRaises(ValueError):
            QuantumState(0)
        
        with self.assertRaises(ValueError):
            QuantumState(-1)
    
    def test_invalid_state_index(self):
        """Test error handling for invalid state indices."""
        with self.assertRaises(IndexError):
            self.state_2q.get_amplitude(4)
        
        with self.assertRaises(IndexError):
            self.state_2q.set_amplitude(4, 1.0)
        
        with self.assertRaises(IndexError):
            self.state_2q.get_qubit_states(4)
    
    def test_invalid_qubit_states(self):
        """Test error handling for invalid qubit states."""
        with self.assertRaises(ValueError):
            self.state_2q.get_state_index([0, 2])  # Invalid qubit state
        
        with self.assertRaises(ValueError):
            self.state_2q.get_state_index([0])  # Wrong number of qubits


if __name__ == '__main__':
    unittest.main()
