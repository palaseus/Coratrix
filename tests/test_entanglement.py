"""
Unit tests for quantum entanglement and Bell states.

This module tests the creation and detection of entangled states,
particularly Bell states, and demonstrates 2-qubit entanglement.
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qubit import QuantumState
from core.gates import HGate, CNOTGate, XGate, ZGate
from core.circuit import QuantumCircuit
from core.measurement import Measurement


class TestEntanglement(unittest.TestCase):
    """Test cases for quantum entanglement and Bell states."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.circuit = QuantumCircuit(2)
        self.h_gate = HGate()
        self.cnot_gate = CNOTGate()
        self.x_gate = XGate()
        self.z_gate = ZGate()
    
    def test_bell_state_phi_plus(self):
        """Test creation of |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 Bell state."""
        # Apply H to qubit 0, then CNOT
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        
        # Check the resulting state
        state_vector = self.circuit.get_state_vector()
        expected_state = np.array([1.0/np.sqrt(2), 0.0, 0.0, 1.0/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(state_vector, expected_state)
        
        # Check probabilities
        probabilities = self.circuit.get_probabilities()
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        np.testing.assert_array_almost_equal(probabilities, expected_probs)
    
    def test_bell_state_phi_minus(self):
        """Test creation of |Φ⁻⟩ = (|00⟩ - |11⟩)/√2 Bell state."""
        # Apply H to qubit 0, then CNOT, then Z to qubit 0
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        self.circuit.apply_gate(self.z_gate, [0])
        
        # Check the resulting state
        state_vector = self.circuit.get_state_vector()
        expected_state = np.array([1.0/np.sqrt(2), 0.0, 0.0, -1.0/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(state_vector, expected_state)
        
        # Check probabilities
        probabilities = self.circuit.get_probabilities()
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        np.testing.assert_array_almost_equal(probabilities, expected_probs)
    
    def test_bell_state_psi_plus(self):
        """Test creation of |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 Bell state."""
        # Apply H to qubit 0, then CNOT, then X to qubit 1
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        self.circuit.apply_gate(self.x_gate, [1])
        
        # Check the resulting state
        state_vector = self.circuit.get_state_vector()
        expected_state = np.array([0.0, 1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(state_vector, expected_state)
        
        # Check probabilities
        probabilities = self.circuit.get_probabilities()
        expected_probs = [0.0, 0.5, 0.5, 0.0]
        np.testing.assert_array_almost_equal(probabilities, expected_probs)
    
    def test_bell_state_psi_minus(self):
        """Test creation of |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 Bell state."""
        # Apply H to qubit 0, then CNOT, then X to qubit 1, then Z to qubit 0
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        self.circuit.apply_gate(self.x_gate, [1])
        self.circuit.apply_gate(self.z_gate, [0])
        
        # Check the resulting state
        state_vector = self.circuit.get_state_vector()
        expected_state = np.array([0.0, 1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(state_vector, expected_state)
        
        # Check probabilities
        probabilities = self.circuit.get_probabilities()
        expected_probs = [0.0, 0.5, 0.5, 0.0]
        np.testing.assert_array_almost_equal(probabilities, expected_probs)
    
    def test_entanglement_measurement_correlation(self):
        """Test that entangled states show measurement correlations."""
        # Create Bell state |Φ⁺⟩
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        
        # Measure multiple times to check correlations
        measurement_results = []
        for _ in range(100):
            # Reset to Bell state
            self.circuit.reset()
            self.circuit.apply_gate(self.h_gate, [0])
            self.circuit.apply_gate(self.cnot_gate, [0, 1])
            
            # Measure both qubits
            measurement = Measurement(self.circuit.get_state())
            results = measurement.measure_all()
            measurement_results.append(results)
        
        # Check that measurements are correlated (both 0 or both 1)
        for result in measurement_results:
            self.assertEqual(result[0], result[1], "Entangled qubits should have correlated measurements")
    
    def test_entanglement_versus_separable(self):
        """Test that entangled states are different from separable states."""
        # Create entangled state (Bell state)
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        entangled_state = self.circuit.get_state_vector()
        
        # Create separable state (|0⟩ ⊗ |+⟩)
        self.circuit.reset()
        self.circuit.apply_gate(self.h_gate, [1])
        separable_state = self.circuit.get_state_vector()
        
        # States should be different
        self.assertFalse(np.allclose(entangled_state, separable_state))
        
        # Entangled state should have non-zero amplitudes for |00⟩ and |11⟩
        self.assertNotAlmostEqual(abs(entangled_state[0]), 0.0)
        self.assertNotAlmostEqual(abs(entangled_state[3]), 0.0)
        
        # Separable state should have non-zero amplitudes for |00⟩ and |01⟩
        self.assertNotAlmostEqual(abs(separable_state[0]), 0.0)
        self.assertNotAlmostEqual(abs(separable_state[1]), 0.0)
    
    def test_bell_state_measurement_statistics(self):
        """Test that Bell state measurements follow expected statistics."""
        # Create Bell state |Φ⁺⟩
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        
        # Measure many times
        measurement_results = []
        for _ in range(1000):
            # Reset to Bell state
            self.circuit.reset()
            self.circuit.apply_gate(self.h_gate, [0])
            self.circuit.apply_gate(self.cnot_gate, [0, 1])
            
            # Measure
            measurement = Measurement(self.circuit.get_state())
            results = measurement.measure_all()
            measurement_results.append(results)
        
        # Count outcomes
        outcome_counts = {}
        for result in measurement_results:
            outcome = tuple(result)
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        # Should only see |00⟩ and |11⟩ outcomes
        self.assertIn((0, 0), outcome_counts)
        self.assertIn((1, 1), outcome_counts)
        self.assertNotIn((0, 1), outcome_counts)
        self.assertNotIn((1, 0), outcome_counts)
        
        # Both outcomes should be roughly equally likely
        count_00 = outcome_counts.get((0, 0), 0)
        count_11 = outcome_counts.get((1, 1), 0)
        total = count_00 + count_11
        
        # Allow for statistical variation (within 10%)
        self.assertGreater(count_00 / total, 0.4)
        self.assertLess(count_00 / total, 0.6)
        self.assertGreater(count_11 / total, 0.4)
        self.assertLess(count_11 / total, 0.6)
    
    def test_entanglement_preservation_under_gates(self):
        """Test that entanglement is preserved under certain gate operations."""
        # Create Bell state
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        
        # Apply Z gate to qubit 0 (should preserve entanglement)
        self.circuit.apply_gate(self.z_gate, [0])
        
        # State should still be entangled (now |Φ⁻⟩)
        state_vector = self.circuit.get_state_vector()
        expected_state = np.array([1.0/np.sqrt(2), 0.0, 0.0, -1.0/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(state_vector, expected_state)
        
        # Check that it's still a Bell state (maximally entangled)
        probabilities = self.circuit.get_probabilities()
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        np.testing.assert_array_almost_equal(probabilities, expected_probs)
    
    def test_entanglement_breaking_measurement(self):
        """Test that measuring one qubit breaks entanglement."""
        # Create Bell state
        self.circuit.apply_gate(self.h_gate, [0])
        self.circuit.apply_gate(self.cnot_gate, [0, 1])
        
        # Measure qubit 0
        measurement = Measurement(self.circuit.get_state())
        result = measurement.measure_qubit(0)
        
        # After measurement, the state should be separable
        # If qubit 0 was measured as 0, qubit 1 should be in |0⟩
        # If qubit 0 was measured as 1, qubit 1 should be in |1⟩
        state_vector = self.circuit.get_state_vector()
        
        if result == 0:
            # State should be |00⟩
            expected_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        else:
            # State should be |11⟩
            expected_state = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
        
        np.testing.assert_array_almost_equal(state_vector, expected_state)


if __name__ == '__main__':
    unittest.main()
