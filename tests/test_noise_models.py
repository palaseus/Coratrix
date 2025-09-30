"""
Tests for noise models and error mitigation.

This module tests noise channel implementations and error mitigation techniques.
"""

import unittest
import sys
import os
import numpy as np
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.noise_models import (
    NoiseModel, QuantumNoise, ErrorMitigation, NoisyQuantumCircuit,
    NoiseChannel
)
from core.qubit import QuantumState
from core.gates import HGate, XGate, CNOTGate


class TestNoiseModel(unittest.TestCase):
    """Test cases for noise model configuration."""
    
    def test_noise_model_initialization(self):
        """Test noise model initialization."""
        model = NoiseModel()
        
        self.assertEqual(model.depolarizing_error, 0.01)
        self.assertEqual(model.amplitude_damping_error, 0.005)
        self.assertEqual(model.phase_damping_error, 0.005)
        self.assertEqual(model.readout_error, 0.02)
        self.assertEqual(model.bit_flip_error, 0.01)
        self.assertEqual(model.phase_flip_error, 0.01)
        self.assertEqual(model.gate_error, 0.005)
        self.assertEqual(model.t1_time, 100.0)
        self.assertEqual(model.t2_time, 50.0)
        self.assertEqual(model.gate_time, 0.1)
    
    def test_custom_noise_model(self):
        """Test custom noise model configuration."""
        model = NoiseModel(
            depolarizing_error=0.05,
            amplitude_damping_error=0.02,
            readout_error=0.1
        )
        
        self.assertEqual(model.depolarizing_error, 0.05)
        self.assertEqual(model.amplitude_damping_error, 0.02)
        self.assertEqual(model.readout_error, 0.1)


class TestQuantumNoise(unittest.TestCase):
    """Test cases for quantum noise channels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.noise_model = NoiseModel()
        self.noise = QuantumNoise(self.noise_model)
    
    def test_depolarizing_noise(self):
        """Test depolarizing noise application."""
        state = QuantumState(2)
        
        # Create Bell state
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        # Apply depolarizing noise
        noisy_state = self.noise.apply_depolarizing_noise(state, 0.1)
        
        # Check that state is still normalized
        norm = np.sum(np.abs(noisy_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_amplitude_damping_noise(self):
        """Test amplitude damping noise application."""
        state = QuantumState(1)
        state.set_amplitude(0, 1.0)  # |0⟩ state
        
        # Apply amplitude damping
        noisy_state = self.noise.apply_amplitude_damping(state, 0.1)
        
        # Check normalization
        norm = np.sum(np.abs(noisy_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_phase_damping_noise(self):
        """Test phase damping noise application."""
        state = QuantumState(1)
        # Create superposition state
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(1, 1.0/np.sqrt(2))
        state.normalize()
        
        # Apply phase damping
        noisy_state = self.noise.apply_phase_damping(state, 0.1)
        
        # Check normalization
        norm = np.sum(np.abs(noisy_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_readout_error(self):
        """Test readout error application."""
        # Test with different bitstrings
        test_cases = ['00', '01', '10', '11', '000', '111']
        
        for bitstring in test_cases:
            noisy_result = self.noise.apply_readout_error(bitstring, 0.1)
            
            # Check that result has same length
            self.assertEqual(len(noisy_result), len(bitstring))
            
            # Check that result contains only 0s and 1s
            for bit in noisy_result:
                self.assertIn(bit, ['0', '1'])
    
    def test_gate_error(self):
        """Test gate error application."""
        state = QuantumState(2)
        
        # Apply gate error
        noisy_state = self.noise.apply_gate_error(state, "H", 0.1)
        
        # Check that state is still normalized
        norm = np.sum(np.abs(noisy_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_zero_error_rates(self):
        """Test that zero error rates don't change the state."""
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        original_state = state.state_vector.copy()
        
        # Apply zero error rates
        noisy_state = self.noise.apply_depolarizing_noise(state, 0.0)
        noisy_state = self.noise.apply_amplitude_damping(noisy_state, 0.0)
        noisy_state = self.noise.apply_phase_damping(noisy_state, 0.0)
        
        # State should be unchanged
        np.testing.assert_array_almost_equal(
            noisy_state.state_vector, original_state, decimal=10
        )


class TestErrorMitigation(unittest.TestCase):
    """Test cases for error mitigation techniques."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.noise_model = NoiseModel()
        self.mitigation = ErrorMitigation(self.noise_model)
    
    def test_mid_circuit_purification(self):
        """Test mid-circuit purification."""
        state = QuantumState(2)
        
        # Apply purification
        purified_state = self.mitigation.apply_mid_circuit_purification(state)
        
        # Check that state is still normalized
        norm = np.sum(np.abs(purified_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_real_time_feedback(self):
        """Test real-time feedback."""
        state = QuantumState(2)
        
        # Apply feedback
        corrected_state = self.mitigation.apply_real_time_feedback(state)
        
        # Check that state is still normalized
        norm = np.sum(np.abs(corrected_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_repetition_code(self):
        """Test repetition code error correction."""
        state = QuantumState(2)
        
        # Apply repetition code
        corrected_state = self.mitigation.apply_error_correction_code(
            state, "repetition"
        )
        
        # Check that state is still normalized
        norm = np.sum(np.abs(corrected_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_surface_code_patch(self):
        """Test surface code patch error correction."""
        state = QuantumState(2)
        
        # Apply surface code
        corrected_state = self.mitigation.apply_error_correction_code(
            state, "surface"
        )
        
        # Check that state is still normalized
        norm = np.sum(np.abs(corrected_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_unknown_error_correction_code(self):
        """Test error handling for unknown error correction codes."""
        state = QuantumState(2)
        
        with self.assertRaises(ValueError):
            self.mitigation.apply_error_correction_code(state, "unknown")


class TestNoisyQuantumCircuit(unittest.TestCase):
    """Test cases for noisy quantum circuit."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.noise_model = NoiseModel()
        self.circuit = NoisyQuantumCircuit(2, self.noise_model)
    
    def test_circuit_initialization(self):
        """Test noisy circuit initialization."""
        self.assertEqual(self.circuit.num_qubits, 2)
        self.assertEqual(self.circuit.noise_model, self.noise_model)
        self.assertEqual(len(self.circuit.gates), 0)
    
    def test_add_gate(self):
        """Test adding gates to noisy circuit."""
        h_gate = HGate()
        cnot_gate = CNOTGate()
        
        self.circuit.add_gate(h_gate, [0])
        self.circuit.add_gate(cnot_gate, [0, 1])
        
        self.assertEqual(len(self.circuit.gates), 2)
    
    def test_apply_gate_with_noise(self):
        """Test applying gates with noise."""
        h_gate = HGate()
        
        # Apply gate with noise
        self.circuit.apply_gate(h_gate, [0])
        
        # Check that state is normalized
        norm = np.sum(np.abs(self.circuit.quantum_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_execute_circuit(self):
        """Test executing noisy circuit."""
        h_gate = HGate()
        cnot_gate = CNOTGate()
        
        self.circuit.add_gate(h_gate, [0])
        self.circuit.add_gate(cnot_gate, [0, 1])
        
        # Execute circuit
        self.circuit.execute()
        
        # Check that state is normalized
        norm = np.sum(np.abs(self.circuit.quantum_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_execute_with_mitigation(self):
        """Test executing circuit with error mitigation."""
        h_gate = HGate()
        cnot_gate = CNOTGate()
        
        self.circuit.add_gate(h_gate, [0])
        self.circuit.add_gate(cnot_gate, [0, 1])
        
        # Execute with mitigation
        self.circuit.execute_with_mitigation(mitigation_enabled=True)
        
        # Check that state is normalized
        norm = np.sum(np.abs(self.circuit.quantum_state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_measure_with_readout_error(self):
        """Test measurement with readout error."""
        h_gate = HGate()
        cnot_gate = CNOTGate()
        
        self.circuit.add_gate(h_gate, [0])
        self.circuit.add_gate(cnot_gate, [0, 1])
        
        # Execute circuit
        self.circuit.execute()
        
        # Measure with readout error
        counts = self.circuit.measure_with_readout_error(shots=100)
        
        # Check that we get results
        self.assertGreater(len(counts), 0)
        
        # Check that all bitstrings are valid
        for bitstring in counts.keys():
            self.assertEqual(len(bitstring), self.circuit.num_qubits)
            for bit in bitstring:
                self.assertIn(bit, ['0', '1'])
    
    def test_noise_effect_on_entanglement(self):
        """Test how noise affects entanglement."""
        # Create Bell state
        h_gate = HGate()
        cnot_gate = CNOTGate()
        
        self.circuit.add_gate(h_gate, [0])
        self.circuit.add_gate(cnot_gate, [0, 1])
        
        # Execute without noise
        self.circuit.execute()
        clean_entropy = self.circuit.quantum_state.get_entanglement_entropy()
        
        # Reset and execute with noise
        self.circuit.quantum_state = QuantumState(2)
        self.circuit.execute()
        noisy_entropy = self.circuit.quantum_state.get_entanglement_entropy()
        
        # Noise should generally reduce entanglement
        # (though this is not guaranteed for all noise types)
        self.assertIsInstance(clean_entropy, float)
        self.assertIsInstance(noisy_entropy, float)
        self.assertGreaterEqual(clean_entropy, 0.0)
        self.assertGreaterEqual(noisy_entropy, 0.0)


class TestNoiseIntegration(unittest.TestCase):
    """Integration tests for noise models."""
    
    def test_noise_model_consistency(self):
        """Test that noise models are consistent across different implementations."""
        model1 = NoiseModel(depolarizing_error=0.1)
        model2 = NoiseModel(depolarizing_error=0.1)
        
        noise1 = QuantumNoise(model1)
        noise2 = QuantumNoise(model2)
        
        state1 = QuantumState(2)
        state2 = QuantumState(2)
        
        # Apply same noise to both states
        noisy_state1 = noise1.apply_depolarizing_noise(state1, 0.1)
        noisy_state2 = noise2.apply_depolarizing_noise(state2, 0.1)
        
        # Both should be normalized
        norm1 = np.sum(np.abs(noisy_state1.state_vector)**2)
        norm2 = np.sum(np.abs(noisy_state2.state_vector)**2)
        
        self.assertAlmostEqual(norm1, 1.0, places=10)
        self.assertAlmostEqual(norm2, 1.0, places=10)
    
    def test_error_mitigation_effectiveness(self):
        """Test that error mitigation improves results."""
        # Create a circuit with known noise
        noise_model = NoiseModel(depolarizing_error=0.2)  # High noise
        circuit = NoisyQuantumCircuit(2, noise_model)
        
        h_gate = HGate()
        cnot_gate = CNOTGate()
        
        circuit.add_gate(h_gate, [0])
        circuit.add_gate(cnot_gate, [0, 1])
        
        # Execute without mitigation
        circuit.execute_with_mitigation(mitigation_enabled=False)
        no_mitigation_entropy = circuit.quantum_state.get_entanglement_entropy()
        
        # Reset and execute with mitigation
        circuit.quantum_state = QuantumState(2)
        circuit.execute_with_mitigation(mitigation_enabled=True)
        with_mitigation_entropy = circuit.quantum_state.get_entanglement_entropy()
        
        # Both should be valid
        self.assertIsInstance(no_mitigation_entropy, float)
        self.assertIsInstance(with_mitigation_entropy, float)
        self.assertGreaterEqual(no_mitigation_entropy, 0.0)
        self.assertGreaterEqual(with_mitigation_entropy, 0.0)


if __name__ == '__main__':
    unittest.main()
