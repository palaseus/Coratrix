"""
Tests for advanced quantum algorithms and analysis tools.

This module tests tomography, fidelity estimation, entanglement monotones,
and other advanced quantum analysis capabilities.
"""

import unittest
import sys
import os
import numpy as np
import tempfile
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.advanced_algorithms import (
    QuantumStateTomography, FidelityEstimator, EntanglementMonotones,
    EntanglementNetwork, AdvancedQuantumAnalysis, TomographyResult
)
from core.qubit import QuantumState
from core.gates import HGate, CNOTGate


class TestQuantumStateTomography(unittest.TestCase):
    """Test cases for quantum state tomography."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 2
        self.tomography = QuantumStateTomography(self.num_qubits)
    
    def test_tomography_initialization(self):
        """Test tomography initialization."""
        self.assertEqual(self.tomography.num_qubits, 2)
        self.assertEqual(self.tomography.dimension, 4)
        self.assertEqual(len(self.tomography.measurement_basis), 16)  # 4^2 for 2 qubits
    
    def test_tomography_bell_state(self):
        """Test tomography on Bell state."""
        # Create Bell state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        # Perform tomography
        result = self.tomography.perform_tomography(state)
        
        # Check results
        self.assertTrue(result.success)
        self.assertGreater(result.fidelity, 0.8)  # Should be high fidelity
        self.assertGreater(result.purity, 0.8)    # Should be high purity
        self.assertEqual(result.measurement_count, 16 * 1000)  # Default shots
    
    def test_tomography_superposition_state(self):
        """Test tomography on superposition state."""
        # Create superposition state (2-qubit)
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(1, 1.0/np.sqrt(2))
        state.normalize()
        
        # Perform tomography
        result = self.tomography.perform_tomography(state)
        
        # Check results
        self.assertTrue(result.success)
        self.assertGreater(result.fidelity, 0.8)
        self.assertGreater(result.purity, 0.8)
    
    def test_tomography_error_handling(self):
        """Test tomography error handling."""
        # Create invalid state (should still work)
        state = QuantumState(2)
        state.set_amplitude(0, 1.0)
        
        # Perform tomography
        result = self.tomography.perform_tomography(state)
        
        # Should succeed even with simple state
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.fidelity, 0.0)
        self.assertGreaterEqual(result.purity, 0.0)


class TestFidelityEstimator(unittest.TestCase):
    """Test cases for fidelity estimation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fidelity_estimator = FidelityEstimator()
    
    def test_state_fidelity_identical_states(self):
        """Test fidelity between identical states."""
        state1 = QuantumState(2)
        state1.set_amplitude(0, 1.0/np.sqrt(2))
        state1.set_amplitude(3, 1.0/np.sqrt(2))
        state1.normalize()
        
        state2 = QuantumState(2)
        state2.set_amplitude(0, 1.0/np.sqrt(2))
        state2.set_amplitude(3, 1.0/np.sqrt(2))
        state2.normalize()
        
        fidelity = self.fidelity_estimator.estimate_state_fidelity(state1, state2)
        
        # Should be very close to 1.0
        self.assertAlmostEqual(fidelity, 1.0, places=10)
    
    def test_state_fidelity_orthogonal_states(self):
        """Test fidelity between orthogonal states."""
        state1 = QuantumState(1)
        state1.set_amplitude(0, 1.0)
        state1.set_amplitude(1, 0.0)  # Clear the initial state

        state2 = QuantumState(1)
        state2.set_amplitude(0, 0.0)  # Clear the initial state
        state2.set_amplitude(1, 1.0)
        
        fidelity = self.fidelity_estimator.estimate_state_fidelity(state1, state2)
        
        # Should be 0.0 for orthogonal states
        self.assertAlmostEqual(fidelity, 0.0, places=10)
    
    def test_gate_fidelity_identical_gates(self):
        """Test fidelity between identical gates."""
        # Create identity gate
        identity = np.eye(2, dtype=complex)
        
        fidelity = self.fidelity_estimator.estimate_gate_fidelity(identity, identity)
        
        # Should be 1.0
        self.assertAlmostEqual(fidelity, 1.0, places=10)
    
    def test_gate_fidelity_different_gates(self):
        """Test fidelity between different gates."""
        # Create X gate
        x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Create identity gate
        identity = np.eye(2, dtype=complex)
        
        fidelity = self.fidelity_estimator.estimate_gate_fidelity(identity, x_gate)
        
        # Should be less than 1.0
        self.assertLess(fidelity, 1.0)
        self.assertGreaterEqual(fidelity, 0.0)


class TestEntanglementMonotones(unittest.TestCase):
    """Test cases for entanglement monotones."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.entanglement_monotones = EntanglementMonotones()
    
    def test_negativity_bell_state(self):
        """Test negativity calculation for Bell state."""
        # Create Bell state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        negativity = self.entanglement_monotones.calculate_negativity(state, (0, 1))
        
        # Bell state should have non-zero negativity
        self.assertGreater(negativity, 0.0)
    
    def test_negativity_separable_state(self):
        """Test negativity calculation for separable state."""
        # Create separable state |00⟩
        state = QuantumState(2)
        state.set_amplitude(0, 1.0)
        
        negativity = self.entanglement_monotones.calculate_negativity(state, (0, 1))
        
        # Separable state should have zero negativity
        self.assertAlmostEqual(negativity, 0.0, places=10)
    
    def test_concurrence_bell_state(self):
        """Test concurrence calculation for Bell state."""
        # Create Bell state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        concurrence = self.entanglement_monotones.calculate_concurrence(state)
        
        # Bell state should have maximum concurrence
        self.assertAlmostEqual(concurrence, 1.0, places=10)
    
    def test_concurrence_separable_state(self):
        """Test concurrence calculation for separable state."""
        # Create separable state |00⟩
        state = QuantumState(2)
        state.set_amplitude(0, 1.0)
        
        concurrence = self.entanglement_monotones.calculate_concurrence(state)
        
        # Separable state should have zero concurrence
        self.assertAlmostEqual(concurrence, 0.0, places=10)
    
    def test_concurrence_invalid_state(self):
        """Test concurrence calculation for invalid state."""
        # Create 3-qubit state (concurrence only defined for 2-qubit states)
        state = QuantumState(3)
        
        with self.assertRaises(ValueError):
            self.entanglement_monotones.calculate_concurrence(state)
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        # Create Bell state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        entropy = self.entanglement_monotones.calculate_entanglement_entropy(state, (0, 1))
        
        # Bell state should have maximum entropy
        self.assertAlmostEqual(entropy, 1.0, places=10)
    
    def test_entanglement_rank(self):
        """Test entanglement rank calculation."""
        # Create Bell state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        rank = self.entanglement_monotones.calculate_entanglement_rank(state)
        
        # Bell state should have rank 2
        self.assertEqual(rank, 2)


class TestEntanglementNetwork(unittest.TestCase):
    """Test cases for entanglement network analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 3
        self.entanglement_network = EntanglementNetwork(self.num_qubits)
    
    def test_network_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.entanglement_network.num_qubits, 3)
        self.assertEqual(self.entanglement_network.entanglement_matrix.shape, (3, 3))
    
    def test_entanglement_graph_calculation(self):
        """Test entanglement graph calculation."""
        # Create GHZ state
        state = QuantumState(3)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(7, 1.0/np.sqrt(2))
        state.normalize()
        
        # Calculate entanglement graph
        graph_data = self.entanglement_network.calculate_entanglement_graph(state)
        
        # Check results
        self.assertIn('entanglement_matrix', graph_data)
        self.assertIn('centrality_metrics', graph_data)
        self.assertIn('num_qubits', graph_data)
        self.assertIn('total_entanglement', graph_data)
        
        # Check matrix dimensions
        matrix = np.array(graph_data['entanglement_matrix'])
        self.assertEqual(matrix.shape, (3, 3))
        
        # Check centrality metrics
        centrality = graph_data['centrality_metrics']
        self.assertIn('degree_centrality', centrality)
        self.assertIn('betweenness_centrality', centrality)
        self.assertIn('closeness_centrality', centrality)
    
    def test_export_to_graphml(self):
        """Test GraphML export functionality."""
        # Create test entanglement data
        entanglement_data = {
            'entanglement_matrix': [[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]],
            'centrality_metrics': {
                'degree_centrality': [0.8, 0.9, 0.7],
                'betweenness_centrality': [0.1, 0.2, 0.1],
                'closeness_centrality': [0.5, 0.6, 0.4]
            },
            'num_qubits': 3,
            'total_entanglement': 1.2
        }
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
            temp_file = f.name
        
        try:
            self.entanglement_network.export_to_graphml(entanglement_data, temp_file)
            
            # Check that file was created
            self.assertTrue(os.path.exists(temp_file))
            
            # Check file content
            with open(temp_file, 'r') as f:
                content = f.read()
                self.assertIn('graphml', content)
                self.assertIn('entanglement_network', content)
                self.assertIn('q0', content)
                self.assertIn('q1', content)
                self.assertIn('q2', content)
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestAdvancedQuantumAnalysis(unittest.TestCase):
    """Test cases for advanced quantum analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 2
        self.analysis = AdvancedQuantumAnalysis(self.num_qubits)
    
    def test_analysis_initialization(self):
        """Test analysis initialization."""
        self.assertEqual(self.analysis.num_qubits, 2)
        self.assertIsNotNone(self.analysis.tomography)
        self.assertIsNotNone(self.analysis.fidelity_estimator)
        self.assertIsNotNone(self.analysis.entanglement_monotones)
        self.assertIsNotNone(self.analysis.entanglement_network)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive quantum state analysis."""
        # Create Bell state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        # Perform comprehensive analysis
        analysis = self.analysis.analyze_quantum_state(state)
        
        # Check that all analysis components are present
        self.assertIn('tomography', analysis)
        self.assertIn('entanglement', analysis)
        self.assertIn('network', analysis)
        
        # Check tomography results
        tomography = analysis['tomography']
        self.assertIn('fidelity', tomography)
        self.assertIn('purity', tomography)
        self.assertIn('success', tomography)
        
        # Check entanglement results
        entanglement = analysis['entanglement']
        self.assertIn('entropy', entanglement)
        self.assertIn('negativity', entanglement)
        self.assertIn('concurrence', entanglement)
        self.assertIn('entanglement_rank', entanglement)
        
        # Check network results
        network = analysis['network']
        self.assertIn('entanglement_matrix', network)
        self.assertIn('centrality_metrics', network)
        self.assertIn('num_qubits', network)
        self.assertIn('total_entanglement', network)
    
    def test_analysis_separable_state(self):
        """Test analysis on separable state."""
        # Create separable state |00⟩
        state = QuantumState(2)
        state.set_amplitude(0, 1.0)
        
        # Perform analysis
        analysis = self.analysis.analyze_quantum_state(state)
        
        # Check that analysis completes successfully
        self.assertIn('tomography', analysis)
        self.assertIn('entanglement', analysis)
        self.assertIn('network', analysis)
        
        # Check entanglement properties
        entanglement = analysis['entanglement']
        self.assertAlmostEqual(entanglement['entropy'], 0.0, places=10)
        self.assertAlmostEqual(entanglement['negativity'], 0.0, places=10)
        self.assertAlmostEqual(entanglement['concurrence'], 0.0, places=10)
    
    def test_analysis_ghz_state(self):
        """Test analysis on GHZ state."""
        # Create GHZ state for 3 qubits
        analysis_3q = AdvancedQuantumAnalysis(3)
        state = QuantumState(3)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(7, 1.0/np.sqrt(2))
        state.normalize()
        
        # Perform analysis
        analysis = analysis_3q.analyze_quantum_state(state)
        
        # Check that analysis completes successfully
        self.assertIn('tomography', analysis)
        self.assertIn('entanglement', analysis)
        self.assertIn('network', analysis)
        
        # Check entanglement properties
        entanglement = analysis['entanglement']
        self.assertGreater(entanglement['entropy'], 0.0)
        self.assertGreater(entanglement['negativity'], 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for advanced algorithms."""
    
    def test_tomography_fidelity_consistency(self):
        """Test consistency between tomography and fidelity estimation."""
        # Create test state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        # Perform tomography
        tomography = QuantumStateTomography(2)
        tomography_result = tomography.perform_tomography(state)
        
        # Estimate fidelity
        fidelity_estimator = FidelityEstimator()
        fidelity = fidelity_estimator.estimate_state_fidelity(state, state)
        
        # Both should give high fidelity
        self.assertGreater(tomography_result.fidelity, 0.8)
        self.assertAlmostEqual(fidelity, 1.0, places=10)
    
    def test_entanglement_measures_consistency(self):
        """Test consistency between different entanglement measures."""
        # Create Bell state
        state = QuantumState(2)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(3, 1.0/np.sqrt(2))
        state.normalize()
        
        # Calculate different entanglement measures
        entanglement_monotones = EntanglementMonotones()
        negativity = entanglement_monotones.calculate_negativity(state, (0, 1))
        concurrence = entanglement_monotones.calculate_concurrence(state)
        entropy = entanglement_monotones.calculate_entanglement_rank(state)
        
        # All measures should indicate entanglement
        self.assertGreater(negativity, 0.0)
        self.assertGreater(concurrence, 0.0)
        self.assertGreater(entropy, 0.0)
    
    def test_network_analysis_completeness(self):
        """Test completeness of network analysis."""
        # Create test state
        state = QuantumState(3)
        state.set_amplitude(0, 1.0/np.sqrt(2))
        state.set_amplitude(7, 1.0/np.sqrt(2))
        state.normalize()
        
        # Perform network analysis
        network = EntanglementNetwork(3)
        graph_data = network.calculate_entanglement_graph(state)
        
        # Check completeness
        self.assertIn('entanglement_matrix', graph_data)
        self.assertIn('centrality_metrics', graph_data)
        self.assertIn('num_qubits', graph_data)
        self.assertIn('total_entanglement', graph_data)
        
        # Check matrix properties
        matrix = np.array(graph_data['entanglement_matrix'])
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.allclose(matrix, matrix.T))  # Should be symmetric


if __name__ == '__main__':
    unittest.main()
