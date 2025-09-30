"""
Tests for multi-subspace Grover search implementation.

This module tests the multi-subspace Grover search with interference diagnostics.
"""

import unittest
import sys
import os
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.multi_subspace_grover import (
    MultiSubspaceGrover, SubspaceConfig, SubspaceType, GroverResult,
    InterferenceDiagnostics
)


class TestSubspaceConfig(unittest.TestCase):
    """Test cases for subspace configuration."""
    
    def test_subspace_config_initialization(self):
        """Test subspace configuration initialization."""
        config = SubspaceConfig(
            subspace_type=SubspaceType.GHZ,
            qubit_indices=[0, 1, 2],
            target_state="000",
            entanglement_threshold=0.8,
            interference_weight=1.0
        )
        
        self.assertEqual(config.subspace_type, SubspaceType.GHZ)
        self.assertEqual(config.qubit_indices, [0, 1, 2])
        self.assertEqual(config.target_state, "000")
        self.assertEqual(config.entanglement_threshold, 0.8)
        self.assertEqual(config.interference_weight, 1.0)
    
    def test_different_subspace_types(self):
        """Test different subspace types."""
        types = [SubspaceType.GHZ, SubspaceType.W, SubspaceType.CLUSTER, 
                SubspaceType.BELL, SubspaceType.CUSTOM]
        
        for subspace_type in types:
            config = SubspaceConfig(
                subspace_type=subspace_type,
                qubit_indices=[0, 1],
                entanglement_threshold=0.5
            )
            
            self.assertEqual(config.subspace_type, subspace_type)
            self.assertEqual(config.qubit_indices, [0, 1])
            self.assertEqual(config.entanglement_threshold, 0.5)


class TestMultiSubspaceGrover(unittest.TestCase):
    """Test cases for multi-subspace Grover search."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 4
        self.subspaces = [
            SubspaceConfig(SubspaceType.GHZ, [0, 1], entanglement_threshold=0.7),
            SubspaceConfig(SubspaceType.W, [2, 3], entanglement_threshold=0.6),
            SubspaceConfig(SubspaceType.CLUSTER, [1, 2], entanglement_threshold=0.8)
        ]
        self.grover = MultiSubspaceGrover(self.num_qubits, self.subspaces)
    
    def test_grover_initialization(self):
        """Test Grover search initialization."""
        self.assertEqual(self.grover.num_qubits, 4)
        self.assertEqual(self.grover.num_subspaces, 3)
        self.assertEqual(self.grover.dimension, 16)  # 2^4
        self.assertEqual(len(self.grover.subspaces), 3)
    
    def test_create_superposition(self):
        """Test creating initial superposition state."""
        from core.scalable_quantum_state import ScalableQuantumState
        
        state = ScalableQuantumState(4, use_gpu=False, sparse_threshold=8)
        self.grover._create_superposition(state)
        
        # Check that state is normalized
        norm = np.sum(np.abs(state.to_dense())**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
        
        # Check that all amplitudes are non-zero (superposition)
        state_vector = state.to_dense()
        self.assertTrue(np.all(np.abs(state_vector) > 0))
    
    def test_subspace_grover_application(self):
        """Test applying Grover operator to subspace."""
        from core.scalable_quantum_state import ScalableQuantumState
        
        state = ScalableQuantumState(4, use_gpu=False, sparse_threshold=8)
        self.grover._create_superposition(state)
        
        target_items = ["0000", "1111"]
        subspace = self.subspaces[0]
        
        result = self.grover._apply_subspace_grover(state, subspace, target_items, 0)
        
        # Check result structure
        self.assertIn('subspace_type', result)
        self.assertIn('entanglement', result)
        self.assertIn('success_probability', result)
        self.assertIn('iteration', result)
        self.assertIn('qubit_indices', result)
        
        self.assertEqual(result['subspace_type'], 'ghz')
        self.assertEqual(result['iteration'], 0)
        self.assertEqual(result['qubit_indices'], [0, 1])
    
    def test_oracle_creation(self):
        """Test oracle creation for different subspace types."""
        target_items = ["0000", "1111"]
        
        # Test GHZ oracle
        ghz_subspace = SubspaceConfig(SubspaceType.GHZ, [0, 1])
        ghz_oracle = self.grover._create_subspace_oracle(ghz_subspace, target_items)
        
        self.assertIn('target_items', ghz_oracle)
        self.assertIn('subspace_type', ghz_oracle)
        self.assertIn('qubit_indices', ghz_oracle)
        self.assertIn('ghz_signature', ghz_oracle)
        
        # Test W oracle
        w_subspace = SubspaceConfig(SubspaceType.W, [2, 3])
        w_oracle = self.grover._create_subspace_oracle(w_subspace, target_items)
        
        self.assertIn('w_signature', w_oracle)
        
        # Test cluster oracle
        cluster_subspace = SubspaceConfig(SubspaceType.CLUSTER, [1, 2])
        cluster_oracle = self.grover._create_subspace_oracle(cluster_subspace, target_items)
        
        self.assertIn('cluster_signature', cluster_oracle)
    
    def test_oracle_signature_creation(self):
        """Test oracle signature creation."""
        # Test GHZ signature
        ghz_subspace = SubspaceConfig(SubspaceType.GHZ, [0, 1])
        ghz_signature = self.grover._create_ghz_oracle_signature(ghz_subspace)
        
        # GHZ signature should include |00⟩ and |11⟩
        self.assertIn(0, ghz_signature)  # |00⟩
        self.assertIn(3, ghz_signature)  # |11⟩
        
        # Test W signature
        w_subspace = SubspaceConfig(SubspaceType.W, [0, 1])
        w_signature = self.grover._create_w_oracle_signature(w_subspace)
        
        # W signature should include |01⟩ and |10⟩
        self.assertIn(1, w_signature)  # |01⟩
        self.assertIn(2, w_signature)  # |10⟩
        
        # Test cluster signature
        cluster_subspace = SubspaceConfig(SubspaceType.CLUSTER, [0, 1])
        cluster_signature = self.grover._create_cluster_oracle_signature(cluster_subspace)
        
        # Cluster signature should include some states
        self.assertIsInstance(cluster_signature, list)
    
    def test_state_matching_oracle(self):
        """Test state matching against oracle."""
        ghz_subspace = SubspaceConfig(SubspaceType.GHZ, [0, 1])
        oracle = self.grover._create_subspace_oracle(ghz_subspace, ["0000", "1111"])
        
        # Test state |0000⟩ (should match GHZ pattern)
        self.assertTrue(self.grover._state_matches_oracle(0, oracle, [0, 1]))
        
        # Test state |0011⟩ (should match GHZ pattern)
        self.assertTrue(self.grover._state_matches_oracle(3, oracle, [0, 1]))
        
        # Test state |0001⟩ (should not match GHZ pattern)
        self.assertFalse(self.grover._state_matches_oracle(1, oracle, [0, 1]))
    
    def test_diffusion_operator(self):
        """Test diffusion operator application."""
        from core.scalable_quantum_state import ScalableQuantumState
        
        state = ScalableQuantumState(4, use_gpu=False, sparse_threshold=8)
        self.grover._create_superposition(state)
        
        initial_state = state.to_dense().copy()
        
        # Apply diffusion operator
        self.grover._apply_diffusion_operator(state, [0, 1])
        
        # State should be normalized
        norm = np.sum(np.abs(state.to_dense())**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
        
        # State should be different from initial (diffusion effect)
        final_state = state.to_dense()
        self.assertFalse(np.allclose(initial_state, final_state))
    
    def test_entanglement_calculation(self):
        """Test entanglement calculation within subspace."""
        from core.scalable_quantum_state import ScalableQuantumState
        
        state = ScalableQuantumState(4, use_gpu=False, sparse_threshold=8)
        self.grover._create_superposition(state)
        
        subspace = self.subspaces[0]
        entanglement = self.grover._calculate_subspace_entanglement(state, subspace)
        
        # Entanglement should be a valid float
        self.assertIsInstance(entanglement, float)
        self.assertGreaterEqual(entanglement, 0.0)
    
    def test_success_probability_calculation(self):
        """Test success probability calculation."""
        from core.scalable_quantum_state import ScalableQuantumState
        
        state = ScalableQuantumState(4, use_gpu=False, sparse_threshold=8)
        self.grover._create_superposition(state)
        
        subspace = self.subspaces[0]
        target_items = ["0000", "1111"]
        
        success_prob = self.grover._calculate_subspace_success_probability(
            state, subspace, target_items
        )
        
        # Success probability should be valid
        self.assertIsInstance(success_prob, float)
        self.assertGreaterEqual(success_prob, 0.0)
        self.assertLessEqual(success_prob, 1.0)
    
    def test_interference_calculation(self):
        """Test interference calculation between subspaces."""
        from core.scalable_quantum_state import ScalableQuantumState
        
        state = ScalableQuantumState(4, use_gpu=False, sparse_threshold=8)
        self.grover._create_superposition(state)
        
        # Create mock subspace results
        subspace_results = [
            {'entanglement': 0.8, 'success_probability': 0.6},
            {'entanglement': 0.6, 'success_probability': 0.7},
            {'entanglement': 0.9, 'success_probability': 0.5}
        ]
        
        interference_metrics = self.grover._calculate_interference(state, subspace_results)
        
        # Check interference metrics structure
        self.assertIn('interference_matrix', interference_metrics)
        self.assertIn('coherence_metrics', interference_metrics)
        self.assertIn('total_interference', interference_metrics)
        
        # Check interference matrix
        interference_matrix = interference_metrics['interference_matrix']
        self.assertEqual(len(interference_matrix), 3)
        self.assertEqual(len(interference_matrix[0]), 3)
        
        # Check coherence metrics
        coherence_metrics = interference_metrics['coherence_metrics']
        self.assertIn('total_interference', coherence_metrics)
        self.assertIn('max_interference', coherence_metrics)
        self.assertIn('mean_interference', coherence_metrics)
        self.assertIn('coherence_length', coherence_metrics)
    
    def test_full_grover_search(self):
        """Test full multi-subspace Grover search."""
        target_items = ["0000", "1111"]
        
        result = self.grover.search(target_items, max_iterations=10, shots=100)
        
        # Check result structure
        self.assertIsInstance(result, GroverResult)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.success_probability, float)
        self.assertIsInstance(result.iterations, int)
        self.assertIsInstance(result.execution_time, float)
        self.assertIsInstance(result.measurement_counts, dict)
        self.assertIsInstance(result.interference_metrics, dict)
        self.assertIsInstance(result.subspace_results, list)
        
        # Check value ranges
        self.assertGreaterEqual(result.success_probability, 0.0)
        self.assertLessEqual(result.success_probability, 1.0)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertGreater(result.execution_time, 0.0)
        
        # Check subspace results
        self.assertEqual(len(result.subspace_results), len(self.subspaces))
        for subspace_result in result.subspace_results:
            self.assertIn('subspace_type', subspace_result)
            self.assertIn('entanglement', subspace_result)
            self.assertIn('success_probability', subspace_result)
    
    def test_grover_search_with_interference(self):
        """Test Grover search with interference enabled."""
        target_items = ["0000", "1111"]
        
        result = self.grover.search(target_items, max_iterations=5, shots=50, 
                                  enable_interference=True)
        
        # Check that interference metrics are present
        self.assertIn('interference_evolution', result.interference_metrics)
        self.assertIn('entanglement_evolution', result.interference_metrics)
        self.assertIn('success_probability_evolution', result.interference_metrics)
        
        # Check evolution lists
        interference_evolution = result.interference_metrics['interference_evolution']
        entanglement_evolution = result.interference_metrics['entanglement_evolution']
        success_prob_evolution = result.interference_metrics['success_probability_evolution']
        
        self.assertIsInstance(interference_evolution, list)
        self.assertIsInstance(entanglement_evolution, list)
        self.assertIsInstance(success_prob_evolution, list)
        
        # All evolution lists should have same length
        self.assertEqual(len(interference_evolution), result.iterations)
        self.assertEqual(len(entanglement_evolution), result.iterations)
        self.assertEqual(len(success_prob_evolution), result.iterations)
    
    def test_grover_search_without_interference(self):
        """Test Grover search with interference disabled."""
        target_items = ["0000", "1111"]
        
        result = self.grover.search(target_items, max_iterations=5, shots=50, 
                                  enable_interference=False)
        
        # Should still complete successfully
        self.assertIsInstance(result, GroverResult)
        self.assertGreaterEqual(result.success_probability, 0.0)
    
    def test_measurement_counts(self):
        """Test measurement counts from Grover search."""
        target_items = ["0000", "1111"]
        
        result = self.grover.search(target_items, max_iterations=5, shots=100)
        
        # Check measurement counts
        self.assertIsInstance(result.measurement_counts, dict)
        self.assertGreater(len(result.measurement_counts), 0)
        
        # Check that all bitstrings are valid
        for bitstring in result.measurement_counts.keys():
            self.assertEqual(len(bitstring), self.num_qubits)
            for bit in bitstring:
                self.assertIn(bit, ['0', '1'])
    
    def test_export_results(self):
        """Test exporting Grover search results."""
        target_items = ["0000", "1111"]
        result = self.grover.search(target_items, max_iterations=5, shots=50)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Export results
            self.grover.export_results(result, temp_file)
            
            # Check that file was created
            self.assertTrue(os.path.exists(temp_file))
            
            # Check file content
            with open(temp_file, 'r') as f:
                import json
                data = json.load(f)
                
                # Check required fields
                self.assertIn('success', data)
                self.assertIn('success_probability', data)
                self.assertIn('iterations', data)
                self.assertIn('execution_time', data)
                self.assertIn('measurement_counts', data)
                self.assertIn('interference_metrics', data)
                self.assertIn('subspace_results', data)
                self.assertIn('num_qubits', data)
                self.assertIn('num_subspaces', data)
                self.assertIn('subspace_configs', data)
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_visualization_generation(self):
        """Test visualization generation."""
        target_items = ["0000", "1111"]
        result = self.grover.search(target_items, max_iterations=5, shots=50)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate visualizations
            self.grover.generate_interference_visualization(result, temp_dir)
            
            # Check that visualization files were created
            expected_files = [
                'interference_heatmap.png',
                'time_evolution.png',
                'entanglement_network.png'
            ]
            
            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(filepath))


class TestInterferenceDiagnostics(unittest.TestCase):
    """Test cases for interference diagnostics."""
    
    def test_interference_diagnostics_initialization(self):
        """Test interference diagnostics initialization."""
        interference_matrix = np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]])
        coherence_metrics = {'total_interference': 1.2, 'max_interference': 0.5}
        entanglement_evolution = [0.8, 0.9, 0.7]
        success_prob_evolution = [0.6, 0.7, 0.8]
        heatmap_data = {'matrices': [interference_matrix.tolist()]}
        
        diagnostics = InterferenceDiagnostics(
            interference_matrix=interference_matrix,
            coherence_metrics=coherence_metrics,
            entanglement_evolution=entanglement_evolution,
            success_probability_evolution=success_prob_evolution,
            interference_heatmap_data=heatmap_data
        )
        
        self.assertEqual(diagnostics.interference_matrix.shape, (3, 3))
        self.assertEqual(diagnostics.coherence_metrics['total_interference'], 1.2)
        self.assertEqual(len(diagnostics.entanglement_evolution), 3)
        self.assertEqual(len(diagnostics.success_probability_evolution), 3)
        self.assertIn('matrices', diagnostics.interference_heatmap_data)


class TestIntegration(unittest.TestCase):
    """Integration tests for multi-subspace Grover search."""
    
    def test_different_subspace_combinations(self):
        """Test different combinations of subspaces."""
        # Test with only GHZ subspaces
        ghz_subspaces = [
            SubspaceConfig(SubspaceType.GHZ, [0, 1]),
            SubspaceConfig(SubspaceType.GHZ, [2, 3])
        ]
        ghz_grover = MultiSubspaceGrover(4, ghz_subspaces)
        
        result = ghz_grover.search(["0000", "1111"], max_iterations=5, shots=50)
        self.assertIsInstance(result, GroverResult)
        
        # Test with mixed subspace types
        mixed_subspaces = [
            SubspaceConfig(SubspaceType.GHZ, [0, 1]),
            SubspaceConfig(SubspaceType.W, [2, 3]),
            SubspaceConfig(SubspaceType.CLUSTER, [1, 2])
        ]
        mixed_grover = MultiSubspaceGrover(4, mixed_subspaces)
        
        result = mixed_grover.search(["0000", "1111"], max_iterations=5, shots=50)
        self.assertIsInstance(result, GroverResult)
    
    def test_large_system_performance(self):
        """Test performance with larger quantum systems."""
        # Test with 6 qubits
        large_subspaces = [
            SubspaceConfig(SubspaceType.GHZ, [0, 1, 2]),
            SubspaceConfig(SubspaceType.W, [3, 4, 5])
        ]
        large_grover = MultiSubspaceGrover(6, large_subspaces)
        
        result = large_grover.search(["000000", "111111"], max_iterations=3, shots=20)
        self.assertIsInstance(result, GroverResult)
        self.assertGreater(result.execution_time, 0.0)
    
    def test_convergence_behavior(self):
        """Test convergence behavior of Grover search."""
        subspaces = [SubspaceConfig(SubspaceType.GHZ, [0, 1])]
        grover = MultiSubspaceGrover(2, subspaces)
        
        # Run multiple searches and check convergence
        for _ in range(3):
            result = grover.search(["00", "11"], max_iterations=10, shots=50)
            
            # Should complete successfully
            self.assertIsInstance(result, GroverResult)
            self.assertGreaterEqual(result.success_probability, 0.0)
            self.assertGreaterEqual(result.iterations, 1)


if __name__ == '__main__':
    unittest.main()
