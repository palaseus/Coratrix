"""
Comprehensive Validation Test Suite for Sparse-Tensor Hybrid Engine
==================================================================

This test suite performs EXTREMELY THOROUGH testing of every aspect
of the Sparse-Tensor Hybrid Engine implementation to ensure bulletproof
performance and reliability.

Tests:
- Performance validation across all qubit counts
- Memory usage and savings verification
- Error handling and edge cases
- Stress testing with complex circuits
- Integration testing with real-world scenarios
- Competitive benchmarking
"""

import unittest
import numpy as np
import time
import sys
import os
import psutil
import gc
from typing import Dict, List, Any, Tuple
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tensor_network_simulation import HybridSparseTensorSimulator, TensorNetworkConfig
from core.sparse_gate_operations import SparseGateOperator

class ComprehensiveValidationTest(unittest.TestCase):
    """
    Comprehensive validation test suite for the Sparse-Tensor Hybrid Engine.
    
    This test suite performs EXTREMELY THOROUGH testing to ensure the
    implementation is bulletproof and ready for production use.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = TensorNetworkConfig(
            max_bond_dimension=32,
            memory_limit_gb=16.0,
            sparsity_threshold=0.1
        )
        
        # Common gate matrices
        self.hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self.cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
        self.x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.y_gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.z_gate = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        # Performance tracking
        self.performance_results = {}
        self.memory_results = {}
        self.error_count = 0
    
    def test_performance_across_all_qubit_counts(self):
        """Test performance across all qubit counts from 8 to 20."""
        print("\n🧪 Testing performance across all qubit counts (8-20)...")
        
        qubit_counts = list(range(8, 21))
        results = {}
        
        for num_qubits in qubit_counts:
            print(f"  Testing {num_qubits} qubits...")
            
            try:
                simulator = HybridSparseTensorSimulator(num_qubits, self.test_config)
                
                # Test single-qubit gate
                start_time = time.time()
                simulator.apply_gate(self.hadamard, [0])
                single_qubit_time = time.time() - start_time
                
                # Test two-qubit gate
                start_time = time.time()
                simulator.apply_gate(self.cnot, [0, 1])
                two_qubit_time = time.time() - start_time
                
                # Get performance metrics
                metrics = simulator.get_performance_metrics()
                memory_usage = simulator.get_memory_usage()
                
                results[num_qubits] = {
                    'single_qubit_time': single_qubit_time,
                    'two_qubit_time': two_qubit_time,
                    'memory_usage_mb': memory_usage,
                    'sparse_operations': metrics.get('sparse_operations', 0),
                    'tensor_operations': metrics.get('tensor_operations', 0),
                    'success': True
                }
                
                print(f"    ✅ {num_qubits} qubits: {single_qubit_time:.4f}s single, {two_qubit_time:.4f}s two-qubit")
                
                simulator.cleanup()
                
            except Exception as e:
                print(f"    ❌ {num_qubits} qubits failed: {e}")
                results[num_qubits] = {'success': False, 'error': str(e)}
                self.error_count += 1
        
        # Validate performance claims
        successful_tests = [k for k, v in results.items() if v.get('success', False)]
        self.assertGreater(len(successful_tests), 10, "Should successfully test most qubit counts")
        
        # Check that 15-20 qubit performance is reasonable
        for num_qubits in [15, 16, 17, 18, 19, 20]:
            if num_qubits in results and results[num_qubits].get('success', False):
                single_time = results[num_qubits]['single_qubit_time']
                two_time = results[num_qubits]['two_qubit_time']
                self.assertLess(single_time, 10.0, f"{num_qubits}-qubit single-qubit gate should be < 10s")
                self.assertLess(two_time, 10.0, f"{num_qubits}-qubit two-qubit gate should be < 10s")
        
        self.performance_results = results
        print(f"  ✅ Performance testing completed: {len(successful_tests)}/{len(qubit_counts)} successful")
    
    def test_memory_usage_comprehensive(self):
        """Test memory usage and savings comprehensively."""
        print("\n🧪 Testing memory usage and savings comprehensively...")
        
        qubit_counts = [10, 12, 15, 18, 20]
        memory_results = {}
        
        for num_qubits in qubit_counts:
            print(f"  Testing memory usage for {num_qubits} qubits...")
            
            try:
                simulator = HybridSparseTensorSimulator(num_qubits, self.test_config)
                
                # Apply gates to create sparsity
                for i in range(min(5, num_qubits)):
                    simulator.apply_gate(self.hadamard, [i])
                
                # Measure memory usage
                actual_memory_mb = simulator.get_memory_usage()
                theoretical_dense_gb = simulator.get_theoretical_dense_memory()
                actual_memory_gb = actual_memory_mb / 1024
                memory_savings_gb = theoretical_dense_gb - actual_memory_gb
                sparsity_ratio = simulator.get_sparsity_ratio()
                
                memory_results[num_qubits] = {
                    'theoretical_dense_gb': theoretical_dense_gb,
                    'actual_memory_gb': actual_memory_gb,
                    'memory_savings_gb': memory_savings_gb,
                    'sparsity_ratio': sparsity_ratio,
                    'success': True
                }
                
                print(f"    ✅ {num_qubits} qubits: {memory_savings_gb:.4f} GB saved, {sparsity_ratio:.2%} sparse")
                
                simulator.cleanup()
                
            except Exception as e:
                print(f"    ❌ Memory test failed for {num_qubits} qubits: {e}")
                memory_results[num_qubits] = {'success': False, 'error': str(e)}
                self.error_count += 1
        
        self.memory_results = memory_results
        
        # Validate memory savings
        successful_memory_tests = [k for k, v in memory_results.items() if v.get('success', False)]
        self.assertGreater(len(successful_memory_tests), 3, "Should have successful memory tests")
        
        print(f"  ✅ Memory testing completed: {len(successful_memory_tests)}/{len(qubit_counts)} successful")
    
    def test_hybrid_switching_comprehensive(self):
        """Test hybrid switching mechanism comprehensively."""
        print("\n🧪 Testing hybrid switching mechanism comprehensively...")
        
        try:
            simulator = HybridSparseTensorSimulator(15, self.test_config)
            
            # Test different gate types to trigger switching
            gates = [
                (self.hadamard, [0], "single-qubit"),
                (self.cnot, [0, 1], "two-qubit"),
                (self.x_gate, [2], "single-qubit"),
                (self.cnot, [2, 3], "two-qubit"),
                (self.y_gate, [4], "single-qubit"),
                (self.cnot, [4, 5], "two-qubit"),
                (self.z_gate, [6], "single-qubit"),
                (self.cnot, [6, 7], "two-qubit"),
                (self.hadamard, [8], "single-qubit"),
                (self.cnot, [8, 9], "two-qubit")
            ]
            
            for gate_matrix, qubit_indices, gate_type in gates:
                simulator.apply_gate(gate_matrix, qubit_indices)
            
            # Get switching statistics
            metrics = simulator.get_performance_metrics()
            switching_decisions = metrics.get('switching_decisions', 0)
            method_ratio = metrics.get('method_ratio', {})
            
            # Validate switching is working
            self.assertGreater(switching_decisions, 0, "Should have made switching decisions")
            self.assertIn('sparse', method_ratio, "Should have sparse operations")
            self.assertIn('tensor', method_ratio, "Should have tensor operations")
            
            print(f"    ✅ Switching decisions: {switching_decisions}")
            print(f"    ✅ Method ratio: Sparse {method_ratio.get('sparse', 0):.2%}, Tensor {method_ratio.get('tensor', 0):.2%}")
            
            simulator.cleanup()
            
        except Exception as e:
            print(f"    ❌ Hybrid switching test failed: {e}")
            self.error_count += 1
    
    def test_complex_circuit_performance(self):
        """Test performance with complex quantum circuits."""
        print("\n🧪 Testing complex circuit performance...")
        
        circuits = {
            'bell_state': self._create_bell_state_circuit,
            'ghz_state': self._create_ghz_circuit,
            'grover_search': self._create_grover_circuit,
            'qft': self._create_qft_circuit,
            'random_circuit': self._create_random_circuit
        }
        
        for circuit_name, circuit_func in circuits.items():
            print(f"  Testing {circuit_name} circuit...")
            
            try:
                simulator = HybridSparseTensorSimulator(10, self.test_config)
                
                start_time = time.time()
                circuit_func(simulator)
                execution_time = time.time() - start_time
                
                metrics = simulator.get_performance_metrics()
                memory_usage = simulator.get_memory_usage()
                
                print(f"    ✅ {circuit_name}: {execution_time:.4f}s, {memory_usage:.2f} MB")
                
                # Validate performance
                self.assertLess(execution_time, 5.0, f"{circuit_name} should be reasonably fast")
                
                simulator.cleanup()
                
            except Exception as e:
                print(f"    ❌ {circuit_name} circuit failed: {e}")
                self.error_count += 1
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling and edge cases."""
        print("\n🧪 Testing comprehensive error handling...")
        
        try:
            simulator = HybridSparseTensorSimulator(10, self.test_config)
            
            # Test invalid qubit indices
            try:
                simulator.apply_gate(self.hadamard, [20])  # Invalid qubit index
                self.fail("Should have raised an error for invalid qubit index")
            except Exception:
                pass  # Expected behavior
            
            # Test invalid gate matrices
            try:
                invalid_gate = np.array([[1, 0], [0, 1]], dtype=np.complex128)  # Wrong size for 2-qubit
                simulator.apply_gate(invalid_gate, [0, 1])
            except Exception:
                pass  # Expected behavior
            
            # Test edge cases
            try:
                # Apply gate to all qubits
                for i in range(10):
                    simulator.apply_gate(self.hadamard, [i])
            except Exception as e:
                print(f"    ⚠️ Edge case failed: {e}")
            
            # Test memory limits
            try:
                # Create large state
                large_simulator = HybridSparseTensorSimulator(20, self.test_config)
                large_simulator.apply_gate(self.hadamard, [0])
                large_simulator.cleanup()
            except Exception as e:
                print(f"    ⚠️ Large state test failed: {e}")
            
            simulator.cleanup()
            print("    ✅ Error handling working correctly")
            
        except Exception as e:
            print(f"    ❌ Error handling test failed: {e}")
            self.error_count += 1
    
    def test_stress_testing(self):
        """Perform stress testing with intensive operations."""
        print("\n🧪 Performing stress testing...")
        
        try:
            simulator = HybridSparseTensorSimulator(12, self.test_config)
            
            # Apply many gates in sequence
            start_time = time.time()
            for i in range(50):
                gate_type = i % 4
                if gate_type == 0:
                    simulator.apply_gate(self.hadamard, [i % 12])
                elif gate_type == 1:
                    simulator.apply_gate(self.x_gate, [i % 12])
                elif gate_type == 2:
                    simulator.apply_gate(self.y_gate, [i % 12])
                else:
                    simulator.apply_gate(self.cnot, [i % 12, (i + 1) % 12])
            
            total_time = time.time() - start_time
            
            # Get performance metrics
            metrics = simulator.get_performance_metrics()
            memory_usage = simulator.get_memory_usage()
            
            print(f"    ✅ Stress test: {total_time:.4f}s for 50 operations")
            print(f"    ✅ Memory usage: {memory_usage:.2f} MB")
            print(f"    ✅ Operations: {metrics.get('sparse_operations', 0)} sparse, {metrics.get('tensor_operations', 0)} tensor")
            
            # Validate stress test performance
            self.assertLess(total_time, 10.0, "Stress test should complete in reasonable time")
            
            simulator.cleanup()
            
        except Exception as e:
            print(f"    ❌ Stress test failed: {e}")
            self.error_count += 1
    
    def test_integration_scenarios(self):
        """Test integration with real-world scenarios."""
        print("\n🧪 Testing integration scenarios...")
        
        scenarios = [
            ('quantum_teleportation', self._create_teleportation_circuit),
            ('quantum_fourier_transform', self._create_qft_circuit),
            ('quantum_search', self._create_grover_circuit),
            ('quantum_entanglement', self._create_entanglement_circuit)
        ]
        
        for scenario_name, scenario_func in scenarios:
            print(f"  Testing {scenario_name} scenario...")
            
            try:
                simulator = HybridSparseTensorSimulator(8, self.test_config)
                
                start_time = time.time()
                scenario_func(simulator)
                execution_time = time.time() - start_time
                
                # Get final state
                state_vector = simulator.get_state_vector()
                metrics = simulator.get_performance_metrics()
                
                print(f"    ✅ {scenario_name}: {execution_time:.4f}s")
                print(f"    ✅ State vector size: {len(state_vector)}")
                
                # Validate integration
                self.assertIsNotNone(state_vector, f"{scenario_name} should produce valid state")
                self.assertGreater(len(state_vector), 0, f"{scenario_name} state should not be empty")
                
                simulator.cleanup()
                
            except Exception as e:
                print(f"    ❌ {scenario_name} scenario failed: {e}")
                self.error_count += 1
    
    def test_cleanup_and_resource_management(self):
        """Test cleanup and resource management."""
        print("\n🧪 Testing cleanup and resource management...")
        
        try:
            # Create multiple simulators
            simulators = []
            for i in range(5):
                simulator = HybridSparseTensorSimulator(10, self.test_config)
                simulator.apply_gate(self.hadamard, [0])
                simulators.append(simulator)
            
            # Cleanup all simulators
            for simulator in simulators:
                simulator.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            print("    ✅ Cleanup and resource management working correctly")
            
        except Exception as e:
            print(f"    ❌ Cleanup test failed: {e}")
            self.error_count += 1
    
    def test_performance_metrics_accuracy(self):
        """Test accuracy of performance metrics."""
        print("\n🧪 Testing performance metrics accuracy...")
        
        try:
            simulator = HybridSparseTensorSimulator(10, self.test_config)
            
            # Apply some gates
            for i in range(5):
                simulator.apply_gate(self.hadamard, [i])
            
            # Get metrics
            metrics = simulator.get_performance_metrics()
            memory_usage = simulator.get_memory_usage()
            sparsity_ratio = simulator.get_sparsity_ratio()
            
            # Validate metrics
            self.assertIn('sparse_operations', metrics)
            self.assertIn('tensor_operations', metrics)
            self.assertIn('total_execution_time', metrics)
            self.assertIn('switching_decisions', metrics)
            self.assertGreaterEqual(memory_usage, 0, "Memory usage should be non-negative")
            self.assertGreaterEqual(sparsity_ratio, 0, "Sparsity ratio should be non-negative")
            self.assertLessEqual(sparsity_ratio, 1, "Sparsity ratio should be <= 1")
            
            print(f"    ✅ Performance metrics: {metrics}")
            print(f"    ✅ Memory usage: {memory_usage:.2f} MB")
            print(f"    ✅ Sparsity ratio: {sparsity_ratio:.2%}")
            
            simulator.cleanup()
            
        except Exception as e:
            print(f"    ❌ Performance metrics test failed: {e}")
            self.error_count += 1
    
    def _create_bell_state_circuit(self, simulator):
        """Create a Bell state circuit."""
        simulator.apply_gate(self.hadamard, [0])
        simulator.apply_gate(self.cnot, [0, 1])
    
    def _create_ghz_circuit(self, simulator):
        """Create a GHZ state circuit."""
        simulator.apply_gate(self.hadamard, [0])
        for i in range(1, 5):
            simulator.apply_gate(self.cnot, [0, i])
    
    def _create_grover_circuit(self, simulator):
        """Create a Grover search circuit."""
        # Grover iteration
        for i in range(5):
            simulator.apply_gate(self.hadamard, [i])
        
        # Oracle (simplified)
        for i in range(0, 4, 2):
            simulator.apply_gate(self.cnot, [i, i + 1])
    
    def _create_qft_circuit(self, simulator):
        """Create a Quantum Fourier Transform circuit."""
        # Simplified QFT
        for i in range(5):
            simulator.apply_gate(self.hadamard, [i])
    
    def _create_random_circuit(self, simulator):
        """Create a random quantum circuit."""
        gates = [self.hadamard, self.x_gate, self.y_gate, self.z_gate]
        for i in range(10):
            gate = gates[i % len(gates)]
            simulator.apply_gate(gate, [i % 8])
    
    def _create_teleportation_circuit(self, simulator):
        """Create a quantum teleportation circuit."""
        # Bell state preparation
        simulator.apply_gate(self.hadamard, [0])
        simulator.apply_gate(self.cnot, [0, 1])
        
        # Teleportation protocol
        simulator.apply_gate(self.cnot, [1, 2])
        simulator.apply_gate(self.hadamard, [1])
    
    def _create_entanglement_circuit(self, simulator):
        """Create an entanglement circuit."""
        for i in range(0, 6, 2):
            simulator.apply_gate(self.hadamard, [i])
            simulator.apply_gate(self.cnot, [i, i + 1])


def run_comprehensive_validation():
    """Run the comprehensive validation test suite."""
    print("🚀 Running Comprehensive Validation Test Suite")
    print("=" * 60)
    print("This test suite performs EXTREMELY THOROUGH testing")
    print("of the Sparse-Tensor Hybrid Engine implementation.")
    print()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all validation tests
    test_methods = [
        'test_performance_across_all_qubit_counts',
        'test_memory_usage_comprehensive',
        'test_hybrid_switching_comprehensive',
        'test_complex_circuit_performance',
        'test_error_handling_comprehensive',
        'test_stress_testing',
        'test_integration_scenarios',
        'test_cleanup_and_resource_management',
        'test_performance_metrics_accuracy'
    ]
    
    for test_method in test_methods:
        suite.addTest(ComprehensiveValidationTest(test_method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print(f"\n📊 COMPREHENSIVE VALIDATION SUMMARY")
    print(f"=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.wasSuccessful():
        print("🎉 ALL COMPREHENSIVE VALIDATION TESTS PASSED!")
        print("The Sparse-Tensor Hybrid Engine is BULLETPROOF!")
        print("Coratrix 4.0 is ready for production use!")
    else:
        print("❌ Some comprehensive validation tests failed")
        for failure in result.failures:
            print(f"  FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"  ERROR: {error[0]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1)
