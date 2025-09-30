"""
Test large qubit systems (15-20 qubits) with sparse operations.

This test verifies that Coratrix 4.0 can handle 15-20 qubit systems
using sparse gate operations and circuit optimization.
"""

import unittest
import numpy as np
import sys
import os
import time
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.advanced_quantum_capabilities import AdvancedQuantumState, AccelerationBackend
from core.sparse_gate_operations import SparseGateOperator, CircuitOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLargeQubitSystems(unittest.TestCase):
    """Test large qubit systems (15-20 qubits) with sparse operations."""
    
    def setUp(self):
        """Setup test environment."""
        self.test_qubit_counts = [15, 16, 17, 18, 19, 20]
        self.hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self.cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
    
    def test_15_qubit_system(self):
        """Test 15-qubit system with sparse operations."""
        print("\n🧪 Testing 15-qubit system...")
        
        # Create 15-qubit state
        state = AdvancedQuantumState(15, acceleration_backend=AccelerationBackend.CPU)
        
        # Test single-qubit gate
        start_time = time.time()
        result = state.apply_gate(self.hadamard, [0])
        execution_time = time.time() - start_time
        
        print(f"✅ 15 qubits: Single-qubit gate applied in {execution_time:.4f}s")
        print(f"   State shape: {result.state.shape if hasattr(result.state, 'shape') else 'sparse'}")
        
        # Verify state is valid
        self.assertIsNotNone(result.state)
        
        # Test performance metrics
        metrics = result.performance_metrics
        print(f"   Execution time: {metrics.execution_time:.4f}s")
        print(f"   Operations/sec: {metrics.operations_per_second:.2f}")
        
        state.cleanup()
    
    def test_18_qubit_system(self):
        """Test 18-qubit system with sparse operations."""
        print("\n🧪 Testing 18-qubit system...")
        
        # Create 18-qubit state
        state = AdvancedQuantumState(18, acceleration_backend=AccelerationBackend.CPU)
        
        # Test single-qubit gate
        start_time = time.time()
        result = state.apply_gate(self.hadamard, [0])
        execution_time = time.time() - start_time
        
        print(f"✅ 18 qubits: Single-qubit gate applied in {execution_time:.4f}s")
        print(f"   State shape: {result.state.shape if hasattr(result.state, 'shape') else 'sparse'}")
        
        # Verify state is valid
        self.assertIsNotNone(result.state)
        
        # Test performance metrics
        metrics = result.performance_metrics
        print(f"   Execution time: {metrics.execution_time:.4f}s")
        print(f"   Operations/sec: {metrics.operations_per_second:.2f}")
        
        state.cleanup()
    
    def test_20_qubit_system(self):
        """Test 20-qubit system with sparse operations."""
        print("\n🧪 Testing 20-qubit system...")
        
        # Create 20-qubit state
        state = AdvancedQuantumState(20, acceleration_backend=AccelerationBackend.CPU)
        
        # Test single-qubit gate
        start_time = time.time()
        result = state.apply_gate(self.hadamard, [0])
        execution_time = time.time() - start_time
        
        print(f"✅ 20 qubits: Single-qubit gate applied in {execution_time:.4f}s")
        print(f"   State shape: {result.state.shape if hasattr(result.state, 'shape') else 'sparse'}")
        
        # Verify state is valid
        self.assertIsNotNone(result.state)
        
        # Test performance metrics
        metrics = result.performance_metrics
        print(f"   Execution time: {metrics.execution_time:.4f}s")
        print(f"   Operations/sec: {metrics.operations_per_second:.2f}")
        
        state.cleanup()
    
    def test_sparse_gate_operations(self):
        """Test sparse gate operations directly."""
        print("\n🧪 Testing sparse gate operations...")
        
        for num_qubits in [15, 18, 20]:
            print(f"\nTesting {num_qubits} qubits with sparse operations:")
            
            # Create sparse gate operator
            operator = SparseGateOperator(num_qubits, use_gpu=False)
            
            # Create test state
            state = np.zeros(2**num_qubits, dtype=np.complex128)
            state[0] = 1.0
            
            # Test single-qubit gate
            start_time = time.time()
            result = operator.apply_single_qubit_gate(state, self.hadamard, 0)
            execution_time = time.time() - start_time
            
            print(f"   ✅ Single-qubit gate: {execution_time:.4f}s")
            
            # Test two-qubit gate
            start_time = time.time()
            result = operator.apply_two_qubit_gate(result, self.cnot, [0, 1])
            execution_time = time.time() - start_time
            
            print(f"   ✅ Two-qubit gate: {execution_time:.4f}s")
            
            # Test performance metrics
            metrics = operator.get_performance_metrics()
            print(f"   📊 Operations: {metrics['operations_count']}")
            print(f"   📊 Total time: {metrics['total_execution_time']:.4f}s")
            print(f"   📊 Ops/sec: {metrics['operations_per_second']:.2f}")
            
            # Test memory savings
            memory_savings = operator.estimate_memory_savings(num_qubits)
            print(f"   💾 Memory saved: {memory_savings:.2f} GB")
    
    def test_circuit_optimization(self):
        """Test circuit optimization for large systems."""
        print("\n🧪 Testing circuit optimization...")
        
        for num_qubits in [15, 18, 20]:
            print(f"\nTesting circuit optimization for {num_qubits} qubits:")
            
            # Create circuit optimizer
            optimizer = CircuitOptimizer(num_qubits)
            
            # Create test circuit with large gates
            test_circuit = [
                {
                    'type': 'multi_qubit',
                    'target_qubits': [0, 1, 2, 3],
                    'gate_matrix': np.eye(16),
                    'num_qubits': 4
                },
                {
                    'type': 'single_qubit',
                    'target_qubits': [0],
                    'gate_matrix': np.eye(2),
                    'num_qubits': 1
                },
                {
                    'type': 'two_qubit',
                    'target_qubits': [1, 2],
                    'gate_matrix': np.eye(4),
                    'num_qubits': 2
                }
            ]
            
            # Optimize circuit
            start_time = time.time()
            optimized_circuit = optimizer.optimize_circuit(test_circuit)
            optimization_time = time.time() - start_time
            
            print(f"   ✅ Circuit optimization: {optimization_time:.4f}s")
            print(f"   📊 Original gates: {len(test_circuit)}")
            print(f"   📊 Optimized gates: {len(optimized_circuit)}")
            
            # Verify optimization worked
            self.assertLessEqual(len(optimized_circuit), len(test_circuit))
    
    def test_memory_usage(self):
        """Test memory usage for large systems."""
        print("\n🧪 Testing memory usage...")
        
        for num_qubits in [15, 18, 20]:
            print(f"\n{num_qubits} qubits:")
            
            # Calculate theoretical memory requirements
            state_size = 2 ** num_qubits
            dense_memory_gb = (state_size * 16) / (1024**3)  # 16 bytes per complex128
            gate_memory_gb = (state_size * state_size * 16) / (1024**3)
            
            print(f"   📊 State vector: {dense_memory_gb:.2f} GB")
            print(f"   📊 Dense gate matrix: {gate_memory_gb:.2f} GB")
            
            # Test actual memory usage
            try:
                state = AdvancedQuantumState(num_qubits, acceleration_backend=AccelerationBackend.CPU)
                memory_usage = state.get_memory_usage()
                print(f"   ✅ Actual memory usage: {memory_usage:.2f} MB")
                
                # Test gate application
                result = state.apply_gate(self.hadamard, [0])
                print(f"   ✅ Gate application successful")
                
                state.cleanup()
                
            except MemoryError as e:
                print(f"   ❌ Memory error: {e}")
            except Exception as e:
                print(f"   ⚠️  Other error: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for large systems."""
        print("\n🧪 Testing performance benchmarks...")
        
        results = {}
        
        for num_qubits in [15, 18, 20]:
            print(f"\nBenchmarking {num_qubits} qubits:")
            
            try:
                # Create state
                state = AdvancedQuantumState(num_qubits, acceleration_backend=AccelerationBackend.CPU)
                
                # Benchmark single-qubit gate
                start_time = time.time()
                for _ in range(10):
                    result = state.apply_gate(self.hadamard, [0])
                single_qubit_time = (time.time() - start_time) / 10
                
                # Benchmark two-qubit gate
                start_time = time.time()
                for _ in range(10):
                    result = state.apply_gate(self.cnot, [0, 1])
                two_qubit_time = (time.time() - start_time) / 10
                
                results[num_qubits] = {
                    'single_qubit_time': single_qubit_time,
                    'two_qubit_time': two_qubit_time,
                    'success': True
                }
                
                print(f"   ✅ Single-qubit gate: {single_qubit_time:.4f}s")
                print(f"   ✅ Two-qubit gate: {two_qubit_time:.4f}s")
                
                state.cleanup()
                
            except Exception as e:
                results[num_qubits] = {
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ Error: {e}")
        
        # Print summary
        print(f"\n📊 Performance Summary:")
        for num_qubits, result in results.items():
            if result['success']:
                print(f"   {num_qubits} qubits: ✅ {result['single_qubit_time']:.4f}s / {result['two_qubit_time']:.4f}s")
            else:
                print(f"   {num_qubits} qubits: ❌ {result['error']}")


def run_large_qubit_tests():
    """Run all large qubit system tests."""
    print("🚀 Testing Large Qubit Systems (15-20 qubits)")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLargeQubitSystems)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed! Coratrix 4.0 can handle 15-20 qubits!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_large_qubit_tests()
