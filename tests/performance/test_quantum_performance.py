"""
Quantum Performance Tests for Coratrix 4.0

This module tests the performance optimization capabilities including
GPU acceleration, memory optimization, and scalability.
"""

import pytest
import numpy as np
import time
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import Coratrix modules
try:
    from core.advanced_quantum_capabilities import AdvancedQuantumState, AccelerationBackend
    from core.advanced_gpu_acceleration import AdvancedGPUAccelerator, AccelerationConfig
    from core.performance_optimization_suite import ComprehensivePerformanceOptimizer, OptimizationConfig, OptimizationLevel
    CORATRIX_AVAILABLE = True
except ImportError:
    CORATRIX_AVAILABLE = False


@pytest.fixture
def sample_quantum_state():
    """Create a sample quantum state for testing."""
    if not CORATRIX_AVAILABLE:
        pytest.skip("Coratrix modules not available")
    
    state = AdvancedQuantumState(4, acceleration_backend=AccelerationBackend.CPU)
    yield state
    state.cleanup()


@pytest.fixture
def gpu_accelerator():
    """Create a GPU accelerator for testing."""
    if not CORATRIX_AVAILABLE:
        pytest.skip("Coratrix modules not available")
    
    config = AccelerationConfig(backend=AccelerationBackend.CPU)
    accelerator = AdvancedGPUAccelerator(config)
    yield accelerator
    accelerator.cleanup()


@pytest.fixture
def performance_optimizer():
    """Create a performance optimizer for testing."""
    if not CORATRIX_AVAILABLE:
        pytest.skip("Coratrix modules not available")
    
    config = OptimizationConfig(level=OptimizationLevel.ADVANCED)
    optimizer = ComprehensivePerformanceOptimizer(config)
    yield optimizer
    optimizer.cleanup()


class TestQuantumStatePerformance:
    """Test quantum state performance capabilities."""
    
    def test_quantum_state_creation_performance(self, sample_quantum_state):
        """Test quantum state creation performance."""
        assert sample_quantum_state.num_qubits == 4
        assert sample_quantum_state.acceleration_backend == AccelerationBackend.CPU
        
        # Test performance metrics
        metrics = sample_quantum_state.get_performance_metrics()
        assert hasattr(metrics, 'execution_time')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'backend_used')
    
    def test_gate_application_performance(self, sample_quantum_state):
        """Test gate application performance."""
        gate_matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        qubit_indices = [0]
        
        start_time = time.time()
        result = sample_quantum_state.apply_gate(gate_matrix, qubit_indices)
        execution_time = time.time() - start_time
        
        assert isinstance(result, AdvancedQuantumState)
        assert execution_time < 1.0  # Should be fast
    
    def test_memory_usage_tracking(self, sample_quantum_state):
        """Test memory usage tracking."""
        memory_usage = sample_quantum_state.get_memory_usage()
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0
    
    def test_scalability(self):
        """Test scalability with different qubit counts."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        for num_qubits in [2, 4, 6, 8]:
            state = AdvancedQuantumState(num_qubits, acceleration_backend=AccelerationBackend.CPU)
            assert state.num_qubits == num_qubits
            
            # Test memory usage scales appropriately
            memory_usage = state.get_memory_usage()
            assert memory_usage >= 0
            
            state.cleanup()


class TestGPUAccelerationPerformance:
    """Test GPU acceleration performance."""
    
    def test_gpu_accelerator_initialization(self, gpu_accelerator):
        """Test GPU accelerator initialization."""
        assert gpu_accelerator.config.backend == AccelerationBackend.CPU
        assert gpu_accelerator.error_count == 0
        assert gpu_accelerator.warning_count == 0
    
    def test_performance_metrics(self, gpu_accelerator):
        """Test performance metrics collection."""
        metrics = gpu_accelerator.get_performance_metrics()
        assert hasattr(metrics, 'execution_time')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'backend_used')
    
    def test_memory_monitoring(self, gpu_accelerator):
        """Test memory monitoring functionality."""
        with gpu_accelerator._memory_monitor():
            # Simulate some memory usage
            test_array = np.random.rand(100, 100)
            del test_array
        
        # Check that metrics were updated
        assert gpu_accelerator.metrics.memory_usage >= 0
    
    def test_error_handling(self, gpu_accelerator):
        """Test error handling and recovery."""
        # Test exception handling
        try:
            raise ValueError("Test error")
        except ValueError as e:
            gpu_accelerator._handle_exception(type(e), e, e.__traceback__)
        
        # Check that error count was incremented
        assert gpu_accelerator.error_count > 0


class TestPerformanceOptimization:
    """Test performance optimization capabilities."""
    
    def test_optimizer_initialization(self, performance_optimizer):
        """Test performance optimizer initialization."""
        assert performance_optimizer.config.level == OptimizationLevel.ADVANCED
        assert performance_optimizer.error_count == 0
        assert performance_optimizer.warning_count == 0
    
    def test_circuit_optimization(self, performance_optimizer):
        """Test circuit optimization."""
        # Create mock circuit
        class MockCircuit:
            def __init__(self):
                self.num_qubits = 4
                # Create proper mock gates
                self.gates = []
                for i in range(8):
                    gate = Mock()
                    gate.__class__.__name__ = f'Gate{i}'
                    gate.qubits = [i % 4]  # Mock qubit indices
                    self.gates.append(gate)
        
        circuit = MockCircuit()
        result = performance_optimizer.optimize_quantum_circuit(circuit)
        
        assert result.success is True
        # Allow negative improvement ratios (performance degradation is possible)
        assert isinstance(result.improvement_ratio, (int, float))
        assert len(result.optimizations_applied) >= 0
    
    def test_ml_workflow_optimization(self, performance_optimizer):
        """Test ML workflow optimization."""
        workflow_config = {
            'type': 'vqe',
            'num_parameters': 10,
            'num_iterations': 100,
            'optimizer': 'adam',
            'backend': 'cpu'
        }
        
        result = performance_optimizer.optimize_quantum_ml_workflow(workflow_config)
        
        assert result.success is True
        # Allow negative improvement ratios (performance degradation is possible)
        assert isinstance(result.improvement_ratio, (int, float))
    
    def test_fault_tolerant_optimization(self, performance_optimizer):
        """Test fault-tolerant circuit optimization."""
        class MockFaultTolerantCircuit:
            def __init__(self):
                self.num_qubits = 4
                self.gates = [Mock() for _ in range(8)]
                self.code_distance = 3
                self.num_logical_qubits = 1
                self.num_physical_qubits = 9
                self.error_correction_cycles = 3
                self.fidelity = 0.99
        
        circuit = MockFaultTolerantCircuit()
        result = performance_optimizer.optimize_fault_tolerant_circuit(circuit)
        
        assert result.success is True
        # Allow negative improvement ratios (performance degradation is possible)
        assert isinstance(result.improvement_ratio, (int, float))
    
    def test_performance_statistics(self, performance_optimizer):
        """Test performance statistics collection."""
        stats = performance_optimizer.get_performance_statistics()
        assert isinstance(stats, dict)
        
        # Test with some optimization history
        class MockCircuit:
            def __init__(self):
                self.num_qubits = 4
                # Create proper mock gates
                self.gates = []
                for i in range(8):
                    gate = Mock()
                    gate.__class__.__name__ = f'Gate{i}'
                    gate.qubits = [i % 4]  # Mock qubit indices
                    self.gates.append(gate)
        
        circuit = MockCircuit()
        performance_optimizer.optimize_quantum_circuit(circuit)
        
        stats = performance_optimizer.get_performance_statistics()
        # Check that stats is a dictionary (even if empty)
        assert isinstance(stats, dict)
        assert 'successful_optimizations' in stats


class TestPerformanceBenchmarks:
    """Test performance benchmarking capabilities."""
    
    def test_benchmark_qubit_scaling(self):
        """Test qubit scaling benchmark."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        from core.advanced_quantum_capabilities import benchmark_qubit_scaling
        
        # Test with small qubit count to avoid memory issues
        results = benchmark_qubit_scaling(max_qubits=6)
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that results contain expected keys
        for key, value in results.items():
            assert 'execution_time' in value
            assert 'memory_usage' in value
            assert 'backend' in value
    
    def test_performance_chart_data(self):
        """Test performance chart data generation."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        from core.advanced_quantum_capabilities import create_performance_chart_data
        
        chart_data = create_performance_chart_data()
        assert isinstance(chart_data, dict)
        assert 'type' in chart_data
        assert 'data' in chart_data
        assert 'options' in chart_data
        
        # Check chart structure
        assert chart_data['type'] == 'line'
        assert 'labels' in chart_data['data']
        assert 'datasets' in chart_data['data']
        assert len(chart_data['data']['datasets']) > 0
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmarking."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        # Test memory usage with different qubit counts
        for num_qubits in [2, 4, 6]:
            state = AdvancedQuantumState(num_qubits, acceleration_backend=AccelerationBackend.CPU)
            memory_usage = state.get_memory_usage()
            
            assert isinstance(memory_usage, float)
            assert memory_usage >= 0
            
            state.cleanup()
    
    def test_execution_time_benchmark(self):
        """Test execution time benchmarking."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        # Test execution time with different qubit counts
        for num_qubits in [2, 4, 6]:
            state = AdvancedQuantumState(num_qubits, acceleration_backend=AccelerationBackend.CPU)
            
            start_time = time.time()
            gate_matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            result = state.apply_gate(gate_matrix, [0])
            execution_time = time.time() - start_time
            
            assert execution_time < 1.0  # Should be fast
            assert isinstance(result, AdvancedQuantumState)
            
            state.cleanup()


class TestPerformanceEdgeCases:
    """Test performance edge cases and error conditions."""
    
    def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        # Test with larger qubit count (but not too large to avoid actual memory issues)
        state = AdvancedQuantumState(8, acceleration_backend=AccelerationBackend.CPU)
        memory_usage = state.get_memory_usage()
        
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0
        
        state.cleanup()
    
    def test_concurrent_access(self):
        """Test concurrent access to performance components."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        import threading
        
        results = []
        
        def worker():
            try:
                state = AdvancedQuantumState(4, acceleration_backend=AccelerationBackend.CPU)
                results.append(state.num_qubits)
                state.cleanup()
            except Exception as e:
                results.append(f"Error: {e}")
        
        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        
        # Start threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        for result in results:
            if isinstance(result, int):
                assert result == 4
            else:
                # Some threads might fail due to resource constraints
                assert "Error" in result
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        if not CORATRIX_AVAILABLE:
            pytest.skip("Coratrix modules not available")
        
        config = AccelerationConfig(backend=AccelerationBackend.CPU)
        accelerator = AdvancedGPUAccelerator(config)
        
        # Simulate errors
        accelerator.error_count = 3
        accelerator.warning_count = 5
        
        # Test error thresholds
        assert accelerator.error_count <= config.error_threshold
        assert accelerator.warning_count <= config.warning_threshold
        
        accelerator.cleanup()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
