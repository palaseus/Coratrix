#!/usr/bin/env python3
"""
Benchmarking script for Coratrix quantum state scaling.

This script benchmarks the performance of quantum state operations
across different qubit counts, GPU/CPU modes, and sparse representations.
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, List, Any
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scalable_quantum_state import ScalableQuantumState
from core.gates import HGate, CNOTGate, XGate, YGate, ZGate
from core.circuit import QuantumCircuit


class QuantumBenchmark:
    """Benchmarking suite for quantum state operations."""
    
    def __init__(self, output_file: str = "benchmark_results.json"):
        self.output_file = output_file
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self._get_system_info(),
            'benchmarks': []
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the benchmark."""
        import platform
        import psutil
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        # Check for GPU availability
        try:
            import cupy as cp
            info['gpu_available'] = True
            info['gpu_memory_gb'] = cp.cuda.Device().mem_info[1] / (1024**3)
        except:
            info['gpu_available'] = False
            info['gpu_memory_gb'] = 0
        
        return info
    
    def benchmark_state_creation(self, qubit_counts: List[int], 
                               modes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Benchmark quantum state creation across different configurations."""
        results = []
        
        for num_qubits in qubit_counts:
            for mode in modes:
                print(f"Benchmarking {num_qubits} qubits, mode: {mode}")
                
                start_time = time.time()
                try:
                    state = ScalableQuantumState(
                        num_qubits=num_qubits,
                        use_gpu=mode.get('use_gpu', False),
                        sparse_threshold=mode.get('sparse_threshold', 8),
                        sparse_format=mode.get('sparse_format', 'csr'),
                        deterministic=True
                    )
                    creation_time = time.time() - start_time
                    
                    # Get memory usage
                    memory_info = state.get_memory_usage()
                    performance_metrics = state.get_performance_metrics()
                    
                    result = {
                        'num_qubits': num_qubits,
                        'mode': mode,
                        'creation_time_ms': creation_time * 1000,
                        'memory_usage_mb': memory_info['memory_mb'],
                        'dimension': state.dimension,
                        'use_gpu': state.use_gpu,
                        'use_sparse': state.use_sparse,
                        'sparse_format': state.sparse_format,
                        'success': True
                    }
                    
                    # Add GPU-specific metrics if available
                    if state.use_gpu and 'gpu_memory_used_mb' in performance_metrics:
                        result['gpu_memory_used_mb'] = performance_metrics['gpu_memory_used_mb']
                        result['gpu_memory_utilization'] = performance_metrics['gpu_memory_utilization']
                    
                    results.append(result)
                    
                except Exception as e:
                    result = {
                        'num_qubits': num_qubits,
                        'mode': mode,
                        'creation_time_ms': 0,
                        'memory_usage_mb': 0,
                        'dimension': 2 ** num_qubits,
                        'use_gpu': mode.get('use_gpu', False),
                        'use_sparse': mode.get('sparse_threshold', 8) <= num_qubits,
                        'sparse_format': mode.get('sparse_format', 'csr'),
                        'success': False,
                        'error': str(e)
                    }
                    results.append(result)
        
        return results
    
    def benchmark_gate_operations(self, qubit_counts: List[int], 
                                modes: List[Dict[str, Any]], 
                                num_operations: int = 100) -> List[Dict[str, Any]]:
        """Benchmark gate operations across different configurations."""
        results = []
        
        for num_qubits in qubit_counts:
            for mode in modes:
                print(f"Benchmarking {num_qubits} qubits gate operations, mode: {mode}")
                
                try:
                    state = ScalableQuantumState(
                        num_qubits=num_qubits,
                        use_gpu=mode.get('use_gpu', False),
                        sparse_threshold=mode.get('sparse_threshold', 8),
                        sparse_format=mode.get('sparse_format', 'csr'),
                        deterministic=True
                    )
                    
                    # Create a simple circuit
                    circuit = QuantumCircuit(num_qubits)
                    
                    # Add gates
                    h_gate = HGate()
                    x_gate = XGate()
                    cnot_gate = CNOTGate()
                    
                    # Apply Hadamard to first qubit
                    circuit.apply_gate(h_gate, [0])
                    
                    # Apply X gates to create some complexity
                    for i in range(min(num_qubits, 3)):
                        circuit.apply_gate(x_gate, [i])
                    
                    # Apply CNOT gates for entanglement
                    for i in range(min(num_qubits - 1, 2)):
                        circuit.apply_gate(cnot_gate, [i, i + 1])
                    
                    # Benchmark repeated operations
                    start_time = time.time()
                    
                    for _ in range(num_operations):
                        # Apply a simple gate operation
                        circuit.apply_gate(h_gate, [0])
                        circuit.apply_gate(x_gate, [0])
                        circuit.apply_gate(h_gate, [0])
                    
                    operation_time = time.time() - start_time
                    
                    # Get performance metrics
                    performance_metrics = state.get_performance_metrics()
                    memory_info = state.get_memory_usage()
                    
                    result = {
                        'num_qubits': num_qubits,
                        'mode': mode,
                        'num_operations': num_operations,
                        'total_time_ms': operation_time * 1000,
                        'time_per_operation_ms': (operation_time * 1000) / num_operations,
                        'operations_per_second': num_operations / operation_time,
                        'memory_usage_mb': memory_info['memory_mb'],
                        'dimension': state.dimension,
                        'use_gpu': state.use_gpu,
                        'use_sparse': state.use_sparse,
                        'sparse_format': state.sparse_format,
                        'success': True
                    }
                    
                    # Add GPU-specific metrics if available
                    if state.use_gpu and 'gpu_memory_used_mb' in performance_metrics:
                        result['gpu_memory_used_mb'] = performance_metrics['gpu_memory_used_mb']
                        result['gpu_memory_utilization'] = performance_metrics['gpu_memory_utilization']
                    
                    results.append(result)
                    
                except Exception as e:
                    result = {
                        'num_qubits': num_qubits,
                        'mode': mode,
                        'num_operations': num_operations,
                        'total_time_ms': 0,
                        'time_per_operation_ms': 0,
                        'operations_per_second': 0,
                        'memory_usage_mb': 0,
                        'dimension': 2 ** num_qubits,
                        'use_gpu': mode.get('use_gpu', False),
                        'use_sparse': mode.get('sparse_threshold', 8) <= num_qubits,
                        'sparse_format': mode.get('sparse_format', 'csr'),
                        'success': False,
                        'error': str(e)
                    }
                    results.append(result)
        
        return results
    
    def benchmark_entanglement_analysis(self, qubit_counts: List[int], 
                                       modes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Benchmark entanglement analysis operations."""
        results = []
        
        for num_qubits in qubit_counts:
            for mode in modes:
                print(f"Benchmarking {num_qubits} qubits entanglement analysis, mode: {mode}")
                
                try:
                    state = ScalableQuantumState(
                        num_qubits=num_qubits,
                        use_gpu=mode.get('use_gpu', False),
                        sparse_threshold=mode.get('sparse_threshold', 8),
                        sparse_format=mode.get('sparse_format', 'csr'),
                        deterministic=True
                    )
                    
                    # Create a Bell state for entanglement analysis
                    if num_qubits >= 2:
                        # Create |00⟩ + |11⟩ state
                        state.set_amplitude(0, 1.0/np.sqrt(2))
                        state.set_amplitude(2**num_qubits - 1, 1.0/np.sqrt(2))
                        state.normalize()
                    
                    # Benchmark entanglement calculations
                    start_time = time.time()
                    
                    # Calculate entanglement entropy
                    entropy = state.get_entanglement_entropy()
                    
                    # Check separability
                    is_separable = state.is_separable()
                    
                    # Get probabilities
                    probabilities = state.get_probabilities()
                    
                    analysis_time = time.time() - start_time
                    
                    # Get performance metrics
                    performance_metrics = state.get_performance_metrics()
                    memory_info = state.get_memory_usage()
                    
                    result = {
                        'num_qubits': num_qubits,
                        'mode': mode,
                        'analysis_time_ms': analysis_time * 1000,
                        'entanglement_entropy': entropy,
                        'is_separable': is_separable,
                        'memory_usage_mb': memory_info['memory_mb'],
                        'dimension': state.dimension,
                        'use_gpu': state.use_gpu,
                        'use_sparse': state.use_sparse,
                        'sparse_format': state.sparse_format,
                        'success': True
                    }
                    
                    # Add GPU-specific metrics if available
                    if state.use_gpu and 'gpu_memory_used_mb' in performance_metrics:
                        result['gpu_memory_used_mb'] = performance_metrics['gpu_memory_used_mb']
                        result['gpu_memory_utilization'] = performance_metrics['gpu_memory_utilization']
                    
                    results.append(result)
                    
                except Exception as e:
                    result = {
                        'num_qubits': num_qubits,
                        'mode': mode,
                        'analysis_time_ms': 0,
                        'entanglement_entropy': 0,
                        'is_separable': True,
                        'memory_usage_mb': 0,
                        'dimension': 2 ** num_qubits,
                        'use_gpu': mode.get('use_gpu', False),
                        'use_sparse': mode.get('sparse_threshold', 8) <= num_qubits,
                        'sparse_format': mode.get('sparse_format', 'csr'),
                        'success': False,
                        'error': str(e)
                    }
                    results.append(result)
        
        return results
    
    def run_full_benchmark(self, max_qubits: int = 12, gpu_available: bool = True):
        """Run the full benchmark suite."""
        print("Starting Coratrix quantum state benchmark...")
        
        # Define qubit counts to test
        qubit_counts = list(range(2, max_qubits + 1))
        
        # Define test modes
        modes = [
            {'use_gpu': False, 'sparse_threshold': 8, 'sparse_format': 'csr', 'name': 'CPU_Dense'},
            {'use_gpu': False, 'sparse_threshold': 6, 'sparse_format': 'csr', 'name': 'CPU_Sparse_CSR'},
            {'use_gpu': False, 'sparse_threshold': 6, 'sparse_format': 'coo', 'name': 'CPU_Sparse_COO'},
            {'use_gpu': False, 'sparse_threshold': 6, 'sparse_format': 'lil', 'name': 'CPU_Sparse_LIL'},
        ]
        
        if gpu_available:
            modes.extend([
                {'use_gpu': True, 'sparse_threshold': 8, 'sparse_format': 'csr', 'name': 'GPU_Dense'},
                {'use_gpu': True, 'sparse_threshold': 6, 'sparse_format': 'csr', 'name': 'GPU_Sparse_CSR'},
            ])
        
        # Run benchmarks
        print("Running state creation benchmarks...")
        creation_results = self.benchmark_state_creation(qubit_counts, modes)
        
        print("Running gate operation benchmarks...")
        operation_results = self.benchmark_gate_operations(qubit_counts, modes)
        
        print("Running entanglement analysis benchmarks...")
        entanglement_results = self.benchmark_entanglement_analysis(qubit_counts, modes)
        
        # Compile results
        self.results['benchmarks'] = {
            'state_creation': creation_results,
            'gate_operations': operation_results,
            'entanglement_analysis': entanglement_results
        }
        
        # Save results
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark completed. Results saved to {self.output_file}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        benchmarks = self.results['benchmarks']
        
        # State creation summary
        print("\nState Creation Performance:")
        print("-" * 40)
        for result in benchmarks['state_creation']:
            if result['success']:
                print(f"{result['num_qubits']} qubits, {result['mode']['name']}: "
                      f"{result['creation_time_ms']:.2f}ms, "
                      f"{result['memory_usage_mb']:.2f}MB")
            else:
                print(f"{result['num_qubits']} qubits, {result['mode']['name']}: FAILED")
        
        # Gate operations summary
        print("\nGate Operations Performance:")
        print("-" * 40)
        for result in benchmarks['gate_operations']:
            if result['success']:
                print(f"{result['num_qubits']} qubits, {result['mode']['name']}: "
                      f"{result['operations_per_second']:.2f} ops/sec, "
                      f"{result['time_per_operation_ms']:.4f}ms/op")
            else:
                print(f"{result['num_qubits']} qubits, {result['mode']['name']}: FAILED")
        
        # Entanglement analysis summary
        print("\nEntanglement Analysis Performance:")
        print("-" * 40)
        for result in benchmarks['entanglement_analysis']:
            if result['success']:
                print(f"{result['num_qubits']} qubits, {result['mode']['name']}: "
                      f"{result['analysis_time_ms']:.2f}ms, "
                      f"entropy={result['entanglement_entropy']:.4f}")
            else:
                print(f"{result['num_qubits']} qubits, {result['mode']['name']}: FAILED")


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark Coratrix quantum state scaling')
    parser.add_argument('--max-qubits', type=int, default=12, 
                       help='Maximum number of qubits to test')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for benchmark results')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU testing')
    
    args = parser.parse_args()
    
    # Check GPU availability
    gpu_available = True
    try:
        import cupy as cp
        print(f"GPU available: {cp.cuda.Device().mem_info[1] / (1024**3):.2f} GB")
    except:
        gpu_available = False
        print("GPU not available")
    
    if args.no_gpu:
        gpu_available = False
    
    # Run benchmark
    benchmark = QuantumBenchmark(args.output)
    benchmark.run_full_benchmark(max_qubits=args.max_qubits, gpu_available=gpu_available)


if __name__ == '__main__':
    main()
