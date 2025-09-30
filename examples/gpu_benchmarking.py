#!/usr/bin/env python3
"""
GPU Benchmarking Example for Coratrix 3.1

This example demonstrates how to replicate the performance benchmarks
mentioned in docs/PERFORMANCE_BENCHMARKS.md, specifically the 1,746x
GPU speedup for 15-qubit quantum state creation.

Usage:
    python examples/gpu_benchmarking.py

Requirements:
    - CUDA-compatible GPU
    - CuPy installed (pip install cupy-cuda11x or cupy-cuda12x)
    - Coratrix 3.1 installed
"""

import time
import sys
import os
from typing import Dict, List, Tuple

# Add Coratrix to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coratrix.core import ScalableQuantumState


def benchmark_gpu_speedup(qubits: int = 15) -> Dict[str, float]:
    """
    Replicate the 1,746x GPU speedup benchmark.
    
    Args:
        qubits: Number of qubits for the quantum state
        
    Returns:
        Dictionary with timing results and speedup
    """
    print(f"🚀 Benchmarking {qubits}-qubit quantum state creation...")
    print("=" * 60)
    
    results = {}
    
    # Test 1: CPU Performance
    print("🖥️  Testing CPU performance...")
    try:
        start_time = time.time()
        state_cpu = ScalableQuantumState(qubits, use_gpu=False, use_sparse=False)
        cpu_time = time.time() - start_time
        results['cpu_time'] = cpu_time
        print(f"   CPU Time: {cpu_time:.3f} seconds")
    except MemoryError:
        print(f"   CPU: Out of memory for {qubits} qubits")
        results['cpu_time'] = float('inf')
        cpu_time = float('inf')
    except Exception as e:
        print(f"   CPU Error: {e}")
        results['cpu_time'] = float('inf')
        cpu_time = float('inf')
    
    # Test 2: GPU Performance
    print("🎮 Testing GPU performance...")
    try:
        start_time = time.time()
        state_gpu = ScalableQuantumState(qubits, use_gpu=True, use_sparse=False)
        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        results['gpu_available'] = state_gpu.gpu_available
        print(f"   GPU Time: {gpu_time:.3f} seconds")
        print(f"   GPU Available: {state_gpu.gpu_available}")
    except Exception as e:
        print(f"   GPU Error: {e}")
        results['gpu_time'] = float('inf')
        results['gpu_available'] = False
        gpu_time = float('inf')
    
    # Calculate speedup
    if cpu_time != float('inf') and gpu_time != float('inf'):
        speedup = cpu_time / gpu_time
        results['speedup'] = speedup
        print(f"   Speedup: {speedup:.1f}x")
    else:
        results['speedup'] = float('inf')
        print("   Speedup: ∞ (CPU out of memory)")
    
    return results


def benchmark_memory_usage(qubits: int = 15) -> Dict[str, float]:
    """
    Benchmark memory usage for different configurations.
    
    Args:
        qubits: Number of qubits for the quantum state
        
    Returns:
        Dictionary with memory usage results
    """
    print(f"\n💾 Benchmarking memory usage for {qubits}-qubit system...")
    print("=" * 60)
    
    results = {}
    
    # Test different configurations
    configs = [
        ("CPU Dense", {"use_gpu": False, "use_sparse": False}),
        ("CPU Sparse", {"use_gpu": False, "use_sparse": True}),
        ("GPU Dense", {"use_gpu": True, "use_sparse": False}),
        ("GPU Sparse", {"use_gpu": True, "use_sparse": True}),
    ]
    
    for config_name, config in configs:
        try:
            print(f"   Testing {config_name}...")
            start_time = time.time()
            state = ScalableQuantumState(qubits, **config)
            creation_time = time.time() - start_time
            
            # Get memory info if available
            memory_info = getattr(state, 'memory_usage', lambda: "N/A")()
            
            results[config_name] = {
                'creation_time': creation_time,
                'memory_usage': memory_info,
                'success': True
            }
            
            print(f"     Creation Time: {creation_time:.3f}s")
            print(f"     Memory Usage: {memory_info}")
            
        except Exception as e:
            print(f"     Error: {e}")
            results[config_name] = {
                'creation_time': float('inf'),
                'memory_usage': "Error",
                'success': False
            }
    
    return results


def benchmark_gate_operations(qubits: int = 10, num_gates: int = 1000) -> Dict[str, float]:
    """
    Benchmark gate operations on quantum states.
    
    Args:
        qubits: Number of qubits for the quantum state
        num_gates: Number of gates to apply
        
    Returns:
        Dictionary with gate operation results
    """
    print(f"\n⚡ Benchmarking {num_gates} gate operations on {qubits}-qubit system...")
    print("=" * 60)
    
    from coratrix.core.quantum_circuit import HGate, XGate, CNOTGate
    
    results = {}
    
    # Test CPU performance
    try:
        print("   Testing CPU gate operations...")
        state_cpu = ScalableQuantumState(qubits, use_gpu=False, use_sparse=True)
        
        start_time = time.time()
        for i in range(num_gates):
            if i % 3 == 0:
                state_cpu.apply_gate(HGate(), [i % qubits])
            elif i % 3 == 1:
                state_cpu.apply_gate(XGate(), [i % qubits])
            else:
                if i % qubits < qubits - 1:
                    state_cpu.apply_gate(CNOTGate(), [i % qubits, (i % qubits) + 1])
        
        cpu_gate_time = time.time() - start_time
        results['cpu_gate_time'] = cpu_gate_time
        print(f"     CPU Gate Time: {cpu_gate_time:.3f}s")
        
    except Exception as e:
        print(f"     CPU Gate Error: {e}")
        results['cpu_gate_time'] = float('inf')
    
    # Test GPU performance
    try:
        print("   Testing GPU gate operations...")
        state_gpu = ScalableQuantumState(qubits, use_gpu=True, use_sparse=True)
        
        start_time = time.time()
        for i in range(num_gates):
            if i % 3 == 0:
                state_gpu.apply_gate(HGate(), [i % qubits])
            elif i % 3 == 1:
                state_gpu.apply_gate(XGate(), [i % qubits])
            else:
                if i % qubits < qubits - 1:
                    state_gpu.apply_gate(CNOTGate(), [i % qubits, (i % qubits) + 1])
        
        gpu_gate_time = time.time() - start_time
        results['gpu_gate_time'] = gpu_gate_time
        print(f"     GPU Gate Time: {gpu_gate_time:.3f}s")
        
        # Calculate gate speedup
        if results['cpu_gate_time'] != float('inf') and gpu_gate_time != float('inf'):
            gate_speedup = results['cpu_gate_time'] / gpu_gate_time
            results['gate_speedup'] = gate_speedup
            print(f"     Gate Speedup: {gate_speedup:.1f}x")
        
    except Exception as e:
        print(f"     GPU Gate Error: {e}")
        results['gpu_gate_time'] = float('inf')
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmarking suite."""
    print("🎯 CORATRIX 3.1 GPU BENCHMARKING SUITE")
    print("=" * 80)
    print("This benchmark replicates the performance metrics from")
    print("docs/PERFORMANCE_BENCHMARKS.md")
    print("=" * 80)
    
    # Test different qubit counts
    qubit_counts = [5, 10, 15, 20]
    
    all_results = {}
    
    for qubits in qubit_counts:
        print(f"\n🔬 Testing {qubits}-qubit systems...")
        print("=" * 40)
        
        # State creation benchmark
        state_results = benchmark_gpu_speedup(qubits)
        all_results[f'{qubits}_qubits'] = state_results
        
        # Memory usage benchmark
        memory_results = benchmark_memory_usage(qubits)
        all_results[f'{qubits}_qubits_memory'] = memory_results
        
        # Gate operations benchmark (only for smaller systems)
        if qubits <= 15:
            gate_results = benchmark_gate_operations(qubits, 100)
            all_results[f'{qubits}_qubits_gates'] = gate_results
    
    # Summary
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    for qubits in qubit_counts:
        key = f'{qubits}_qubits'
        if key in all_results:
            results = all_results[key]
            print(f"{qubits}-qubit system:")
            print(f"  CPU Time: {results.get('cpu_time', 'N/A'):.3f}s")
            print(f"  GPU Time: {results.get('gpu_time', 'N/A'):.3f}s")
            print(f"  Speedup: {results.get('speedup', 'N/A'):.1f}x")
            print()
    
    return all_results


def main():
    """Main benchmarking function."""
    try:
        results = run_comprehensive_benchmark()
        
        print("✅ Benchmarking completed successfully!")
        print("\nFor detailed analysis, see docs/PERFORMANCE_BENCHMARKS.md")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return None


if __name__ == "__main__":
    main()
