"""
Coratrix Python API - Production-Ready Quantum Computing Platform

This module provides a stable, well-documented Python API for Coratrix,
a high-performance quantum computing simulation and research platform.

Key Features:
- Scalable quantum state simulation (2-12 qubits)
- GPU acceleration with CuPy fallback
- Sparse matrix optimization for large systems
- Hardware backend interfaces (OpenQASM, Qiskit, PennyLane)
- Advanced quantum algorithms and analysis
- Noise models and error mitigation
- Optimization engine for parameterized circuits
- Multi-subspace Grover search with interference diagnostics
- Reproducibility and security features

Example Usage:
    >>> from coratrix import QuantumCircuit, HGate, CNOTGate
    >>> from coratrix import ScalableQuantumState
    >>> from coratrix import MultiSubspaceGrover, SubspaceConfig, SubspaceType
    >>> 
    >>> # Create a quantum circuit
    >>> circuit = QuantumCircuit(2)
    >>> circuit.add_gate(HGate(), [0])
    >>> circuit.add_gate(CNOTGate(), [0, 1])
    >>> 
    >>> # Execute the circuit
    >>> circuit.execute()
    >>> print(circuit.get_state())
    >>> 
    >>> # Use scalable quantum state
    >>> state = ScalableQuantumState(4, use_gpu=True, use_sparse=True)
    >>> 
    >>> # Multi-subspace Grover search
    >>> subspaces = [
    >>>     SubspaceConfig(SubspaceType.GHZ, [0, 1]),
    >>>     SubspaceConfig(SubspaceType.W, [2, 3])
    >>> ]
    >>> grover = MultiSubspaceGrover(4, subspaces)
    >>> result = grover.search(["0000", "1111"], max_iterations=100)

Version: 2.3.0
Author: Coratrix Development Team
License: MIT
"""

# Core quantum computing modules
from core.qubit import QuantumState
from core.scalable_quantum_state import ScalableQuantumState
from core.circuit import QuantumCircuit
from core.gates import XGate, YGate, ZGate, HGate, CNOTGate, ToffoliGate, SWAPGate
from core.advanced_gates import RxGate, RyGate, RzGate, CPhaseGate
from core.measurement import QuantumMeasurement
from core.entanglement_analysis import EntanglementAnalyzer

# Scalable and high-performance modules
from core.noise_models import NoiseModel, QuantumNoise, ErrorMitigation, NoisyQuantumCircuit

# Advanced algorithms and analysis
from core.advanced_algorithms import (
    QuantumStateTomography, FidelityEstimator, EntanglementMonotones,
    EntanglementNetwork, AdvancedQuantumAnalysis
)

# Optimization and parameterized circuits
from core.optimization_engine import (
    ParameterizedCircuit, OptimizationEngine, OptimizationConfig,
    OptimizationMethod, OptimizationResult, NoiseAwareOptimization,
    ConstrainedOptimization
)

# Hardware interfaces
from hardware.openqasm_interface import OpenQASMInterface, OpenQASMParser, OpenQASMExporter
from hardware.backend_interface import (
    BackendManager, QuantumBackend, BackendResult, BackendCapabilities,
    CoratrixSimulatorBackend, NoisySimulatorBackend, IBMQStubBackend
)

# Multi-subspace Grover search
from algorithms.multi_subspace_grover import (
    MultiSubspaceGrover, SubspaceConfig, SubspaceType, GroverResult,
    InterferenceDiagnostics
)

# Reproducibility and security
from core.reproducibility import (
    ReproducibilityManager, SecurityManager, DeterministicRandom,
    ReproducibilityChecker, SystemMetadata, ExperimentMetadata
)

# Version information
__version__ = "2.3.0"
__author__ = "Coratrix Development Team"
__email__ = "info@coratrix.org"
__license__ = "MIT"

# API exports
__all__ = [
    # Core quantum computing
    'QuantumState',
    'ScalableQuantumState', 
    'QuantumCircuit',
    'QuantumMeasurement',
    'EntanglementAnalyzer',
    
    # Quantum gates
    'XGate', 'YGate', 'ZGate', 'HGate', 'CNOTGate', 'ToffoliGate', 'SWAPGate',
    'RxGate', 'RyGate', 'RzGate', 'CPhaseGate',
    
    # Noise models and error mitigation
    'NoiseModel', 'QuantumNoise', 'ErrorMitigation', 'NoisyQuantumCircuit',
    
    # Advanced algorithms
    'QuantumStateTomography', 'FidelityEstimator', 'EntanglementMonotones',
    'EntanglementNetwork', 'AdvancedQuantumAnalysis',
    
    # Optimization
    'ParameterizedCircuit', 'OptimizationEngine', 'OptimizationConfig',
    'OptimizationMethod', 'OptimizationResult', 'NoiseAwareOptimization',
    'ConstrainedOptimization',
    
    # Hardware interfaces
    'OpenQASMInterface', 'OpenQASMParser', 'OpenQASMExporter',
    'BackendManager', 'QuantumBackend', 'BackendResult', 'BackendCapabilities',
    'CoratrixSimulatorBackend', 'NoisySimulatorBackend', 'IBMQStubBackend',
    
    # Multi-subspace Grover search
    'MultiSubspaceGrover', 'SubspaceConfig', 'SubspaceType', 'GroverResult',
    'InterferenceDiagnostics',
    
    # Reproducibility and security
    'ReproducibilityManager', 'SecurityManager', 'DeterministicRandom',
    'ReproducibilityChecker', 'SystemMetadata', 'ExperimentMetadata',
    
    # Version info
    '__version__', '__author__', '__email__', '__license__'
]

# Convenience functions for common operations
def create_bell_state() -> QuantumCircuit:
    """
    Create a Bell state circuit.
    
    Returns:
        QuantumCircuit: A circuit that creates the Bell state (|00⟩ + |11⟩)/√2
        
    Example:
        >>> bell_circuit = create_bell_state()
        >>> bell_circuit.execute()
        >>> print(bell_circuit.get_state())
    """
    circuit = QuantumCircuit(2)
    circuit.add_gate(HGate(), [0])
    circuit.add_gate(CNOTGate(), [0, 1])
    return circuit

def create_ghz_state(num_qubits: int) -> QuantumCircuit:
    """
    Create a GHZ state circuit.
    
    Args:
        num_qubits: Number of qubits in the GHZ state
        
    Returns:
        QuantumCircuit: A circuit that creates the GHZ state
        
    Example:
        >>> ghz_circuit = create_ghz_state(3)
        >>> ghz_circuit.execute()
        >>> print(ghz_circuit.get_state())
    """
    circuit = QuantumCircuit(num_qubits)
    circuit.add_gate(HGate(), [0])
    for i in range(num_qubits - 1):
        circuit.add_gate(CNOTGate(), [i, i + 1])
    return circuit

def create_w_state(num_qubits: int) -> QuantumCircuit:
    """
    Create a W state circuit.
    
    Args:
        num_qubits: Number of qubits in the W state
        
    Returns:
        QuantumCircuit: A circuit that creates the W state
        
    Example:
        >>> w_circuit = create_w_state(3)
        >>> w_circuit.execute()
        >>> print(w_circuit.get_state())
    """
    circuit = QuantumCircuit(num_qubits)
    circuit.add_gate(HGate(), [0])
    for i in range(1, num_qubits):
        circuit.add_gate(HGate(), [i])
        circuit.add_gate(CNOTGate(), [i - 1, i])
    return circuit

def benchmark_quantum_operations(num_qubits: int, shots: int = 1000) -> dict:
    """
    Benchmark quantum operations for a given number of qubits.
    
    Args:
        num_qubits: Number of qubits to benchmark
        shots: Number of measurement shots
        
    Returns:
        dict: Benchmark results including timing and memory usage
        
    Example:
        >>> results = benchmark_quantum_operations(4, shots=1000)
        >>> print(f"Execution time: {results['execution_time']:.3f}s")
    """
    import time
    
    start_time = time.time()
    
    # Create scalable quantum state
    state = ScalableQuantumState(num_qubits, use_gpu=False, use_sparse=num_qubits >= 8)
    
    # Apply some gates
    h_gate = HGate()
    x_gate = XGate()
    cnot_gate = CNOTGate()
    
    for i in range(num_qubits):
        h_gate.apply(state, [i])
        if i < num_qubits - 1:
            cnot_gate.apply(state, [i, i + 1])
    
    # Perform measurements
    measurement = QuantumMeasurement(num_qubits)
    counts = measurement.measure_multiple(state, shots)
    
    execution_time = time.time() - start_time
    
    return {
        'num_qubits': num_qubits,
        'shots': shots,
        'execution_time': execution_time,
        'memory_usage': state.get_memory_usage(),
        'measurement_counts': counts,
        'entanglement_entropy': state.get_entanglement_entropy()
    }

def get_system_info() -> dict:
    """
    Get system information for reproducibility.
    
    Returns:
        dict: System information including hardware and software details
        
    Example:
        >>> info = get_system_info()
        >>> print(f"Python version: {info['python_version']}")
        >>> print(f"GPU available: {info['gpu_available']}")
    """
    manager = ReproducibilityManager()
    metadata = manager.system_metadata
    
    return {
        'python_version': metadata.python_version,
        'platform': metadata.platform,
        'architecture': metadata.architecture,
        'cpu_count': metadata.cpu_count,
        'memory_total_gb': metadata.memory_total_gb,
        'gpu_available': metadata.gpu_available,
        'gpu_memory_gb': metadata.gpu_memory_gb,
        'numpy_version': metadata.numpy_version,
        'scipy_version': metadata.scipy_version,
        'cupy_version': metadata.cupy_version,
        'git_commit_hash': metadata.git_commit_hash,
        'git_branch': metadata.git_branch
    }

# Add convenience functions to __all__
__all__.extend([
    'create_bell_state',
    'create_ghz_state', 
    'create_w_state',
    'benchmark_quantum_operations',
    'get_system_info'
])
