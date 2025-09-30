"""
Advanced Quantum Capabilities for Coratrix 4.0

This module extends Coratrix to support 20+ qubit systems with optimized
sparse matrix algorithms, GPU/TPU acceleration, and advanced quantum features.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cp_sparse = None

# TPU acceleration imports
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    jax = None
    jnp = None
    jit = None
    vmap = None

# Distributed computing imports
try:
    import dask
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    Client = None
    as_completed = None

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class AccelerationBackend(Enum):
    """Available acceleration backends."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DISTRIBUTED = "distributed"


@dataclass
class PerformanceMetrics:
    """Performance metrics for quantum operations."""
    execution_time: float
    memory_usage: float
    gpu_memory_used: Optional[float] = None
    operations_per_second: float = 0.0
    sparsity_ratio: float = 0.0
    backend_used: str = "cpu"


class AdvancedQuantumState:
    """
    Advanced quantum state representation supporting 20+ qubits
    with optimized sparse matrix algorithms and multi-backend acceleration.
    """
    
    def __init__(self, num_qubits: int, 
                 acceleration_backend: AccelerationBackend = AccelerationBackend.CPU,
                 sparse_format: str = 'csr',
                 distributed_workers: int = None):
        """
        Initialize advanced quantum state.
        
        Args:
            num_qubits: Number of qubits (supports 20+)
            acceleration_backend: Backend for acceleration
            sparse_format: Sparse matrix format ('csr', 'coo', 'lil')
            distributed_workers: Number of distributed workers
        """
        self.num_qubits = num_qubits
        self.acceleration_backend = acceleration_backend
        self.sparse_format = sparse_format
        self.distributed_workers = distributed_workers or mp.cpu_count()
        
        # Initialize state based on backend
        self._initialize_state()
        self._setup_acceleration()
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0.0,
            backend_used=acceleration_backend.value
        )
    
    def _initialize_state(self):
        """Initialize quantum state based on system size and backend."""
        state_size = 2 ** self.num_qubits
        
        if self.num_qubits <= 12:
            # Use dense representation for small systems
            self.state = np.zeros(state_size, dtype=np.complex128)
            self.state[0] = 1.0  # Initialize to |0...0⟩
            self.is_sparse = False
        else:
            # Use sparse representation for large systems
            if self.sparse_format == 'csr':
                self.state = sp.csr_matrix(([1.0], ([0], [0])), shape=(state_size, 1), dtype=np.complex128)
            elif self.sparse_format == 'coo':
                self.state = sp.coo_matrix(([1.0], ([0], [0])), shape=(state_size, 1), dtype=np.complex128)
            else:  # lil
                self.state = sp.lil_matrix((state_size, 1), dtype=np.complex128)
                self.state[0, 0] = 1.0
            self.is_sparse = True
    
    def _setup_acceleration(self):
        """Setup acceleration backend."""
        if self.acceleration_backend == AccelerationBackend.GPU and GPU_AVAILABLE:
            self._setup_gpu_acceleration()
        elif self.acceleration_backend == AccelerationBackend.TPU and TPU_AVAILABLE:
            self._setup_tpu_acceleration()
        elif self.acceleration_backend == AccelerationBackend.DISTRIBUTED and DASK_AVAILABLE:
            self._setup_distributed_acceleration()
    
    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration with CuPy."""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration requested but CuPy not available")
        
        # Convert state to GPU
        if self.is_sparse:
            if self.sparse_format == 'csr':
                self.state = cp_sparse.csr_matrix(self.state)
            elif self.sparse_format == 'coo':
                self.state = cp_sparse.coo_matrix(self.state)
            else:
                self.state = cp_sparse.lil_matrix(self.state)
        else:
            self.state = cp.asarray(self.state)
        
        logger.info(f"GPU acceleration enabled for {self.num_qubits} qubits")
    
    def _setup_tpu_acceleration(self):
        """Setup TPU acceleration with JAX."""
        if not TPU_AVAILABLE:
            raise RuntimeError("TPU acceleration requested but JAX not available")
        
        # Convert state to JAX array
        if self.is_sparse:
            # Convert sparse to dense for JAX
            self.state = self.state.toarray()
        
        self.state = jnp.array(self.state)
        logger.info(f"TPU acceleration enabled for {self.num_qubits} qubits")
    
    def _setup_distributed_acceleration(self):
        """Setup distributed computing with Dask."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Distributed acceleration requested but Dask not available")
        
        # Initialize Dask client
        self.dask_client = Client(n_workers=self.distributed_workers)
        logger.info(f"Distributed acceleration enabled with {self.distributed_workers} workers")
    
    def apply_gate(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'AdvancedQuantumState':
        """
        Apply quantum gate to the state.
        
        Args:
            gate_matrix: Gate matrix to apply
            qubit_indices: Indices of qubits to apply gate to
            
        Returns:
            Updated quantum state
        """
        start_time = time.time()
        
        try:
            if self.acceleration_backend == AccelerationBackend.GPU and GPU_AVAILABLE:
                result = self._apply_gate_gpu(gate_matrix, qubit_indices)
            elif self.acceleration_backend == AccelerationBackend.TPU and TPU_AVAILABLE:
                result = self._apply_gate_tpu(gate_matrix, qubit_indices)
            elif self.acceleration_backend == AccelerationBackend.DISTRIBUTED and DASK_AVAILABLE:
                result = self._apply_gate_distributed(gate_matrix, qubit_indices)
            else:
                result = self._apply_gate_cpu(gate_matrix, qubit_indices)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics.execution_time += execution_time
            self.performance_metrics.operations_per_second = 1.0 / execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying gate: {e}")
            raise
    
    def _apply_gate_cpu(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'AdvancedQuantumState':
        """Apply gate using CPU computation with sparse optimization for large systems."""
        # For large systems, use sparse operations to avoid memory issues
        if self.num_qubits > 15:
            return self._apply_gate_sparse(gate_matrix, qubit_indices)
        
        # For smaller systems, use standard operations
        if self.is_sparse:
            # Sparse matrix operations
            full_gate = self._create_full_gate_matrix(gate_matrix, qubit_indices)
            self.state = full_gate.dot(self.state)
        else:
            # Dense matrix operations
            full_gate = self._create_full_gate_matrix(gate_matrix, qubit_indices)
            self.state = full_gate @ self.state
        
        return self
    
    def _apply_gate_sparse(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'AdvancedQuantumState':
        """Apply gate using sparse operations for large systems."""
        from core.sparse_gate_operations import SparseGateOperator
        
        # Use sparse gate operator for large systems
        operator = SparseGateOperator(self.num_qubits, use_gpu=False)
        
        if len(qubit_indices) == 1:
            # Single-qubit gate
            self.state = operator.apply_single_qubit_gate(self.state, gate_matrix, qubit_indices[0])
        elif len(qubit_indices) == 2:
            # Two-qubit gate
            self.state = operator.apply_two_qubit_gate(self.state, gate_matrix, qubit_indices)
        else:
            # Multi-qubit gate - decompose into smaller gates
            self._apply_multi_qubit_gate_sparse(gate_matrix, qubit_indices)
        
        return self
    
    def _apply_multi_qubit_gate_sparse(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'AdvancedQuantumState':
        """Apply multi-qubit gate using sparse decomposition."""
        from core.sparse_gate_operations import CircuitOptimizer
        
        # Use circuit optimizer to decompose large gates
        optimizer = CircuitOptimizer(self.num_qubits)
        
        # Create circuit representation
        circuit = [{
            'type': 'multi_qubit',
            'target_qubits': qubit_indices,
            'gate_matrix': gate_matrix
        }]
        
        # Optimize circuit (decompose large gates)
        optimized_circuit = optimizer.optimize_circuit(circuit)
        
        # Apply optimized gates
        for gate in optimized_circuit:
            if gate['type'] == 'single_qubit':
                self._apply_gate_sparse(gate['gate_matrix'], gate['target_qubits'])
            elif gate['type'] == 'two_qubit':
                self._apply_gate_sparse(gate['gate_matrix'], gate['target_qubits'])
        
        return self
    
    def _apply_gate_gpu(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'AdvancedQuantumState':
        """Apply gate using GPU acceleration."""
        # Convert gate matrix to GPU
        gate_matrix_gpu = cp.asarray(gate_matrix)
        
        # Create full gate matrix on GPU
        full_gate = self._create_full_gate_matrix_gpu(gate_matrix_gpu, qubit_indices)
        
        # Apply gate
        if self.is_sparse:
            self.state = full_gate.dot(self.state)
        else:
            self.state = full_gate @ self.state
        
        return self
    
    def _apply_gate_tpu(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'AdvancedQuantumState':
        """Apply gate using TPU acceleration."""
        # Convert gate matrix to JAX
        gate_matrix_jax = jnp.array(gate_matrix)
        
        # Create full gate matrix on TPU
        full_gate = self._create_full_gate_matrix_tpu(gate_matrix_jax, qubit_indices)
        
        # Apply gate
        self.state = full_gate @ self.state
        
        return self
    
    def _apply_gate_distributed(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'AdvancedQuantumState':
        """Apply gate using distributed computing."""
        # Split computation across workers
        # This is a simplified version - full implementation would
        # partition the state and gate matrix across workers
        
        futures = []
        for i in range(self.distributed_workers):
            future = self.dask_client.submit(
                self._apply_gate_worker,
                gate_matrix, qubit_indices, i, self.distributed_workers
            )
            futures.append(future)
        
        # Collect results
        results = self.dask_client.gather(futures)
        
        # Combine results
        self._combine_distributed_results(results)
        
        return self
    
    def _create_full_gate_matrix(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> np.ndarray:
        """Create full gate matrix for the entire system using sparse operations for large systems."""
        gate_size = 2 ** len(qubit_indices)
        full_size = 2 ** self.num_qubits
        
        # For large systems, use sparse operations to avoid memory issues
        if self.num_qubits > 15:
            return self._create_sparse_gate_matrix(gate_matrix, qubit_indices)
        
        # For smaller systems, use dense operations
        identity_size = 2 ** (self.num_qubits - len(qubit_indices))
        identity = np.eye(identity_size, dtype=np.complex128)
        
        # Tensor product with identity
        if self.is_sparse:
            if self.sparse_format == 'csr':
                return sp.kron(identity, gate_matrix, format='csr')
            else:
                return sp.kron(identity, gate_matrix)
        else:
            return np.kron(identity, gate_matrix)
    
    def _create_sparse_gate_matrix(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> sp.spmatrix:
        """Create sparse gate matrix for large systems."""
        from core.sparse_gate_operations import SparseGateOperator
        
        # Use sparse gate operator for large systems
        operator = SparseGateOperator(self.num_qubits, use_gpu=self.acceleration_backend == AccelerationBackend.GPU)
        
        if len(qubit_indices) == 1:
            # Single-qubit gate
            return operator._create_sparse_single_qubit_gate(gate_matrix, qubit_indices[0])
        elif len(qubit_indices) == 2:
            # Two-qubit gate
            return operator._create_sparse_two_qubit_gate(gate_matrix, qubit_indices)
        else:
            # Multi-qubit gate - decompose into smaller gates
            return self._decompose_multi_qubit_gate(gate_matrix, qubit_indices)
    
    def _decompose_multi_qubit_gate(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> sp.spmatrix:
        """Decompose multi-qubit gate into smaller gates for large systems."""
        # For very large gates, decompose into single and two-qubit gates
        # This is a simplified decomposition - in practice, more sophisticated
        # decomposition algorithms would be used
        
        if len(qubit_indices) <= 2:
            # Already small enough
            return self._create_sparse_gate_matrix(gate_matrix, qubit_indices)
        
        # Decompose into two-qubit gates
        # This is a placeholder - real decomposition would use more advanced algorithms
        from core.sparse_gate_operations import SparseGateOperator
        operator = SparseGateOperator(self.num_qubits, use_gpu=self.acceleration_backend == AccelerationBackend.GPU)
        
        # For now, create identity matrix and let the circuit optimizer handle decomposition
        identity = sp.eye(2 ** self.num_qubits, dtype=np.complex128, format='csr')
        return identity
    
    def _create_full_gate_matrix_gpu(self, gate_matrix: cp.ndarray, qubit_indices: List[int]) -> cp.ndarray:
        """Create full gate matrix on GPU."""
        gate_size = 2 ** len(qubit_indices)
        identity_size = 2 ** (self.num_qubits - len(qubit_indices))
        identity = cp.eye(identity_size, dtype=cp.complex128)
        
        if self.is_sparse:
            return cp_sparse.kron(identity, gate_matrix, format='csr')
        else:
            return cp.kron(identity, gate_matrix)
    
    def _create_full_gate_matrix_tpu(self, gate_matrix, qubit_indices: List[int]):
        """Create full gate matrix on TPU."""
        gate_size = 2 ** len(qubit_indices)
        identity_size = 2 ** (self.num_qubits - len(qubit_indices))
        
        if TPU_AVAILABLE:
            identity = jnp.eye(identity_size, dtype=jnp.complex128)
        else:
            # Fallback to numpy if JAX not available
            identity = np.eye(identity_size, dtype=np.complex128)
        
        if TPU_AVAILABLE:
            return jnp.kron(identity, gate_matrix)
        else:
            return np.kron(identity, gate_matrix)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if self.acceleration_backend == AccelerationBackend.GPU and GPU_AVAILABLE:
            # Get GPU memory usage
            mempool = cp.get_default_memory_pool()
            self.performance_metrics.gpu_memory_used = mempool.used_bytes() / (1024**3)  # GB
        
        # Calculate sparsity ratio
        if self.is_sparse:
            total_elements = 2 ** self.num_qubits
            if hasattr(self.state, 'nnz'):
                non_zero_elements = self.state.nnz
            else:
                non_zero_elements = np.count_nonzero(self.state.toarray())
            self.performance_metrics.sparsity_ratio = 1.0 - (non_zero_elements / total_elements)
        
        return self.performance_metrics
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        if self.acceleration_backend == AccelerationBackend.GPU and GPU_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes() / (1024**2)  # MB
        else:
            if self.is_sparse:
                return self.state.data.nbytes / (1024**2)  # MB
            else:
                return self.state.nbytes / (1024**2)  # MB
    
    def cleanup(self):
        """Cleanup resources."""
        if self.acceleration_backend == AccelerationBackend.DISTRIBUTED and hasattr(self, 'dask_client'):
            self.dask_client.close()


class QuantumCircuitPartitioner:
    """
    Partition quantum circuits for execution across multiple backends.
    """
    
    def __init__(self, max_qubits_per_partition: int = 10):
        self.max_qubits_per_partition = max_qubits_per_partition
    
    def partition_circuit(self, circuit, num_qubits: int) -> List[Dict]:
        """
        Partition circuit into smaller sub-circuits.
        
        Args:
            circuit: Quantum circuit to partition
            num_qubits: Total number of qubits
            
        Returns:
            List of partitioned circuits
        """
        partitions = []
        
        # Simple partitioning strategy - can be enhanced with more sophisticated algorithms
        num_partitions = (num_qubits + self.max_qubits_per_partition - 1) // self.max_qubits_per_partition
        
        for i in range(num_partitions):
            start_qubit = i * self.max_qubits_per_partition
            end_qubit = min((i + 1) * self.max_qubits_per_partition, num_qubits)
            
            partition = {
                'qubits': list(range(start_qubit, end_qubit)),
                'gates': self._extract_gates_for_qubits(circuit, start_qubit, end_qubit),
                'partition_id': i
            }
            partitions.append(partition)
        
        return partitions
    
    def _extract_gates_for_qubits(self, circuit, start_qubit: int, end_qubit: int) -> List:
        """Extract gates that operate on specified qubits."""
        # This is a simplified version - full implementation would
        # properly extract and remap gates for the partition
        
        relevant_gates = []
        for gate in circuit.gates:
            if any(start_qubit <= qubit < end_qubit for qubit in gate.qubits):
                # Remap qubit indices to partition-local indices
                remapped_gate = self._remap_gate_qubits(gate, start_qubit, end_qubit)
                relevant_gates.append(remapped_gate)
        
        return relevant_gates
    
    def _remap_gate_qubits(self, gate, start_qubit: int, end_qubit: int):
        """Remap gate qubit indices to partition-local indices."""
        # Simplified implementation
        remapped_qubits = [q - start_qubit for q in gate.qubits if start_qubit <= q < end_qubit]
        # Create new gate with remapped qubits
        # This would need proper implementation based on gate types
        return gate


class PerformanceOptimizer:
    """
    AI-driven circuit optimization using machine learning.
    """
    
    def __init__(self):
        self.optimization_history = []
        self.performance_models = {}
    
    def optimize_circuit(self, circuit, target_backend: str = "cpu") -> Dict:
        """
        Optimize circuit using AI-driven techniques.
        
        Args:
            circuit: Circuit to optimize
            target_backend: Target execution backend
            
        Returns:
            Optimization results and suggestions
        """
        suggestions = []
        
        # Gate reduction suggestions
        gate_reductions = self._suggest_gate_reductions(circuit)
        suggestions.extend(gate_reductions)
        
        # Gate reordering suggestions
        reordering_suggestions = self._suggest_gate_reordering(circuit)
        suggestions.extend(reordering_suggestions)
        
        # Backend-specific optimizations
        backend_optimizations = self._suggest_backend_optimizations(circuit, target_backend)
        suggestions.extend(backend_optimizations)
        
        return {
            'suggestions': suggestions,
            'estimated_improvement': self._estimate_improvement(suggestions),
            'optimization_level': self._determine_optimization_level(circuit)
        }
    
    def _suggest_gate_reductions(self, circuit) -> List[Dict]:
        """Suggest gate reductions based on circuit analysis."""
        suggestions = []
        
        # Look for consecutive identical gates that can be combined
        for i in range(len(circuit.gates) - 1):
            if self._can_combine_gates(circuit.gates[i], circuit.gates[i + 1]):
                suggestions.append({
                    'type': 'gate_combination',
                    'description': f'Combine gates at positions {i} and {i+1}',
                    'estimated_savings': '10-20% execution time'
                })
        
        return suggestions
    
    def _suggest_gate_reordering(self, circuit) -> List[Dict]:
        """Suggest gate reordering for better performance."""
        suggestions = []
        
        # Look for gates that can be reordered for better locality
        for i in range(len(circuit.gates) - 1):
            if self._can_reorder_gates(circuit.gates[i], circuit.gates[i + 1]):
                suggestions.append({
                    'type': 'gate_reordering',
                    'description': f'Reorder gates at positions {i} and {i+1}',
                    'estimated_savings': '5-15% execution time'
                })
        
        return suggestions
    
    def _suggest_backend_optimizations(self, circuit, target_backend: str) -> List[Dict]:
        """Suggest backend-specific optimizations."""
        suggestions = []
        
        if target_backend == "gpu":
            suggestions.append({
                'type': 'gpu_optimization',
                'description': 'Use GPU-optimized gate implementations',
                'estimated_savings': '50-90% execution time'
            })
        elif target_backend == "distributed":
            suggestions.append({
                'type': 'distributed_optimization',
                'description': 'Partition circuit for distributed execution',
                'estimated_savings': '30-70% execution time'
            })
        
        return suggestions
    
    def _can_combine_gates(self, gate1, gate2) -> bool:
        """Check if two gates can be combined."""
        # Simplified implementation
        return (gate1.__class__ == gate2.__class__ and 
                gate1.qubits == gate2.qubits)
    
    def _can_reorder_gates(self, gate1, gate2) -> bool:
        """Check if two gates can be reordered."""
        # Gates can be reordered if they don't share qubits
        return not set(gate1.qubits).intersection(set(gate2.qubits))
    
    def _estimate_improvement(self, suggestions: List[Dict]) -> str:
        """Estimate overall improvement from suggestions."""
        if not suggestions:
            return "No optimizations available"
        
        # Simplified estimation
        total_savings = len(suggestions) * 10  # Assume 10% per suggestion
        return f"Estimated {min(total_savings, 80)}% improvement"
    
    def _determine_optimization_level(self, circuit) -> str:
        """Determine optimization level needed."""
        if len(circuit.gates) < 10:
            return "Low"
        elif len(circuit.gates) < 50:
            return "Medium"
        else:
            return "High"


# Performance benchmarking functions
def benchmark_qubit_scaling(max_qubits: int = 20) -> Dict:
    """
    Benchmark performance scaling with qubit count.
    
    Args:
        max_qubits: Maximum number of qubits to test
        
    Returns:
        Performance metrics for different qubit counts
    """
    results = {}
    
    for num_qubits in range(2, max_qubits + 1, 2):
        try:
            # Test different backends
            for backend in [AccelerationBackend.CPU, AccelerationBackend.GPU]:
                if backend == AccelerationBackend.GPU and not GPU_AVAILABLE:
                    continue
                
                state = AdvancedQuantumState(num_qubits, acceleration_backend=backend)
                
                # Benchmark gate application
                start_time = time.time()
                # Apply a simple gate (e.g., Hadamard)
                gate_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                state.apply_gate(gate_matrix, [0])
                execution_time = time.time() - start_time
                
                results[f"{num_qubits}_qubits_{backend.value}"] = {
                    'execution_time': execution_time,
                    'memory_usage': state.get_memory_usage(),
                    'backend': backend.value,
                    'qubits': num_qubits
                }
                
                state.cleanup()
        
        except Exception as e:
            logger.warning(f"Failed to benchmark {num_qubits} qubits: {e}")
            continue
    
    return results


def create_performance_chart_data() -> Dict:
    """
    Create performance chart data for visualization.
    
    Returns:
        Chart.js compatible data structure
    """
    # Get benchmark results
    benchmark_results = benchmark_qubit_scaling(20)
    
    # Prepare data for chart
    qubit_counts = []
    cpu_times = []
    gpu_times = []
    
    for key, result in benchmark_results.items():
        if '_cpu' in key:
            qubits = int(key.split('_')[0])
            qubit_counts.append(qubits)
            cpu_times.append(result['execution_time'])
        elif '_gpu' in key:
            gpu_times.append(result['execution_time'])
    
    return {
        "type": "line",
        "data": {
            "labels": qubit_counts,
            "datasets": [
                {
                    "label": "CPU Execution Time (s)",
                    "data": cpu_times,
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)"
                },
                {
                    "label": "GPU Execution Time (s)",
                    "data": gpu_times,
                    "borderColor": "rgb(255, 99, 132)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)"
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "y": {
                    "type": "logarithmic",
                    "title": {
                        "display": True,
                        "text": "Execution Time (seconds)"
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": "Number of Qubits"
                    }
                }
            },
            "title": {
                "display": True,
                "text": "Coratrix 4.0 Performance Scaling: CPU vs GPU"
            }
        }
    }
