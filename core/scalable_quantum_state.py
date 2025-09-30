"""
Scalable quantum state representation with sparse matrices and GPU acceleration.

This module provides efficient quantum state representation for large systems
using sparse matrices and optional GPU acceleration for performance.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Union, Optional, Dict, Any
import math
import warnings
import time
import psutil
import os

# Optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration disabled.")

# Optional sparse matrix optimization
try:
    import scipy.sparse.linalg as spla
    SPARSE_LINALG_AVAILABLE = True
except ImportError:
    SPARSE_LINALG_AVAILABLE = False


class ScalableQuantumState:
    """
    Scalable quantum state representation for n-qubit systems.
    
    Supports both CPU and GPU computation with sparse matrix operations
    for efficient memory usage in large quantum systems.
    """
    
    def __init__(self, num_qubits: int, use_gpu: bool = False, use_sparse: bool = None, 
                 sparse_threshold: int = 8, sparse_format: str = 'csr', deterministic: bool = True):
        """
        Initialize a scalable quantum state.
        
        Args:
            num_qubits: Number of qubits in the system
            use_gpu: Whether to use GPU acceleration (requires CuPy)
            use_sparse: Whether to use sparse representation (deprecated, use sparse_threshold instead)
            sparse_threshold: Minimum number of qubits to use sparse representation
            sparse_format: Sparse matrix format ('csr', 'coo', 'lil')
            deterministic: Whether to use deterministic operations for reproducibility
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.sparse_threshold = sparse_threshold
        self.sparse_format = sparse_format
        self.deterministic = deterministic
        
        # Handle deprecated use_sparse parameter
        if use_sparse is not None:
            import warnings
            warnings.warn("use_sparse parameter is deprecated. Use sparse_threshold instead.", 
                        DeprecationWarning, stacklevel=2)
            self.use_sparse = use_sparse
        else:
            self.use_sparse = num_qubits >= sparse_threshold
        
        # Set random seed for reproducibility if deterministic
        if self.deterministic:
            np.random.seed(42)
            if self.use_gpu and GPU_AVAILABLE:
                cp.random.seed(42)
        
        # Initialize state vector with optimized representation
        if self.use_gpu:
            self.state_vector = cp.zeros(self.dimension, dtype=cp.complex128)
        else:
            if self.use_sparse:
                # Use optimized sparse representation for large systems
                if sparse_format == 'csr':
                    self.state_vector = sp.csr_matrix((1, self.dimension), dtype=complex)
                elif sparse_format == 'coo':
                    self.state_vector = sp.coo_matrix((1, self.dimension), dtype=complex)
                else:  # lil
                    self.state_vector = sp.lil_matrix((1, self.dimension), dtype=complex)
            else:
                self.state_vector = np.zeros(self.dimension, dtype=complex)
        
        # Set initial state |00...0⟩
        self._set_amplitude(0, 1.0)
        
        # Performance metrics
        self.operation_count = 0
        self.memory_usage = self._calculate_memory_usage()
        self.start_time = time.time()
        
        # GPU memory management
        if self.use_gpu and GPU_AVAILABLE:
            self.gpu_memory_pool = cp.get_default_memory_pool()
            self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
    
    def _set_amplitude(self, state_index: int, amplitude: complex):
        """Set amplitude for a specific state index."""
        if self.use_gpu:
            self.state_vector[state_index] = amplitude
        elif self.use_sparse:
            # For sparse matrices, use LIL format for efficient element setting
            if self.sparse_format == 'csr':
                # Convert to LIL for efficient setting, then back to CSR
                if not hasattr(self, '_lil_matrix') or self._lil_matrix is None:
                    self._lil_matrix = self.state_vector.tolil()
                self._lil_matrix[0, state_index] = amplitude
                self.state_vector = self._lil_matrix.tocsr()
            else:
                self.state_vector[0, state_index] = amplitude
        else:
            self.state_vector[state_index] = amplitude
    
    def _get_amplitude(self, state_index: int) -> complex:
        """Get amplitude for a specific state index."""
        if self.use_gpu:
            return complex(self.state_vector[state_index])
        elif self.use_sparse:
            return complex(self.state_vector[0, state_index])
        else:
            return self.state_vector[state_index]
    
    def get_amplitude(self, state_index: int) -> complex:
        """Get the amplitude for a specific computational basis state."""
        if 0 <= state_index < self.dimension:
            return self._get_amplitude(state_index)
        else:
            raise IndexError(f"State index {state_index} out of range [0, {self.dimension-1}]")
    
    def set_amplitude(self, state_index: int, amplitude: complex):
        """Set the amplitude for a specific computational basis state."""
        if 0 <= state_index < self.dimension:
            self._set_amplitude(state_index, amplitude)
        else:
            raise IndexError(f"State index {state_index} out of range [0, {self.dimension-1}]")
    
    def normalize(self):
        """Normalize the state vector to ensure probabilities sum to 1."""
        if self.use_gpu:
            # GPU-optimized normalization
            norm = cp.sqrt(cp.sum(cp.abs(self.state_vector)**2))
            if norm > 0:
                self.state_vector /= norm
        elif self.use_sparse:
            # Optimized sparse normalization
            if self.sparse_format == 'csr':
                # Use sparse operations for CSR format
                data = np.array(self.state_vector.data)
                norm = np.sqrt(np.sum(np.abs(data)**2))
                if norm > 0:
                    self.state_vector.data = data / norm
            else:
                # Convert to dense for other sparse formats
                dense_vector = self.state_vector.toarray().flatten()
                norm = np.sqrt(np.sum(np.abs(dense_vector)**2))
                if norm > 0:
                    dense_vector /= norm
                    # Convert back to sparse format
                    if self.sparse_format == 'coo':
                        self.state_vector = sp.coo_matrix(dense_vector.reshape(1, -1))
                    else:  # lil
                        self.state_vector = sp.lil_matrix(dense_vector.reshape(1, -1))
        else:
            # Dense vector normalization
            norm = np.sqrt(np.sum(np.abs(self.state_vector)**2))
            if norm > 0:
                self.state_vector /= norm
    
    def get_probabilities(self) -> np.ndarray:
        """Get the probability distribution over all basis states."""
        if self.use_gpu:
            probs = cp.abs(self.state_vector)**2
            return cp.asnumpy(probs)
        elif self.use_sparse:
            dense_vector = self.state_vector.toarray().flatten()
            return np.abs(dense_vector)**2
        else:
            return np.abs(self.state_vector)**2
    
    def get_state_index(self, qubit_states: List[int]) -> int:
        """Convert qubit states to state vector index."""
        if len(qubit_states) != self.num_qubits:
            raise ValueError(f"Expected {self.num_qubits} qubit states, got {len(qubit_states)}")
        
        index = 0
        for i, state in enumerate(qubit_states):
            if state not in [0, 1]:
                raise ValueError(f"Qubit state must be 0 or 1, got {state}")
            index += state * (2 ** (self.num_qubits - 1 - i))
        
        return index
    
    def get_qubit_states(self, state_index: int) -> List[int]:
        """Convert state vector index to qubit states."""
        if not (0 <= state_index < self.dimension):
            raise IndexError(f"State index {state_index} out of range [0, {self.dimension-1}]")
        
        binary = format(state_index, f'0{self.num_qubits}b')
        return [int(bit) for bit in binary]
    
    def get_entanglement_entropy(self) -> float:
        """
        Calculate the entanglement entropy of the quantum state.
        
        For a pure state, this measures the entanglement between subsystems.
        """
        if self.num_qubits < 2:
            return 0.0
        
        # For simplicity, calculate entropy of the first qubit
        # by tracing out the other qubits
        probs = self.get_probabilities()
        
        # Calculate reduced density matrix for first qubit
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, prob in enumerate(probs):
            if i < self.dimension // 2:  # First qubit is 0
                prob_0 += prob
            else:  # First qubit is 1
                prob_1 += prob
        
        # Calculate von Neumann entropy
        if prob_0 > 0 and prob_1 > 0:
            entropy = -prob_0 * math.log2(prob_0) - prob_1 * math.log2(prob_1)
            return float(entropy)
        else:
            return 0.0
    
    def is_separable(self) -> bool:
        """
        Check if the quantum state is separable (not entangled).
        
        This is a simplified check for 2-qubit systems.
        """
        if self.num_qubits != 2:
            return True  # Only check 2-qubit separability for now
        
        # Get state vector
        if self.use_gpu:
            state_vector = cp.asnumpy(self.state_vector)
        elif self.use_sparse:
            state_vector = self.state_vector.toarray().flatten()
        else:
            state_vector = self.state_vector
        
        # Check if state can be written as |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩
        # For a 2-qubit system, this means checking if the state vector
        # can be factored into a product of single-qubit states
        
        # Simple heuristic: check if the state has the form
        # [a₁b₁, a₁b₂, a₂b₁, a₂b₂] where |ψ₁⟩ = [a₁, a₂] and |ψ₂⟩ = [b₁, b₂]
        
        # If any of the amplitudes are zero in a specific pattern, it might be separable
        if (abs(state_vector[0]) < 1e-10 and abs(state_vector[3]) < 1e-10) or \
           (abs(state_vector[1]) < 1e-10 and abs(state_vector[2]) < 1e-10):
            return True
        
        # If the state has non-zero amplitudes in a pattern that suggests entanglement
        # (like Bell states), it's likely entangled
        return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        if self.use_gpu:
            memory_bytes = self.state_vector.nbytes
        elif self.use_sparse:
            # For LIL matrix, calculate memory usage differently
            memory_bytes = self.state_vector.data.nbytes
            if hasattr(self.state_vector, 'indices'):
                memory_bytes += self.state_vector.indices.nbytes
            if hasattr(self.state_vector, 'indptr'):
                memory_bytes += self.state_vector.indptr.nbytes
            # For LIL matrix, also account for row data
            if hasattr(self.state_vector, 'rows'):
                memory_bytes += sum(len(row) * 8 for row in self.state_vector.rows)  # 8 bytes per index
        else:
            memory_bytes = self.state_vector.nbytes
        
        return {
            'memory_bytes': memory_bytes,
            'memory_mb': memory_bytes / (1024 * 1024),
            'use_gpu': self.use_gpu,
            'use_sparse': self.use_sparse,
            'dimension': self.dimension,
            'num_qubits': self.num_qubits
        }
    
    def _calculate_memory_usage(self) -> int:
        """Calculate current memory usage in bytes."""
        usage_info = self.get_memory_usage()
        return usage_info['memory_bytes']
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the quantum state."""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        metrics = {
            'operation_count': self.operation_count,
            'memory_usage': self.get_memory_usage(),
            'use_gpu': self.use_gpu,
            'use_sparse': self.use_sparse,
            'sparse_format': self.sparse_format,
            'dimension': self.dimension,
            'num_qubits': self.num_qubits,
            'runtime_seconds': runtime,
            'operations_per_second': self.operation_count / max(runtime, 1e-6)
        }
        
        # Add GPU-specific metrics
        if self.use_gpu and GPU_AVAILABLE:
            try:
                metrics['gpu_memory_used_mb'] = self.gpu_memory_pool.used_bytes() / (1024 * 1024)
                metrics['gpu_memory_total_mb'] = self.gpu_memory_pool.total_bytes() / (1024 * 1024)
                metrics['gpu_memory_utilization'] = self.gpu_memory_pool.used_bytes() / max(self.gpu_memory_pool.total_bytes(), 1)
            except:
                metrics['gpu_memory_used_mb'] = 0
                metrics['gpu_memory_total_mb'] = 0
                metrics['gpu_memory_utilization'] = 0
        
        # Add system metrics
        try:
            process = psutil.Process(os.getpid())
            metrics['cpu_percent'] = process.cpu_percent()
            metrics['memory_percent'] = process.memory_percent()
            metrics['system_memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
        except:
            metrics['cpu_percent'] = 0
            metrics['memory_percent'] = 0
            metrics['system_memory_available_gb'] = 0
        
        return metrics
    
    def __str__(self) -> str:
        """String representation of the quantum state."""
        if self.use_gpu:
            state_vector = cp.asnumpy(self.state_vector)
        elif self.use_sparse:
            state_vector = self.state_vector.toarray().flatten()
        else:
            state_vector = self.state_vector
        
        result = []
        for i, amplitude in enumerate(state_vector):
            if abs(amplitude) > 1e-10:  # Only show non-zero amplitudes
                binary = format(i, f'0{self.num_qubits}b')
                result.append(f"{amplitude:.3f}|{binary}⟩")
        
        return " + ".join(result) if result else "|0⟩"
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array representation."""
        if self.use_gpu:
            return cp.asnumpy(self.state_vector)
        elif self.use_sparse:
            return self.state_vector.toarray().flatten()
        else:
            return self.state_vector.copy()
    
    def from_dense(self, state_vector: np.ndarray):
        """Load state from dense numpy array."""
        if len(state_vector) != self.dimension:
            raise ValueError(f"State vector length {len(state_vector)} doesn't match dimension {self.dimension}")
        
        if self.use_gpu:
            self.state_vector = cp.asarray(state_vector)
        elif self.use_sparse:
            if self.sparse_format == 'csr':
                self.state_vector = sp.csr_matrix(state_vector.reshape(1, -1))
            elif self.sparse_format == 'coo':
                self.state_vector = sp.coo_matrix(state_vector.reshape(1, -1))
            else:  # lil
                self.state_vector = sp.lil_matrix(state_vector.reshape(1, -1))
        else:
            self.state_vector = state_vector.copy()
        
        self.normalize()
    
    def optimize_memory(self):
        """Optimize memory usage by converting between formats."""
        if self.use_gpu and GPU_AVAILABLE:
            # Free unused GPU memory
            self.gpu_memory_pool.free_all_blocks()
            self.pinned_memory_pool.free_all_blocks()
        elif self.use_sparse:
            # Convert to most efficient sparse format
            if self.sparse_format == 'lil':
                # LIL is good for construction, CSR is good for operations
                self.state_vector = self.state_vector.tocsr()
                self.sparse_format = 'csr'
            elif self.sparse_format == 'coo':
                # COO is good for construction, CSR is good for operations
                self.state_vector = self.state_vector.tocsr()
                self.sparse_format = 'csr'
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache to free up memory."""
        if self.use_gpu and GPU_AVAILABLE:
            self.gpu_memory_pool.free_all_blocks()
            self.pinned_memory_pool.free_all_blocks()
    
    def get_sparsity_ratio(self) -> float:
        """Get the sparsity ratio of the state vector."""
        if self.use_gpu:
            non_zero = cp.count_nonzero(self.state_vector)
            return float(non_zero) / self.dimension
        elif self.use_sparse:
            return self.state_vector.nnz / self.dimension
        else:
            return np.count_nonzero(self.state_vector) / self.dimension
    
    def is_sparse_optimal(self) -> bool:
        """Check if sparse representation is optimal for current state."""
        sparsity = self.get_sparsity_ratio()
        # Use sparse if less than 10% of elements are non-zero
        return sparsity < 0.1
    
    def get_density_matrix(self) -> np.ndarray:
        """Get the density matrix of the quantum state."""
        if self.use_gpu:
            # Convert to CPU for density matrix calculation
            state_vector = cp.asnumpy(self.state_vector)
        elif self.use_sparse:
            # Convert sparse to dense
            state_vector = self.state_vector.toarray().flatten()
        else:
            state_vector = self.state_vector
        
        # Calculate density matrix: ρ = |ψ⟩⟨ψ|
        psi = state_vector.reshape(-1, 1)
        rho = np.outer(psi, psi.conj())
        return rho
    
    def auto_optimize_format(self):
        """Automatically optimize the representation format."""
        if self.use_sparse and not self.is_sparse_optimal():
            # Convert to dense if sparse is not optimal
            self.state_vector = self.to_dense()
            self.use_sparse = False
        elif not self.use_sparse and self.is_sparse_optimal():
            # Convert to sparse if dense is not optimal
            if self.sparse_format == 'csr':
                self.state_vector = sp.csr_matrix(self.state_vector.reshape(1, -1))
            elif self.sparse_format == 'coo':
                self.state_vector = sp.coo_matrix(self.state_vector.reshape(1, -1))
            else:
                self.state_vector = sp.lil_matrix(self.state_vector.reshape(1, -1))
            self.use_sparse = True
    
    def apply_gate(self, gate, target_qubits: List[int]):
        """Apply a quantum gate to the state vector."""
        # Get the gate matrix
        gate_matrix = gate.get_matrix(self.num_qubits, target_qubits)
        
        if self.use_gpu:
            # GPU-optimized gate application
            gate_matrix_gpu = cp.asarray(gate_matrix)
            self.state_vector = gate_matrix_gpu @ self.state_vector
        elif self.use_sparse:
            # Sparse-optimized gate application
            # Convert to dense for matrix multiplication, then back to sparse
            dense_state = self.to_dense()
            result = gate_matrix @ dense_state
            self.from_dense(result)
        else:
            # Dense matrix multiplication
            self.state_vector = gate_matrix @ self.state_vector
        
        self.operation_count += 1
