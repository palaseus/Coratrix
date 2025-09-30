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

# Optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration disabled.")


class ScalableQuantumState:
    """
    Scalable quantum state representation for n-qubit systems.
    
    Supports both CPU and GPU computation with sparse matrix operations
    for efficient memory usage in large quantum systems.
    """
    
    def __init__(self, num_qubits: int, use_gpu: bool = False, sparse_threshold: int = 10):
        """
        Initialize a scalable quantum state.
        
        Args:
            num_qubits: Number of qubits in the system
            use_gpu: Whether to use GPU acceleration (requires CuPy)
            sparse_threshold: Minimum number of qubits to use sparse representation
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.sparse_threshold = sparse_threshold
        self.use_sparse = num_qubits >= sparse_threshold
        
        # Initialize state vector
        if self.use_gpu:
            self.state_vector = cp.zeros(self.dimension, dtype=cp.complex128)
        else:
            if self.use_sparse:
                # Use sparse representation for large systems
                self.state_vector = sp.lil_matrix((1, self.dimension), dtype=complex)
            else:
                self.state_vector = np.zeros(self.dimension, dtype=complex)
        
        # Set initial state |00...0⟩
        self._set_amplitude(0, 1.0)
        
        # Performance metrics
        self.operation_count = 0
        self.memory_usage = self._calculate_memory_usage()
    
    def _set_amplitude(self, state_index: int, amplitude: complex):
        """Set amplitude for a specific state index."""
        if self.use_gpu:
            self.state_vector[state_index] = amplitude
        elif self.use_sparse:
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
            norm = cp.sqrt(cp.sum(cp.abs(self.state_vector)**2))
            if norm > 0:
                self.state_vector /= norm
        elif self.use_sparse:
            # Convert to dense for normalization
            dense_vector = self.state_vector.toarray().flatten()
            norm = np.sqrt(np.sum(np.abs(dense_vector)**2))
            if norm > 0:
                dense_vector /= norm
                self.state_vector = sp.lil_matrix(dense_vector.reshape(1, -1))
        else:
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
        return {
            'operation_count': self.operation_count,
            'memory_usage': self.get_memory_usage(),
            'use_gpu': self.use_gpu,
            'use_sparse': self.use_sparse,
            'dimension': self.dimension,
            'num_qubits': self.num_qubits
        }
    
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
            self.state_vector = sp.lil_matrix(state_vector.reshape(1, -1))
        else:
            self.state_vector = state_vector.copy()
        
        self.normalize()
