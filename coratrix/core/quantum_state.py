"""
Scalable Quantum State Implementation

This module provides the core quantum state representation with support for:
- Dense and sparse matrix representations
- GPU acceleration via CuPy
- Memory-efficient large system simulation
- Automatic format optimization
"""

import numpy as np
import scipy.sparse as sp
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class StateConfig:
    """Configuration for quantum state representation."""
    use_gpu: bool = False
    use_sparse: bool = False
    sparse_threshold: int = 8
    memory_limit_gb: float = 8.0
    optimization_level: int = 2


class ScalableQuantumState:
    """
    Scalable quantum state representation with multiple backends.
    
    Supports:
    - Dense NumPy arrays for small systems
    - Sparse matrices (CSR, COO, LIL) for large systems
    - GPU acceleration via CuPy
    - Automatic format optimization
    """
    
    def __init__(self, num_qubits: int, config: Optional[StateConfig] = None, 
                 use_gpu: bool = False, use_sparse: bool = False, sparse_threshold: int = 8):
        """
        Initialize a quantum state.
        
        Args:
            num_qubits: Number of qubits in the system
            config: Configuration for state representation
            use_gpu: Whether to use GPU acceleration
            use_sparse: Whether to use sparse representation
            sparse_threshold: Threshold for sparse representation
        """
        self.num_qubits = num_qubits
        self.config = config or StateConfig()
        self.state_vector = None
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.use_sparse = use_sparse
        self.sparse_threshold = sparse_threshold
        
        # Performance metrics
        self.operation_count = 0
        self.last_operation_time = 0.0
        self.memory_usage = 0.0
        
        # Initialize state
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize the quantum state vector."""
        dim = 2 ** self.num_qubits
        
        if self.use_gpu:
            self.state_vector = cp.zeros(dim, dtype=cp.complex128)
            self.state_vector[0] = 1.0  # |00...0⟩
        elif self.use_sparse and self.num_qubits >= self.sparse_threshold:
            # Use sparse representation for large systems
            self.state_vector = sp.lil_matrix((dim, 1), dtype=np.complex128)
            self.state_vector[0, 0] = 1.0
        else:
            # Use dense representation
            self.state_vector = np.zeros(dim, dtype=np.complex128)
            self.state_vector[0] = 1.0
    
    def get_amplitude(self, index: int) -> complex:
        """Get the amplitude of a specific basis state."""
        if self.use_gpu:
            return complex(self.state_vector[index])
        elif self.use_sparse:
            return complex(self.state_vector[index, 0])
        else:
            return complex(self.state_vector[index])
    
    def set_amplitude(self, index: int, amplitude: complex):
        """Set the amplitude of a specific basis state."""
        if self.use_gpu:
            self.state_vector[index] = amplitude
        elif self.use_sparse:
            self.state_vector[index, 0] = amplitude
        else:
            self.state_vector[index] = amplitude
    
    def apply_gate(self, gate: 'QuantumGate', target_qubits: List[int]):
        """Apply a quantum gate to the state."""
        start_time = time.time()
        
        # Get gate matrix
        gate_matrix = gate.get_matrix(self.num_qubits, target_qubits)
        
        # Apply gate
        if self.use_gpu:
            gate_matrix_gpu = cp.asarray(gate_matrix)
            self.state_vector = gate_matrix_gpu @ self.state_vector
        elif self.use_sparse:
            gate_matrix_sparse = sp.csr_matrix(gate_matrix)
            self.state_vector = gate_matrix_sparse @ self.state_vector
        else:
            self.state_vector = gate_matrix @ self.state_vector
        
        # Update metrics
        self.operation_count += 1
        self.last_operation_time = time.time() - start_time
        self._update_memory_usage()
    
    def normalize(self):
        """Normalize the quantum state."""
        if self.use_gpu:
            norm = cp.linalg.norm(self.state_vector)
            if norm > 1e-10:
                self.state_vector /= norm
        elif self.use_sparse:
            norm = np.linalg.norm(self.state_vector.data)
            if norm > 1e-10:
                self.state_vector /= norm
        else:
            norm = np.linalg.norm(self.state_vector)
            if norm > 1e-10:
                self.state_vector /= norm
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states."""
        if self.use_gpu:
            probs = cp.abs(self.state_vector) ** 2
            return cp.asnumpy(probs)
        elif self.use_sparse:
            probs = np.abs(self.state_vector.toarray().flatten()) ** 2
            return probs
        else:
            return np.abs(self.state_vector) ** 2
    
    def measure(self, qubit: int) -> int:
        """Measure a single qubit and return the result."""
        # Calculate measurement probabilities
        probs = self.get_probabilities()
        
        # Find states where the qubit is |0⟩ and |1⟩
        qubit_mask = 1 << (self.num_qubits - 1 - qubit)
        prob_0 = np.sum(probs[probs & qubit_mask == 0])
        prob_1 = np.sum(probs[probs & qubit_mask != 0])
        
        # Normalize probabilities
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # Sample measurement result
        result = np.random.choice([0, 1], p=[prob_0, prob_1])
        
        # Collapse state
        self._collapse_state(qubit, result)
        
        return result
    
    def _collapse_state(self, qubit: int, result: int):
        """Collapse the state after measurement."""
        qubit_mask = 1 << (self.num_qubits - 1 - qubit)
        
        if self.use_gpu:
            # Set amplitudes to zero for states that don't match measurement
            for i in range(len(self.state_vector)):
                if ((i & qubit_mask) != 0) == (result == 1):
                    self.state_vector[i] = 0
        elif self.use_sparse:
            # For sparse matrices, we need to rebuild
            new_state = sp.lil_matrix(self.state_vector.shape, dtype=np.complex128)
            for i in range(self.state_vector.shape[0]):
                if ((i & qubit_mask) != 0) == (result == 1):
                    new_state[i, 0] = self.state_vector[i, 0]
            self.state_vector = new_state
        else:
            # Set amplitudes to zero for states that don't match measurement
            for i in range(len(self.state_vector)):
                if ((i & qubit_mask) != 0) == (result == 1):
                    self.state_vector[i] = 0
        
        # Renormalize
        self.normalize()
    
    def get_density_matrix(self) -> np.ndarray:
        """Get the density matrix of the quantum state."""
        if self.use_gpu:
            state_vector = cp.asnumpy(self.state_vector)
        elif self.use_sparse:
            state_vector = self.state_vector.toarray().flatten()
        else:
            state_vector = self.state_vector
        
        # Calculate density matrix: ρ = |ψ⟩⟨ψ|
        psi = state_vector.reshape(-1, 1)
        rho = np.outer(psi, psi.conj())
        return rho
    
    def _update_memory_usage(self):
        """Update memory usage metrics."""
        if self.use_gpu:
            self.memory_usage = self.state_vector.nbytes / (1024**3)  # GB
        elif self.use_sparse:
            self.memory_usage = self.state_vector.data.nbytes / (1024**3)
        else:
            self.memory_usage = self.state_vector.nbytes / (1024**3)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'operation_count': self.operation_count,
            'last_operation_time': self.last_operation_time,
            'memory_usage_gb': self.memory_usage,
            'use_gpu': self.use_gpu,
            'use_sparse': self.use_sparse,
            'num_qubits': self.num_qubits
        }
    
    def __str__(self) -> str:
        """String representation of the quantum state."""
        return f"ScalableQuantumState(qubits={self.num_qubits}, gpu={self.use_gpu}, sparse={self.use_sparse})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ScalableQuantumState(num_qubits={self.num_qubits}, config={self.config})"
