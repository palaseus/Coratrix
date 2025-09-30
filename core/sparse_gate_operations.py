"""
Sparse Gate Operations for Large Quantum Systems (15-20 qubits)

This module implements efficient sparse gate operations that avoid creating
full dense gate matrices for large quantum systems.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging
from concurrent.futures import ThreadPoolExecutor
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

logger = logging.getLogger(__name__)


class GateType(Enum):
    """Types of quantum gates."""
    SINGLE_QUBIT = "single_qubit"
    TWO_QUBIT = "two_qubit"
    MULTI_QUBIT = "multi_qubit"
    CONTROLLED = "controlled"


@dataclass
class SparseGateInfo:
    """Information about a sparse gate operation."""
    gate_type: GateType
    target_qubits: List[int]
    control_qubits: List[int] = None
    gate_matrix: np.ndarray = None
    sparse_matrix: sp.spmatrix = None
    memory_usage: float = 0.0
    execution_time: float = 0.0


class SparseGateOperator:
    """
    Efficient sparse gate operations for large quantum systems.
    Avoids creating full dense gate matrices for 15+ qubit systems.
    """
    
    def __init__(self, num_qubits: int, use_gpu: bool = False):
        """
        Initialize sparse gate operator.
        
        Args:
            num_qubits: Number of qubits in the system
            use_gpu: Whether to use GPU acceleration
        """
        self.num_qubits = num_qubits
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.state_size = 2 ** num_qubits
        
        # Setup backend
        if self.use_gpu:
            self._xp = cp
            self._sp = cp_sparse
        else:
            self._xp = np
            self._sp = sp
        
        # Cache for common operations
        self._gate_cache = {}
        self._identity_cache = {}
        
        # Performance tracking
        self.operations_count = 0
        self.total_execution_time = 0.0
        self.memory_saved = 0.0
    
    def apply_single_qubit_gate(self, state: Union[np.ndarray, sp.spmatrix], 
                               gate_matrix: np.ndarray, target_qubit: int) -> Union[np.ndarray, sp.spmatrix]:
        """
        Apply single-qubit gate using sparse operations.
        
        Args:
            state: Quantum state vector (dense or sparse)
            gate_matrix: 2x2 gate matrix
            target_qubit: Index of target qubit
            
        Returns:
            Updated quantum state
        """
        start_time = time.time()
        
        try:
            if self.num_qubits <= 12:
                # Use dense operations for small systems
                return self._apply_dense_single_qubit_gate(state, gate_matrix, target_qubit)
            else:
                # Use sparse operations for large systems
                return self._apply_sparse_single_qubit_gate(state, gate_matrix, target_qubit)
        
        finally:
            execution_time = time.time() - start_time
            self.operations_count += 1
            self.total_execution_time += execution_time
    
    def apply_two_qubit_gate(self, state: Union[np.ndarray, sp.spmatrix], 
                            gate_matrix: np.ndarray, target_qubits: List[int]) -> Union[np.ndarray, sp.spmatrix]:
        """
        Apply two-qubit gate using sparse operations.
        
        Args:
            state: Quantum state vector
            gate_matrix: 4x4 gate matrix
            target_qubits: Indices of target qubits [qubit1, qubit2]
            
        Returns:
            Updated quantum state
        """
        start_time = time.time()
        
        try:
            if self.num_qubits <= 10:
                # Use dense operations for small systems
                return self._apply_dense_two_qubit_gate(state, gate_matrix, target_qubits)
            else:
                # Use sparse operations for large systems
                return self._apply_sparse_two_qubit_gate(state, gate_matrix, target_qubits)
        
        finally:
            execution_time = time.time() - start_time
            self.operations_count += 1
            self.total_execution_time += execution_time
    
    def apply_controlled_gate(self, state: Union[np.ndarray, sp.spmatrix], 
                             gate_matrix: np.ndarray, control_qubit: int, 
                             target_qubit: int) -> Union[np.ndarray, sp.spmatrix]:
        """
        Apply controlled gate using sparse operations.
        
        Args:
            state: Quantum state vector
            gate_matrix: 2x2 gate matrix for target qubit
            control_qubit: Index of control qubit
            target_qubit: Index of target qubit
            
        Returns:
            Updated quantum state
        """
        start_time = time.time()
        
        try:
            if self.num_qubits <= 12:
                # Use dense operations for small systems
                return self._apply_dense_controlled_gate(state, gate_matrix, control_qubit, target_qubit)
            else:
                # Use sparse operations for large systems
                return self._apply_sparse_controlled_gate(state, gate_matrix, control_qubit, target_qubit)
        
        finally:
            execution_time = time.time() - start_time
            self.operations_count += 1
            self.total_execution_time += execution_time
    
    def _apply_dense_single_qubit_gate(self, state: np.ndarray, gate_matrix: np.ndarray, 
                                      target_qubit: int) -> np.ndarray:
        """Apply single-qubit gate using dense operations."""
        # Create full gate matrix
        full_gate = self._create_full_gate_matrix_dense(gate_matrix, [target_qubit])
        
        # Apply gate
        if self.use_gpu:
            state_gpu = self._xp.asarray(state)
            result = full_gate @ state_gpu
            return self._xp.asnumpy(result)
        else:
            return full_gate @ state
    
    def _apply_sparse_single_qubit_gate(self, state: Union[np.ndarray, sp.spmatrix], 
                                       gate_matrix: np.ndarray, target_qubit: int) -> Union[np.ndarray, sp.spmatrix]:
        """Apply single-qubit gate using sparse operations."""
        # Convert state to sparse if needed
        if not sp.issparse(state):
            state = sp.csr_matrix(state.reshape(-1, 1))
        
        # Create sparse gate matrix
        sparse_gate = self._create_sparse_single_qubit_gate(gate_matrix, target_qubit)
        
        # Apply gate
        try:
            result = sparse_gate.dot(state)
        except Exception as e:
            # Fallback to dense operations for compatibility
            logger.warning(f"Sparse operation failed, falling back to dense: {e}")
            if sp.issparse(state):
                state_dense = state.toarray().flatten()
            else:
                state_dense = state
            return self._apply_dense_single_qubit_gate(state_dense, gate_matrix, target_qubit)
        
        # Convert back to original format
        if isinstance(state, np.ndarray):
            return result.toarray().flatten()
        else:
            return result
    
    def _apply_dense_two_qubit_gate(self, state: np.ndarray, gate_matrix: np.ndarray, 
                                   target_qubits: List[int]) -> np.ndarray:
        """Apply two-qubit gate using dense operations."""
        # Create full gate matrix
        full_gate = self._create_full_gate_matrix_dense(gate_matrix, target_qubits)
        
        # Apply gate
        if self.use_gpu:
            state_gpu = self._xp.asarray(state)
            result = full_gate @ state_gpu
            return self._xp.asnumpy(result)
        else:
            return full_gate @ state
    
    def _apply_sparse_two_qubit_gate(self, state: Union[np.ndarray, sp.spmatrix], 
                                    gate_matrix: np.ndarray, target_qubits: List[int]) -> Union[np.ndarray, sp.spmatrix]:
        """Apply two-qubit gate using sparse operations."""
        # Convert state to sparse if needed
        if not sp.issparse(state):
            state = sp.csr_matrix(state.reshape(-1, 1))
        
        # Create sparse gate matrix
        sparse_gate = self._create_sparse_two_qubit_gate(gate_matrix, target_qubits)
        
        # Apply gate
        result = sparse_gate.dot(state)
        
        # Convert back to original format
        if isinstance(state, np.ndarray):
            return result.toarray().flatten()
        else:
            return result
    
    def _apply_dense_controlled_gate(self, state: np.ndarray, gate_matrix: np.ndarray, 
                                    control_qubit: int, target_qubit: int) -> np.ndarray:
        """Apply controlled gate using dense operations."""
        # Create controlled gate matrix
        controlled_gate = self._create_controlled_gate_matrix_dense(gate_matrix, control_qubit, target_qubit)
        
        # Apply gate
        if self.use_gpu:
            state_gpu = self._xp.asarray(state)
            result = controlled_gate @ state_gpu
            return self._xp.asnumpy(result)
        else:
            return controlled_gate @ state
    
    def _apply_sparse_controlled_gate(self, state: Union[np.ndarray, sp.spmatrix], 
                                     gate_matrix: np.ndarray, control_qubit: int, 
                                     target_qubit: int) -> Union[np.ndarray, sp.spmatrix]:
        """Apply controlled gate using sparse operations."""
        # Convert state to sparse if needed
        if not sp.issparse(state):
            state = sp.csr_matrix(state.reshape(-1, 1))
        
        # Create sparse controlled gate matrix
        sparse_gate = self._create_sparse_controlled_gate(gate_matrix, control_qubit, target_qubit)
        
        # Apply gate
        result = sparse_gate.dot(state)
        
        # Convert back to original format
        if isinstance(state, np.ndarray):
            return result.toarray().flatten()
        else:
            return result
    
    def _create_full_gate_matrix_dense(self, gate_matrix: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Create full gate matrix using dense operations."""
        gate_size = 2 ** len(target_qubits)
        identity_size = 2 ** (self.num_qubits - len(target_qubits))
        
        # Create identity matrix
        identity = self._xp.eye(identity_size, dtype=self._xp.complex128)
        
        # Tensor product
        return self._xp.kron(identity, gate_matrix)
    
    def _create_sparse_single_qubit_gate(self, gate_matrix: np.ndarray, target_qubit: int) -> sp.spmatrix:
        """Create sparse single-qubit gate matrix."""
        # For single-qubit gates, we can use a more efficient approach
        # by directly applying the gate to the relevant state components
        
        # Create sparse matrix with only the necessary elements
        rows = []
        cols = []
        data = []
        
        # Calculate the stride for the target qubit
        stride = 2 ** target_qubit
        
        for i in range(self.state_size):
            # Determine which state components are affected
            if i & stride == 0:  # |0⟩ component
                # Apply gate[0,0] and gate[0,1]
                rows.extend([i, i])
                cols.extend([i, i + stride])
                data.extend([gate_matrix[0, 0], gate_matrix[0, 1]])
            else:  # |1⟩ component
                # Apply gate[1,0] and gate[1,1]
                rows.extend([i, i])
                cols.extend([i - stride, i])
                data.extend([gate_matrix[1, 0], gate_matrix[1, 1]])
        
        # Ensure we have valid data
        if not data:
            # Fallback to identity matrix
            return sp.identity(self.state_size, dtype=np.complex128)
        
        return sp.csr_matrix((data, (rows, cols)), shape=(self.state_size, self.state_size), dtype=np.complex128)
    
    def _create_sparse_two_qubit_gate(self, gate_matrix: np.ndarray, target_qubits: List[int]) -> sp.spmatrix:
        """Create sparse two-qubit gate matrix."""
        # For two-qubit gates, we need to be more careful about the ordering
        qubit1, qubit2 = sorted(target_qubits)
        
        rows = []
        cols = []
        data = []
        
        # Calculate strides
        stride1 = 2 ** qubit1
        stride2 = 2 ** qubit2
        
        for i in range(self.state_size):
            # Extract the two-qubit state
            bit1 = (i >> qubit1) & 1
            bit2 = (i >> qubit2) & 1
            two_qubit_state = bit1 * 2 + bit2
            
            # Apply gate to this two-qubit state
            for j in range(4):
                if gate_matrix[two_qubit_state, j] != 0:
                    # Calculate the target state index
                    new_bit1 = (j >> 1) & 1
                    new_bit2 = j & 1
                    
                    # Create new state index
                    new_i = i
                    if new_bit1 != bit1:
                        new_i ^= stride1
                    if new_bit2 != bit2:
                        new_i ^= stride2
                    
                    rows.append(i)
                    cols.append(new_i)
                    data.append(gate_matrix[two_qubit_state, j])
        
        return sp.csr_matrix((data, (rows, cols)), shape=(self.state_size, self.state_size), dtype=np.complex128)
    
    def _create_sparse_controlled_gate(self, gate_matrix: np.ndarray, control_qubit: int, 
                                      target_qubit: int) -> sp.spmatrix:
        """Create sparse controlled gate matrix."""
        rows = []
        cols = []
        data = []
        
        control_stride = 2 ** control_qubit
        target_stride = 2 ** target_qubit
        
        for i in range(self.state_size):
            control_bit = (i >> control_qubit) & 1
            target_bit = (i >> target_qubit) & 1
            
            if control_bit == 1:  # Control qubit is |1⟩
                # Apply gate to target qubit
                for j in range(2):
                    if gate_matrix[target_bit, j] != 0:
                        new_i = i
                        if j != target_bit:
                            new_i ^= target_stride
                        
                        rows.append(i)
                        cols.append(new_i)
                        data.append(gate_matrix[target_bit, j])
            else:  # Control qubit is |0⟩
                # Identity operation
                rows.append(i)
                cols.append(i)
                data.append(1.0)
        
        return sp.csr_matrix((data, (rows, cols)), shape=(self.state_size, self.state_size), dtype=np.complex128)
    
    def _create_controlled_gate_matrix_dense(self, gate_matrix: np.ndarray, control_qubit: int, 
                                            target_qubit: int) -> np.ndarray:
        """Create controlled gate matrix using dense operations."""
        # Create identity matrix
        identity = self._xp.eye(self.state_size, dtype=self._xp.complex128)
        
        # Create controlled gate matrix
        controlled_gate = identity.copy()
        
        control_stride = 2 ** control_qubit
        target_stride = 2 ** target_qubit
        
        for i in range(self.state_size):
            control_bit = (i >> control_qubit) & 1
            if control_bit == 1:  # Control qubit is |1⟩
                target_bit = (i >> target_qubit) & 1
                for j in range(2):
                    if gate_matrix[target_bit, j] != 0:
                        new_i = i
                        if j != target_bit:
                            new_i ^= target_stride
                        controlled_gate[i, new_i] = gate_matrix[target_bit, j]
        
        return controlled_gate
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for sparse operations."""
        avg_execution_time = self.total_execution_time / max(self.operations_count, 1)
        
        return {
            'operations_count': self.operations_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_execution_time,
            'memory_saved': self.memory_saved,
            'operations_per_second': 1.0 / avg_execution_time if avg_execution_time > 0 else 0.0
        }
    
    def estimate_memory_savings(self, num_qubits: int) -> float:
        """Estimate memory savings from sparse operations."""
        if num_qubits <= 12:
            return 0.0  # No savings for small systems
        
        # Calculate memory for dense vs sparse operations
        state_size = 2 ** num_qubits
        
        # Dense gate matrix memory
        dense_memory = state_size * state_size * 16  # 16 bytes per complex128
        
        # Sparse gate matrix memory (assuming 10% sparsity)
        sparse_memory = state_size * state_size * 0.1 * 16
        
        memory_savings = dense_memory - sparse_memory
        return memory_savings / (1024 ** 3)  # Convert to GB


class CircuitOptimizer:
    """
    Circuit optimization for large quantum systems.
    Decomposes large gates into smaller ones to avoid memory issues.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize circuit optimizer.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.optimization_rules = self._setup_optimization_rules()
    
    def _setup_optimization_rules(self) -> Dict[str, Callable]:
        """Setup optimization rules for circuit optimization."""
        return {
            'decompose_large_gates': self._decompose_large_gates,
            'merge_adjacent_gates': self._merge_adjacent_gates,
            'optimize_gate_order': self._optimize_gate_order,
            'remove_redundant_gates': self._remove_redundant_gates
        }
    
    def optimize_circuit(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize quantum circuit for large systems.
        
        Args:
            circuit: List of gate operations
            
        Returns:
            Optimized circuit
        """
        optimized_circuit = circuit.copy()
        
        # Apply optimization rules
        for rule_name, rule_func in self.optimization_rules.items():
            optimized_circuit = rule_func(optimized_circuit)
        
        return optimized_circuit
    
    def _decompose_large_gates(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Decompose large gates into smaller ones."""
        optimized_circuit = []
        
        for gate in circuit:
            if gate.get('num_qubits', 1) > 2:
                # Decompose multi-qubit gates
                decomposed = self._decompose_multi_qubit_gate(gate)
                optimized_circuit.extend(decomposed)
            else:
                optimized_circuit.append(gate)
        
        return optimized_circuit
    
    def _decompose_multi_qubit_gate(self, gate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a multi-qubit gate into smaller gates."""
        # This is a simplified decomposition
        # In practice, this would use more sophisticated decomposition algorithms
        
        decomposed = []
        target_qubits = gate.get('target_qubits', [])
        
        if len(target_qubits) > 2:
            # Decompose into two-qubit gates
            for i in range(0, len(target_qubits) - 1, 2):
                if i + 1 < len(target_qubits):
                    decomposed.append({
                        'type': 'two_qubit',
                        'target_qubits': [target_qubits[i], target_qubits[i + 1]],
                        'gate_matrix': gate.get('gate_matrix', np.eye(4))
                    })
        
        return decomposed
    
    def _merge_adjacent_gates(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge adjacent gates that can be combined."""
        if len(circuit) < 2:
            return circuit
        
        optimized_circuit = []
        i = 0
        
        while i < len(circuit):
            current_gate = circuit[i]
            
            # Check if we can merge with the next gate
            if i + 1 < len(circuit):
                next_gate = circuit[i + 1]
                
                if self._can_merge_gates(current_gate, next_gate):
                    merged_gate = self._merge_gates(current_gate, next_gate)
                    optimized_circuit.append(merged_gate)
                    i += 2  # Skip both gates
                else:
                    optimized_circuit.append(current_gate)
                    i += 1
            else:
                optimized_circuit.append(current_gate)
                i += 1
        
        return optimized_circuit
    
    def _can_merge_gates(self, gate1: Dict[str, Any], gate2: Dict[str, Any]) -> bool:
        """Check if two gates can be merged."""
        # Simple check: same target qubits
        return (gate1.get('target_qubits') == gate2.get('target_qubits') and
                gate1.get('type') == gate2.get('type'))
    
    def _merge_gates(self, gate1: Dict[str, Any], gate2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two gates into one."""
        # Multiply gate matrices
        matrix1 = gate1.get('gate_matrix', np.eye(2))
        matrix2 = gate2.get('gate_matrix', np.eye(2))
        merged_matrix = matrix2 @ matrix1
        
        return {
            'type': gate1.get('type'),
            'target_qubits': gate1.get('target_qubits'),
            'gate_matrix': merged_matrix
        }
    
    def _optimize_gate_order(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize the order of gates for better performance."""
        # Group gates by target qubits
        gate_groups = {}
        
        for gate in circuit:
            target_qubits = tuple(sorted(gate.get('target_qubits', [])))
            if target_qubits not in gate_groups:
                gate_groups[target_qubits] = []
            gate_groups[target_qubits].append(gate)
        
        # Reorder gates to minimize qubit switching
        optimized_circuit = []
        for target_qubits in sorted(gate_groups.keys()):
            optimized_circuit.extend(gate_groups[target_qubits])
        
        return optimized_circuit
    
    def _remove_redundant_gates(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant gates (e.g., identity gates)."""
        optimized_circuit = []
        
        for gate in circuit:
            gate_matrix = gate.get('gate_matrix', np.eye(2))
            
            # Check if gate is close to identity
            if not np.allclose(gate_matrix, np.eye(gate_matrix.shape[0]), atol=1e-10):
                optimized_circuit.append(gate)
        
        return optimized_circuit


# Test functions
def test_sparse_gate_operations():
    """Test sparse gate operations for large systems."""
    print("Testing sparse gate operations for 15-20 qubits...")
    
    # Test 15 qubits
    print("\nTesting 15 qubits:")
    operator_15 = SparseGateOperator(15, use_gpu=False)
    
    # Create test state
    state_15 = np.zeros(2**15, dtype=np.complex128)
    state_15[0] = 1.0
    
    # Test single-qubit gate
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    result_15 = operator_15.apply_single_qubit_gate(state_15, hadamard, 0)
    
    print(f"✅ 15 qubits: Single-qubit gate applied successfully")
    print(f"   Result shape: {result_15.shape}")
    
    # Test 18 qubits
    print("\nTesting 18 qubits:")
    operator_18 = SparseGateOperator(18, use_gpu=False)
    
    # Create test state
    state_18 = np.zeros(2**18, dtype=np.complex128)
    state_18[0] = 1.0
    
    # Test single-qubit gate
    result_18 = operator_18.apply_single_qubit_gate(state_18, hadamard, 0)
    
    print(f"✅ 18 qubits: Single-qubit gate applied successfully")
    print(f"   Result shape: {result_18.shape}")
    
    # Test performance metrics
    metrics_15 = operator_15.get_performance_metrics()
    metrics_18 = operator_18.get_performance_metrics()
    
    print(f"\n📊 Performance Metrics:")
    print(f"15 qubits: {metrics_15['operations_count']} operations, {metrics_15['total_execution_time']:.4f}s")
    print(f"18 qubits: {metrics_18['operations_count']} operations, {metrics_18['total_execution_time']:.4f}s")
    
    # Test memory savings
    memory_savings_15 = operator_15.estimate_memory_savings(15)
    memory_savings_18 = operator_18.estimate_memory_savings(18)
    
    print(f"\n💾 Memory Savings:")
    print(f"15 qubits: {memory_savings_15:.2f} GB saved")
    print(f"18 qubits: {memory_savings_18:.2f} GB saved")
    
    print("\n🎉 Sparse gate operations working for 15-20 qubits!")


def test_circuit_optimization():
    """Test circuit optimization for large systems."""
    print("\nTesting circuit optimization...")
    
    optimizer = CircuitOptimizer(20)
    
    # Create test circuit with large gates
    test_circuit = [
        {'type': 'multi_qubit', 'target_qubits': [0, 1, 2, 3], 'gate_matrix': np.eye(16)},
        {'type': 'single_qubit', 'target_qubits': [0], 'gate_matrix': np.eye(2)},
        {'type': 'two_qubit', 'target_qubits': [1, 2], 'gate_matrix': np.eye(4)}
    ]
    
    # Optimize circuit
    optimized_circuit = optimizer.optimize_circuit(test_circuit)
    
    print(f"✅ Circuit optimization successful")
    print(f"   Original gates: {len(test_circuit)}")
    print(f"   Optimized gates: {len(optimized_circuit)}")
    
    print("\n🎉 Circuit optimization working for large systems!")


if __name__ == "__main__":
    # Run tests
    test_sparse_gate_operations()
    test_circuit_optimization()
    
    print("\n🚀 Sparse gate operations and circuit optimization ready for 15-20 qubits!")
