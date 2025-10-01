"""
Quantum gates implementation with matrix operations.

This module implements the fundamental quantum gates:
- Single-qubit gates: X (Pauli-X), Y (Pauli-Y), Z (Pauli-Z), H (Hadamard)
- Two-qubit gates: CNOT (Controlled-NOT)

Each gate is represented by its unitary matrix and can be applied to quantum states.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
import math


class QuantumGate(ABC):
    """
    Abstract base class for quantum gates.
    
    All quantum gates must be unitary (U†U = I) to preserve the
    normalization of quantum states.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """
        Get the unitary matrix representation of the gate.
        
        Args:
            num_qubits: Total number of qubits in the system
            target_qubits: List of qubit indices the gate acts on
        
        Returns:
            Unitary matrix representing the gate operation
        """
        pass
    
    def apply(self, quantum_state, target_qubits: List[int]):
        """
        Apply the gate to the quantum state.
        
        Args:
            quantum_state: QuantumState object to apply the gate to
            target_qubits: List of qubit indices the gate acts on
        """
        matrix = self.get_matrix(quantum_state.num_qubits, target_qubits)
        quantum_state.state_vector = matrix @ quantum_state.state_vector
        quantum_state.normalize()


class XGate(QuantumGate):
    """
    Pauli-X gate (quantum NOT gate).
    
    Matrix representation:
    X = [0  1]
        [1  0]
    
    X|0⟩ = |1⟩, X|1⟩ = |0⟩
    """
    
    def __init__(self):
        super().__init__("X")
        self.single_qubit_matrix = np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the X gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("X gate acts on exactly one qubit")
        
        return self._build_full_matrix(num_qubits, target_qubits[0], self.single_qubit_matrix)
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Build the full matrix for a single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        # Apply the gate to the target qubit
        for i in range(dimension):
            for j in range(dimension):
                # Check if the gate should be applied based on qubit states
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    # Extract the relevant 2x2 submatrix
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if the gate should be applied between states i and j."""
        # All other qubits must be in the same state
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class YGate(QuantumGate):
    """
    Pauli-Y gate.
    
    Matrix representation:
    Y = [0  -i]
        [i   0]
    
    Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    """
    
    def __init__(self):
        super().__init__("Y")
        self.single_qubit_matrix = np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the Y gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("Y gate acts on exactly one qubit")
        
        return self._build_full_matrix(num_qubits, target_qubits[0], self.single_qubit_matrix)
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Build the full matrix for a single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        for i in range(dimension):
            for j in range(dimension):
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if the gate should be applied between states i and j."""
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class ZGate(QuantumGate):
    """
    Pauli-Z gate.
    
    Matrix representation:
    Z = [1  0]
        [0 -1]
    
    Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    """
    
    def __init__(self):
        super().__init__("Z")
        self.single_qubit_matrix = np.array([
            [1, 0],
            [0, -1]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the Z gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("Z gate acts on exactly one qubit")
        
        return self._build_full_matrix(num_qubits, target_qubits[0], self.single_qubit_matrix)
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Build the full matrix for a single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        for i in range(dimension):
            for j in range(dimension):
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if the gate should be applied between states i and j."""
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class HGate(QuantumGate):
    """
    Hadamard gate.
    
    Matrix representation:
    H = (1/√2) [1   1]
               [1  -1]
    
    H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2
    Creates superposition states.
    """
    
    def __init__(self):
        super().__init__("H")
        self.single_qubit_matrix = (1/math.sqrt(2)) * np.array([
            [1, 1],
            [1, -1]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the H gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("H gate acts on exactly one qubit")
        
        return self._build_full_matrix(num_qubits, target_qubits[0], self.single_qubit_matrix)
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Build the full matrix for a single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        for i in range(dimension):
            for j in range(dimension):
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if the gate should be applied between states i and j."""
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class CNOTGate(QuantumGate):
    """
    Controlled-NOT gate (CNOT).
    
    Matrix representation (for qubits 0 and 1):
    CNOT = [1 0 0 0]
           [0 1 0 0]
           [0 0 0 1]
           [0 0 1 0]
    
    CNOT|00⟩ = |00⟩, CNOT|01⟩ = |01⟩, CNOT|10⟩ = |11⟩, CNOT|11⟩ = |10⟩
    Flips the target qubit if the control qubit is |1⟩.
    """
    
    def __init__(self):
        super().__init__("CNOT")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the CNOT gate matrix for the specified qubits."""
        if len(target_qubits) != 2:
            raise ValueError("CNOT gate acts on exactly two qubits")
        
        control_qubit, target_qubit = target_qubits[0], target_qubits[1]
        dimension = 2 ** num_qubits
        matrix = np.eye(dimension, dtype=complex)
        
        # Apply CNOT logic: flip target if control is |1⟩
        for i in range(dimension):
            for j in range(dimension):
                # Check if control qubit is |1⟩ in both states
                control_i = (i >> (num_qubits - 1 - control_qubit)) & 1
                control_j = (j >> (num_qubits - 1 - control_qubit)) & 1
                
                if control_i == 1 and control_j == 1:
                    # Control is |1⟩, so target should be flipped
                    target_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    target_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    
                    # Check if target is flipped and other qubits are the same
                    if target_i == 1 - target_j:
                        # Check that all other qubits are the same
                        same_other_qubits = True
                        for q in range(num_qubits):
                            if q != control_qubit and q != target_qubit:
                                bit_i = (i >> (num_qubits - 1 - q)) & 1
                                bit_j = (j >> (num_qubits - 1 - q)) & 1
                                if bit_i != bit_j:
                                    same_other_qubits = False
                                    break
                        
                        if same_other_qubits:
                            matrix[i, j] = 1.0
                        else:
                            matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = 0.0
                elif control_i == control_j:
                    # Control qubit is the same in both states
                    # Check if all qubits are the same
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        matrix[i, j] = 0.0
                else:
                    matrix[i, j] = 0.0
        
        return matrix


class RYGate(QuantumGate):
    """
    Rotation around Y-axis gate.
    
    Matrix representation:
    RY(θ) = [cos(θ/2)  -sin(θ/2)]
            [sin(θ/2)   cos(θ/2)]
    
    RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    RY(θ)|1⟩ = -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
    """
    
    def __init__(self, angle: float):
        super().__init__("RY")
        self.angle = angle
        self.single_qubit_matrix = np.array([
            [np.cos(angle/2), -np.sin(angle/2)],
            [np.sin(angle/2), np.cos(angle/2)]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the RY gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("RY gate acts on exactly one qubit")
        
        return self._build_full_matrix(num_qubits, target_qubits[0], self.single_qubit_matrix)
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Build the full matrix for a single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        for i in range(dimension):
            for j in range(dimension):
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if the gate should be applied between states i and j."""
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class RZGate(QuantumGate):
    """
    Rotation around Z-axis gate.
    
    Matrix representation:
    RZ(θ) = [e^(-iθ/2)     0    ]
            [0        e^(iθ/2)]
    
    RZ(θ)|0⟩ = e^(-iθ/2)|0⟩
    RZ(θ)|1⟩ = e^(iθ/2)|1⟩
    """
    
    def __init__(self, angle: float):
        super().__init__("RZ")
        self.angle = angle
        self.single_qubit_matrix = np.array([
            [np.exp(-1j * angle/2), 0],
            [0, np.exp(1j * angle/2)]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the RZ gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("RZ gate acts on exactly one qubit")
        
        return self._build_full_matrix(num_qubits, target_qubits[0], self.single_qubit_matrix)
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Build the full matrix for a single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        for i in range(dimension):
            for j in range(dimension):
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if the gate should be applied between states i and j."""
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class CZGate(QuantumGate):
    """
    Controlled-Z gate (CZ).
    
    Matrix representation (for qubits 0 and 1):
    CZ = [1 0  0  0]
         [0 1  0  0]
         [0 0  1  0]
         [0 0  0 -1]
    
    CZ|00⟩ = |00⟩, CZ|01⟩ = |01⟩, CZ|10⟩ = |10⟩, CZ|11⟩ = -|11⟩
    Applies a phase flip to the target qubit if the control qubit is |1⟩.
    """
    
    def __init__(self):
        super().__init__("CZ")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the CZ gate matrix for the specified qubits."""
        if len(target_qubits) != 2:
            raise ValueError("CZ gate acts on exactly two qubits")
        
        control_qubit, target_qubit = target_qubits[0], target_qubits[1]
        dimension = 2 ** num_qubits
        matrix = np.eye(dimension, dtype=complex)
        
        # Apply CZ logic: phase flip target if control is |1⟩
        for i in range(dimension):
            for j in range(dimension):
                # Check if control qubit is |1⟩ in both states
                control_i = (i >> (num_qubits - 1 - control_qubit)) & 1
                control_j = (j >> (num_qubits - 1 - control_qubit)) & 1
                
                if control_i == 1 and control_j == 1:
                    # Control is |1⟩, so target should have phase flip
                    target_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    target_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    
                    # Check if target is the same and other qubits are the same
                    if target_i == target_j:
                        # Check that all other qubits are the same
                        same_other_qubits = True
                        for q in range(num_qubits):
                            if q != control_qubit and q != target_qubit:
                                bit_i = (i >> (num_qubits - 1 - q)) & 1
                                bit_j = (j >> (num_qubits - 1 - q)) & 1
                                if bit_i != bit_j:
                                    same_other_qubits = False
                                    break
                        
                        if same_other_qubits:
                            # Apply phase flip if target is |1⟩
                            if target_i == 1:
                                matrix[i, j] = -1.0
                            else:
                                matrix[i, j] = 1.0
                        else:
                            matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = 0.0
                elif control_i == control_j:
                    # Control qubit is the same in both states
                    # Check if all qubits are the same
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        matrix[i, j] = 0.0
                else:
                    matrix[i, j] = 0.0
        
        return matrix
