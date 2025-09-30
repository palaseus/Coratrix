"""
Advanced quantum gates library.

This module implements an extensive collection of quantum gates including
Toffoli, SWAP, phase rotations, controlled gates, and parameterized gates.
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from .gates import QuantumGate


class ParameterizedGate(QuantumGate):
    """
    Abstract base class for parameterized quantum gates.
    
    Parameterized gates can have their parameters adjusted,
    making them useful for quantum algorithms and optimization.
    """
    
    def __init__(self, name: str, parameters: Dict[str, float]):
        super().__init__(name)
        self.parameters = parameters.copy()
    
    def set_parameter(self, param_name: str, value: float):
        """Set a parameter value."""
        if param_name not in self.parameters:
            raise ValueError(f"Parameter {param_name} not found in {self.name} gate")
        self.parameters[param_name] = value
    
    def get_parameter(self, param_name: str) -> float:
        """Get a parameter value."""
        if param_name not in self.parameters:
            raise ValueError(f"Parameter {param_name} not found in {self.name} gate")
        return self.parameters[param_name]


class ToffoliGate(QuantumGate):
    """
    Toffoli gate (CCNOT) - controlled-controlled-NOT gate.
    
    Matrix representation (for qubits 0, 1, 2):
    Toffoli = [1 0 0 0 0 0 0 0]
              [0 1 0 0 0 0 0 0]
              [0 0 1 0 0 0 0 0]
              [0 0 0 1 0 0 0 0]
              [0 0 0 0 1 0 0 0]
              [0 0 0 0 0 1 0 0]
              [0 0 0 0 0 0 0 1]
              [0 0 0 0 0 0 1 0]
    
    Flips the target qubit if both control qubits are |1⟩.
    """
    
    def __init__(self):
        super().__init__("Toffoli")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the Toffoli gate matrix for the specified qubits."""
        if len(target_qubits) != 3:
            raise ValueError("Toffoli gate acts on exactly three qubits")
        
        control1, control2, target = target_qubits[0], target_qubits[1], target_qubits[2]
        dimension = 2 ** num_qubits
        matrix = np.eye(dimension, dtype=complex)
        
        # Apply Toffoli logic: flip target if both controls are |1⟩
        for i in range(dimension):
            for j in range(dimension):
                # Check if both control qubits are |1⟩ in both states
                control1_i = (i >> (num_qubits - 1 - control1)) & 1
                control1_j = (j >> (num_qubits - 1 - control1)) & 1
                control2_i = (i >> (num_qubits - 1 - control2)) & 1
                control2_j = (j >> (num_qubits - 1 - control2)) & 1
                
                if control1_i == 1 and control1_j == 1 and control2_i == 1 and control2_j == 1:
                    # Both controls are |1⟩, so target should be flipped
                    target_i = (i >> (num_qubits - 1 - target)) & 1
                    target_j = (j >> (num_qubits - 1 - target)) & 1
                    
                    # Check if target is flipped and other qubits are the same
                    if target_i == 1 - target_j:
                        # Check that all other qubits are the same
                        same_other_qubits = True
                        for q in range(num_qubits):
                            if q not in [control1, control2, target]:
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
                elif control1_i == control1_j and control2_i == control2_j:
                    # Control qubits are the same in both states
                    # Check if all qubits are the same
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        matrix[i, j] = 0.0
                else:
                    matrix[i, j] = 0.0
        
        return matrix


class SWAPGate(QuantumGate):
    """
    SWAP gate - swaps two qubits.
    
    Matrix representation (for qubits 0 and 1):
    SWAP = [1 0 0 0]
           [0 0 1 0]
           [0 1 0 0]
           [0 0 0 1]
    
    SWAP|01⟩ = |10⟩, SWAP|10⟩ = |01⟩
    """
    
    def __init__(self):
        super().__init__("SWAP")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the SWAP gate matrix for the specified qubits."""
        if len(target_qubits) != 2:
            raise ValueError("SWAP gate acts on exactly two qubits")
        
        qubit1, qubit2 = target_qubits[0], target_qubits[1]
        dimension = 2 ** num_qubits
        matrix = np.eye(dimension, dtype=complex)
        
        # Apply SWAP logic: swap the two qubits
        for i in range(dimension):
            # Calculate the swapped state
            swapped_i = self._swap_qubits(i, qubit1, qubit2, num_qubits)
            matrix[i, swapped_i] = 1.0
            if i != swapped_i:
                matrix[i, i] = 0.0
        
        return matrix
    
    def _swap_qubits(self, state_index: int, qubit1: int, qubit2: int, num_qubits: int) -> int:
        """Calculate the state index after swapping two qubits."""
        # Get the binary representation
        binary = format(state_index, f'0{num_qubits}b')
        binary_list = list(binary)
        
        # Swap the qubits
        binary_list[qubit1], binary_list[qubit2] = binary_list[qubit2], binary_list[qubit1]
        
        # Convert back to integer
        return int(''.join(binary_list), 2)


class RxGate(ParameterizedGate):
    """
    Rotation around X-axis gate.
    
    Matrix representation:
    Rx(θ) = [cos(θ/2)    -i*sin(θ/2)]
            [-i*sin(θ/2)  cos(θ/2)   ]
    
    Parameters:
        theta: Rotation angle in radians
    """
    
    def __init__(self, theta: float = 0.0):
        super().__init__("Rx", {"theta": theta})
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the Rx gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("Rx gate acts on exactly one qubit")
        
        theta = self.parameters["theta"]
        cos_theta_2 = math.cos(theta / 2)
        sin_theta_2 = math.sin(theta / 2)
        
        single_qubit_matrix = np.array([
            [cos_theta_2, -1j * sin_theta_2],
            [-1j * sin_theta_2, cos_theta_2]
        ], dtype=complex)
        
        return self._build_full_matrix(num_qubits, target_qubits[0], single_qubit_matrix)
    
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


class RyGate(ParameterizedGate):
    """
    Rotation around Y-axis gate.
    
    Matrix representation:
    Ry(θ) = [cos(θ/2)  -sin(θ/2)]
            [sin(θ/2)   cos(θ/2)]
    
    Parameters:
        theta: Rotation angle in radians
    """
    
    def __init__(self, theta: float = 0.0):
        super().__init__("Ry", {"theta": theta})
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the Ry gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("Ry gate acts on exactly one qubit")
        
        theta = self.parameters["theta"]
        cos_theta_2 = math.cos(theta / 2)
        sin_theta_2 = math.sin(theta / 2)
        
        single_qubit_matrix = np.array([
            [cos_theta_2, -sin_theta_2],
            [sin_theta_2, cos_theta_2]
        ], dtype=complex)
        
        return self._build_full_matrix(num_qubits, target_qubits[0], single_qubit_matrix)
    
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


class RzGate(ParameterizedGate):
    """
    Rotation around Z-axis gate.
    
    Matrix representation:
    Rz(θ) = [exp(-iθ/2)     0     ]
            [    0      exp(iθ/2)]
    
    Parameters:
        theta: Rotation angle in radians
    """
    
    def __init__(self, theta: float = 0.0):
        super().__init__("Rz", {"theta": theta})
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the Rz gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("Rz gate acts on exactly one qubit")
        
        theta = self.parameters["theta"]
        exp_neg_i_theta_2 = np.exp(-1j * theta / 2)
        exp_pos_i_theta_2 = np.exp(1j * theta / 2)
        
        single_qubit_matrix = np.array([
            [exp_neg_i_theta_2, 0],
            [0, exp_pos_i_theta_2]
        ], dtype=complex)
        
        return self._build_full_matrix(num_qubits, target_qubits[0], single_qubit_matrix)
    
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


class CPhaseGate(ParameterizedGate):
    """
    Controlled phase gate.
    
    Matrix representation (for qubits 0 and 1):
    CPhase(φ) = [1 0 0 0    ]
                 [0 1 0 0    ]
                 [0 0 1 0    ]
                 [0 0 0 exp(iφ)]
    
    Parameters:
        phi: Phase angle in radians
    """
    
    def __init__(self, phi: float = 0.0):
        super().__init__("CPhase", {"phi": phi})
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the CPhase gate matrix for the specified qubits."""
        if len(target_qubits) != 2:
            raise ValueError("CPhase gate acts on exactly two qubits")
        
        control_qubit, target_qubit = target_qubits[0], target_qubits[1]
        phi = self.parameters["phi"]
        exp_i_phi = np.exp(1j * phi)
        
        dimension = 2 ** num_qubits
        matrix = np.eye(dimension, dtype=complex)
        
        # Apply CPhase logic: apply phase if both qubits are |1⟩
        for i in range(dimension):
            for j in range(dimension):
                # Check if both qubits are |1⟩ in both states
                control_i = (i >> (num_qubits - 1 - control_qubit)) & 1
                control_j = (j >> (num_qubits - 1 - control_qubit)) & 1
                target_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                target_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                
                if control_i == 1 and control_j == 1 and target_i == 1 and target_j == 1:
                    # Both qubits are |1⟩, apply phase
                    # Check that all other qubits are the same
                    same_other_qubits = True
                    for q in range(num_qubits):
                        if q not in [control_qubit, target_qubit]:
                            bit_i = (i >> (num_qubits - 1 - q)) & 1
                            bit_j = (j >> (num_qubits - 1 - q)) & 1
                            if bit_i != bit_j:
                                same_other_qubits = False
                                break
                    
                    if same_other_qubits:
                        matrix[i, j] = exp_i_phi
                    else:
                        matrix[i, j] = 0.0
                elif i == j:
                    # All qubits are the same
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = 0.0
        
        return matrix


class SGate(QuantumGate):
    """
    S gate (π/2 phase gate).
    
    Matrix representation:
    S = [1  0]
        [0  i]
    
    S|0⟩ = |0⟩, S|1⟩ = i|1⟩
    """
    
    def __init__(self):
        super().__init__("S")
        self.single_qubit_matrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the S gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("S gate acts on exactly one qubit")
        
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


class TGate(QuantumGate):
    """
    T gate (π/4 phase gate).
    
    Matrix representation:
    T = [1     0    ]
        [0  exp(iπ/4)]
    
    T|0⟩ = |0⟩, T|1⟩ = exp(iπ/4)|1⟩
    """
    
    def __init__(self):
        super().__init__("T")
        self.single_qubit_matrix = np.array([
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ], dtype=complex)
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the T gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("T gate acts on exactly one qubit")
        
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


class FredkinGate(QuantumGate):
    """
    Fredkin gate (controlled-SWAP gate).
    
    Matrix representation (for qubits 0, 1, 2):
    Fredkin = [1 0 0 0 0 0 0 0]
              [0 1 0 0 0 0 0 0]
              [0 0 1 0 0 0 0 0]
              [0 0 0 1 0 0 0 0]
              [0 0 0 0 1 0 0 0]
              [0 0 0 0 0 0 1 0]
              [0 0 0 0 0 1 0 0]
              [0 0 0 0 0 0 0 1]
    
    Swaps the target qubits if the control qubit is |1⟩.
    """
    
    def __init__(self):
        super().__init__("Fredkin")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the Fredkin gate matrix for the specified qubits."""
        if len(target_qubits) != 3:
            raise ValueError("Fredkin gate acts on exactly three qubits")
        
        control, target1, target2 = target_qubits[0], target_qubits[1], target_qubits[2]
        dimension = 2 ** num_qubits
        matrix = np.eye(dimension, dtype=complex)
        
        # Apply Fredkin logic: swap targets if control is |1⟩
        for i in range(dimension):
            for j in range(dimension):
                # Check if control qubit is |1⟩ in both states
                control_i = (i >> (num_qubits - 1 - control)) & 1
                control_j = (j >> (num_qubits - 1 - control)) & 1
                
                if control_i == 1 and control_j == 1:
                    # Control is |1⟩, so targets should be swapped
                    target1_i = (i >> (num_qubits - 1 - target1)) & 1
                    target1_j = (j >> (num_qubits - 1 - target1)) & 1
                    target2_i = (i >> (num_qubits - 1 - target2)) & 1
                    target2_j = (j >> (num_qubits - 1 - target2)) & 1
                    
                    # Check if targets are swapped and other qubits are the same
                    if target1_i == target2_j and target2_i == target1_j:
                        # Check that all other qubits are the same
                        same_other_qubits = True
                        for q in range(num_qubits):
                            if q not in [control, target1, target2]:
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
