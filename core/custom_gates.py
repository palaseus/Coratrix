"""
Custom quantum gates for Coratrix.

This module contains custom quantum gates that extend the standard gate set
with specialized operations for research and experimentation.
"""

import numpy as np
import math
from typing import List
from .advanced_gates import ParameterizedGate


class CustomCPhaseGate(ParameterizedGate):
    """
    Custom Controlled Phase Gate with non-standard angle.
    
    This gate applies a phase of exp(i*phi) to the |11⟩ state,
    where phi is a custom angle (not the standard π).
    
    Matrix representation:
    [1  0  0  0 ]
    [0  1  0  0 ]
    [0  0  1  0 ]
    [0  0  0  exp(i*phi)]
    
    Parameters:
        phi: Custom phase angle in radians (default: π/3 for 60 degrees)
    """
    
    def __init__(self, phi: float = np.pi / 3):
        super().__init__("CustomCPhase", {"phi": phi})
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the custom CPhase gate matrix for the specified qubits."""
        if len(target_qubits) != 2:
            raise ValueError("CustomCPhase gate acts on exactly two qubits")
        
        control_qubit, target_qubit = target_qubits[0], target_qubits[1]
        phi = self.parameters["phi"]
        exp_i_phi = np.exp(1j * phi)
        
        dimension = 2 ** num_qubits
        matrix = np.eye(dimension, dtype=complex)
        
        # Apply phase to |11⟩ state
        for i in range(dimension):
            # Check if both control and target qubits are in |1⟩ state
            control_bit = (i >> (num_qubits - 1 - control_qubit)) & 1
            target_bit = (i >> (num_qubits - 1 - target_qubit)) & 1
            
            if control_bit == 1 and target_bit == 1:
                matrix[i, i] = exp_i_phi
        
        return matrix


class CustomRotationGate(ParameterizedGate):
    """
    Custom rotation gate with arbitrary axis and angle.
    
    This gate performs a rotation around an arbitrary axis by a custom angle.
    Useful for quantum state preparation and custom algorithms.
    
    Parameters:
        theta: Rotation angle in radians
        axis: Rotation axis (x, y, or z)
    """
    
    def __init__(self, theta: float = np.pi / 4, axis: str = "x"):
        super().__init__("CustomRotation", {"theta": theta, "axis": axis})
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Get the custom rotation gate matrix for the specified qubits."""
        if len(target_qubits) != 1:
            raise ValueError("CustomRotation gate acts on exactly one qubit")
        
        theta = self.parameters["theta"]
        axis = self.parameters["axis"].lower()
        
        if axis == "x":
            cos_theta_2 = math.cos(theta / 2)
            sin_theta_2 = math.sin(theta / 2)
            matrix = np.array([
                [cos_theta_2, -1j * sin_theta_2],
                [-1j * sin_theta_2, cos_theta_2]
            ], dtype=complex)
        elif axis == "y":
            cos_theta_2 = math.cos(theta / 2)
            sin_theta_2 = math.sin(theta / 2)
            matrix = np.array([
                [cos_theta_2, -sin_theta_2],
                [sin_theta_2, cos_theta_2]
            ], dtype=complex)
        elif axis == "z":
            exp_neg_i_theta_2 = np.exp(-1j * theta / 2)
            exp_pos_i_theta_2 = np.exp(1j * theta / 2)
            matrix = np.array([
                [exp_neg_i_theta_2, 0],
                [0, exp_pos_i_theta_2]
            ], dtype=complex)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        return self._build_full_matrix(num_qubits, target_qubits[0], matrix)
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Build the full matrix for the custom rotation gate."""
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
