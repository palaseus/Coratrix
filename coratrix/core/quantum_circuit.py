"""
Quantum Circuit Implementation

This module provides quantum circuit construction and execution capabilities.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


class QuantumGate(ABC):
    """Base class for quantum gates."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> 'np.ndarray':
        """Get the matrix representation of the gate."""
        pass
    
    def __str__(self) -> str:
        return f"{self.name}"


class QuantumCircuit:
    """Quantum circuit representation."""
    
    def __init__(self, num_qubits: int, name: str = "circuit"):
        self.num_qubits = num_qubits
        self.name = name
        self.gates: List[Dict[str, Any]] = []
        self.measurements: List[Dict[str, Any]] = []
    
    def add_gate(self, gate: QuantumGate, target_qubits: List[int], 
                 parameters: List[float] = None):
        """Add a gate to the circuit."""
        gate_info = {
            'gate': gate,
            'target_qubits': target_qubits,
            'parameters': parameters or []
        }
        self.gates.append(gate_info)
    
    def add_measurement(self, qubit: int, classical_bit: int):
        """Add a measurement to the circuit."""
        measurement_info = {
            'qubit': qubit,
            'classical_bit': classical_bit
        }
        self.measurements.append(measurement_info)
    
    def execute(self, state: 'ScalableQuantumState'):
        """Execute the circuit on a quantum state."""
        for gate_info in self.gates:
            gate = gate_info['gate']
            target_qubits = gate_info['target_qubits']
            state.apply_gate(gate, target_qubits)
    
    def __str__(self) -> str:
        return f"QuantumCircuit(name='{self.name}', qubits={self.num_qubits}, gates={len(self.gates)})"
    
    def __repr__(self) -> str:
        return f"QuantumCircuit(num_qubits={self.num_qubits}, name='{self.name}')"


class GateFactory:
    """Factory for creating quantum gates."""
    
    @staticmethod
    def create_gate(gate_type: str, **kwargs) -> QuantumGate:
        """Create a quantum gate of the specified type."""
        if gate_type == "H":
            return HGate()
        elif gate_type == "X":
            return XGate()
        elif gate_type == "Y":
            return YGate()
        elif gate_type == "Z":
            return ZGate()
        elif gate_type == "CNOT":
            return CNOTGate()
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")


# Basic gate implementations
class HGate(QuantumGate):
    """Hadamard gate."""
    
    def __init__(self):
        super().__init__("H")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> 'np.ndarray':
        import numpy as np
        dim = 2 ** num_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        
        # Apply H gate to target qubit
        target_qubit = target_qubits[0]
        
        # Create Hadamard matrix for the target qubit
        h_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # Apply to all states
        for i in range(dim):
            for j in range(dim):
                # Check if only the target qubit differs
                if bin(i ^ j).count('1') == 1:
                    target_bit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    target_bit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    
                    if target_bit_i != target_bit_j:
                        # Check if other qubits are the same
                        other_bits_i = i ^ (target_bit_i << (num_qubits - 1 - target_qubit))
                        other_bits_j = j ^ (target_bit_j << (num_qubits - 1 - target_qubit))
                        
                        if other_bits_i == other_bits_j:
                            matrix[i, j] = h_matrix[target_bit_i, target_bit_j]
                elif i == j:
                    # Diagonal elements
                    target_bit = (i >> (num_qubits - 1 - target_qubit)) & 1
                    matrix[i, j] = h_matrix[target_bit, target_bit]
        
        return matrix


class XGate(QuantumGate):
    """Pauli-X gate."""
    
    def __init__(self):
        super().__init__("X")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> 'np.ndarray':
        import numpy as np
        dim = 2 ** num_qubits
        matrix = np.eye(dim, dtype=complex)
        
        # Apply X gate to target qubit
        target_qubit = target_qubits[0]
        for i in range(dim):
            # Flip the target qubit
            flipped = i ^ (1 << (num_qubits - 1 - target_qubit))
            matrix[i, flipped] = 1
        
        return matrix


class YGate(QuantumGate):
    """Pauli-Y gate."""
    
    def __init__(self):
        super().__init__("Y")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> 'np.ndarray':
        import numpy as np
        dim = 2 ** num_qubits
        matrix = np.eye(dim, dtype=complex)
        
        # Apply Y gate to target qubit
        target_qubit = target_qubits[0]
        for i in range(dim):
            # Flip the target qubit and apply phase
            flipped = i ^ (1 << (num_qubits - 1 - target_qubit))
            target_bit = (i >> (num_qubits - 1 - target_qubit)) & 1
            matrix[i, flipped] = 1j * (-1)**target_bit
        
        return matrix


class ZGate(QuantumGate):
    """Pauli-Z gate."""
    
    def __init__(self):
        super().__init__("Z")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> 'np.ndarray':
        import numpy as np
        dim = 2 ** num_qubits
        matrix = np.eye(dim, dtype=complex)
        
        # Apply Z gate to target qubit
        target_qubit = target_qubits[0]
        for i in range(dim):
            target_bit = (i >> (num_qubits - 1 - target_qubit)) & 1
            matrix[i, i] = (-1)**target_bit
        
        return matrix


class CNOTGate(QuantumGate):
    """Controlled-NOT gate."""
    
    def __init__(self):
        super().__init__("CNOT")
    
    def get_matrix(self, num_qubits: int, target_qubits: List[int]) -> 'np.ndarray':
        import numpy as np
        dim = 2 ** num_qubits
        matrix = np.eye(dim, dtype=complex)
        
        # Apply CNOT gate
        control_qubit = target_qubits[0]
        target_qubit = target_qubits[1]
        
        for i in range(dim):
            control_bit = (i >> (num_qubits - 1 - control_qubit)) & 1
            target_bit = (i >> (num_qubits - 1 - target_qubit)) & 1
            
            if control_bit == 1:  # Control qubit is |1⟩
                # Flip target qubit
                flipped = i ^ (1 << (num_qubits - 1 - target_qubit))
                matrix[i, flipped] = 1
            else:
                matrix[i, i] = 1
        
        return matrix
