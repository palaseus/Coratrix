"""
Quantum circuit implementation for applying gates to quantum states.

This module provides the QuantumCircuit class that manages the application
of quantum gates to quantum states in a sequential manner.
"""

from typing import List, Tuple, Union
from core.qubit import QuantumState
from core.gates import QuantumGate


class QuantumCircuit:
    """
    A quantum circuit that applies gates to quantum states.
    
    The circuit maintains a sequence of gates and can apply them
    to quantum states in order. This provides the interface between
    the gate simulator backend and higher-level quantum programming.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize a quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
        """
        self.num_qubits = num_qubits
        self.gates = []  # List of (gate, target_qubits) tuples
        self.quantum_state = QuantumState(num_qubits)
    
    def add_gate(self, gate: QuantumGate, target_qubits: List[int]):
        """
        Add a gate to the circuit.
        
        Args:
            gate: The quantum gate to add
            target_qubits: List of qubit indices the gate acts on
        """
        # Validate qubit indices
        for qubit in target_qubits:
            if not (0 <= qubit < self.num_qubits):
                raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
        
        self.gates.append((gate, target_qubits))
    
    def apply_gate(self, gate: QuantumGate, target_qubits: List[int]):
        """
        Apply a gate directly to the current quantum state.
        
        Args:
            gate: The quantum gate to apply
            target_qubits: List of qubit indices the gate acts on
        """
        # Validate qubit indices
        for qubit in target_qubits:
            if not (0 <= qubit < self.num_qubits):
                raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
        
        gate.apply(self.quantum_state, target_qubits)
    
    def execute(self):
        """
        Execute all gates in the circuit in sequence.
        
        This applies all gates that have been added to the circuit
        to the quantum state in the order they were added.
        """
        for gate, target_qubits in self.gates:
            gate.apply(self.quantum_state, target_qubits)
    
    def reset(self):
        """Reset the circuit to the initial |00...0⟩ state."""
        self.quantum_state = QuantumState(self.num_qubits)
    
    def get_state(self) -> QuantumState:
        """Get the current quantum state."""
        return self.quantum_state
    
    def get_state_vector(self) -> List[complex]:
        """Get the current state vector as a list."""
        return self.quantum_state.state_vector.tolist()
    
    def get_probabilities(self) -> List[float]:
        """Get the probability distribution over all basis states."""
        return self.quantum_state.get_probabilities().tolist()
    
    def __str__(self) -> str:
        """String representation of the circuit."""
        if not self.gates:
            return f"Empty circuit with {self.num_qubits} qubits"
        
        gate_strings = []
        for gate, target_qubits in self.gates:
            if len(target_qubits) == 1:
                gate_strings.append(f"{gate.name} q{target_qubits[0]}")
            else:
                qubit_list = ",".join(f"q{q}" for q in target_qubits)
                gate_strings.append(f"{gate.name} {qubit_list}")
        
        return f"Circuit: {' → '.join(gate_strings)}"
    
    def __len__(self) -> int:
        """Return the number of gates in the circuit."""
        return len(self.gates)
