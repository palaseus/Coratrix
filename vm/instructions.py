"""
Quantum instruction definitions.

This module defines the instruction set for the quantum virtual machine,
including gate instructions and measurement instructions.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict
from enum import Enum


class InstructionType(Enum):
    """Types of quantum instructions."""
    GATE = "gate"
    MEASURE = "measure"
    COMMENT = "comment"


class QuantumInstruction(ABC):
    """
    Abstract base class for quantum instructions.
    
    All quantum instructions must implement the execute method
    to perform their operation on the quantum state.
    """
    
    def __init__(self, instruction_type: InstructionType):
        self.instruction_type = instruction_type
    
    @abstractmethod
    def execute(self, executor) -> Any:
        """
        Execute the instruction.
        
        Args:
            executor: QuantumExecutor instance to execute the instruction
        
        Returns:
            Result of the instruction execution
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the instruction."""
        pass


class GateInstruction(QuantumInstruction):
    """
    Instruction for applying quantum gates.
    
    Supports single-qubit gates (X, Y, Z, H) and two-qubit gates (CNOT).
    """
    
    def __init__(self, gate_name: str, target_qubits: List[int]):
        """
        Initialize a gate instruction.
        
        Args:
            gate_name: Name of the gate (X, Y, Z, H, CNOT)
            target_qubits: List of qubit indices the gate acts on
        """
        super().__init__(InstructionType.GATE)
        self.gate_name = gate_name.upper()
        self.target_qubits = target_qubits
        
        # Validate gate name
        valid_gates = {'X', 'Y', 'Z', 'H', 'CNOT'}
        if self.gate_name not in valid_gates:
            raise ValueError(f"Invalid gate name: {gate_name}. Valid gates: {valid_gates}")
        
        # Validate number of target qubits
        if self.gate_name == 'CNOT' and len(target_qubits) != 2:
            raise ValueError("CNOT gate requires exactly 2 target qubits")
        elif self.gate_name in {'X', 'Y', 'Z', 'H'} and len(target_qubits) != 1:
            raise ValueError(f"{self.gate_name} gate requires exactly 1 target qubit")
    
    def execute(self, executor) -> None:
        """Execute the gate instruction."""
        executor.apply_gate(self.gate_name, self.target_qubits)
    
    def __str__(self) -> str:
        """String representation of the gate instruction."""
        if len(self.target_qubits) == 1:
            return f"{self.gate_name} q{self.target_qubits[0]}"
        else:
            qubit_list = ",".join(f"q{q}" for q in self.target_qubits)
            return f"{self.gate_name} {qubit_list}"


class MeasureInstruction(QuantumInstruction):
    """
    Instruction for measuring qubits.
    
    Can measure all qubits or specific qubits.
    """
    
    def __init__(self, target_qubits: List[int] = None):
        """
        Initialize a measurement instruction.
        
        Args:
            target_qubits: List of qubit indices to measure (None for all qubits)
        """
        super().__init__(InstructionType.MEASURE)
        self.target_qubits = target_qubits
    
    def execute(self, executor) -> List[int]:
        """Execute the measurement instruction."""
        if self.target_qubits is None:
            return executor.measure_all()
        else:
            results = []
            for qubit in self.target_qubits:
                result = executor.measure_qubit(qubit)
                results.append(result)
            return results
    
    def __str__(self) -> str:
        """String representation of the measure instruction."""
        if self.target_qubits is None:
            return "MEASURE"
        else:
            qubit_list = ",".join(f"q{q}" for q in self.target_qubits)
            return f"MEASURE {qubit_list}"


class CommentInstruction(QuantumInstruction):
    """
    Instruction for comments (no operation).
    
    Used for documentation and debugging purposes.
    """
    
    def __init__(self, comment: str):
        """
        Initialize a comment instruction.
        
        Args:
            comment: The comment text
        """
        super().__init__(InstructionType.COMMENT)
        self.comment = comment
    
    def execute(self, executor) -> None:
        """Execute the comment instruction (no operation)."""
        pass
    
    def __str__(self) -> str:
        """String representation of the comment instruction."""
        return f"# {self.comment}"
