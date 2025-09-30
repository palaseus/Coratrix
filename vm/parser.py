"""
Quantum instruction parser.

This module provides the QuantumParser class for parsing quantum
instruction scripts into executable instruction objects.
"""

import re
from typing import List, Optional, Union, Dict
from vm.instructions import QuantumInstruction, GateInstruction, MeasureInstruction, CommentInstruction


class QuantumParser:
    """
    Parser for quantum instruction scripts.
    
    Parses text-based quantum instructions into executable
    instruction objects for the quantum virtual machine.
    """
    
    def __init__(self):
        """Initialize the quantum parser."""
        # Define instruction patterns
        self.patterns = {
            'gate': re.compile(r'^(X|Y|Z|H|CNOT)\s+q(\d+)(?:\s*,\s*q(\d+))?$', re.IGNORECASE),
            'measure': re.compile(r'^MEASURE(?:\s+q(\d+)(?:\s*,\s*q(\d+))*)?$', re.IGNORECASE),
            'comment': re.compile(r'^#\s*(.*)$')
        }
    
    def parse_line(self, line: str) -> Optional[QuantumInstruction]:
        """
        Parse a single line of quantum instructions.
        
        Args:
            line: Line of text containing quantum instructions
        
        Returns:
            Parsed instruction object or None for empty lines
        """
        # Strip whitespace and handle empty lines
        line = line.strip()
        if not line:
            return None
        
        # Try to match comment pattern
        comment_match = self.patterns['comment'].match(line)
        if comment_match:
            return CommentInstruction(comment_match.group(1))
        
        # Try to match gate pattern
        gate_match = self.patterns['gate'].match(line)
        if gate_match:
            gate_name = gate_match.group(1).upper()
            qubit1 = int(gate_match.group(2))
            
            if gate_name == 'CNOT':
                qubit2 = int(gate_match.group(3))
                if qubit2 is None:
                    raise ValueError("CNOT gate requires two qubits")
                return GateInstruction(gate_name, [qubit1, qubit2])
            else:
                return GateInstruction(gate_name, [qubit1])
        
        # Try to match measure pattern
        measure_match = self.patterns['measure'].match(line)
        if measure_match:
            # Extract all qubit indices from the match
            qubits = []
            for i in range(1, len(measure_match.groups()) + 1):
                if measure_match.group(i) is not None:
                    qubits.append(int(measure_match.group(i)))
            
            if not qubits:
                return MeasureInstruction()  # Measure all qubits
            else:
                return MeasureInstruction(qubits)
        
        # If no pattern matches, raise an error
        raise ValueError(f"Invalid instruction: {line}")
    
    def parse_script(self, script: str) -> List[QuantumInstruction]:
        """
        Parse a complete quantum script.
        
        Args:
            script: Multi-line string containing quantum instructions
        
        Returns:
            List of parsed instruction objects
        """
        instructions = []
        lines = script.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            try:
                instruction = self.parse_line(line)
                if instruction is not None:
                    instructions.append(instruction)
            except ValueError as e:
                raise ValueError(f"Parse error at line {line_num}: {e}")
        
        return instructions
    
    def parse_file(self, filename: str) -> List[QuantumInstruction]:
        """
        Parse a quantum script from a file.
        
        Args:
            filename: Path to the quantum script file
        
        Returns:
            List of parsed instruction objects
        """
        try:
            with open(filename, 'r') as file:
                script = file.read()
            return self.parse_script(script)
        except FileNotFoundError:
            raise FileNotFoundError(f"Quantum script file not found: {filename}")
        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {e}")
    
    def validate_instructions(self, instructions: List[QuantumInstruction], num_qubits: int) -> None:
        """
        Validate that all instructions are compatible with the number of qubits.
        
        Args:
            instructions: List of instruction objects to validate
            num_qubits: Number of qubits in the system
        
        Raises:
            ValueError: If any instruction references invalid qubit indices
        """
        for i, instruction in enumerate(instructions):
            if isinstance(instruction, GateInstruction):
                for qubit in instruction.target_qubits:
                    if not (0 <= qubit < num_qubits):
                        raise ValueError(f"Instruction {i+1}: Qubit {qubit} out of range [0, {num_qubits-1}]")
            elif isinstance(instruction, MeasureInstruction):
                if instruction.target_qubits is not None:
                    for qubit in instruction.target_qubits:
                        if not (0 <= qubit < num_qubits):
                            raise ValueError(f"Instruction {i+1}: Qubit {qubit} out of range [0, {num_qubits-1}]")
    
    def get_supported_instructions(self) -> Dict[str, str]:
        """
        Get information about supported instructions.
        
        Returns:
            Dictionary mapping instruction names to descriptions
        """
        return {
            'X': 'Pauli-X gate (quantum NOT)',
            'Y': 'Pauli-Y gate',
            'Z': 'Pauli-Z gate',
            'H': 'Hadamard gate (creates superposition)',
            'CNOT': 'Controlled-NOT gate (entanglement)',
            'MEASURE': 'Measure qubits (causes state collapse)',
            '#': 'Comment (ignored during execution)'
        }
