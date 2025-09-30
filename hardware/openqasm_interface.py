"""
OpenQASM import/export interface for Coratrix.

This module provides functionality to import and export quantum circuits
in OpenQASM format, enabling interoperability with other quantum computing frameworks.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from core.circuit import QuantumCircuit
from core.gates import XGate, YGate, ZGate, HGate, CNOTGate
from core.advanced_gates import ToffoliGate, SWAPGate
from core.advanced_gates import RxGate, RyGate, RzGate, CPhaseGate


class OpenQASMVersion(Enum):
    """OpenQASM version enumeration."""
    V2_0 = "2.0"
    V3_0 = "3.0"


@dataclass
class OpenQASMInstruction:
    """Represents a single OpenQASM instruction."""
    gate_name: str
    qubits: List[int]
    parameters: List[float]
    classical_bits: Optional[List[int]] = None
    condition: Optional[str] = None


class OpenQASMParser:
    """Parser for OpenQASM quantum assembly language."""
    
    def __init__(self, version: OpenQASMVersion = OpenQASMVersion.V2_0):
        self.version = version
        self.gate_map = {
            'x': XGate,
            'y': YGate,
            'z': ZGate,
            'h': HGate,
            'cx': CNOTGate,
            'cnot': CNOTGate,
            'ccx': ToffoliGate,
            'toffoli': ToffoliGate,
            'swap': SWAPGate,
            'rx': RxGate,
            'ry': RyGate,
            'rz': RzGate,
            'cp': CPhaseGate,
            'cphase': CPhaseGate
        }
    
    def parse_file(self, filename: str) -> QuantumCircuit:
        """Parse an OpenQASM file and return a QuantumCircuit."""
        with open(filename, 'r') as f:
            content = f.read()
        return self.parse_string(content)
    
    def parse_string(self, qasm_string: str) -> QuantumCircuit:
        """Parse an OpenQASM string and return a QuantumCircuit."""
        lines = qasm_string.strip().split('\n')
        
        # Extract header information
        num_qubits = self._extract_num_qubits(lines)
        num_classical_bits = self._extract_num_classical_bits(lines)
        
        # Create circuit
        circuit = QuantumCircuit(num_qubits)
        
        # Parse instructions
        instructions = self._parse_instructions(lines)
        
        # Apply instructions to circuit
        for instruction in instructions:
            self._apply_instruction(circuit, instruction)
        
        return circuit
    
    def _extract_num_qubits(self, lines: List[str]) -> int:
        """Extract number of qubits from OpenQASM header."""
        for line in lines:
            if 'qreg' in line:
                match = re.search(r'qreg\s+(\w+)\[(\d+)\]', line)
                if match:
                    return int(match.group(2))
        return 2  # Default to 2 qubits
    
    def _extract_num_classical_bits(self, lines: List[str]) -> int:
        """Extract number of classical bits from OpenQASM header."""
        for line in lines:
            if 'creg' in line:
                match = re.search(r'creg\s+(\w+)\[(\d+)\]', line)
                if match:
                    return int(match.group(2))
        return 0
    
    def _parse_instructions(self, lines: List[str]) -> List[OpenQASMInstruction]:
        """Parse OpenQASM instructions from lines."""
        instructions = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('OPENQASM') or line.startswith('include'):
                continue
            
            # Parse gate instructions
            if any(gate in line for gate in self.gate_map.keys()):
                instruction = self._parse_gate_instruction(line)
                if instruction:
                    instructions.append(instruction)
            elif line and not line.startswith('qreg') and not line.startswith('creg'):
                # This is a gate instruction but not a known gate
                raise ValueError(f"Unknown gate in line: {line}")
        
        return instructions
    
    def _parse_gate_instruction(self, line: str) -> Optional[OpenQASMInstruction]:
        """Parse a single gate instruction."""
        # Remove comments
        line = line.split('//')[0].strip()
        
        # Parse gate name and parameters
        gate_match = re.match(r'(\w+)(?:\(([^)]+)\))?\s+(.+)', line)
        if not gate_match:
            return None
        
        gate_name = gate_match.group(1)
        param_str = gate_match.group(2)
        qubit_str = gate_match.group(3)
        
        # Parse parameters
        parameters = []
        if param_str:
            param_values = [float(p.strip()) for p in param_str.split(',')]
            parameters = param_values
        
        # Parse qubits
        qubits = self._parse_qubits(qubit_str)
        
        return OpenQASMInstruction(
            gate_name=gate_name,
            qubits=qubits,
            parameters=parameters
        )
    
    def _parse_qubits(self, qubit_str: str) -> List[int]:
        """Parse qubit indices from string."""
        qubits = []
        
        # Handle comma-separated qubits
        for qubit_part in qubit_str.split(','):
            qubit_part = qubit_part.strip()
            
            # Handle qubit ranges like q[0:2]
            if '[' in qubit_part and ':' in qubit_part:
                match = re.match(r'(\w+)\[(\d+):(\d+)\]', qubit_part)
                if match:
                    start = int(match.group(2))
                    end = int(match.group(3))
                    qubits.extend(range(start, end + 1))
            # Handle single qubit like q[0]
            elif '[' in qubit_part:
                match = re.match(r'(\w+)\[(\d+)\]', qubit_part)
                if match:
                    qubits.append(int(match.group(2)))
            # Handle simple qubit names
            else:
                # Extract number from qubit name
                match = re.search(r'(\d+)', qubit_part)
                if match:
                    qubits.append(int(match.group(1)))
        
        return qubits
    
    def _apply_instruction(self, circuit: QuantumCircuit, instruction: OpenQASMInstruction):
        """Apply an OpenQASM instruction to a circuit."""
        gate_name = instruction.gate_name.lower()
        
        if gate_name not in self.gate_map:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        gate_class = self.gate_map[gate_name]
        
        # Create gate instance with parameters if needed
        if instruction.parameters:
            if gate_name in ['rx', 'ry', 'rz']:
                gate = gate_class(instruction.parameters[0])
            elif gate_name in ['cp', 'cphase']:
                gate = gate_class(instruction.parameters[0])
            else:
                gate = gate_class()
        else:
            gate = gate_class()
        
        # Apply gate to circuit
        circuit.add_gate(gate, instruction.qubits)


class OpenQASMExporter:
    """Exporter for OpenQASM quantum assembly language."""
    
    def __init__(self, version: OpenQASMVersion = OpenQASMVersion.V2_0):
        self.version = version
        self.gate_names = {
            XGate: 'x',
            YGate: 'y',
            ZGate: 'z',
            HGate: 'h',
            CNOTGate: 'cx',
            ToffoliGate: 'ccx',
            SWAPGate: 'swap',
            RxGate: 'rx',
            RyGate: 'ry',
            RzGate: 'rz',
            CPhaseGate: 'cp'
        }
    
    def export_circuit(self, circuit: QuantumCircuit, filename: str):
        """Export a QuantumCircuit to OpenQASM file."""
        qasm_string = self.circuit_to_qasm(circuit)
        with open(filename, 'w') as f:
            f.write(qasm_string)
    
    def circuit_to_qasm(self, circuit: QuantumCircuit) -> str:
        """Convert a QuantumCircuit to OpenQASM string."""
        lines = []
        
        # Add header
        lines.append(f"OPENQASM {self.version.value};")
        lines.append("include \"qelib1.inc\";")
        lines.append("")
        
        # Add quantum register
        lines.append(f"qreg q[{circuit.num_qubits}];")
        lines.append("")
        
        # Add circuit instructions
        for gate, target_qubits in circuit.gates:
            instruction = self._gate_to_instruction(gate, target_qubits)
            if instruction:
                lines.append(instruction)
        
        return '\n'.join(lines)
    
    def _gate_to_instruction(self, gate, target_qubits: List[int]) -> Optional[str]:
        """Convert a gate to OpenQASM instruction string."""
        gate_type = type(gate)
        
        if gate_type not in self.gate_names:
            return None
        
        gate_name = self.gate_names[gate_type]
        
        # Format qubit indices
        qubit_str = ','.join(f"q[{i}]" for i in target_qubits)
        
        # Handle parameterized gates
        if hasattr(gate, 'parameters') and 'theta' in gate.parameters:
            if gate_type in [RxGate, RyGate, RzGate]:
                return f"{gate_name}({gate.parameters['theta']}) {qubit_str};"
            elif gate_type == CPhaseGate:
                return f"{gate_name}({gate.parameters['theta']}) {qubit_str};"
        
        # Handle multi-qubit gates
        if gate_type == CNOTGate and len(target_qubits) == 2:
            return f"cx {qubit_str};"
        elif gate_type == ToffoliGate and len(target_qubits) == 3:
            return f"ccx {qubit_str};"
        elif gate_type == SWAPGate and len(target_qubits) == 2:
            return f"swap {qubit_str};"
        
        # Handle single-qubit gates
        if len(target_qubits) == 1:
            return f"{gate_name} {qubit_str};"
        
        return None


class OpenQASMInterface:
    """Main interface for OpenQASM operations."""
    
    def __init__(self, version: OpenQASMVersion = OpenQASMVersion.V2_0):
        self.version = version
        self.parser = OpenQASMParser(version)
        self.exporter = OpenQASMExporter(version)
    
    def import_circuit(self, filename: str) -> QuantumCircuit:
        """Import a circuit from OpenQASM file."""
        return self.parser.parse_file(filename)
    
    def import_circuit_string(self, qasm_string: str) -> QuantumCircuit:
        """Import a circuit from OpenQASM string."""
        return self.parser.parse_string(qasm_string)
    
    def export_circuit(self, circuit: QuantumCircuit, filename: str):
        """Export a circuit to OpenQASM file."""
        self.exporter.export_circuit(circuit, filename)
    
    def circuit_to_qasm(self, circuit: QuantumCircuit) -> str:
        """Convert a circuit to OpenQASM string."""
        return self.exporter.circuit_to_qasm(circuit)
    
    def validate_qasm(self, qasm_string: str) -> Tuple[bool, List[str]]:
        """Validate OpenQASM syntax and return errors if any."""
        errors = []
        
        try:
            # Try to parse the QASM string
            circuit = self.parser.parse_string(qasm_string)
            
            # Basic validation
            if circuit.num_qubits <= 0:
                errors.append("Invalid number of qubits")
            
            # Check for unsupported gates
            for gate, target_qubits in circuit.gates:
                if type(gate) not in self.parser.gate_map.values():
                    errors.append(f"Unsupported gate: {type(gate).__name__}")
        
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
        
        return len(errors) == 0, errors
