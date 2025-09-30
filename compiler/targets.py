"""
Target code generators for the Coratrix compiler.

This module provides generators for various target formats including
OpenQASM, Qiskit, PennyLane, and other quantum programming frameworks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .ir import CoratrixIR, IRCircuit, IRFunction, IRStatement, IRExpression
from .ir import IROperation, IRType, IRVariable, IROperand, IRValue


class TargetFormat(Enum):
    """Supported target formats."""
    OPENQASM = "openqasm"
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"
    QSHARP = "qsharp"
    BRAKET = "braket"


@dataclass
class TargetResult:
    """Result of target code generation."""
    success: bool
    code: str
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class TargetGenerator(ABC):
    """Base class for target code generators."""
    
    def __init__(self, target_format: TargetFormat):
        self.target_format = target_format
    
    @abstractmethod
    def generate(self, ir: CoratrixIR) -> TargetResult:
        """Generate target code from IR."""
        pass
    
    @abstractmethod
    def generate_circuit(self, circuit: IRCircuit) -> str:
        """Generate code for a single circuit."""
        pass


class QASMTarget(TargetGenerator):
    """Generator for OpenQASM target."""
    
    def __init__(self):
        super().__init__(TargetFormat.OPENQASM)
        self.version = "2.0"
        self.include_header = True
    
    def generate(self, ir: CoratrixIR) -> TargetResult:
        """Generate OpenQASM code from IR."""
        try:
            code_lines = []
            
            # Header
            if self.include_header:
                code_lines.append(f"OPENQASM {self.version};")
                code_lines.append('include "qelib1.inc";')
                code_lines.append("")
            
            # Generate circuits
            for circuit in ir.circuits:
                circuit_code = self.generate_circuit(circuit)
                code_lines.append(circuit_code)
                code_lines.append("")
            
            # Generate functions (as custom gates)
            for function in ir.functions:
                function_code = self._generate_function_as_gate(function)
                code_lines.append(function_code)
                code_lines.append("")
            
            code = "\n".join(code_lines)
            return TargetResult(success=True, code=code)
            
        except Exception as e:
            return TargetResult(success=False, code="", errors=[str(e)])
    
    def generate_circuit(self, circuit: IRCircuit) -> str:
        """Generate OpenQASM code for a circuit."""
        lines = []
        
        # Circuit declaration
        lines.append(f"// Circuit: {circuit.name}")
        
        # Declare qubits and classical bits
        num_qubits = len(circuit.qubits)
        num_classical = len(circuit.classical_bits)
        
        lines.append(f"qreg q[{num_qubits}];")
        if num_classical > 0:
            lines.append(f"creg c[{num_classical}];")
        lines.append("")
        
        # Generate statements
        for statement in circuit.body.statements:
            stmt_code = self._generate_statement(statement)
            if stmt_code:
                lines.append(stmt_code)
        
        return "\n".join(lines)
    
    def _generate_statement(self, statement: IRStatement) -> str:
        """Generate OpenQASM code for a statement."""
        if statement.operation == IROperation.H:
            return self._generate_h_gate(statement)
        elif statement.operation == IROperation.X:
            return self._generate_x_gate(statement)
        elif statement.operation == IROperation.Y:
            return self._generate_y_gate(statement)
        elif statement.operation == IROperation.Z:
            return self._generate_z_gate(statement)
        elif statement.operation == IROperation.CNOT:
            return self._generate_cnot_gate(statement)
        elif statement.operation == IROperation.CZ:
            return self._generate_cz_gate(statement)
        elif statement.operation == IROperation.CPHASE:
            return self._generate_cphase_gate(statement)
        elif statement.operation == IROperation.MEASURE:
            return self._generate_measurement(statement)
        elif statement.operation == IROperation.RX:
            return self._generate_rx_gate(statement)
        elif statement.operation == IROperation.RY:
            return self._generate_ry_gate(statement)
        elif statement.operation == IROperation.RZ:
            return self._generate_rz_gate(statement)
        else:
            return f"// {statement.operation.value} gate"
    
    def _generate_h_gate(self, statement: IRStatement) -> str:
        """Generate H gate."""
        qubit = self._get_qubit_name(statement.operands[0])
        return f"h q[{qubit}];"
    
    def _generate_x_gate(self, statement: IRStatement) -> str:
        """Generate X gate."""
        qubit = self._get_qubit_name(statement.operands[0])
        return f"x q[{qubit}];"
    
    def _generate_y_gate(self, statement: IRStatement) -> str:
        """Generate Y gate."""
        qubit = self._get_qubit_name(statement.operands[0])
        return f"y q[{qubit}];"
    
    def _generate_z_gate(self, statement: IRStatement) -> str:
        """Generate Z gate."""
        qubit = self._get_qubit_name(statement.operands[0])
        return f"z q[{qubit}];"
    
    def _generate_cnot_gate(self, statement: IRStatement) -> str:
        """Generate CNOT gate."""
        control = self._get_qubit_name(statement.operands[0])
        target = self._get_qubit_name(statement.operands[1])
        return f"cx q[{control}], q[{target}];"
    
    def _generate_cz_gate(self, statement: IRStatement) -> str:
        """Generate CZ gate."""
        control = self._get_qubit_name(statement.operands[0])
        target = self._get_qubit_name(statement.operands[1])
        return f"cz q[{control}], q[{target}];"
    
    def _generate_cphase_gate(self, statement: IRStatement) -> str:
        """Generate CPhase gate."""
        control = self._get_qubit_name(statement.operands[0])
        target = self._get_qubit_name(statement.operands[1])
        if len(statement.operands) > 2:
            phase = self._get_parameter_value(statement.operands[2])
            return f"cp({phase}) q[{control}], q[{target}];"
        else:
            return f"cp q[{control}], q[{target}];"
    
    def _generate_measurement(self, statement: IRStatement) -> str:
        """Generate measurement."""
        qubit = self._get_qubit_name(statement.operands[0])
        classical = self._get_classical_name(statement.operands[1])
        return f"measure q[{qubit}] -> c[{classical}];"
    
    def _generate_rx_gate(self, statement: IRStatement) -> str:
        """Generate RX gate."""
        qubit = self._get_qubit_name(statement.operands[0])
        angle = self._get_parameter_value(statement.operands[1])
        return f"rx({angle}) q[{qubit}];"
    
    def _generate_ry_gate(self, statement: IRStatement) -> str:
        """Generate RY gate."""
        qubit = self._get_qubit_name(statement.operands[0])
        angle = self._get_parameter_value(statement.operands[1])
        return f"ry({angle}) q[{qubit}];"
    
    def _generate_rz_gate(self, statement: IRStatement) -> str:
        """Generate RZ gate."""
        qubit = self._get_qubit_name(statement.operands[0])
        angle = self._get_parameter_value(statement.operands[1])
        return f"rz({angle}) q[{qubit}];"
    
    def _get_qubit_name(self, operand: IROperand) -> int:
        """Get qubit index from operand."""
        if operand.variable:
            # Extract index from variable name (e.g., "q0" -> 0)
            name = operand.variable.name
            if name.startswith('q'):
                try:
                    return int(name[1:])
                except ValueError:
                    pass
        return 0  # Default
    
    def _get_classical_name(self, operand: IROperand) -> int:
        """Get classical bit index from operand."""
        if operand.variable:
            name = operand.variable.name
            if name.startswith('c'):
                try:
                    return int(name[1:])
                except ValueError:
                    pass
        return 0  # Default
    
    def _get_parameter_value(self, operand: IROperand) -> float:
        """Get parameter value from operand."""
        if operand.value:
            return operand.value.value
        return 0.0  # Default
    
    def _generate_function_as_gate(self, function: IRFunction) -> str:
        """Generate function as custom gate."""
        lines = []
        
        # Gate definition
        param_str = ", ".join(f"{param.name}" for param in function.parameters)
        qubit_str = ", ".join(f"q{i}" for i in range(4))  # Default qubits
        
        lines.append(f"gate {function.name}({param_str}) {qubit_str} {{")
        
        # Generate function body
        for statement in function.body.statements:
            stmt_code = self._generate_statement(statement)
            if stmt_code:
                lines.append(f"    {stmt_code}")
        
        lines.append("}")
        
        return "\n".join(lines)


class QiskitTarget(TargetGenerator):
    """Generator for Qiskit target."""
    
    def __init__(self):
        super().__init__(TargetFormat.QISKIT)
    
    def generate(self, ir: CoratrixIR) -> TargetResult:
        """Generate Qiskit code from IR."""
        try:
            code_lines = []
            
            # Imports
            code_lines.append("from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister")
            code_lines.append("from qiskit.circuit import Parameter")
            code_lines.append("")
            
            # Generate circuits
            for circuit in ir.circuits:
                circuit_code = self.generate_circuit(circuit)
                code_lines.append(circuit_code)
                code_lines.append("")
            
            code = "\n".join(code_lines)
            return TargetResult(success=True, code=code)
            
        except Exception as e:
            return TargetResult(success=False, code="", errors=[str(e)])
    
    def generate_circuit(self, circuit: IRCircuit) -> str:
        """Generate Qiskit code for a circuit."""
        lines = []
        
        # Circuit creation
        num_qubits = len(circuit.qubits)
        num_classical = len(circuit.classical_bits)
        
        lines.append(f"# Circuit: {circuit.name}")
        lines.append(f"qc = QuantumCircuit({num_qubits})")
        
        if num_classical > 0:
            lines.append(f"qc.add_register(ClassicalRegister({num_classical}, 'c'))")
        
        lines.append("")
        
        # Generate statements
        for statement in circuit.body.statements:
            stmt_code = self._generate_statement(statement)
            if stmt_code:
                lines.append(stmt_code)
        
        return "\n".join(lines)
    
    def _generate_statement(self, statement: IRStatement) -> str:
        """Generate Qiskit code for a statement."""
        if statement.operation == IROperation.H:
            return self._generate_h_gate(statement)
        elif statement.operation == IROperation.X:
            return self._generate_x_gate(statement)
        elif statement.operation == IROperation.Y:
            return self._generate_y_gate(statement)
        elif statement.operation == IROperation.Z:
            return self._generate_z_gate(statement)
        elif statement.operation == IROperation.CNOT:
            return self._generate_cnot_gate(statement)
        elif statement.operation == IROperation.MEASURE:
            return self._generate_measurement(statement)
        else:
            return f"# {statement.operation.value} gate"
    
    def _generate_h_gate(self, statement: IRStatement) -> str:
        """Generate H gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qc.h({qubit})"
    
    def _generate_x_gate(self, statement: IRStatement) -> str:
        """Generate X gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qc.x({qubit})"
    
    def _generate_y_gate(self, statement: IRStatement) -> str:
        """Generate Y gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qc.y({qubit})"
    
    def _generate_z_gate(self, statement: IRStatement) -> str:
        """Generate Z gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qc.z({qubit})"
    
    def _generate_cnot_gate(self, statement: IRStatement) -> str:
        """Generate CNOT gate."""
        control = self._get_qubit_index(statement.operands[0])
        target = self._get_qubit_index(statement.operands[1])
        return f"qc.cx({control}, {target})"
    
    def _generate_measurement(self, statement: IRStatement) -> str:
        """Generate measurement."""
        qubit = self._get_qubit_index(statement.operands[0])
        classical = self._get_classical_index(statement.operands[1])
        return f"qc.measure({qubit}, {classical})"
    
    def _get_qubit_index(self, operand: IROperand) -> int:
        """Get qubit index from operand."""
        if operand.variable:
            name = operand.variable.name
            if name.startswith('q'):
                try:
                    return int(name[1:])
                except ValueError:
                    pass
        return 0
    
    def _get_classical_index(self, operand: IROperand) -> int:
        """Get classical bit index from operand."""
        if operand.variable:
            name = operand.variable.name
            if name.startswith('c'):
                try:
                    return int(name[1:])
                except ValueError:
                    pass
        return 0


class PennyLaneTarget(TargetGenerator):
    """Generator for PennyLane target."""
    
    def __init__(self):
        super().__init__(TargetFormat.PENNYLANE)
    
    def generate(self, ir: CoratrixIR) -> TargetResult:
        """Generate PennyLane code from IR."""
        try:
            code_lines = []
            
            # Imports
            code_lines.append("import pennylane as qml")
            code_lines.append("")
            
            # Generate circuits
            for circuit in ir.circuits:
                circuit_code = self.generate_circuit(circuit)
                code_lines.append(circuit_code)
                code_lines.append("")
            
            code = "\n".join(code_lines)
            return TargetResult(success=True, code=code)
            
        except Exception as e:
            return TargetResult(success=False, code="", errors=[str(e)])
    
    def generate_circuit(self, circuit: IRCircuit) -> str:
        """Generate PennyLane code for a circuit."""
        lines = []
        
        # Circuit definition
        num_qubits = len(circuit.qubits)
        
        lines.append(f"# Circuit: {circuit.name}")
        lines.append(f"@qml.qnode(dev)")
        lines.append("def circuit():")
        
        # Generate statements
        for statement in circuit.body.statements:
            stmt_code = self._generate_statement(statement)
            if stmt_code:
                lines.append(f"    {stmt_code}")
        
        lines.append("    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]")
        
        return "\n".join(lines)
    
    def _generate_statement(self, statement: IRStatement) -> str:
        """Generate PennyLane code for a statement."""
        if statement.operation == IROperation.H:
            return self._generate_h_gate(statement)
        elif statement.operation == IROperation.X:
            return self._generate_x_gate(statement)
        elif statement.operation == IROperation.Y:
            return self._generate_y_gate(statement)
        elif statement.operation == IROperation.Z:
            return self._generate_z_gate(statement)
        elif statement.operation == IROperation.CNOT:
            return self._generate_cnot_gate(statement)
        else:
            return f"# {statement.operation.value} gate"
    
    def _generate_h_gate(self, statement: IRStatement) -> str:
        """Generate H gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qml.Hadamard(wires={qubit})"
    
    def _generate_x_gate(self, statement: IRStatement) -> str:
        """Generate X gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qml.PauliX(wires={qubit})"
    
    def _generate_y_gate(self, statement: IRStatement) -> str:
        """Generate Y gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qml.PauliY(wires={qubit})"
    
    def _generate_z_gate(self, statement: IRStatement) -> str:
        """Generate Z gate."""
        qubit = self._get_qubit_index(statement.operands[0])
        return f"qml.PauliZ(wires={qubit})"
    
    def _generate_cnot_gate(self, statement: IRStatement) -> str:
        """Generate CNOT gate."""
        control = self._get_qubit_index(statement.operands[0])
        target = self._get_qubit_index(statement.operands[1])
        return f"qml.CNOT(wires=[{control}, {target}])"
    
    def _get_qubit_index(self, operand: IROperand) -> int:
        """Get qubit index from operand."""
        if operand.variable:
            name = operand.variable.name
            if name.startswith('q'):
                try:
                    return int(name[1:])
                except ValueError:
                    pass
        return 0
