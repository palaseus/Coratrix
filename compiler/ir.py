"""
Coratrix Intermediate Representation (IR).

This module defines the intermediate representation used by the Coratrix compiler
to represent quantum circuits in a platform-agnostic way.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


class IROperation(Enum):
    """Types of IR operations."""
    # Quantum gates
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    S = "s"
    T = "t"
    CNOT = "cnot"
    CZ = "cz"
    CPHASE = "cphase"
    SWAP = "swap"
    TOFFOLI = "toffoli"
    
    # Rotations
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    ROTATION = "rotation"
    
    # Custom gates
    CUSTOM = "custom"
    
    # Control flow
    IF = "if"
    FOR = "for"
    WHILE = "while"
    CALL = "call"
    
    # Measurement
    MEASURE = "measure"
    
    # Classical operations
    ASSIGN = "assign"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    
    # Memory operations
    LOAD = "load"
    STORE = "store"


class IRType(Enum):
    """Types in the IR type system."""
    QUBIT = "qubit"
    CLASSICAL = "classical"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    ARRAY = "array"
    FUNCTION = "function"


@dataclass
class IRValue:
    """A value in the IR."""
    type: IRType
    value: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class IRVariable:
    """A variable in the IR."""
    name: str
    type: IRType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope: Optional[str] = None


@dataclass
class IROperand:
    """An operand in an IR operation."""
    variable: Optional[IRVariable] = None
    value: Optional[IRValue] = None
    expression: Optional['IRExpression'] = None


@dataclass
class IRExpression:
    """An expression in the IR."""
    operation: IROperation
    operands: List[IROperand]
    result: Optional[IRVariable] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class IRStatement:
    """A statement in the IR."""
    operation: IROperation
    operands: List[IROperand]
    result: Optional[IRVariable] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRBlock:
    """A block of IR statements."""
    statements: List[IRStatement]
    variables: Dict[str, IRVariable] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRFunction:
    """A function in the IR."""
    name: str
    parameters: List[IRVariable]
    return_type: Optional[IRType]
    body: IRBlock
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRCircuit:
    """A quantum circuit in the IR."""
    name: str
    parameters: List[IRVariable]
    qubits: List[IRVariable]
    classical_bits: List[IRVariable]
    body: IRBlock
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoratrixIR:
    """The complete Coratrix IR representation."""
    circuits: List[IRCircuit] = field(default_factory=list)
    functions: List[IRFunction] = field(default_factory=list)
    global_variables: Dict[str, IRVariable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_circuit(self, circuit: IRCircuit):
        """Add a circuit to the IR."""
        self.circuits.append(circuit)
    
    def add_function(self, function: IRFunction):
        """Add a function to the IR."""
        self.functions.append(function)
    
    def get_circuit(self, name: str) -> Optional[IRCircuit]:
        """Get a circuit by name."""
        for circuit in self.circuits:
            if circuit.name == name:
                return circuit
        return None
    
    def get_function(self, name: str) -> Optional[IRFunction]:
        """Get a function by name."""
        for function in self.functions:
            if function.name == name:
                return function
        return None


class IRBuilder:
    """Builder for creating Coratrix IR."""
    
    def __init__(self):
        self.ir = CoratrixIR()
        self.current_scope = None
        self.variable_map = {}
    
    def create_circuit(self, name: str, parameters: List[str] = None, 
                      qubits: List[str] = None, classical_bits: List[str] = None) -> IRCircuit:
        """Create a new quantum circuit."""
        if parameters is None:
            parameters = []
        if qubits is None:
            qubits = []
        if classical_bits is None:
            classical_bits = []
        
        # Create parameter variables
        param_vars = []
        for param in parameters:
            param_var = IRVariable(param, IRType.INTEGER)
            param_vars.append(param_var)
            self.variable_map[param] = param_var
        
        # Create qubit variables
        qubit_vars = []
        for i, qubit in enumerate(qubits):
            qubit_var = IRVariable(qubit, IRType.QUBIT)
            qubit_vars.append(qubit_var)
            self.variable_map[qubit] = qubit_var
        
        # Create classical bit variables
        classical_vars = []
        for i, bit in enumerate(classical_bits):
            bit_var = IRVariable(bit, IRType.CLASSICAL)
            classical_vars.append(bit_var)
            self.variable_map[bit] = bit_var
        
        circuit = IRCircuit(
            name=name,
            parameters=param_vars,
            qubits=qubit_vars,
            classical_bits=classical_vars,
            body=IRBlock([])
        )
        
        self.ir.add_circuit(circuit)
        self.current_scope = circuit.id
        return circuit
    
    def create_function(self, name: str, parameters: List[tuple], 
                       return_type: Optional[IRType] = None) -> IRFunction:
        """Create a new function."""
        param_vars = []
        for param_name, param_type in parameters:
            param_var = IRVariable(param_name, param_type)
            param_vars.append(param_var)
            self.variable_map[param_name] = param_var
        
        function = IRFunction(
            name=name,
            parameters=param_vars,
            return_type=return_type,
            body=IRBlock([])
        )
        
        self.ir.add_function(function)
        self.current_scope = function.id
        return function
    
    def add_gate(self, gate_type: IROperation, qubits: List[str], 
                 parameters: List[Any] = None) -> IRStatement:
        """Add a gate operation."""
        if parameters is None:
            parameters = []
        
        # Create operands
        operands = []
        for qubit in qubits:
            if qubit in self.variable_map:
                operands.append(IROperand(variable=self.variable_map[qubit]))
            else:
                # Create new qubit variable
                qubit_var = IRVariable(qubit, IRType.QUBIT)
                self.variable_map[qubit] = qubit_var
                operands.append(IROperand(variable=qubit_var))
        
        for param in parameters:
            if isinstance(param, (int, float)):
                value = IRValue(IRType.FLOAT if isinstance(param, float) else IRType.INTEGER, param)
                operands.append(IROperand(value=value))
            else:
                # Assume it's a variable reference
                if param in self.variable_map:
                    operands.append(IROperand(variable=self.variable_map[param]))
                else:
                    # Create new variable
                    param_var = IRVariable(param, IRType.FLOAT)
                    self.variable_map[param] = param_var
                    operands.append(IROperand(variable=param_var))
        
        statement = IRStatement(operation=gate_type, operands=operands)
        
        # Add to current scope
        if self.current_scope:
            for circuit in self.ir.circuits:
                if circuit.id == self.current_scope:
                    circuit.body.statements.append(statement)
                    break
            for function in self.ir.functions:
                if function.id == self.current_scope:
                    function.body.statements.append(statement)
                    break
        
        return statement
    
    def add_measurement(self, qubit: str, classical_bit: str) -> IRStatement:
        """Add a measurement operation."""
        qubit_var = self.variable_map.get(qubit)
        if not qubit_var:
            qubit_var = IRVariable(qubit, IRType.QUBIT)
            self.variable_map[qubit] = qubit_var
        
        classical_var = self.variable_map.get(classical_bit)
        if not classical_var:
            classical_var = IRVariable(classical_bit, IRType.CLASSICAL)
            self.variable_map[classical_bit] = classical_var
        
        statement = IRStatement(
            operation=IROperation.MEASURE,
            operands=[
                IROperand(variable=qubit_var),
                IROperand(variable=classical_var)
            ]
        )
        
        # Add to current scope
        if self.current_scope:
            for circuit in self.ir.circuits:
                if circuit.id == self.current_scope:
                    circuit.body.statements.append(statement)
                    break
            for function in self.ir.functions:
                if function.id == self.current_scope:
                    function.body.statements.append(statement)
                    break
        
        return statement
    
    def add_control_flow(self, operation: IROperation, condition: IRExpression = None,
                        body: List[IRStatement] = None) -> IRStatement:
        """Add control flow operations."""
        if body is None:
            body = []
        
        operands = []
        if condition:
            operands.append(IROperand(expression=condition))
        
        statement = IRStatement(
            operation=operation,
            operands=operands,
            metadata={'body': body}
        )
        
        # Add to current scope
        if self.current_scope:
            for circuit in self.ir.circuits:
                if circuit.id == self.current_scope:
                    circuit.body.statements.append(statement)
                    break
            for function in self.ir.functions:
                if function.id == self.current_scope:
                    function.body.statements.append(statement)
                    break
        
        return statement
    
    def create_expression(self, operation: IROperation, operands: List[IROperand]) -> IRExpression:
        """Create an expression."""
        return IRExpression(operation=operation, operands=operands)
    
    def get_ir(self) -> CoratrixIR:
        """Get the built IR."""
        return self.ir


class IROptimizer:
    """Optimizer for Coratrix IR."""
    
    def __init__(self):
        self.optimizations = [
            self._optimize_gate_merging,
            self._optimize_redundant_operations,
            self._optimize_constant_folding,
            self._optimize_dead_code_elimination
        ]
    
    def optimize(self, ir: CoratrixIR) -> CoratrixIR:
        """Apply all optimizations to the IR."""
        optimized_ir = ir
        
        for optimization in self.optimizations:
            optimized_ir = optimization(optimized_ir)
        
        return optimized_ir
    
    def _optimize_gate_merging(self, ir: CoratrixIR) -> CoratrixIR:
        """Merge adjacent gates that can be combined."""
        for circuit in ir.circuits:
            statements = circuit.body.statements
            i = 0
            while i < len(statements) - 1:
                current = statements[i]
                next_stmt = statements[i + 1]
                
                # Check if gates can be merged
                if (current.operation == next_stmt.operation and
                    current.operation in [IROperation.H, IROperation.X, IROperation.Y, IROperation.Z]):
                    # Merge identical gates (they cancel out)
                    statements.pop(i)
                    statements.pop(i)
                    continue
                
                i += 1
        
        return ir
    
    def _optimize_redundant_operations(self, ir: CoratrixIR) -> CoratrixIR:
        """Remove redundant operations."""
        for circuit in ir.circuits:
            statements = circuit.body.statements
            i = 0
            while i < len(statements):
                stmt = statements[i]
                
                # Remove redundant H gates (H^2 = I)
                if stmt.operation == IROperation.H:
                    # Look for another H gate on the same qubit
                    for j in range(i + 1, len(statements)):
                        if (statements[j].operation == IROperation.H and
                            stmt.operands[0].variable == statements[j].operands[0].variable):
                            # Remove both gates
                            statements.pop(j)
                            statements.pop(i)
                            i -= 1
                            break
                
                i += 1
        
        return ir
    
    def _optimize_constant_folding(self, ir: CoratrixIR) -> CoratrixIR:
        """Fold constant expressions."""
        # This would evaluate constant expressions at compile time
        # For now, it's a placeholder
        return ir
    
    def _optimize_dead_code_elimination(self, ir: CoratrixIR) -> CoratrixIR:
        """Remove unreachable code."""
        # This would remove code that can never be executed
        # For now, it's a placeholder
        return ir
