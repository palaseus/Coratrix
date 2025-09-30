"""
Compiler passes for the Coratrix compiler.

This module defines the compiler pass system that transforms DSL AST
to Coratrix IR and applies various optimizations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .dsl import DSLNode, QuantumProgram, QuantumCircuit, GateDefinition, FunctionDefinition
from .dsl import GateCall, MeasureStatement, IfStatement, ForStatement, WhileStatement
from .dsl import Expression, NumberLiteral, StringLiteral, Identifier, BinaryExpression, UnaryExpression, FunctionCall
from .ir import CoratrixIR, IRBuilder, IRCircuit, IRFunction, IRStatement, IRExpression
from .ir import IROperation, IRType, IRVariable, IROperand, IRValue


class PassType(Enum):
    """Types of compiler passes."""
    FRONTEND = "frontend"      # DSL to IR
    OPTIMIZATION = "optimization"  # IR optimizations
    BACKEND = "backend"        # IR to target code


@dataclass
class PassResult:
    """Result of a compiler pass."""
    success: bool
    output: Any
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class CompilerPass(ABC):
    """Base class for compiler passes."""
    
    def __init__(self, name: str, pass_type: PassType):
        self.name = name
        self.pass_type = pass_type
    
    @abstractmethod
    def run(self, input_data: Any) -> PassResult:
        """Run the compiler pass."""
        pass
    
    def __str__(self) -> str:
        return f"{self.name} ({self.pass_type.value})"


class DSLToIRPass(CompilerPass):
    """Pass to convert DSL AST to Coratrix IR."""
    
    def __init__(self):
        super().__init__("DSL to IR", PassType.FRONTEND)
        self.builder = IRBuilder()
    
    def run(self, ast: QuantumProgram) -> PassResult:
        """Convert DSL AST to IR."""
        try:
            # Process circuits
            for circuit in ast.circuits:
                self._process_circuit(circuit)
            
            # Process functions
            for function in ast.functions:
                self._process_function(function)
            
            # Process gate definitions
            for gate in ast.gates:
                self._process_gate_definition(gate)
            
            ir = self.builder.get_ir()
            return PassResult(success=True, output=ir)
            
        except Exception as e:
            return PassResult(success=False, output=None, errors=[str(e)])
    
    def _process_circuit(self, circuit: QuantumCircuit):
        """Process a circuit definition."""
        # Create circuit in IR
        param_names = [param for param in circuit.parameters]
        qubit_names = [f"q{i}" for i in range(4)]  # Default qubits
        classical_names = [f"c{i}" for i in range(4)]  # Default classical bits
        
        ir_circuit = self.builder.create_circuit(
            circuit.name, param_names, qubit_names, classical_names
        )
        
        # Process circuit body
        for statement in circuit.body:
            self._process_statement(statement, ir_circuit)
    
    def _process_function(self, function: FunctionDefinition):
        """Process a function definition."""
        # Map return type
        return_type = None
        if function.return_type:
            return_type = self._map_dsl_type_to_ir(function.return_type)
        
        # Create function parameters
        parameters = []
        for param in function.parameters:
            param_type = IRType.INTEGER  # Default type
            parameters.append((param, param_type))
        
        ir_function = self.builder.create_function(
            function.name, parameters, return_type
        )
        
        # Process function body
        for statement in function.body:
            self._process_statement(statement, ir_function)
    
    def _process_gate_definition(self, gate: GateDefinition):
        """Process a gate definition."""
        # Gate definitions are handled as functions in the IR
        parameters = []
        for param in gate.parameters:
            parameters.append((param, IRType.FLOAT))
        
        ir_function = self.builder.create_function(
            gate.name, parameters, None
        )
        
        # Process gate body
        for statement in gate.body:
            self._process_statement(statement, ir_function)
    
    def _process_statement(self, statement: Any, scope: Any):
        """Process a statement."""
        if isinstance(statement, GateCall):
            self._process_gate_call(statement)
        elif isinstance(statement, MeasureStatement):
            self._process_measurement(statement)
        elif isinstance(statement, IfStatement):
            self._process_if_statement(statement)
        elif isinstance(statement, ForStatement):
            self._process_for_statement(statement)
        elif isinstance(statement, WhileStatement):
            self._process_while_statement(statement)
    
    def _process_gate_call(self, gate_call: GateCall):
        """Process a gate call."""
        # Map gate name to IR operation
        gate_mapping = {
            "h": IROperation.H,
            "x": IROperation.X,
            "y": IROperation.Y,
            "z": IROperation.Z,
            "cnot": IROperation.CNOT,
            "cz": IROperation.CZ,
            "cphase": IROperation.CPHASE,
            "rotation": IROperation.ROTATION
        }
        
        operation = gate_mapping.get(gate_call.gate_name.lower(), IROperation.CUSTOM)
        
        # Extract qubit names
        qubit_names = []
        for qubit_expr in gate_call.qubits:
            if isinstance(qubit_expr, Identifier):
                qubit_names.append(qubit_expr.name)
            else:
                # Handle more complex expressions
                qubit_names.append(f"q{len(qubit_names)}")
        
        # Extract parameters
        parameters = []
        for param_expr in gate_call.parameters:
            if isinstance(param_expr, NumberLiteral):
                parameters.append(param_expr.value)
            elif isinstance(param_expr, Identifier):
                parameters.append(param_expr.name)
            else:
                parameters.append(0.0)  # Default parameter
        
        # Add gate to IR
        self.builder.add_gate(operation, qubit_names, parameters)
    
    def _process_measurement(self, measure: MeasureStatement):
        """Process a measurement statement."""
        qubit_name = self._extract_identifier_name(measure.qubit)
        classical_name = self._extract_identifier_name(measure.classical_bit)
        
        self.builder.add_measurement(qubit_name, classical_name)
    
    def _process_if_statement(self, if_stmt: IfStatement):
        """Process an if statement."""
        # Create condition expression
        condition = self._process_expression(if_stmt.condition)
        
        # Process then body
        then_body = []
        for stmt in if_stmt.then_body:
            then_body.append(self._process_statement_to_ir_stmt(stmt))
        
        # Process else body
        else_body = []
        if if_stmt.else_body:
            for stmt in if_stmt.else_body:
                else_body.append(self._process_statement_to_ir_stmt(stmt))
        
        self.builder.add_control_flow(IROperation.IF, condition, then_body)
    
    def _process_for_statement(self, for_stmt: ForStatement):
        """Process a for statement."""
        # Create loop body
        body = []
        for stmt in for_stmt.body:
            body.append(self._process_statement_to_ir_stmt(stmt))
        
        self.builder.add_control_flow(IROperation.FOR, None, body)
    
    def _process_while_statement(self, while_stmt: WhileStatement):
        """Process a while statement."""
        # Create condition expression
        condition = self._process_expression(while_stmt.condition)
        
        # Create loop body
        body = []
        for stmt in while_stmt.body:
            body.append(self._process_statement_to_ir_stmt(stmt))
        
        self.builder.add_control_flow(IROperation.WHILE, condition, body)
    
    def _process_expression(self, expr: Expression) -> IRExpression:
        """Process an expression."""
        if isinstance(expr, NumberLiteral):
            value = IRValue(IRType.FLOAT if isinstance(expr.value, float) else IRType.INTEGER, expr.value)
            return IRExpression(IROperation.LOAD, [IROperand(value=value)])
        
        elif isinstance(expr, StringLiteral):
            value = IRValue(IRType.STRING, expr.value)
            return IRExpression(IROperation.LOAD, [IROperand(value=value)])
        
        elif isinstance(expr, Identifier):
            # Look up variable
            if expr.name in self.builder.variable_map:
                var = self.builder.variable_map[expr.name]
                return IRExpression(IROperation.LOAD, [IROperand(variable=var)])
            else:
                # Create new variable
                var = IRVariable(expr.name, IRType.INTEGER)
                self.builder.variable_map[expr.name] = var
                return IRExpression(IROperation.LOAD, [IROperand(variable=var)])
        
        elif isinstance(expr, BinaryExpression):
            left = self._process_expression(expr.left)
            right = self._process_expression(expr.right)
            
            # Map operator to IR operation
            op_mapping = {
                "+": IROperation.ADD,
                "-": IROperation.SUB,
                "*": IROperation.MUL,
                "/": IROperation.DIV
            }
            
            operation = op_mapping.get(expr.operator, IROperation.ADD)
            return IRExpression(operation, [IROperand(expression=left), IROperand(expression=right)])
        
        elif isinstance(expr, UnaryExpression):
            operand = self._process_expression(expr.operand)
            # Handle unary operations
            return operand  # Simplified for now
        
        elif isinstance(expr, FunctionCall):
            # Handle function calls
            return IRExpression(IROperation.CALL, [])
        
        else:
            # Default case
            return IRExpression(IROperation.LOAD, [])
    
    def _process_statement_to_ir_stmt(self, stmt: Any) -> IRStatement:
        """Convert a statement to IR statement."""
        # This is a simplified conversion
        return IRStatement(IROperation.ASSIGN, [])
    
    def _extract_identifier_name(self, expr: Expression) -> str:
        """Extract identifier name from expression."""
        if isinstance(expr, Identifier):
            return expr.name
        else:
            return "unknown"
    
    def _map_dsl_type_to_ir(self, dsl_type: str) -> IRType:
        """Map DSL type to IR type."""
        type_mapping = {
            "int": IRType.INTEGER,
            "float": IRType.FLOAT,
            "bool": IRType.BOOLEAN,
            "string": IRType.STRING,
            "qubit": IRType.QUBIT,
            "classical": IRType.CLASSICAL
        }
        return type_mapping.get(dsl_type.lower(), IRType.INTEGER)


class IROptimizationPass(CompilerPass):
    """Pass to optimize IR."""
    
    def __init__(self, optimization_name: str):
        super().__init__(f"IR Optimization: {optimization_name}", PassType.OPTIMIZATION)
        self.optimization_name = optimization_name
    
    def run(self, ir: CoratrixIR) -> PassResult:
        """Apply optimization to IR."""
        try:
            from .ir import IROptimizer
            optimizer = IROptimizer()
            optimized_ir = optimizer.optimize(ir)
            return PassResult(success=True, output=optimized_ir)
        except Exception as e:
            return PassResult(success=False, output=ir, errors=[str(e)])


class PassManager:
    """Manager for compiler passes."""
    
    def __init__(self):
        self.passes: List[CompilerPass] = []
        self.pass_results: Dict[str, PassResult] = {}
    
    def add_pass(self, pass_obj: CompilerPass):
        """Add a pass to the manager."""
        self.passes.append(pass_obj)
    
    def add_passes(self, passes: List[CompilerPass]):
        """Add multiple passes."""
        self.passes.extend(passes)
    
    def run_passes(self, input_data: Any) -> PassResult:
        """Run all passes in sequence."""
        current_data = input_data
        
        for pass_obj in self.passes:
            result = pass_obj.run(current_data)
            self.pass_results[pass_obj.name] = result
            
            if not result.success:
                return PassResult(
                    success=False,
                    output=current_data,
                    errors=result.errors,
                    warnings=result.warnings
                )
            
            current_data = result.output
        
        return PassResult(success=True, output=current_data)
    
    def get_pass_result(self, pass_name: str) -> Optional[PassResult]:
        """Get the result of a specific pass."""
        return self.pass_results.get(pass_name)
    
    def get_all_results(self) -> Dict[str, PassResult]:
        """Get all pass results."""
        return self.pass_results.copy()
    
    def clear_results(self):
        """Clear all pass results."""
        self.pass_results.clear()
    
    def get_pass_by_name(self, name: str) -> Optional[CompilerPass]:
        """Get a pass by name."""
        for pass_obj in self.passes:
            if pass_obj.name == name:
                return pass_obj
        return None
    
    def remove_pass(self, name: str) -> bool:
        """Remove a pass by name."""
        for i, pass_obj in enumerate(self.passes):
            if pass_obj.name == name:
                del self.passes[i]
                return True
        return False
    
    def get_passes_by_type(self, pass_type: PassType) -> List[CompilerPass]:
        """Get all passes of a specific type."""
        return [pass_obj for pass_obj in self.passes if pass_obj.pass_type == pass_type]
    
    def __str__(self) -> str:
        return f"PassManager with {len(self.passes)} passes"
    
    def __repr__(self) -> str:
        return f"PassManager(passes={[p.name for p in self.passes]})"
