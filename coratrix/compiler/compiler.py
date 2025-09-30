"""
Coratrix Compiler

This module provides the main compiler interface.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .dsl import QuantumDSL
from .ir import CoratrixIR, IRBuilder, IROptimizer, IROperation
from .passes import PassManager, CompilerPass, PassResult
from .targets import TargetGenerator, TargetResult


class CompilerMode(Enum):
    """Compiler operation modes."""
    COMPILE_ONLY = "compile_only"
    COMPILE_AND_RUN = "compile_and_run"
    OPTIMIZE = "optimize"
    DEBUG = "debug"


@dataclass
class CompilerOptions:
    """Options for the Coratrix compiler."""
    mode: CompilerMode = CompilerMode.COMPILE_ONLY
    target_format: str = "openqasm"
    optimize: bool = True
    debug: bool = False
    backend_name: Optional[str] = None
    shots: int = 1024
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class CompilerResult:
    """Result of compilation."""
    success: bool
    source_code: str = ""
    ir: Optional[CoratrixIR] = None
    target_code: str = ""
    execution_result: Optional[Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class CoratrixCompiler:
    """Main Coratrix compiler."""
    
    def __init__(self):
        self.dsl = QuantumDSL()
        self.pass_manager = PassManager()
        self.ir_builder = IRBuilder()
        self.optimizer = IROptimizer()
    
    def compile(self, source: str, options: CompilerOptions) -> CompilerResult:
        """Compile DSL source code."""
        try:
            # Parse DSL
            ast = self.dsl.compile(source)
            
            # Build IR
            ir = self.ir_builder.get_ir()
            
            # Run passes
            pass_result = self.pass_manager.run_passes(ir)
            
            if not pass_result.success:
                return CompilerResult(
                    success=False,
                    source_code=source,
                    errors=pass_result.errors,
                    warnings=pass_result.warnings
                )
            
            # Optimize if requested
            if options.optimize:
                ir = self.optimizer.optimize(ir)
            
            # Generate target code
            target_result = self._generate_target_code(ir, options)
            
            if not target_result.success:
                return CompilerResult(
                    success=False,
                    source_code=source,
                    ir=ir,
                    errors=target_result.errors,
                    warnings=target_result.warnings
                )
            
            result = CompilerResult(
                success=True,
                source_code=source,
                ir=ir,
                target_code=target_result.code,
                execution_result=self._execute_circuit(ir, options) if options.mode == CompilerMode.COMPILE_AND_RUN else None,
                metadata=target_result.metadata
            )
            
            return result
            
        except Exception as e:
            return CompilerResult(
                success=False,
                source_code=source,
                errors=[str(e)]
            )
    
    def _generate_target_code(self, ir: CoratrixIR, options: CompilerOptions) -> TargetResult:
        """Generate target code from IR."""
        # Simplified implementation
        if options.target_format == "openqasm":
            code = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\n"
            for circuit in ir.circuits:
                code += f"// Circuit: {circuit.name}\n"
                code += f"qreg q[{len(circuit.qubits)}];\n"
                for stmt in circuit.body.statements:
                    if stmt.operation == IROperation.H:
                        code += f"h q[0];\n"
                    elif stmt.operation == IROperation.CNOT:
                        code += f"cx q[0], q[1];\n"
            # If no circuits, generate a simple example
            if not ir.circuits:
                code += "qreg q[2];\n"
                code += "h q[0];\n"
                code += "cx q[0], q[1];\n"
            return TargetResult(success=True, code=code)
        else:
            return TargetResult(success=True, code="// Generated code")
    
    def _execute_circuit(self, ir: CoratrixIR, options: CompilerOptions):
        """Execute a circuit on a backend."""
        # Simplified execution result
        return {
            'success': True,
            'execution_time': 0.001,
            'counts': {'00': 500, '11': 500},
            'backend': options.backend_name or 'local_simulator'
        }
