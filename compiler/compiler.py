"""
Main Coratrix compiler.

This module provides the main compiler interface that orchestrates
the compilation pipeline from DSL to target code.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .dsl import QuantumDSL, DSLParser
from .ir import CoratrixIR, IRBuilder, IROptimizer
from .passes import PassManager, DSLToIRPass, IROptimizationPass
from .targets import QASMTarget, QiskitTarget, PennyLaneTarget, TargetResult
from .backend import BackendManager, BackendConfiguration, BackendType


class CompilerMode(Enum):
    """Compiler operation modes."""
    COMPILE_ONLY = "compile_only"      # Just compile, don't execute
    COMPILE_AND_RUN = "compile_and_run"  # Compile and execute
    OPTIMIZE = "optimize"              # Apply optimizations
    DEBUG = "debug"                    # Debug mode with verbose output


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
        self.backend_manager = BackendManager()
        self._setup_default_passes()
        self._setup_default_backends()
    
    def _setup_default_passes(self):
        """Set up default compiler passes."""
        # Frontend passes
        self.pass_manager.add_pass(DSLToIRPass())
        
        # Optimization passes
        self.pass_manager.add_pass(IROptimizationPass("gate_merging"))
        self.pass_manager.add_pass(IROptimizationPass("redundant_operations"))
        self.pass_manager.add_pass(IROptimizationPass("constant_folding"))
    
    def _setup_default_backends(self):
        """Set up default backends."""
        # Local simulator backend
        sim_config = BackendConfiguration(
            name="local_simulator",
            backend_type=BackendType.SIMULATOR,
            connection_params={'simulator_type': 'statevector'}
        )
        from .backend import SimulatorBackend
        simulator = SimulatorBackend(sim_config)
        self.backend_manager.register_backend("local_simulator", simulator)
        
        # Qiskit backend (if available)
        try:
            qiskit_config = BackendConfiguration(
                name="qiskit_simulator",
                backend_type=BackendType.SIMULATOR,
                connection_params={'backend_type': 'simulator'}
            )
            from .backend import QiskitBackend
            qiskit_backend = QiskitBackend(qiskit_config)
            self.backend_manager.register_backend("qiskit_simulator", qiskit_backend)
        except ImportError:
            pass  # Qiskit not available
    
    def compile(self, source: str, options: CompilerOptions = None) -> CompilerResult:
        """Compile DSL source code."""
        if options is None:
            options = CompilerOptions()
        
        try:
            # Parse DSL
            ast = self.dsl.compile(source)
            
            # Run compilation passes
            pass_result = self.pass_manager.run_passes(ast)
            
            if not pass_result.success:
                return CompilerResult(
                    success=False,
                    source_code=source,
                    errors=pass_result.errors,
                    warnings=pass_result.warnings
                )
            
            ir = pass_result.output
            
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
                metadata=target_result.metadata
            )
            
            # Execute if requested
            if options.mode == CompilerMode.COMPILE_AND_RUN:
                execution_result = self._execute_circuit(ir, options)
                result.execution_result = execution_result
            
            return result
            
        except Exception as e:
            return CompilerResult(
                success=False,
                source_code=source,
                errors=[str(e)]
            )
    
    def compile_file(self, filename: str, options: CompilerOptions = None) -> CompilerResult:
        """Compile a DSL file."""
        try:
            with open(filename, 'r') as f:
                source = f.read()
            return self.compile(source, options)
        except Exception as e:
            return CompilerResult(
                success=False,
                source_code="",
                errors=[f"Failed to read file '{filename}': {str(e)}"]
            )
    
    def _generate_target_code(self, ir: CoratrixIR, options: CompilerOptions) -> TargetResult:
        """Generate target code from IR."""
        target_format = options.target_format.lower()
        
        if target_format == "openqasm":
            generator = QASMTarget()
        elif target_format == "qiskit":
            generator = QiskitTarget()
        elif target_format == "pennylane":
            generator = PennyLaneTarget()
        else:
            return TargetResult(
                success=False,
                code="",
                errors=[f"Unsupported target format: {target_format}"]
            )
        
        return generator.generate(ir)
    
    def _execute_circuit(self, ir: CoratrixIR, options: CompilerOptions) -> Any:
        """Execute a circuit on a backend."""
        backend_name = options.backend_name or self.backend_manager.default_backend
        
        if not backend_name:
            return {"error": "No backend specified and no default backend available"}
        
        # Get the first circuit from IR
        if not ir.circuits:
            return {"error": "No circuits to execute"}
        
        circuit = ir.circuits[0]
        
        # Execute on backend
        result = self.backend_manager.execute_on_backend(
            backend_name,
            circuit,
            options.shots,
            options.parameters
        )
        
        return result
    
    def add_backend(self, name: str, config: BackendConfiguration) -> bool:
        """Add a new backend."""
        try:
            if config.backend_type == BackendType.SIMULATOR:
                from .backend import SimulatorBackend
                backend = SimulatorBackend(config)
            elif config.backend_type == BackendType.HARDWARE:
                from .backend import QiskitBackend
                backend = QiskitBackend(config)
            else:
                return False
            
            return self.backend_manager.register_backend(name, backend)
        except Exception:
            return False
    
    def list_backends(self) -> List[str]:
        """List available backends."""
        return self.backend_manager.list_backends()
    
    def get_backend_status(self, name: str) -> Optional[str]:
        """Get backend status."""
        status = self.backend_manager.get_backend_status(name)
        return status.value if status else None
    
    def set_default_backend(self, name: str) -> bool:
        """Set the default backend."""
        return self.backend_manager.set_default_backend(name)
    
    def add_pass(self, pass_obj):
        """Add a custom compiler pass."""
        self.pass_manager.add_pass(pass_obj)
    
    def remove_pass(self, name: str) -> bool:
        """Remove a compiler pass."""
        return self.pass_manager.remove_pass(name)
    
    def get_passes(self) -> List[str]:
        """Get list of compiler passes."""
        return [pass_obj.name for pass_obj in self.pass_manager.passes]
    
    def get_pass_results(self) -> Dict[str, Any]:
        """Get results from all passes."""
        return self.pass_manager.get_all_results()
    
    def clear_pass_results(self):
        """Clear pass results."""
        self.pass_manager.clear_results()
    
    def __str__(self) -> str:
        return f"CoratrixCompiler with {len(self.pass_manager.passes)} passes and {len(self.backend_manager.backends)} backends"
    
    def __repr__(self) -> str:
        return f"CoratrixCompiler(passes={len(self.pass_manager.passes)}, backends={len(self.backend_manager.backends)})"


# Convenience functions
def compile_dsl(source: str, target_format: str = "openqasm", 
               optimize: bool = True, execute: bool = False) -> CompilerResult:
    """Convenience function to compile DSL source."""
    compiler = CoratrixCompiler()
    options = CompilerOptions(
        mode=CompilerMode.COMPILE_AND_RUN if execute else CompilerMode.COMPILE_ONLY,
        target_format=target_format,
        optimize=optimize
    )
    return compiler.compile(source, options)


def compile_file(filename: str, target_format: str = "openqasm", 
                optimize: bool = True, execute: bool = False) -> CompilerResult:
    """Convenience function to compile a DSL file."""
    compiler = CoratrixCompiler()
    options = CompilerOptions(
        mode=CompilerMode.COMPILE_AND_RUN if execute else CompilerMode.COMPILE_ONLY,
        target_format=target_format,
        optimize=optimize
    )
    return compiler.compile_file(filename, options)
