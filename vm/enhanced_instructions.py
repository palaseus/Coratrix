"""
Enhanced quantum instructions with advanced features.

This module defines enhanced instruction types for the quantum virtual machine,
including loops, subroutines, parameterized gates, and conditional execution.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Optional
from enum import Enum
from vm.instructions import QuantumInstruction, InstructionType


class EnhancedInstructionType(Enum):
    """Types of enhanced quantum instructions."""
    GATE = "gate"
    MEASURE = "measure"
    COMMENT = "comment"
    LOOP = "loop"
    SUBROUTINE = "subroutine"
    CALL = "call"
    CONDITIONAL = "conditional"
    ERROR_HANDLING = "error_handling"
    VARIABLE = "variable"
    INCLUDE = "include"


class LoopInstruction(QuantumInstruction):
    """
    Instruction for creating loops in quantum programs.
    
    Allows repeating a sequence of instructions a specified number of times.
    """
    
    def __init__(self, iterations: int, loop_body: str):
        """
        Initialize a loop instruction.
        
        Args:
            iterations: Number of times to repeat the loop
            loop_body: Instructions to repeat
        """
        super().__init__(InstructionType.GATE)  # Use base type for compatibility
        self.iterations = iterations
        self.loop_body = loop_body
        self.current_iteration = 0
    
    def execute(self, executor) -> None:
        """Execute the loop instruction."""
        for i in range(self.iterations):
            self.current_iteration = i
            # Parse and execute the loop body
            from .enhanced_parser import EnhancedQuantumParser
            parser = EnhancedQuantumParser()
            instructions = parser.parse_script(self.loop_body)
            executor.execute_instructions(instructions)
    
    def __str__(self) -> str:
        """String representation of the loop instruction."""
        return f"LOOP {self.iterations}: {self.loop_body}"


class SubroutineInstruction(QuantumInstruction):
    """
    Instruction for calling subroutines.
    
    Allows defining and calling reusable sequences of quantum instructions.
    """
    
    def __init__(self, sub_name: str, parameters: Dict[str, Any] = None):
        """
        Initialize a subroutine call instruction.
        
        Args:
            sub_name: Name of the subroutine to call
            parameters: Parameters to pass to the subroutine
        """
        super().__init__(InstructionType.GATE)  # Use base type for compatibility
        self.sub_name = sub_name
        self.parameters = parameters or {}
    
    def execute(self, executor) -> Any:
        """Execute the subroutine call."""
        # Get the subroutine definition from the parser
        if hasattr(executor, 'parser') and hasattr(executor.parser, 'subroutines'):
            if self.sub_name in executor.parser.subroutines:
                sub_body = executor.parser.subroutines[self.sub_name]
                
                # Apply parameters if any
                if self.parameters:
                    for param_name, param_value in self.parameters.items():
                        sub_body = sub_body.replace(param_name, str(param_value))
                
                # Parse and execute the subroutine body
                from .enhanced_parser import EnhancedQuantumParser
                parser = EnhancedQuantumParser()
                instructions = parser.parse_script(sub_body)
                return executor.execute_instructions(instructions)
            else:
                raise ValueError(f"Subroutine {self.sub_name} not defined")
        else:
            raise ValueError("Subroutine support not available in executor")
    
    def __str__(self) -> str:
        """String representation of the subroutine call."""
        if self.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            return f"CALL {self.sub_name} WITH {param_str}"
        else:
            return f"CALL {self.sub_name}"


class ParameterizedGateInstruction(QuantumInstruction):
    """
    Instruction for applying parameterized quantum gates.
    
    Supports gates with adjustable parameters like rotation angles.
    """
    
    def __init__(self, gate_name: str, target_qubits: List[int], parameters: Dict[str, float]):
        """
        Initialize a parameterized gate instruction.
        
        Args:
            gate_name: Name of the parameterized gate
            target_qubits: List of qubit indices the gate acts on
            parameters: Dictionary of parameter names and values
        """
        super().__init__(InstructionType.GATE)
        self.gate_name = gate_name.upper()
        self.target_qubits = target_qubits
        self.parameters = parameters
        
        # Validate gate name
        valid_gates = {'RX', 'RY', 'RZ', 'CPHASE', 'PHASE'}
        if self.gate_name not in valid_gates:
            raise ValueError(f"Invalid parameterized gate name: {gate_name}. Valid gates: {valid_gates}")
    
    def execute(self, executor) -> None:
        """Execute the parameterized gate instruction."""
        # Create the appropriate parameterized gate
        from ..core.advanced_gates import RxGate, RyGate, RzGate, CPhaseGate
        
        gate_map = {
            'RX': RxGate,
            'RY': RyGate,
            'RZ': RzGate,
            'CPHASE': CPhaseGate,
            'PHASE': CPhaseGate
        }
        
        if self.gate_name in gate_map:
            gate_class = gate_map[self.gate_name]
            gate = gate_class()
            
            # Set parameters
            for param_name, param_value in self.parameters.items():
                gate.set_parameter(param_name, param_value)
            
            executor.apply_gate(gate, self.target_qubits)
        else:
            raise ValueError(f"Unknown parameterized gate: {self.gate_name}")
    
    def __str__(self) -> str:
        """String representation of the parameterized gate instruction."""
        if len(self.target_qubits) == 1:
            qubit_str = f"q{self.target_qubits[0]}"
        else:
            qubit_str = ",".join(f"q{q}" for q in self.target_qubits)
        
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.gate_name} {qubit_str}({param_str})"


class ConditionalInstruction(QuantumInstruction):
    """
    Instruction for conditional execution.
    
    Allows executing instructions based on variable values or measurement results.
    """
    
    def __init__(self, variable_name: str, expected_value: Any, conditional_body: str):
        """
        Initialize a conditional instruction.
        
        Args:
            variable_name: Name of the variable to check
            expected_value: Expected value for the variable
            conditional_body: Instructions to execute if condition is met
        """
        super().__init__(InstructionType.GATE)  # Use base type for compatibility
        self.variable_name = variable_name
        self.expected_value = expected_value
        self.conditional_body = conditional_body
    
    def execute(self, executor) -> Any:
        """Execute the conditional instruction."""
        # Check if the condition is met
        if hasattr(executor, 'parser') and hasattr(executor.parser, 'variables'):
            if self.variable_name in executor.parser.variables:
                current_value = executor.parser.variables[self.variable_name]
                if current_value == self.expected_value:
                    # Parse and execute the conditional body
                    from .enhanced_parser import EnhancedQuantumParser
                    parser = EnhancedQuantumParser()
                    instructions = parser.parse_script(self.conditional_body)
                    return executor.execute_instructions(instructions)
        return None
    
    def __str__(self) -> str:
        """String representation of the conditional instruction."""
        return f"IF {self.variable_name} = {self.expected_value}: {self.conditional_body}"


class ErrorHandlingInstruction(QuantumInstruction):
    """
    Instruction for error handling.
    
    Allows defining error handling behavior for quantum operations.
    """
    
    def __init__(self, error_body: str):
        """
        Initialize an error handling instruction.
        
        Args:
            error_body: Instructions to execute on error
        """
        super().__init__(InstructionType.GATE)  # Use base type for compatibility
        self.error_body = error_body
    
    def execute(self, executor) -> Any:
        """Execute the error handling instruction."""
        # Store error handler in executor
        if hasattr(executor, 'error_handlers'):
            executor.error_handlers['default'] = self.error_body
        return None
    
    def __str__(self) -> str:
        """String representation of the error handling instruction."""
        return f"ON_ERROR {self.error_body}"


class VariableInstruction(QuantumInstruction):
    """
    Instruction for variable assignment.
    
    Allows setting and using variables in quantum programs.
    """
    
    def __init__(self, variable_name: str, value: Any):
        """
        Initialize a variable instruction.
        
        Args:
            variable_name: Name of the variable
            value: Value to assign to the variable
        """
        super().__init__(InstructionType.GATE)  # Use base type for compatibility
        self.variable_name = variable_name
        self.value = value
    
    def execute(self, executor) -> None:
        """Execute the variable instruction."""
        if hasattr(executor, 'parser') and hasattr(executor.parser, 'variables'):
            executor.parser.variables[self.variable_name] = self.value
    
    def __str__(self) -> str:
        """String representation of the variable instruction."""
        return f"SET {self.variable_name} = {self.value}"


class IncludeInstruction(QuantumInstruction):
    """
    Instruction for including external files.
    
    Allows including and executing external quantum script files.
    """
    
    def __init__(self, filename: str):
        """
        Initialize an include instruction.
        
        Args:
            filename: Path to the file to include
        """
        super().__init__(InstructionType.GATE)  # Use base type for compatibility
        self.filename = filename
    
    def execute(self, executor) -> Any:
        """Execute the include instruction."""
        try:
            with open(self.filename, 'r') as file:
                content = file.read()
            
            # Parse and execute the included content
            from .enhanced_parser import EnhancedQuantumParser
            parser = EnhancedQuantumParser()
            instructions = parser.parse_script(content)
            return executor.execute_instructions(instructions)
        except FileNotFoundError:
            raise ValueError(f"Include file not found: {self.filename}")
        except Exception as e:
            raise ValueError(f"Error including file {self.filename}: {e}")
    
    def __str__(self) -> str:
        """String representation of the include instruction."""
        return f"INCLUDE {self.filename}"


class QuantumAlgorithmInstruction(QuantumInstruction):
    """
    Instruction for quantum algorithms.
    
    Allows executing predefined quantum algorithms with parameters.
    """
    
    def __init__(self, algorithm_name: str, parameters: Dict[str, Any] = None):
        """
        Initialize a quantum algorithm instruction.
        
        Args:
            algorithm_name: Name of the algorithm to execute
            parameters: Parameters for the algorithm
        """
        super().__init__(InstructionType.GATE)  # Use base type for compatibility
        self.algorithm_name = algorithm_name.upper()
        self.parameters = parameters or {}
    
    def execute(self, executor) -> Any:
        """Execute the quantum algorithm instruction."""
        # Import algorithm implementations
        from algorithms.quantum_algorithms import (
            GroverAlgorithm, QuantumFourierTransform, 
            QuantumTeleportation, GHZState, WState
        )
        
        algorithm_map = {
            'GROVER': GroverAlgorithm,
            'QFT': QuantumFourierTransform,
            'TELEPORTATION': QuantumTeleportation,
            'GHZ': GHZState,
            'W_STATE': WState
        }
        
        if self.algorithm_name in algorithm_map:
            algorithm_class = algorithm_map[self.algorithm_name]
            algorithm = algorithm_class()
            return algorithm.execute(executor, self.parameters)
        else:
            raise ValueError(f"Unknown quantum algorithm: {self.algorithm_name}")
    
    def __str__(self) -> str:
        """String representation of the quantum algorithm instruction."""
        if self.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            return f"ALGORITHM {self.algorithm_name}({param_str})"
        else:
            return f"ALGORITHM {self.algorithm_name}"
