"""
Enhanced quantum instruction parser with advanced features.

This module provides an enhanced parser for quantum instruction scripts
supporting loops, subroutines, parameterized gates, and advanced error handling.
"""

import re
import math
from typing import List, Optional, Union, Dict, Any, Tuple
from vm.instructions import QuantumInstruction, GateInstruction, MeasureInstruction, CommentInstruction
from vm.enhanced_instructions import (
    LoopInstruction, SubroutineInstruction, ParameterizedGateInstruction,
    ConditionalInstruction, ErrorHandlingInstruction
)


class EnhancedQuantumParser:
    """
    Enhanced parser for quantum instruction scripts with advanced features.
    
    Supports:
    - Loops and iterations
    - Subroutines and function calls
    - Parameterized gates
    - Conditional execution
    - Advanced error handling
    - Comments and documentation
    """
    
    def __init__(self):
        """Initialize the enhanced quantum parser."""
        # Define enhanced instruction patterns
        self.patterns = {
            'gate': re.compile(r'^(X|Y|Z|H|CNOT|Toffoli|SWAP|Rx|Ry|Rz|CPhase|S|T|Fredkin)\s+q(\d+)(?:\s*,\s*q(\d+))?(?:\s*,\s*q(\d+))?(?:\s*\(\s*([^)]+)\s*\))?$', re.IGNORECASE),
            'measure': re.compile(r'^MEASURE(?:\s+q(\d+)(?:\s*,\s*q(\d+))*)?$', re.IGNORECASE),
            'comment': re.compile(r'^#\s*(.*)$'),
            'loop': re.compile(r'^LOOP\s+(\d+)\s*:\s*(.*)$', re.IGNORECASE),
            'endloop': re.compile(r'^ENDLOOP$', re.IGNORECASE),
            'subroutine': re.compile(r'^SUBROUTINE\s+(\w+)\s*:\s*(.*)$', re.IGNORECASE),
            'endsubroutine': re.compile(r'^ENDSUBROUTINE$', re.IGNORECASE),
            'call': re.compile(r'^CALL\s+(\w+)(?:\s+WITH\s+(.*))?$', re.IGNORECASE),
            'conditional': re.compile(r'^IF\s+(\w+)\s*=\s*(\d+)\s*:\s*(.*)$', re.IGNORECASE),
            'endif': re.compile(r'^ENDIF$', re.IGNORECASE),
            'error_handling': re.compile(r'^ON_ERROR\s+(.*)$', re.IGNORECASE),
            'end_error_handling': re.compile(r'^END_ERROR_HANDLING$', re.IGNORECASE),
            'variable': re.compile(r'^SET\s+(\w+)\s*=\s*(.*)$', re.IGNORECASE),
            'include': re.compile(r'^INCLUDE\s+(.*)$', re.IGNORECASE)
        }
        
        # Variable storage
        self.variables = {}
        
        # Subroutine storage
        self.subroutines = {}
        
        # Error handling
        self.error_handlers = {}
    
    def parse_line(self, line: str) -> Optional[QuantumInstruction]:
        """
        Parse a single line of enhanced quantum instructions.
        
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
        
        # Try to match variable assignment
        variable_match = self.patterns['variable'].match(line)
        if variable_match:
            var_name = variable_match.group(1)
            var_value = self._evaluate_expression(variable_match.group(2))
            self.variables[var_name] = var_value
            return CommentInstruction(f"Set {var_name} = {var_value}")
        
        # Try to match include pattern
        include_match = self.patterns['include'].match(line)
        if include_match:
            filename = include_match.group(1).strip()
            return self._parse_include_file(filename)
        
        # Try to match loop pattern
        loop_match = self.patterns['loop'].match(line)
        if loop_match:
            iterations = int(loop_match.group(1))
            loop_body = loop_match.group(2)
            return LoopInstruction(iterations, loop_body)
        
        # Try to match endloop pattern
        endloop_match = self.patterns['endloop'].match(line)
        if endloop_match:
            return CommentInstruction("End of loop")
        
        # Try to match subroutine pattern
        subroutine_match = self.patterns['subroutine'].match(line)
        if subroutine_match:
            sub_name = subroutine_match.group(1)
            sub_body = subroutine_match.group(2)
            self.subroutines[sub_name] = sub_body
            return CommentInstruction(f"Define subroutine {sub_name}")
        
        # Try to match endsubroutine pattern
        endsubroutine_match = self.patterns['endsubroutine'].match(line)
        if endsubroutine_match:
            return CommentInstruction("End of subroutine")
        
        # Try to match call pattern
        call_match = self.patterns['call'].match(line)
        if call_match:
            sub_name = call_match.group(1)
            params = call_match.group(2) if call_match.group(2) else ""
            return self._parse_subroutine_call(sub_name, params)
        
        # Try to match conditional pattern
        conditional_match = self.patterns['conditional'].match(line)
        if conditional_match:
            var_name = conditional_match.group(1)
            expected_value = int(conditional_match.group(2))
            conditional_body = conditional_match.group(3)
            return ConditionalInstruction(var_name, expected_value, conditional_body)
        
        # Try to match endif pattern
        endif_match = self.patterns['endif'].match(line)
        if endif_match:
            return CommentInstruction("End of conditional")
        
        # Try to match error handling pattern
        error_match = self.patterns['error_handling'].match(line)
        if error_match:
            error_body = error_match.group(1)
            return ErrorHandlingInstruction(error_body)
        
        # Try to match end error handling pattern
        end_error_match = self.patterns['end_error_handling'].match(line)
        if end_error_match:
            return CommentInstruction("End of error handling")
        
        # Try to match enhanced gate pattern
        gate_match = self.patterns['gate'].match(line)
        if gate_match:
            gate_name = gate_match.group(1).upper()
            qubit1 = int(gate_match.group(2))
            qubit2 = int(gate_match.group(3)) if gate_match.group(3) else None
            qubit3 = int(gate_match.group(4)) if gate_match.group(4) else None
            parameters = gate_match.group(5) if gate_match.group(5) else None
            
            # Handle parameterized gates
            if parameters:
                param_dict = self._parse_parameters(parameters)
                return ParameterizedGateInstruction(gate_name, [qubit1, qubit2, qubit3], param_dict)
            else:
                # Handle regular gates
                target_qubits = [q for q in [qubit1, qubit2, qubit3] if q is not None]
                return GateInstruction(gate_name, target_qubits)
        
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
    
    def _evaluate_expression(self, expression: str) -> Union[int, float, str]:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
        
        Returns:
            Evaluated result
        """
        try:
            # Replace variables with their values
            for var_name, var_value in self.variables.items():
                expression = expression.replace(var_name, str(var_value))
            
            # Evaluate the expression
            return eval(expression)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {e}")
    
    def _parse_parameters(self, param_string: str) -> Dict[str, float]:
        """
        Parse parameter string for parameterized gates.
        
        Args:
            param_string: Parameter string (e.g., "theta=pi/2, phi=0.5")
        
        Returns:
            Dictionary of parameter names and values
        """
        parameters = {}
        
        # Split by commas and parse each parameter
        param_pairs = param_string.split(',')
        for pair in param_pairs:
            if '=' in pair:
                name, value = pair.split('=', 1)
                name = name.strip()
                value = value.strip()
                
                # Evaluate the value (can be mathematical expressions)
                try:
                    # Replace common mathematical constants
                    value = value.replace('pi', str(math.pi))
                    value = value.replace('e', str(math.e))
                    
                    parameters[name] = float(eval(value))
                except Exception as e:
                    raise ValueError(f"Error parsing parameter {name}={value}: {e}")
        
        return parameters
    
    def _parse_include_file(self, filename: str) -> Optional[QuantumInstruction]:
        """
        Parse an included file.
        
        Args:
            filename: Path to the file to include
        
        Returns:
            Parsed instruction from the included file
        """
        try:
            with open(filename, 'r') as file:
                content = file.read()
            
            # Parse the included content
            included_instructions = self.parse_script(content)
            return CommentInstruction(f"Included {len(included_instructions)} instructions from {filename}")
        except FileNotFoundError:
            raise ValueError(f"Include file not found: {filename}")
        except Exception as e:
            raise ValueError(f"Error including file {filename}: {e}")
    
    def _parse_subroutine_call(self, sub_name: str, params: str) -> QuantumInstruction:
        """
        Parse a subroutine call.
        
        Args:
            sub_name: Name of the subroutine
            params: Parameters for the subroutine
        
        Returns:
            Parsed subroutine call instruction
        """
        if sub_name not in self.subroutines:
            raise ValueError(f"Subroutine {sub_name} not defined")
        
        # Parse parameters if provided
        param_dict = {}
        if params:
            param_dict = self._parse_parameters(params)
        
        return SubroutineInstruction(sub_name, param_dict)
    
    def parse_script(self, script: str) -> List[QuantumInstruction]:
        """
        Parse a complete enhanced quantum script.
        
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
            if isinstance(instruction, (GateInstruction, ParameterizedGateInstruction)):
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
        Get information about supported enhanced instructions.
        
        Returns:
            Dictionary mapping instruction names to descriptions
        """
        return {
            'X': 'Pauli-X gate (quantum NOT)',
            'Y': 'Pauli-Y gate',
            'Z': 'Pauli-Z gate',
            'H': 'Hadamard gate (creates superposition)',
            'CNOT': 'Controlled-NOT gate (entanglement)',
            'Toffoli': 'Toffoli gate (controlled-controlled-NOT)',
            'SWAP': 'SWAP gate (swaps two qubits)',
            'Rx': 'Rotation around X-axis (parameterized)',
            'Ry': 'Rotation around Y-axis (parameterized)',
            'Rz': 'Rotation around Z-axis (parameterized)',
            'CPhase': 'Controlled phase gate (parameterized)',
            'S': 'S gate (π/2 phase)',
            'T': 'T gate (π/4 phase)',
            'Fredkin': 'Fredkin gate (controlled-SWAP)',
            'MEASURE': 'Measure qubits (causes state collapse)',
            'LOOP': 'Loop instruction (repeat n times)',
            'SUBROUTINE': 'Define subroutine',
            'CALL': 'Call subroutine',
            'IF': 'Conditional execution',
            'SET': 'Set variable',
            'INCLUDE': 'Include external file',
            'ON_ERROR': 'Error handling',
            '#': 'Comment (ignored during execution)'
        }
    
    def get_variables(self) -> Dict[str, Any]:
        """Get current variable values."""
        return self.variables.copy()
    
    def get_subroutines(self) -> Dict[str, str]:
        """Get defined subroutines."""
        return self.subroutines.copy()
    
    def clear_variables(self):
        """Clear all variables."""
        self.variables.clear()
    
    def clear_subroutines(self):
        """Clear all subroutines."""
        self.subroutines.clear()
