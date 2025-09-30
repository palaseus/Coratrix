"""
ASCII circuit diagram generator for quantum circuits.

This module provides functionality to generate ASCII art representations
of quantum circuits, making it easy to visualize quantum algorithms.
"""

from typing import List, Dict, Any, Optional, Tuple
import re


class CircuitDiagram:
    """
    Generator for ASCII circuit diagrams.
    
    Creates visual representations of quantum circuits using ASCII art,
    making it easy to understand and debug quantum algorithms.
    """
    
    def __init__(self, width: int = 80):
        """
        Initialize the circuit diagram generator.
        
        Args:
            width: Maximum width of the diagram
        """
        self.width = width
        self.gate_symbols = {
            'X': 'X',
            'Y': 'Y', 
            'Z': 'Z',
            'H': 'H',
            'CNOT': '⊕',
            'Toffoli': '⊕',
            'SWAP': '×',
            'Rx': 'Rx',
            'Ry': 'Ry',
            'Rz': 'Rz',
            'CPhase': 'P',
            'S': 'S',
            'T': 'T',
            'Fredkin': 'F',
            'MEASURE': 'M'
        }
    
    def generate_diagram(self, instructions: List[Dict[str, Any]], num_qubits: int) -> str:
        """
        Generate an ASCII circuit diagram from a list of instructions.
        
        Args:
            instructions: List of quantum instructions
            num_qubits: Number of qubits in the circuit
        
        Returns:
            ASCII art representation of the circuit
        """
        # Initialize the circuit grid
        grid = self._initialize_grid(num_qubits)
        
        # Add instructions to the grid
        for instruction in instructions:
            self._add_instruction_to_grid(grid, instruction, num_qubits)
        
        # Generate the final diagram
        return self._grid_to_string(grid)
    
    def _initialize_grid(self, num_qubits: int) -> List[List[str]]:
        """Initialize the circuit grid."""
        # Create a grid with enough space for the circuit
        height = num_qubits * 3  # 3 lines per qubit
        width = self.width
        
        grid = []
        for i in range(height):
            grid.append([' '] * width)
        
        # Add qubit lines
        for i in range(num_qubits):
            line_index = i * 3 + 1  # Middle line of each qubit
            for j in range(width):
                grid[line_index][j] = '─'
        
        return grid
    
    def _add_instruction_to_grid(self, grid: List[List[str]], instruction: Dict[str, Any], num_qubits: int):
        """Add an instruction to the circuit grid."""
        gate_type = instruction.get('gate', '')
        target_qubits = instruction.get('qubits', [])
        
        if not target_qubits:
            return
        
        # Find the next available position
        position = self._find_next_position(grid, num_qubits)
        
        if gate_type == 'CNOT':
            self._add_cnot_gate(grid, target_qubits, position, num_qubits)
        elif gate_type == 'Toffoli':
            self._add_toffoli_gate(grid, target_qubits, position, num_qubits)
        elif gate_type == 'SWAP':
            self._add_swap_gate(grid, target_qubits, position, num_qubits)
        elif gate_type == 'MEASURE':
            self._add_measurement(grid, target_qubits, position, num_qubits)
        else:
            # Single or multi-qubit gate
            self._add_single_gate(grid, gate_type, target_qubits, position, num_qubits)
    
    def _find_next_position(self, grid: List[List[str]], num_qubits: int) -> int:
        """Find the next available position in the circuit."""
        # Look for the rightmost occupied position
        max_position = 0
        for i in range(num_qubits):
            line_index = i * 3 + 1
            for j in range(len(grid[line_index])):
                if grid[line_index][j] not in [' ', '─']:
                    max_position = max(max_position, j)
        
        return max_position + 4  # Add some spacing
    
    def _add_single_gate(self, grid: List[List[str]], gate_type: str, target_qubits: List[int], position: int, num_qubits: int):
        """Add a single-qubit gate to the circuit."""
        symbol = self.gate_symbols.get(gate_type, gate_type)
        
        for qubit in target_qubits:
            line_index = qubit * 3 + 1
            # Add the gate symbol
            grid[line_index][position] = symbol
            # Add brackets around the gate
            grid[line_index][position-1] = '['
            grid[line_index][position+1] = ']'
    
    def _add_cnot_gate(self, grid: List[List[str]], target_qubits: List[int], position: int, num_qubits: int):
        """Add a CNOT gate to the circuit."""
        if len(target_qubits) != 2:
            return
        
        control_qubit, target_qubit = target_qubits[0], target_qubits[1]
        
        # Add control qubit (dot)
        control_line = control_qubit * 3 + 1
        grid[control_line][position] = '●'
        
        # Add target qubit (X)
        target_line = target_qubit * 3 + 1
        grid[target_line][position] = '⊕'
        
        # Add vertical line connecting control and target
        start_line = min(control_line, target_line)
        end_line = max(control_line, target_line)
        
        for line in range(start_line + 1, end_line):
            if grid[line][position] == '─':
                grid[line][position] = '│'
    
    def _add_toffoli_gate(self, grid: List[List[str]], target_qubits: List[int], position: int, num_qubits: int):
        """Add a Toffoli gate to the circuit."""
        if len(target_qubits) != 3:
            return
        
        control1, control2, target = target_qubits[0], target_qubits[1], target_qubits[2]
        
        # Add control qubits (dots)
        control1_line = control1 * 3 + 1
        control2_line = control2 * 3 + 1
        grid[control1_line][position] = '●'
        grid[control2_line][position] = '●'
        
        # Add target qubit (X)
        target_line = target * 3 + 1
        grid[target_line][position] = '⊕'
        
        # Add vertical lines
        lines = [control1_line, control2_line, target_line]
        start_line = min(lines)
        end_line = max(lines)
        
        for line in range(start_line + 1, end_line):
            if grid[line][position] == '─':
                grid[line][position] = '│'
    
    def _add_swap_gate(self, grid: List[List[str]], target_qubits: List[int], position: int, num_qubits: int):
        """Add a SWAP gate to the circuit."""
        if len(target_qubits) != 2:
            return
        
        qubit1, qubit2 = target_qubits[0], target_qubits[1]
        
        # Add SWAP symbols
        line1 = qubit1 * 3 + 1
        line2 = qubit2 * 3 + 1
        grid[line1][position] = '×'
        grid[line2][position] = '×'
        
        # Add connecting line
        start_line = min(line1, line2)
        end_line = max(line1, line2)
        
        for line in range(start_line + 1, end_line):
            if grid[line][position] == '─':
                grid[line][position] = '│'
    
    def _add_measurement(self, grid: List[List[str]], target_qubits: List[int], position: int, num_qubits: int):
        """Add measurement to the circuit."""
        for qubit in target_qubits:
            line_index = qubit * 3 + 1
            # Add measurement symbol
            grid[line_index][position] = 'M'
            # Add brackets
            grid[line_index][position-1] = '['
            grid[line_index][position+1] = ']'
    
    def _grid_to_string(self, grid: List[List[str]]) -> str:
        """Convert the grid to a string representation."""
        lines = []
        for row in grid:
            # Remove trailing spaces
            line = ''.join(row).rstrip()
            if line:  # Only add non-empty lines
                lines.append(line)
        
        return '\n'.join(lines)
    
    def generate_instruction_diagram(self, instruction: str, num_qubits: int) -> str:
        """
        Generate a diagram for a single instruction.
        
        Args:
            instruction: Quantum instruction string
            num_qubits: Number of qubits
        
        Returns:
            ASCII diagram for the instruction
        """
        # Parse the instruction
        parsed = self._parse_instruction(instruction)
        if not parsed:
            return "Invalid instruction"
        
        # Generate diagram
        return self.generate_diagram([parsed], num_qubits)
    
    def _parse_instruction(self, instruction: str) -> Optional[Dict[str, Any]]:
        """Parse a quantum instruction string."""
        instruction = instruction.strip()
        
        # Handle comments
        if instruction.startswith('#'):
            return None
        
        # Handle gates
        gate_pattern = r'^([A-Za-z]+)\s+q(\d+)(?:\s*,\s*q(\d+))?(?:\s*,\s*q(\d+))?$'
        match = re.match(gate_pattern, instruction)
        
        if match:
            gate_type = match.group(1)
            qubits = [int(match.group(2))]
            if match.group(3):
                qubits.append(int(match.group(3)))
            if match.group(4):
                qubits.append(int(match.group(4)))
            
            return {
                'gate': gate_type,
                'qubits': qubits
            }
        
        # Handle measurement
        if instruction.upper().startswith('MEASURE'):
            if instruction.upper() == 'MEASURE':
                return {
                    'gate': 'MEASURE',
                    'qubits': list(range(num_qubits))
                }
            else:
                # Parse specific qubits
                qubit_pattern = r'q(\d+)'
                qubits = [int(match.group(1)) for match in re.finditer(qubit_pattern, instruction)]
                return {
                    'gate': 'MEASURE',
                    'qubits': qubits
                }
        
        return None
    
    def generate_circuit_summary(self, instructions: List[Dict[str, Any]], num_qubits: int) -> str:
        """
        Generate a summary of the circuit.
        
        Args:
            instructions: List of quantum instructions
            num_qubits: Number of qubits
        
        Returns:
            Text summary of the circuit
        """
        summary = []
        summary.append(f"Quantum Circuit Summary ({num_qubits} qubits)")
        summary.append("=" * 40)
        
        # Count different types of gates
        gate_counts = {}
        for instruction in instructions:
            gate_type = instruction.get('gate', '')
            gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
        
        summary.append("Gate counts:")
        for gate_type, count in sorted(gate_counts.items()):
            summary.append(f"  {gate_type}: {count}")
        
        summary.append(f"\nTotal instructions: {len(instructions)}")
        
        return '\n'.join(summary)
    
    def generate_depth_analysis(self, instructions: List[Dict[str, Any]], num_qubits: int) -> str:
        """
        Generate circuit depth analysis.
        
        Args:
            instructions: List of quantum instructions
            num_qubits: Number of qubits
        
        Returns:
            Depth analysis string
        """
        # Calculate circuit depth
        qubit_depths = [0] * num_qubits
        
        for instruction in instructions:
            gate_type = instruction.get('gate', '')
            target_qubits = instruction.get('qubits', [])
            
            if gate_type == 'MEASURE':
                continue  # Measurement doesn't contribute to depth
            
            # Update depths for affected qubits
            for qubit in target_qubits:
                if 0 <= qubit < num_qubits:
                    qubit_depths[qubit] += 1
        
        max_depth = max(qubit_depths) if qubit_depths else 0
        
        analysis = []
        analysis.append(f"Circuit Depth Analysis")
        analysis.append("=" * 30)
        analysis.append(f"Maximum depth: {max_depth}")
        analysis.append(f"Qubit depths: {qubit_depths}")
        
        return '\n'.join(analysis)
