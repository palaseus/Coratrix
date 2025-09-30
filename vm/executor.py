"""
Quantum instruction executor.

This module provides the QuantumExecutor class that executes
parsed quantum instructions by calling the gate simulator backend.
"""

from typing import List, Optional, Any
from core.qubit import QuantumState
from core.circuit import QuantumCircuit
from core.gates import XGate, YGate, ZGate, HGate, CNOTGate
from core.measurement import Measurement
from .instructions import QuantumInstruction


class QuantumExecutor:
    """
    Executes quantum instructions by interfacing with the gate simulator.
    
    The executor maintains the quantum state and applies gates and
    measurements as specified by the instruction sequence.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize the quantum executor.
        
        Args:
            num_qubits: Number of qubits in the quantum system
        """
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)
        self.measurement = Measurement(self.circuit.get_state())
        self.execution_history = []
        
        # Gate registry for quick lookup
        self.gate_registry = {
            'X': XGate(),
            'Y': YGate(),
            'Z': ZGate(),
            'H': HGate(),
            'CNOT': CNOTGate()
        }
        
        # Add advanced gates if available
        try:
            from core.advanced_gates import CPhaseGate
            self.gate_registry['CPhase'] = CPhaseGate()
        except ImportError:
            pass
    
    def execute_instruction(self, instruction: QuantumInstruction) -> Any:
        """
        Execute a single quantum instruction.
        
        Args:
            instruction: The instruction to execute
        
        Returns:
            Result of the instruction execution
        """
        result = instruction.execute(self)
        self.execution_history.append((instruction, result))
        return result
    
    def execute_instructions(self, instructions: List[QuantumInstruction]) -> List[Any]:
        """
        Execute a sequence of quantum instructions.
        
        Args:
            instructions: List of instructions to execute
        
        Returns:
            List of results from each instruction
        """
        results = []
        for instruction in instructions:
            result = self.execute_instruction(instruction)
            results.append(result)
        return results
    
    def apply_gate(self, gate_name: str, target_qubits: List[int]) -> None:
        """
        Apply a quantum gate to the quantum state.
        
        Args:
            gate_name: Name of the gate to apply
            target_qubits: List of qubit indices the gate acts on
        """
        if gate_name not in self.gate_registry:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        gate = self.gate_registry[gate_name]
        self.circuit.apply_gate(gate, target_qubits)
    
    def measure_all(self) -> List[int]:
        """
        Measure all qubits in the system.
        
        Returns:
            List of measurement results (0s and 1s)
        """
        return self.measurement.measure_all()
    
    def measure_qubit(self, qubit_index: int) -> int:
        """
        Measure a specific qubit.
        
        Args:
            qubit_index: Index of the qubit to measure
        
        Returns:
            Measurement result (0 or 1)
        """
        return self.measurement.measure_qubit(qubit_index)
    
    def get_state(self) -> QuantumState:
        """Get the current quantum state."""
        return self.circuit.get_state()
    
    def get_state_vector(self) -> List[complex]:
        """Get the current state vector."""
        return self.circuit.get_state_vector()
    
    def get_probabilities(self) -> List[float]:
        """Get the probability distribution over all basis states."""
        return self.circuit.get_probabilities()
    
    def get_measurement_history(self) -> List[Any]:
        """Get the history of all measurements performed."""
        return self.measurement.get_measurement_history()
    
    def get_execution_history(self) -> List[tuple]:
        """Get the history of all executed instructions."""
        return self.execution_history.copy()
    
    def reset(self) -> None:
        """Reset the executor to the initial |00...0⟩ state."""
        self.circuit.reset()
        self.measurement = Measurement(self.circuit.get_state())
        self.execution_history = []
    
    def get_circuit_string(self) -> str:
        """Get a string representation of the current circuit."""
        return str(self.circuit)
    
    def get_state_string(self) -> str:
        """Get a string representation of the current quantum state."""
        return str(self.circuit.get_state())
    
    def get_entanglement_info(self) -> dict:
        """
        Get information about entanglement in the current state.
        
        Returns:
            Dictionary with entanglement information
        """
        state = self.circuit.get_state()
        probabilities = state.get_probabilities()
        
        # Check for Bell states (maximally entangled 2-qubit states)
        bell_states = {
            '|Φ⁺⟩': [0.5, 0.0, 0.0, 0.5],  # (|00⟩ + |11⟩)/√2
            '|Φ⁻⟩': [0.5, 0.0, 0.0, -0.5], # (|00⟩ - |11⟩)/√2
            '|Ψ⁺⟩': [0.0, 0.5, 0.5, 0.0],   # (|01⟩ + |10⟩)/√2
            '|Ψ⁻⟩': [0.0, 0.5, -0.5, 0.0]  # (|01⟩ - |10⟩)/√2
        }
        
        # Check if the state matches any Bell state
        for bell_name, bell_probs in bell_states.items():
            if len(probabilities) == 4:  # 2-qubit system
                matches = all(abs(p - bp) < 1e-10 for p, bp in zip(probabilities, bell_probs))
                if matches:
                    return {
                        'is_bell_state': True,
                        'bell_state': bell_name,
                        'entanglement': 'maximal'
                    }
        
        # Check for general entanglement (non-separable states)
        is_entangled = self._check_entanglement(state)
        
        return {
            'is_bell_state': False,
            'bell_state': None,
            'entanglement': 'maximal' if is_entangled else 'none'
        }
    
    def _check_entanglement(self, state: QuantumState) -> bool:
        """
        Check if the quantum state is entangled.
        
        Args:
            state: Quantum state to check
        
        Returns:
            True if the state is entangled, False otherwise
        """
        # Simple heuristic: check if the state can be written as a product state
        # For a 2-qubit system, this means checking if the state vector
        # can be factored into a product of single-qubit states
        
        if state.num_qubits != 2:
            return False  # Only check 2-qubit entanglement for now
        
        # For a 2-qubit system, check if the state is separable
        # A separable state has the form |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩
        # This means the state vector can be written as [a₁b₁, a₁b₂, a₂b₁, a₂b₂]
        # where |ψ₁⟩ = [a₁, a₂] and |ψ₂⟩ = [b₁, b₂]
        
        amplitudes = state.state_vector
        
        # Check if the state is separable by trying to factor it
        # This is a simplified check - a full entanglement measure would be more complex
        
        # If any of the amplitudes are zero in a specific pattern, it might be separable
        if (abs(amplitudes[0]) < 1e-10 and abs(amplitudes[3]) < 1e-10) or \
           (abs(amplitudes[1]) < 1e-10 and abs(amplitudes[2]) < 1e-10):
            return False  # Likely separable
        
        # If the state has non-zero amplitudes in a pattern that suggests entanglement
        # (like Bell states), it's likely entangled
        return True
