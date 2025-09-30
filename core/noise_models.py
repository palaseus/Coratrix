"""
Noise models and error mitigation for quantum circuits.

This module provides configurable noise channels and error mitigation
techniques for quantum circuit simulation.
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import random

from core.qubit import QuantumState
from core.scalable_quantum_state import ScalableQuantumState


class NoiseChannel(Enum):
    """Types of noise channels."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    READOUT_ERROR = "readout_error"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"


@dataclass
class NoiseModel:
    """Configuration for noise model."""
    depolarizing_error: float = 0.01
    amplitude_damping_error: float = 0.005
    phase_damping_error: float = 0.005
    readout_error: float = 0.02
    bit_flip_error: float = 0.01
    phase_flip_error: float = 0.01
    gate_error: float = 0.005
    t1_time: float = 100.0  # T1 relaxation time in microseconds
    t2_time: float = 50.0   # T2 dephasing time in microseconds
    gate_time: float = 0.1  # Gate execution time in microseconds


class QuantumNoise:
    """Quantum noise channel implementation."""
    
    def __init__(self, noise_model: Optional[NoiseModel] = None):
        self.noise_model = noise_model or NoiseModel()
        self.random_state = np.random.RandomState(42)  # For reproducibility
    
    def apply_depolarizing_noise(self, state: Union[QuantumState, ScalableQuantumState], 
                                error_rate: float) -> Union[QuantumState, ScalableQuantumState]:
        """Apply depolarizing noise to quantum state."""
        if error_rate <= 0:
            return state
        
        # Get state vector
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        # Apply depolarizing channel
        # With probability error_rate, apply random Pauli error
        if self.random_state.random() < error_rate:
            # Choose random Pauli error (X, Y, or Z)
            pauli_error = self.random_state.choice(['X', 'Y', 'Z'])
            
            if pauli_error == 'X':
                # Apply X gate (bit flip)
                x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
                for i in range(state.num_qubits):
                    if self.random_state.random() < 0.5:  # Apply to random qubit
                        self._apply_single_qubit_gate(state, x_matrix, i)
            
            elif pauli_error == 'Y':
                # Apply Y gate
                y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
                for i in range(state.num_qubits):
                    if self.random_state.random() < 0.5:
                        self._apply_single_qubit_gate(state, y_matrix, i)
            
            elif pauli_error == 'Z':
                # Apply Z gate (phase flip)
                z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
                for i in range(state.num_qubits):
                    if self.random_state.random() < 0.5:
                        self._apply_single_qubit_gate(state, z_matrix, i)
        
        return state
    
    def apply_amplitude_damping(self, state: Union[QuantumState, ScalableQuantumState], 
                               error_rate: float) -> Union[QuantumState, ScalableQuantumState]:
        """Apply amplitude damping noise."""
        if error_rate <= 0:
            return state
        
        # Amplitude damping Kraus operators
        gamma = error_rate
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        
        # Apply to each qubit with probability
        for i in range(state.num_qubits):
            if self.random_state.random() < error_rate:
                self._apply_kraus_operators(state, [E0, E1], i)
        
        return state
    
    def apply_phase_damping(self, state: Union[QuantumState, ScalableQuantumState], 
                           error_rate: float) -> Union[QuantumState, ScalableQuantumState]:
        """Apply phase damping noise."""
        if error_rate <= 0:
            return state
        
        # Phase damping Kraus operators
        gamma = error_rate
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
        
        # Apply to each qubit with probability
        for i in range(state.num_qubits):
            if self.random_state.random() < error_rate:
                self._apply_kraus_operators(state, [E0, E1], i)
        
        return state
    
    def apply_readout_error(self, measurement_result: str, error_rate: float) -> str:
        """Apply readout error to measurement result."""
        if error_rate <= 0:
            return measurement_result
        
        result_list = list(measurement_result)
        for i, bit in enumerate(result_list):
            if self.random_state.random() < error_rate:
                result_list[i] = '1' if bit == '0' else '0'
        
        return ''.join(result_list)
    
    def apply_gate_error(self, state: Union[QuantumState, ScalableQuantumState], 
                        gate_name: str, error_rate: float) -> Union[QuantumState, ScalableQuantumState]:
        """Apply gate-specific error."""
        if error_rate <= 0:
            return state
        
        # Apply depolarizing noise after gate
        return self.apply_depolarizing_noise(state, error_rate)
    
    def _apply_single_qubit_gate(self, state: Union[QuantumState, ScalableQuantumState], 
                                gate_matrix: np.ndarray, qubit_index: int):
        """Apply single-qubit gate to state."""
        if isinstance(state, ScalableQuantumState):
            # For ScalableQuantumState, we need to handle sparse/GPU cases
            state_vector = state.to_dense()
            full_matrix = self._build_full_matrix(state.num_qubits, qubit_index, gate_matrix)
            new_state_vector = full_matrix @ state_vector
            state.from_dense(new_state_vector)
        else:
            # For regular QuantumState
            full_matrix = self._build_full_matrix(state.num_qubits, qubit_index, gate_matrix)
            state.state_vector = full_matrix @ state.state_vector
            state.normalize()
    
    def _apply_kraus_operators(self, state: Union[QuantumState, ScalableQuantumState], 
                              kraus_ops: List[np.ndarray], qubit_index: int):
        """Apply Kraus operators to state."""
        # Choose random Kraus operator
        probabilities = [np.trace(op @ op.conj().T) for op in kraus_ops]
        probabilities = np.array(probabilities) / sum(probabilities)
        
        chosen_op = self.random_state.choice(kraus_ops, p=probabilities)
        
        # Apply chosen operator
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
            full_matrix = self._build_full_matrix(state.num_qubits, qubit_index, chosen_op)
            new_state_vector = full_matrix @ state_vector
            state.from_dense(new_state_vector)
        else:
            full_matrix = self._build_full_matrix(state.num_qubits, qubit_index, chosen_op)
            state.state_vector = full_matrix @ state.state_vector
            state.normalize()
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, 
                          gate_matrix: np.ndarray) -> np.ndarray:
        """Build full matrix for single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        for i in range(dimension):
            for j in range(dimension):
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if gate should be applied between states i and j."""
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class ErrorMitigation:
    """Error mitigation techniques for quantum circuits."""
    
    def __init__(self, noise_model: Optional[NoiseModel] = None):
        self.noise_model = noise_model or NoiseModel()
        self.noise = QuantumNoise(noise_model)
    
    def apply_mid_circuit_purification(self, state: Union[QuantumState, ScalableQuantumState], 
                                      purification_threshold: float = 0.8) -> Union[QuantumState, ScalableQuantumState]:
        """Apply mid-circuit purification to improve state fidelity."""
        # Calculate current fidelity
        fidelity = self._calculate_state_fidelity(state)
        
        if fidelity < purification_threshold:
            # Apply purification protocol
            state = self._purification_protocol(state)
        
        return state
    
    def apply_real_time_feedback(self, state: Union[QuantumState, ScalableQuantumState], 
                                target_fidelity: float = 0.95) -> Union[QuantumState, ScalableQuantumState]:
        """Apply real-time feedback to maintain target fidelity."""
        current_fidelity = self._calculate_state_fidelity(state)
        
        if current_fidelity < target_fidelity:
            # Apply corrective operations
            correction_strength = target_fidelity - current_fidelity
            state = self._apply_correction(state, correction_strength)
        
        return state
    
    def apply_error_correction_code(self, state: Union[QuantumState, ScalableQuantumState], 
                                   code_type: str = "repetition") -> Union[QuantumState, ScalableQuantumState]:
        """Apply error correction code."""
        if code_type == "repetition":
            return self._apply_repetition_code(state)
        elif code_type == "surface":
            return self._apply_surface_code_patch(state)
        else:
            raise ValueError(f"Unknown error correction code: {code_type}")
    
    def _calculate_state_fidelity(self, state: Union[QuantumState, ScalableQuantumState]) -> float:
        """Calculate state fidelity (simplified measure)."""
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        # Calculate purity (simplified fidelity measure)
        purity = np.sum(np.abs(state_vector)**4)
        return float(purity)
    
    def _purification_protocol(self, state: Union[QuantumState, ScalableQuantumState]) -> Union[QuantumState, ScalableQuantumState]:
        """Apply purification protocol."""
        # Simplified purification: apply random corrective operations
        correction_ops = ['X', 'Y', 'Z']
        
        for i in range(state.num_qubits):
            if random.random() < 0.1:  # 10% chance of correction
                op = random.choice(correction_ops)
                if op == 'X':
                    self._apply_x_correction(state, i)
                elif op == 'Y':
                    self._apply_y_correction(state, i)
                elif op == 'Z':
                    self._apply_z_correction(state, i)
        
        return state
    
    def _apply_correction(self, state: Union[QuantumState, ScalableQuantumState], 
                         strength: float) -> Union[QuantumState, ScalableQuantumState]:
        """Apply correction with given strength."""
        # Apply weak corrective operations
        for i in range(state.num_qubits):
            if random.random() < strength:
                self._apply_x_correction(state, i)
        
        return state
    
    def _apply_repetition_code(self, state: Union[QuantumState, ScalableQuantumState]) -> Union[QuantumState, ScalableQuantumState]:
        """Apply repetition code error correction."""
        # Simplified repetition code implementation
        # In practice, this would involve encoding the logical qubit
        # into multiple physical qubits
        
        # For now, just apply some corrective operations
        return self._apply_correction(state, 0.1)
    
    def _apply_surface_code_patch(self, state: Union[QuantumState, ScalableQuantumState]) -> Union[QuantumState, ScalableQuantumState]:
        """Apply surface code patch error correction."""
        # Simplified surface code implementation
        # In practice, this would involve more complex stabilizer measurements
        
        # For now, just apply some corrective operations
        return self._apply_correction(state, 0.05)
    
    def _apply_x_correction(self, state: Union[QuantumState, ScalableQuantumState], qubit_index: int):
        """Apply X correction to specific qubit."""
        x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_single_qubit_gate(state, x_matrix, qubit_index)
    
    def _apply_y_correction(self, state: Union[QuantumState, ScalableQuantumState], qubit_index: int):
        """Apply Y correction to specific qubit."""
        y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_single_qubit_gate(state, y_matrix, qubit_index)
    
    def _apply_z_correction(self, state: Union[QuantumState, ScalableQuantumState], qubit_index: int):
        """Apply Z correction to specific qubit."""
        z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_single_qubit_gate(state, z_matrix, qubit_index)
    
    def _apply_single_qubit_gate(self, state: Union[QuantumState, ScalableQuantumState], 
                                gate_matrix: np.ndarray, qubit_index: int):
        """Apply single-qubit gate to state."""
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
            full_matrix = self._build_full_matrix(state.num_qubits, qubit_index, gate_matrix)
            new_state_vector = full_matrix @ state_vector
            state.from_dense(new_state_vector)
        else:
            full_matrix = self._build_full_matrix(state.num_qubits, qubit_index, gate_matrix)
            state.state_vector = full_matrix @ state.state_vector
            state.normalize()
    
    def _build_full_matrix(self, num_qubits: int, target_qubit: int, 
                          gate_matrix: np.ndarray) -> np.ndarray:
        """Build full matrix for single-qubit gate."""
        dimension = 2 ** num_qubits
        full_matrix = np.eye(dimension, dtype=complex)
        
        for i in range(dimension):
            for j in range(dimension):
                if self._should_apply_gate(i, j, target_qubit, num_qubits):
                    qubit_i = (i >> (num_qubits - 1 - target_qubit)) & 1
                    qubit_j = (j >> (num_qubits - 1 - target_qubit)) & 1
                    full_matrix[i, j] = gate_matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _should_apply_gate(self, i: int, j: int, target_qubit: int, num_qubits: int) -> bool:
        """Check if gate should be applied between states i and j."""
        for q in range(num_qubits):
            if q != target_qubit:
                bit_i = (i >> (num_qubits - 1 - q)) & 1
                bit_j = (j >> (num_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    return False
        return True


class NoisyQuantumCircuit:
    """Quantum circuit with noise simulation."""
    
    def __init__(self, num_qubits: int, noise_model: Optional[NoiseModel] = None):
        self.num_qubits = num_qubits
        self.noise_model = noise_model or NoiseModel()
        self.noise = QuantumNoise(noise_model)
        self.mitigation = ErrorMitigation(noise_model)
        self.gates = []
        self.quantum_state = QuantumState(num_qubits)
    
    def add_gate(self, gate, target_qubits: List[int]):
        """Add a gate to the circuit."""
        self.gates.append((gate, target_qubits))
    
    def apply_gate(self, gate, target_qubits: List[int]):
        """Apply a gate with noise."""
        # Apply the gate
        gate.apply(self.quantum_state, target_qubits)
        
        # Apply gate error
        gate_error = self.noise_model.gate_error
        self.quantum_state = self.noise.apply_gate_error(
            self.quantum_state, gate.name, gate_error
        )
        
        # Apply other noise channels
        self.quantum_state = self.noise.apply_depolarizing_noise(
            self.quantum_state, self.noise_model.depolarizing_error
        )
        
        self.quantum_state = self.noise.apply_amplitude_damping(
            self.quantum_state, self.noise_model.amplitude_damping_error
        )
        
        self.quantum_state = self.noise.apply_phase_damping(
            self.quantum_state, self.noise_model.phase_damping_error
        )
    
    def execute(self):
        """Execute the circuit with noise."""
        for gate, target_qubits in self.gates:
            self.apply_gate(gate, target_qubits)
    
    def execute_with_mitigation(self, mitigation_enabled: bool = True):
        """Execute circuit with optional error mitigation."""
        for gate, target_qubits in self.gates:
            self.apply_gate(gate, target_qubits)
            
            # Apply mid-circuit mitigation
            if mitigation_enabled:
                self.quantum_state = self.mitigation.apply_mid_circuit_purification(
                    self.quantum_state
                )
    
    def measure_with_readout_error(self, shots: int = 1024) -> Dict[str, int]:
        """Measure with readout error."""
        from core.measurement import Measurement
        
        measurement = Measurement(self.quantum_state)
        counts = measurement.measure_multiple(self.quantum_state, shots)
        
        # Apply readout error
        noisy_counts = {}
        for bitstring, count in counts.items():
            for _ in range(count):
                noisy_bitstring = self.noise.apply_readout_error(
                    bitstring, self.noise_model.readout_error
                )
                noisy_counts[noisy_bitstring] = noisy_counts.get(noisy_bitstring, 0) + 1
        
        return noisy_counts
    
    def get_state(self) -> QuantumState:
        """Get the current quantum state."""
        return self.quantum_state
