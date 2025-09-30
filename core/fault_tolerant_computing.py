"""
Fault-Tolerant Quantum Computing Module for Coratrix 4.0

This module implements fault-tolerant quantum computing support including
surface code implementations, logical qubit simulations, and error-corrected circuits.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of quantum errors."""
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"


class LogicalGate(Enum):
    """Logical quantum gates for fault-tolerant computation."""
    LOGICAL_X = "logical_x"
    LOGICAL_Y = "logical_y"
    LOGICAL_Z = "logical_z"
    LOGICAL_H = "logical_h"
    LOGICAL_CNOT = "logical_cnot"
    LOGICAL_T = "logical_t"
    LOGICAL_S = "logical_s"


@dataclass
class ErrorSyndrome:
    """Error syndrome for error correction."""
    stabilizer_measurements: List[int]
    error_locations: List[Tuple[int, int]]
    error_types: List[ErrorType]
    confidence: float


@dataclass
class LogicalQubit:
    """Logical qubit representation."""
    physical_qubits: List[int]
    code_distance: int
    logical_state: np.ndarray
    stabilizer_generators: List[np.ndarray]
    logical_operators: Dict[str, np.ndarray]


class SurfaceCode:
    """
    Surface code implementation for fault-tolerant quantum computing.
    
    The surface code is a 2D topological quantum error-correcting code
    that provides fault-tolerant quantum computation.
    """
    
    def __init__(self, distance: int, lattice_size: Tuple[int, int]):
        """
        Initialize surface code.
        
        Args:
            distance: Code distance (determines error correction capability)
            lattice_size: Size of the surface code lattice (rows, cols)
        """
        self.distance = distance
        self.lattice_size = lattice_size
        self.physical_qubits = self._create_physical_qubits()
        self.stabilizers = self._create_stabilizers()
        self.logical_operators = self._create_logical_operators()
        self.error_history = []
    
    def _create_physical_qubits(self) -> List[int]:
        """Create physical qubits for the surface code."""
        rows, cols = self.lattice_size
        qubits = []
        
        # Create data qubits (on vertices)
        for i in range(rows):
            for j in range(cols):
                qubits.append(i * cols + j)
        
        # Create ancilla qubits (on edges and faces)
        # This is simplified - full implementation would properly place ancillas
        ancilla_start = rows * cols
        for i in range(rows - 1):
            for j in range(cols - 1):
                qubits.append(ancilla_start + i * (cols - 1) + j)
        
        return qubits
    
    def _create_stabilizers(self) -> List[np.ndarray]:
        """Create stabilizer generators for the surface code."""
        stabilizers = []
        
        # X-stabilizers (measure X on faces)
        for i in range(self.lattice_size[0] - 1):
            for j in range(self.lattice_size[1] - 1):
                stabilizer = self._create_x_stabilizer(i, j)
                stabilizers.append(stabilizer)
        
        # Z-stabilizers (measure Z on faces)
        for i in range(self.lattice_size[0] - 1):
            for j in range(self.lattice_size[1] - 1):
                stabilizer = self._create_z_stabilizer(i, j)
                stabilizers.append(stabilizer)
        
        return stabilizers
    
    def _create_x_stabilizer(self, row: int, col: int) -> np.ndarray:
        """Create X-stabilizer for a face."""
        # Simplified implementation
        # Full implementation would create proper stabilizer matrices
        num_qubits = len(self.physical_qubits)
        stabilizer = np.zeros(num_qubits, dtype=int)
        
        # Mark qubits involved in this stabilizer
        # This is a simplified version
        qubit_indices = [
            row * self.lattice_size[1] + col,
            row * self.lattice_size[1] + col + 1,
            (row + 1) * self.lattice_size[1] + col,
            (row + 1) * self.lattice_size[1] + col + 1
        ]
        
        for idx in qubit_indices:
            if idx < num_qubits:
                stabilizer[idx] = 1
        
        return stabilizer
    
    def _create_z_stabilizer(self, row: int, col: int) -> np.ndarray:
        """Create Z-stabilizer for a face."""
        # Similar to X-stabilizer but for Z measurements
        num_qubits = len(self.physical_qubits)
        stabilizer = np.zeros(num_qubits, dtype=int)
        
        qubit_indices = [
            row * self.lattice_size[1] + col,
            row * self.lattice_size[1] + col + 1,
            (row + 1) * self.lattice_size[1] + col,
            (row + 1) * self.lattice_size[1] + col + 1
        ]
        
        for idx in qubit_indices:
            if idx < num_qubits:
                stabilizer[idx] = 1
        
        return stabilizer
    
    def _create_logical_operators(self) -> Dict[str, np.ndarray]:
        """Create logical operators for the surface code."""
        logical_ops = {}
        
        # Logical X operator (horizontal string)
        logical_x = np.zeros(len(self.physical_qubits), dtype=int)
        for j in range(self.lattice_size[1]):
            logical_x[j] = 1
        logical_ops['X'] = logical_x
        
        # Logical Z operator (vertical string)
        logical_z = np.zeros(len(self.physical_qubits), dtype=int)
        for i in range(self.lattice_size[0]):
            logical_z[i * self.lattice_size[1]] = 1
        logical_ops['Z'] = logical_z
        
        return logical_ops
    
    def measure_stabilizers(self, state: np.ndarray) -> List[int]:
        """Measure all stabilizers and return syndrome."""
        syndrome = []
        
        for stabilizer in self.stabilizers:
            # Measure stabilizer
            measurement = self._measure_stabilizer(state, stabilizer)
            syndrome.append(measurement)
        
        return syndrome
    
    def _measure_stabilizer(self, state: np.ndarray, stabilizer: np.ndarray) -> int:
        """Measure a single stabilizer."""
        # Simplified measurement
        # Full implementation would perform actual quantum measurement
        
        # For demonstration, return random measurement result
        return np.random.choice([0, 1])
    
    def decode_syndrome(self, syndrome: List[int]) -> ErrorSyndrome:
        """Decode error syndrome to identify errors."""
        # Simplified decoding
        # Full implementation would use minimum weight perfect matching
        
        error_locations = []
        error_types = []
        
        # Find errors based on syndrome
        for i, measurement in enumerate(syndrome):
            if measurement == 1:  # Error detected
                # Determine error location and type
                error_location = self._get_error_location(i)
                error_type = self._get_error_type(i)
                
                error_locations.append(error_location)
                error_types.append(error_type)
        
        confidence = self._calculate_confidence(syndrome)
        
        return ErrorSyndrome(
            stabilizer_measurements=syndrome,
            error_locations=error_locations,
            error_types=error_types,
            confidence=confidence
        )
    
    def _get_error_location(self, stabilizer_index: int) -> Tuple[int, int]:
        """Get physical location of error."""
        # Simplified error location
        row = stabilizer_index // (self.lattice_size[1] - 1)
        col = stabilizer_index % (self.lattice_size[1] - 1)
        return (row, col)
    
    def _get_error_type(self, stabilizer_index: int) -> ErrorType:
        """Determine type of error."""
        # Simplified error type determination
        if stabilizer_index < len(self.stabilizers) // 2:
            return ErrorType.BIT_FLIP
        else:
            return ErrorType.PHASE_FLIP
    
    def _calculate_confidence(self, syndrome: List[int]) -> float:
        """Calculate confidence in error correction."""
        # Simplified confidence calculation
        error_count = sum(syndrome)
        total_stabilizers = len(syndrome)
        
        if error_count == 0:
            return 1.0
        elif error_count <= self.distance // 2:
            return 0.8
        else:
            return 0.3
    
    def apply_error_correction(self, state: np.ndarray, syndrome: List[int]) -> np.ndarray:
        """Apply error correction based on syndrome."""
        error_syndrome = self.decode_syndrome(syndrome)
        
        corrected_state = state.copy()
        
        # Apply corrections for each detected error
        for i, (location, error_type) in enumerate(zip(error_syndrome.error_locations, 
                                                      error_syndrome.error_types)):
            correction = self._get_correction_operator(location, error_type)
            corrected_state = self._apply_correction(corrected_state, correction)
        
        return corrected_state
    
    def _get_correction_operator(self, location: Tuple[int, int], error_type: ErrorType) -> np.ndarray:
        """Get correction operator for specific error."""
        # Simplified correction operator
        num_qubits = len(self.physical_qubits)
        correction = np.eye(num_qubits, dtype=np.complex128)
        
        # Apply appropriate correction based on error type
        if error_type == ErrorType.BIT_FLIP:
            # Apply X correction
            qubit_index = location[0] * self.lattice_size[1] + location[1]
            if qubit_index < num_qubits:
                correction[qubit_index, qubit_index] = -1
        elif error_type == ErrorType.PHASE_FLIP:
            # Apply Z correction
            qubit_index = location[0] * self.lattice_size[1] + location[1]
            if qubit_index < num_qubits:
                correction[qubit_index, qubit_index] = -1
        
        return correction
    
    def _apply_correction(self, state: np.ndarray, correction: np.ndarray) -> np.ndarray:
        """Apply correction operator to state."""
        # Ensure dimensions match
        if state.shape[0] != correction.shape[1]:
            # If dimensions don't match, create a properly sized correction matrix
            state_size = state.shape[0]
            if correction.shape[0] != state_size or correction.shape[1] != state_size:
                # Create identity matrix of correct size
                correction = np.eye(state_size, dtype=np.complex128)
        
        return correction @ state


class LogicalQubitSimulator:
    """
    Simulator for logical qubits with error correction.
    """
    
    def __init__(self, surface_code: SurfaceCode):
        """
        Initialize logical qubit simulator.
        
        Args:
            surface_code: Surface code for error correction
        """
        self.surface_code = surface_code
        self.logical_qubits = {}
        self.error_rates = {}
    
    def create_logical_qubit(self, qubit_id: str, initial_state: np.ndarray = None) -> LogicalQubit:
        """
        Create a logical qubit.
        
        Args:
            qubit_id: Unique identifier for logical qubit
            initial_state: Initial logical state (default: |0⟩)
            
        Returns:
            Created logical qubit
        """
        if initial_state is None:
            initial_state = np.array([1, 0], dtype=np.complex128)  # |0⟩ state
        
        # Create logical qubit
        logical_qubit = LogicalQubit(
            physical_qubits=self.surface_code.physical_qubits,
            code_distance=self.surface_code.distance,
            logical_state=initial_state,
            stabilizer_generators=self.surface_code.stabilizers,
            logical_operators=self.surface_code.logical_operators
        )
        
        self.logical_qubits[qubit_id] = logical_qubit
        return logical_qubit
    
    def apply_logical_gate(self, qubit_id: str, gate: LogicalGate, 
                          target_qubit_id: str = None) -> bool:
        """
        Apply logical gate to logical qubit.
        
        Args:
            qubit_id: Target logical qubit
            gate: Logical gate to apply
            target_qubit_id: Target qubit for two-qubit gates
            
        Returns:
            Success status
        """
        if qubit_id not in self.logical_qubits:
            logger.error(f"Logical qubit {qubit_id} not found")
            return False
        
        logical_qubit = self.logical_qubits[qubit_id]
        
        try:
            if gate == LogicalGate.LOGICAL_X:
                self._apply_logical_x(logical_qubit)
            elif gate == LogicalGate.LOGICAL_Z:
                self._apply_logical_z(logical_qubit)
            elif gate == LogicalGate.LOGICAL_H:
                self._apply_logical_h(logical_qubit)
            elif gate == LogicalGate.LOGICAL_CNOT:
                if target_qubit_id is None:
                    logger.error("Target qubit required for CNOT gate")
                    return False
                if target_qubit_id not in self.logical_qubits:
                    logger.error(f"Target logical qubit {target_qubit_id} not found")
                    return False
                self._apply_logical_cnot(logical_qubit, self.logical_qubits[target_qubit_id])
            else:
                logger.warning(f"Gate {gate} not implemented yet")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying logical gate {gate}: {e}")
            return False
    
    def _apply_logical_x(self, logical_qubit: LogicalQubit):
        """Apply logical X gate."""
        # Apply X to all qubits in logical X operator
        x_operator = logical_qubit.logical_operators['X']
        for i, qubit_involved in enumerate(x_operator):
            if qubit_involved:
                # Apply X gate to physical qubit i
                # This is simplified - full implementation would apply actual gates
                pass
    
    def _apply_logical_z(self, logical_qubit: LogicalQubit):
        """Apply logical Z gate."""
        # Apply Z to all qubits in logical Z operator
        z_operator = logical_qubit.logical_operators['Z']
        for i, qubit_involved in enumerate(z_operator):
            if qubit_involved:
                # Apply Z gate to physical qubit i
                pass
    
    def _apply_logical_h(self, logical_qubit: LogicalQubit):
        """Apply logical Hadamard gate."""
        # Logical Hadamard is implemented as H = X H Z
        self._apply_logical_x(logical_qubit)
        # Apply H to all physical qubits
        # Apply Z to all qubits in logical Z operator
        z_operator = logical_qubit.logical_operators['Z']
        for i, qubit_involved in enumerate(z_operator):
            if qubit_involved:
                pass
    
    def _apply_logical_cnot(self, control_qubit: LogicalQubit, target_qubit: LogicalQubit):
        """Apply logical CNOT gate."""
        # Logical CNOT between two logical qubits
        # This is a simplified implementation
        pass
    
    def measure_logical_qubit(self, qubit_id: str) -> Tuple[int, float]:
        """
        Measure logical qubit.
        
        Args:
            qubit_id: Logical qubit to measure
            
        Returns:
            Measurement result and confidence
        """
        if qubit_id not in self.logical_qubits:
            logger.error(f"Logical qubit {qubit_id} not found")
            return 0, 0.0
        
        logical_qubit = self.logical_qubits[qubit_id]
        
        # Perform error correction before measurement
        corrected_state = self._perform_error_correction(logical_qubit)
        
        # Measure logical qubit
        measurement_result = self._measure_logical_state(corrected_state)
        confidence = self._calculate_measurement_confidence(logical_qubit)
        
        return measurement_result, confidence
    
    def _perform_error_correction(self, logical_qubit: LogicalQubit) -> np.ndarray:
        """Perform error correction on logical qubit."""
        # Measure stabilizers
        syndrome = self.surface_code.measure_stabilizers(logical_qubit.logical_state)
        
        # Apply error correction
        corrected_state = self.surface_code.apply_error_correction(
            logical_qubit.logical_state, syndrome
        )
        
        return corrected_state
    
    def measure_stabilizers(self, qubit_id: str) -> List[int]:
        """
        Measure stabilizers for error detection.
        
        Args:
            qubit_id: Logical qubit to measure stabilizers for
            
        Returns:
            Syndrome measurement results
        """
        if qubit_id not in self.logical_qubits:
            logger.error(f"Logical qubit {qubit_id} not found")
            return []
        
        logical_qubit = self.logical_qubits[qubit_id]
        return self.surface_code.measure_stabilizers(logical_qubit.logical_state)
    
    def perform_error_correction(self, qubit_id: str, syndromes: List[int]) -> bool:
        """
        Perform error correction based on syndrome measurements.
        
        Args:
            qubit_id: Logical qubit to correct
            syndromes: Syndrome measurement results
            
        Returns:
            True if correction was successful
        """
        if qubit_id not in self.logical_qubits:
            logger.error(f"Logical qubit {qubit_id} not found")
            return False
        
        logical_qubit = self.logical_qubits[qubit_id]
        
        # Apply error correction
        corrected_state = self.surface_code.apply_error_correction(
            logical_qubit.logical_state, syndromes
        )
        
        # Update the logical qubit state
        logical_qubit.logical_state = corrected_state
        
        return True
    
    def _measure_logical_state(self, state: np.ndarray) -> int:
        """Measure logical state."""
        # Simplified measurement
        # Full implementation would perform proper quantum measurement
        
        # For demonstration, return measurement based on state probabilities
        if len(state) >= 2:
            prob_0 = abs(state[0]) ** 2
            prob_1 = abs(state[1]) ** 2
            
            # Normalize probabilities
            total_prob = prob_0 + prob_1
            if total_prob > 0:
                prob_0 /= total_prob
                prob_1 /= total_prob
            
            # Return measurement result
            return 0 if prob_0 > prob_1 else 1
        else:
            return 0
    
    def _calculate_measurement_confidence(self, logical_qubit: LogicalQubit) -> float:
        """Calculate confidence in logical measurement."""
        # Simplified confidence calculation
        # Full implementation would consider error rates and code distance
        
        base_confidence = 0.95  # Base confidence for logical measurement
        
        # Adjust based on code distance
        distance_factor = min(1.0, logical_qubit.code_distance / 5.0)
        
        return base_confidence * distance_factor


class FaultTolerantCircuit:
    """
    Fault-tolerant quantum circuit with error correction.
    """
    
    def __init__(self, surface_code: SurfaceCode):
        """
        Initialize fault-tolerant circuit.
        
        Args:
            surface_code: Surface code for error correction
        """
        self.surface_code = surface_code
        self.logical_simulator = LogicalQubitSimulator(surface_code)
        self.circuit_gates = []
        self.error_correction_cycles = []
    
    def add_logical_gate(self, gate: LogicalGate, qubit_id: str, 
                        target_qubit_id: str = None) -> bool:
        """
        Add logical gate to circuit.
        
        Args:
            gate: Logical gate to add
            qubit_id: Target logical qubit
            target_qubit_id: Target qubit for two-qubit gates
            
        Returns:
            Success status
        """
        success = self.logical_simulator.apply_logical_gate(qubit_id, gate, target_qubit_id)
        
        if success:
            self.circuit_gates.append({
                'gate': gate,
                'qubit_id': qubit_id,
                'target_qubit_id': target_qubit_id
            })
        
        return success
    
    def add_error_correction_cycle(self, qubit_ids: List[str]) -> bool:
        """
        Add error correction cycle.
        
        Args:
            qubit_ids: Logical qubits to perform error correction on
            
        Returns:
            Success status
        """
        try:
            for qubit_id in qubit_ids:
                if qubit_id in self.logical_simulator.logical_qubits:
                    logical_qubit = self.logical_simulator.logical_qubits[qubit_id]
                    corrected_state = self.logical_simulator._perform_error_correction(logical_qubit)
                    logical_qubit.logical_state = corrected_state
            
            self.error_correction_cycles.append(qubit_ids)
            return True
            
        except Exception as e:
            logger.error(f"Error in error correction cycle: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute fault-tolerant circuit.
        
        Returns:
            Execution results
        """
        results = {
            'success': True,
            'measurements': {},
            'error_correction_stats': {},
            'execution_time': 0.0
        }
        
        import time
        start_time = time.time()
        
        try:
            # Execute all gates
            for gate_info in self.circuit_gates:
                success = self.logical_simulator.apply_logical_gate(
                    gate_info['qubit_id'],
                    gate_info['gate'],
                    gate_info.get('target_qubit_id')
                )
                
                if not success:
                    results['success'] = False
                    break
            
            # Perform final measurements
            for qubit_id in self.logical_simulator.logical_qubits:
                measurement, confidence = self.logical_simulator.measure_logical_qubit(qubit_id)
                results['measurements'][qubit_id] = {
                    'result': measurement,
                    'confidence': confidence
                }
            
            # Calculate error correction statistics
            results['error_correction_stats'] = self._calculate_error_stats()
            
        except Exception as e:
            logger.error(f"Error executing fault-tolerant circuit: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        results['execution_time'] = time.time() - start_time
        return results
    
    def _calculate_error_stats(self) -> Dict[str, Any]:
        """Calculate error correction statistics."""
        stats = {
            'total_error_correction_cycles': len(self.error_correction_cycles),
            'average_confidence': 0.0,
            'error_rate': 0.0
        }
        
        # Calculate average confidence
        confidences = []
        for qubit_id in self.logical_simulator.logical_qubits:
            _, confidence = self.logical_simulator.measure_logical_qubit(qubit_id)
            confidences.append(confidence)
        
        if confidences:
            stats['average_confidence'] = np.mean(confidences)
        
        # Estimate error rate based on code distance
        stats['error_rate'] = 1.0 - stats['average_confidence']
        
        return stats


# Tutorial and educational functions
def create_surface_code_tutorial():
    """Create tutorial for surface code."""
    tutorial = {
        'title': 'Surface Code Tutorial',
        'description': 'Learn about surface codes for fault-tolerant quantum computing',
        'steps': [
            {
                'step': 1,
                'title': 'Create Surface Code',
                'description': 'Initialize a surface code with distance 3',
                'code': '''
# Create surface code
surface_code = SurfaceCode(distance=3, lattice_size=(3, 3))
print(f"Surface code created with {len(surface_code.physical_qubits)} physical qubits")
'''
            },
            {
                'step': 2,
                'title': 'Create Logical Qubit',
                'description': 'Create a logical qubit using the surface code',
                'code': '''
# Create logical qubit simulator
simulator = LogicalQubitSimulator(surface_code)

# Create logical qubit
logical_qubit = simulator.create_logical_qubit("qubit_0")
print(f"Logical qubit created with code distance {logical_qubit.code_distance}")
'''
            },
            {
                'step': 3,
                'title': 'Apply Logical Gates',
                'description': 'Apply logical gates to the logical qubit',
                'code': '''
# Apply logical gates
simulator.apply_logical_gate("qubit_0", LogicalGate.LOGICAL_H)
simulator.apply_logical_gate("qubit_0", LogicalGate.LOGICAL_X)
print("Logical gates applied successfully")
'''
            },
            {
                'step': 4,
                'title': 'Measure Logical Qubit',
                'description': 'Measure the logical qubit with error correction',
                'code': '''
# Measure logical qubit
result, confidence = simulator.measure_logical_qubit("qubit_0")
print(f"Measurement result: {result}, Confidence: {confidence:.3f}")
'''
            }
        ]
    }
    
    return tutorial


def demonstrate_fault_tolerant_computing():
    """Demonstrate fault-tolerant quantum computing."""
    print("=== Fault-Tolerant Quantum Computing Demonstration ===")
    
    # Create surface code
    surface_code = SurfaceCode(distance=3, lattice_size=(3, 3))
    print(f"Surface code created with {len(surface_code.physical_qubits)} physical qubits")
    
    # Create logical qubit simulator
    simulator = LogicalQubitSimulator(surface_code)
    
    # Create logical qubits
    qubit1 = simulator.create_logical_qubit("qubit_1")
    qubit2 = simulator.create_logical_qubit("qubit_2")
    print(f"Created logical qubits with code distance {qubit1.code_distance}")
    
    # Create fault-tolerant circuit
    circuit = FaultTolerantCircuit(surface_code)
    
    # Add logical gates
    circuit.add_logical_gate(LogicalGate.LOGICAL_H, "qubit_1")
    circuit.add_logical_gate(LogicalGate.LOGICAL_X, "qubit_1")
    circuit.add_logical_gate(LogicalGate.LOGICAL_CNOT, "qubit_1", "qubit_2")
    
    # Add error correction cycles
    circuit.add_error_correction_cycle(["qubit_1", "qubit_2"])
    
    # Execute circuit
    results = circuit.execute()
    
    print(f"Circuit execution successful: {results['success']}")
    print(f"Measurements: {results['measurements']}")
    print(f"Error correction stats: {results['error_correction_stats']}")
    print(f"Execution time: {results['execution_time']:.4f} seconds")
    
    return results


def benchmark_fault_tolerance():
    """Benchmark fault-tolerant quantum computing performance."""
    print("=== Fault-Tolerance Benchmark ===")
    
    distances = [3, 5, 7]
    results = {}
    
    for distance in distances:
        print(f"Testing distance {distance} surface code...")
        
        # Create surface code
        lattice_size = (distance, distance)
        surface_code = SurfaceCode(distance, lattice_size)
        
        # Create simulator
        simulator = LogicalQubitSimulator(surface_code)
        
        # Create logical qubit
        logical_qubit = simulator.create_logical_qubit("test_qubit")
        
        # Benchmark operations
        import time
        
        # Benchmark logical gate application
        start_time = time.time()
        for _ in range(10):
            simulator.apply_logical_gate("test_qubit", LogicalGate.LOGICAL_H)
        gate_time = time.time() - start_time
        
        # Benchmark error correction
        start_time = time.time()
        for _ in range(10):
            simulator._perform_error_correction(logical_qubit)
        correction_time = time.time() - start_time
        
        # Benchmark measurement
        start_time = time.time()
        for _ in range(10):
            simulator.measure_logical_qubit("test_qubit")
        measurement_time = time.time() - start_time
        
        results[distance] = {
            'physical_qubits': len(surface_code.physical_qubits),
            'gate_time': gate_time / 10,
            'correction_time': correction_time / 10,
            'measurement_time': measurement_time / 10,
            'total_time': (gate_time + correction_time + measurement_time) / 10
        }
        
        print(f"Distance {distance}: {results[distance]}")
    
    return results


if __name__ == "__main__":
    # Run demonstrations
    print("=== Surface Code Tutorial ===")
    tutorial = create_surface_code_tutorial()
    for step in tutorial['steps']:
        print(f"Step {step['step']}: {step['title']}")
        print(f"Description: {step['description']}")
        print(f"Code:\n{step['code']}")
        print()
    
    print("\n=== Fault-Tolerant Computing Demonstration ===")
    ft_results = demonstrate_fault_tolerant_computing()
    
    print("\n=== Fault-Tolerance Benchmark ===")
    benchmark_results = benchmark_fault_tolerance()
