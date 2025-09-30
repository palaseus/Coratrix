"""
Hardware backend interface for Coratrix.

This module provides a pluggable backend interface for running quantum circuits
on different hardware platforms and simulators.
"""

import abc
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.circuit import QuantumCircuit
from core.measurement import Measurement


class BackendType(Enum):
    """Backend type enumeration."""
    SIMULATOR = "simulator"
    HARDWARE = "hardware"
    NOISY_SIMULATOR = "noisy_simulator"


@dataclass
class BackendResult:
    """Result from backend execution."""
    success: bool
    counts: Dict[str, int]
    execution_time: float
    backend_info: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class BackendCapabilities:
    """Backend capabilities information."""
    max_qubits: int
    supported_gates: List[str]
    supports_measurement: bool
    supports_noise: bool
    supports_parameterized_gates: bool
    max_circuit_depth: int
    execution_timeout: float


class QuantumBackend(abc.ABC):
    """Abstract base class for quantum backends."""
    
    def __init__(self, name: str, backend_type: BackendType):
        self.name = name
        self.backend_type = backend_type
        self.capabilities = self._get_capabilities()
    
    @abc.abstractmethod
    def _get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities."""
        pass
    
    @abc.abstractmethod
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> BackendResult:
        """Execute a quantum circuit on the backend."""
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass
    
    def validate_circuit(self, circuit: QuantumCircuit) -> Tuple[bool, List[str]]:
        """Validate that a circuit is compatible with this backend."""
        errors = []
        
        # Check qubit count
        if circuit.num_qubits > self.capabilities.max_qubits:
            errors.append(f"Circuit requires {circuit.num_qubits} qubits, "
                         f"backend supports {self.capabilities.max_qubits}")
        
        # Check circuit depth
        if len(circuit.gates) > self.capabilities.max_circuit_depth:
            errors.append(f"Circuit depth {len(circuit.gates)} exceeds "
                         f"backend limit {self.capabilities.max_circuit_depth}")
        
        # Check supported gates
        for gate, _ in circuit.gates:
            gate_name = type(gate).__name__.lower().replace('gate', '')
            if gate_name not in self.capabilities.supported_gates:
                errors.append(f"Gate {gate_name} not supported by backend")
        
        return len(errors) == 0, errors


class CoratrixSimulatorBackend(QuantumBackend):
    """Coratrix built-in simulator backend."""
    
    def __init__(self):
        super().__init__("coratrix_simulator", BackendType.SIMULATOR)
    
    def _get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=12,
            supported_gates=['x', 'y', 'z', 'h', 'cnot', 'toffoli', 'swap', 
                           'rx', 'ry', 'rz', 'cphase'],
            supports_measurement=True,
            supports_noise=False,
            supports_parameterized_gates=True,
            max_circuit_depth=1000,
            execution_timeout=30.0
        )
    
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> BackendResult:
        """Execute circuit using Coratrix simulator."""
        start_time = time.time()
        
        try:
            # Execute circuit
            circuit.execute()
            
            # Perform measurements
            measurement = Measurement(circuit.get_state())
            counts = measurement.measure_multiple(circuit.get_state(), shots)
            
            execution_time = time.time() - start_time
            
            return BackendResult(
                success=True,
                counts=counts,
                execution_time=execution_time,
                backend_info={
                    'backend_name': self.name,
                    'backend_type': self.backend_type.value,
                    'shots': shots,
                    'circuit_depth': len(circuit.gates)
                }
            )
        
        except Exception as e:
            return BackendResult(
                success=False,
                counts={},
                execution_time=time.time() - start_time,
                backend_info={'backend_name': self.name},
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Coratrix simulator is always available."""
        return True


class NoisySimulatorBackend(QuantumBackend):
    """Noisy simulator backend with configurable noise models."""
    
    def __init__(self, noise_model: Optional[Dict[str, Any]] = None):
        super().__init__("noisy_simulator", BackendType.NOISY_SIMULATOR)
        self.noise_model = noise_model or self._default_noise_model()
    
    def _default_noise_model(self) -> Dict[str, Any]:
        """Default noise model parameters."""
        return {
            'depolarizing_error': 0.01,
            'readout_error': 0.02,
            'gate_error': 0.005
        }
    
    def _get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=10,
            supported_gates=['x', 'y', 'z', 'h', 'cnot', 'toffoli', 'swap'],
            supports_measurement=True,
            supports_noise=True,
            supports_parameterized_gates=False,
            max_circuit_depth=500,
            execution_timeout=60.0
        )
    
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> BackendResult:
        """Execute circuit with noise simulation."""
        start_time = time.time()
        
        try:
            # Execute circuit with noise
            circuit.execute()
            
            # Apply noise model
            noisy_state = self._apply_noise(circuit.get_state())
            
            # Perform measurements with readout error
            measurement = Measurement(noisy_state)
            counts = measurement.measure_multiple(noisy_state, shots)
            
            # Apply readout error
            counts = self._apply_readout_error(counts)
            
            execution_time = time.time() - start_time
            
            return BackendResult(
                success=True,
                counts=counts,
                execution_time=execution_time,
                backend_info={
                    'backend_name': self.name,
                    'backend_type': self.backend_type.value,
                    'shots': shots,
                    'noise_model': self.noise_model,
                    'circuit_depth': len(circuit.gates)
                }
            )
        
        except Exception as e:
            return BackendResult(
                success=False,
                counts={},
                execution_time=time.time() - start_time,
                backend_info={'backend_name': self.name},
                error_message=str(e)
            )
    
    def _apply_noise(self, state):
        """Apply noise model to quantum state."""
        # Simplified noise application
        # In a real implementation, this would apply more sophisticated noise models
        return state
    
    def _apply_readout_error(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Apply readout error to measurement counts."""
        readout_error = self.noise_model['readout_error']
        noisy_counts = {}
        
        for bitstring, count in counts.items():
            # Apply readout error with some probability
            if np.random.random() < readout_error:
                # Flip a random bit
                bitlist = list(bitstring)
                if bitlist:
                    flip_index = np.random.randint(len(bitlist))
                    bitlist[flip_index] = '1' if bitlist[flip_index] == '0' else '0'
                    bitstring = ''.join(bitlist)
            
            noisy_counts[bitstring] = noisy_counts.get(bitstring, 0) + count
        
        return noisy_counts
    
    def is_available(self) -> bool:
        """Noisy simulator is always available."""
        return True


class IBMQStubBackend(QuantumBackend):
    """Stub backend for IBM Quantum (requires Qiskit)."""
    
    def __init__(self):
        super().__init__("ibmq_stub", BackendType.HARDWARE)
        self.qiskit_available = self._check_qiskit_availability()
    
    def _check_qiskit_availability(self) -> bool:
        """Check if Qiskit is available."""
        try:
            import qiskit
            return True
        except ImportError:
            return False
    
    def _get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=5,
            supported_gates=['x', 'y', 'z', 'h', 'cnot', 'rx', 'ry', 'rz'],
            supports_measurement=True,
            supports_noise=True,
            supports_parameterized_gates=True,
            max_circuit_depth=100,
            execution_timeout=300.0
        )
    
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> BackendResult:
        """Execute circuit using IBM Quantum (stub implementation)."""
        if not self.qiskit_available:
            return BackendResult(
                success=False,
                counts={},
                execution_time=0.0,
                backend_info={'backend_name': self.name},
                error_message="Qiskit not available"
            )
        
        start_time = time.time()
        
        try:
            # Convert Coratrix circuit to Qiskit circuit
            qiskit_circuit = self._convert_to_qiskit(circuit)
            
            # Execute on IBM Quantum (stub)
            # In a real implementation, this would use actual IBM Quantum API
            counts = self._simulate_ibmq_execution(qiskit_circuit, shots)
            
            execution_time = time.time() - start_time
            
            return BackendResult(
                success=True,
                counts=counts,
                execution_time=execution_time,
                backend_info={
                    'backend_name': self.name,
                    'backend_type': self.backend_type.value,
                    'shots': shots,
                    'provider': 'IBM Quantum'
                }
            )
        
        except Exception as e:
            return BackendResult(
                success=False,
                counts={},
                execution_time=time.time() - start_time,
                backend_info={'backend_name': self.name},
                error_message=str(e)
            )
    
    def _convert_to_qiskit(self, circuit: QuantumCircuit):
        """Convert Coratrix circuit to Qiskit circuit (stub)."""
        # This would be implemented with actual Qiskit conversion
        pass
    
    def _simulate_ibmq_execution(self, qiskit_circuit, shots: int) -> Dict[str, int]:
        """Simulate IBM Quantum execution (stub)."""
        # Return mock results
        return {'00': shots // 2, '11': shots // 2}
    
    def is_available(self) -> bool:
        """Check if IBM Quantum backend is available."""
        return self.qiskit_available


class BackendManager:
    """Manager for quantum backends."""
    
    def __init__(self):
        self.backends = {}
        self._register_default_backends()
    
    def _register_default_backends(self):
        """Register default backends."""
        # Register Coratrix simulator
        self.register_backend(CoratrixSimulatorBackend())
        
        # Register noisy simulator
        self.register_backend(NoisySimulatorBackend())
        
        # Register IBMQ stub if available
        ibmq_backend = IBMQStubBackend()
        if ibmq_backend.is_available():
            self.register_backend(ibmq_backend)
    
    def register_backend(self, backend: QuantumBackend):
        """Register a backend."""
        self.backends[backend.name] = backend
    
    def get_backend(self, name: str) -> Optional[QuantumBackend]:
        """Get a backend by name."""
        return self.backends.get(name)
    
    def list_backends(self) -> List[str]:
        """List available backends."""
        return [name for name, backend in self.backends.items() if backend.is_available()]
    
    def get_backend_capabilities(self, name: str) -> Optional[BackendCapabilities]:
        """Get backend capabilities."""
        backend = self.get_backend(name)
        return backend.capabilities if backend else None
    
    def execute_circuit(self, circuit: QuantumCircuit, backend_name: str, 
                       shots: int = 1024) -> BackendResult:
        """Execute a circuit on a specific backend."""
        backend = self.get_backend(backend_name)
        if not backend:
            return BackendResult(
                success=False,
                counts={},
                execution_time=0.0,
                backend_info={'backend_name': backend_name},
                error_message=f"Backend {backend_name} not found"
            )
        
        if not backend.is_available():
            return BackendResult(
                success=False,
                counts={},
                execution_time=0.0,
                backend_info={'backend_name': backend_name},
                error_message=f"Backend {backend_name} not available"
            )
        
        # Validate circuit
        is_valid, errors = backend.validate_circuit(circuit)
        if not is_valid:
            return BackendResult(
                success=False,
                counts={},
                execution_time=0.0,
                backend_info={'backend_name': backend_name},
                error_message=f"Circuit validation failed: {', '.join(errors)}"
            )
        
        # Execute circuit
        return backend.execute_circuit(circuit, shots)
