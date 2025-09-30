"""
Modular backend interface for Coratrix.

This module provides a unified interface for connecting to various
quantum hardware backends and cloud simulators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class BackendType(Enum):
    """Types of quantum backends."""
    SIMULATOR = "simulator"
    HARDWARE = "hardware"
    CLOUD = "cloud"
    LOCAL = "local"


class BackendStatus(Enum):
    """Backend status states."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class BackendCapabilities:
    """Capabilities of a quantum backend."""
    max_qubits: int
    max_shots: int
    supports_noise: bool
    supports_parametric_circuits: bool
    supports_measurement: bool
    supports_conditional_operations: bool
    gate_set: List[str] = field(default_factory=list)
    noise_models: List[str] = field(default_factory=list)


@dataclass
class BackendResult:
    """Result from a backend execution."""
    success: bool
    counts: Dict[str, int] = field(default_factory=dict)
    statevector: Optional[List[complex]] = None
    execution_time: float = 0.0
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BackendConfiguration:
    """Configuration for a backend."""
    name: str
    backend_type: BackendType
    connection_params: Dict[str, Any] = field(default_factory=dict)
    capabilities: Optional[BackendCapabilities] = None
    timeout: float = 300.0  # 5 minutes default
    retry_attempts: int = 3
    retry_delay: float = 1.0


class BackendInterface(ABC):
    """Base interface for quantum backends."""
    
    def __init__(self, config: BackendConfiguration):
        self.config = config
        self.status = BackendStatus.OFFLINE
        self.connection = None
        self.last_used = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the backend."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the backend."""
        pass
    
    @abstractmethod
    def execute_circuit(self, circuit: Any, shots: int = 1024, 
                       parameters: Dict[str, Any] = None) -> BackendResult:
        """Execute a quantum circuit."""
        pass
    
    @abstractmethod
    def get_status(self) -> BackendStatus:
        """Get the current status of the backend."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Get the capabilities of the backend."""
        pass
    
    def is_available(self) -> bool:
        """Check if the backend is available."""
        return self.status == BackendStatus.AVAILABLE
    
    def update_last_used(self):
        """Update the last used timestamp."""
        self.last_used = time.time()


class SimulatorBackend(BackendInterface):
    """Backend for local quantum simulators."""
    
    def __init__(self, config: BackendConfiguration):
        super().__init__(config)
        self.simulator = None
    
    def connect(self) -> bool:
        """Connect to the simulator."""
        try:
            # Initialize simulator based on configuration
            simulator_type = self.config.connection_params.get('simulator_type', 'statevector')
            
            if simulator_type == 'statevector':
                from core.scalable_quantum_state import ScalableQuantumState
                self.simulator = ScalableQuantumState
            elif simulator_type == 'density_matrix':
                # Would use density matrix simulator
                pass
            elif simulator_type == 'stabilizer':
                # Would use stabilizer simulator
                pass
            
            self.status = BackendStatus.AVAILABLE
            return True
            
        except Exception as e:
            self.status = BackendStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from the simulator."""
        self.simulator = None
        self.status = BackendStatus.OFFLINE
        return True
    
    def execute_circuit(self, circuit: Any, shots: int = 1024, 
                       parameters: Dict[str, Any] = None) -> BackendResult:
        """Execute a circuit on the simulator."""
        start_time = time.time()
        
        try:
            if parameters is None:
                parameters = {}
            
            # Create quantum state
            num_qubits = getattr(circuit, 'num_qubits', 2)
            state = self.simulator(num_qubits, use_gpu=False, sparse_threshold=8)
            
            # Apply circuit operations
            self._apply_circuit_operations(state, circuit)
            
            # Simulate measurements
            counts = self._simulate_measurements(state, shots)
            
            execution_time = time.time() - start_time
            
            return BackendResult(
                success=True,
                counts=counts,
                statevector=[state.get_amplitude(i) for i in range(2**num_qubits)],
                execution_time=execution_time,
                metadata={'simulator_type': 'statevector'}
            )
            
        except Exception as e:
            return BackendResult(
                success=False,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _apply_circuit_operations(self, state, circuit):
        """Apply circuit operations to the state."""
        # This would apply the actual circuit operations
        # For now, it's a placeholder
        pass
    
    def _simulate_measurements(self, state, shots: int) -> Dict[str, int]:
        """Simulate measurements."""
        # Get probabilities
        probs = state.get_probabilities()
        
        # Sample from distribution
        import numpy as np
        outcomes = np.random.choice(len(probs), size=shots, p=probs)
        
        # Count outcomes
        counts = {}
        for outcome in outcomes:
            binary = format(outcome, '0{}b'.format(int(np.log2(len(probs)))))
            counts[binary] = counts.get(binary, 0) + 1
        
        return counts
    
    def get_status(self) -> BackendStatus:
        """Get the current status."""
        return self.status
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get simulator capabilities."""
        return BackendCapabilities(
            max_qubits=32,
            max_shots=1000000,
            supports_noise=True,
            supports_parametric_circuits=True,
            supports_measurement=True,
            supports_conditional_operations=True,
            gate_set=['h', 'x', 'y', 'z', 'cnot', 'cz', 'cphase', 'rx', 'ry', 'rz'],
            noise_models=['depolarizing', 'amplitude_damping', 'phase_damping']
        )


class QiskitBackend(BackendInterface):
    """Backend for Qiskit-based systems."""
    
    def __init__(self, config: BackendConfiguration):
        super().__init__(config)
        self.provider = None
        self.backend = None
    
    def connect(self) -> bool:
        """Connect to Qiskit backend."""
        try:
            from qiskit import IBMQ
            from qiskit_aer import AerSimulator
            
            # Check if using real hardware or simulator
            if self.config.backend_type == BackendType.HARDWARE:
                # Connect to IBM Quantum
                if 'token' in self.config.connection_params:
                    IBMQ.enable_account(self.config.connection_params['token'])
                    self.provider = IBMQ.get_provider()
                    backend_name = self.config.connection_params.get('backend_name', 'ibmq_qasm_simulator')
                    self.backend = self.provider.get_backend(backend_name)
                else:
                    raise ValueError("IBMQ token required for hardware backend")
            else:
                # Use local simulator
                self.backend = AerSimulator()
            
            self.status = BackendStatus.AVAILABLE
            return True
            
        except ImportError:
            self.status = BackendStatus.ERROR
            return False
        except Exception as e:
            self.status = BackendStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Qiskit backend."""
        self.provider = None
        self.backend = None
        self.status = BackendStatus.OFFLINE
        return True
    
    def execute_circuit(self, circuit: Any, shots: int = 1024, 
                       parameters: Dict[str, Any] = None) -> BackendResult:
        """Execute a circuit on Qiskit backend."""
        start_time = time.time()
        
        try:
            if parameters is None:
                parameters = {}
            
            # Convert circuit to Qiskit format if needed
            if not hasattr(circuit, 'measure_all'):
                # Assume it's a Coratrix circuit, convert to Qiskit
                qiskit_circuit = self._convert_to_qiskit(circuit)
            else:
                qiskit_circuit = circuit
            
            # Execute on backend
            job = self.backend.run(qiskit_circuit, shots=shots)
            result = job.result()
            
            execution_time = time.time() - start_time
            
            return BackendResult(
                success=True,
                counts=result.get_counts(),
                execution_time=execution_time,
                job_id=job.job_id() if hasattr(job, 'job_id') else None,
                metadata={'backend': str(self.backend)}
            )
            
        except Exception as e:
            return BackendResult(
                success=False,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _convert_to_qiskit(self, circuit) -> Any:
        """Convert Coratrix circuit to Qiskit format."""
        # This would convert the circuit format
        # For now, return a placeholder
        from qiskit import QuantumCircuit
        return QuantumCircuit(2)
    
    def get_status(self) -> BackendStatus:
        """Get the current status."""
        if self.backend is None:
            return BackendStatus.OFFLINE
        
        try:
            status = self.backend.status()
            if status.operational:
                return BackendStatus.AVAILABLE
            else:
                return BackendStatus.BUSY
        except:
            return BackendStatus.ERROR
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities."""
        if self.backend is None:
            return BackendCapabilities(0, 0, False, False, False, False)
        
        try:
            config = self.backend.configuration()
            return BackendCapabilities(
                max_qubits=config.n_qubits,
                max_shots=config.max_shots,
                supports_noise=hasattr(config, 'noise_model'),
                supports_parametric_circuits=True,
                supports_measurement=True,
                supports_conditional_operations=hasattr(config, 'conditional_operations'),
                gate_set=config.basis_gates if hasattr(config, 'basis_gates') else []
            )
        except:
            return BackendCapabilities(0, 0, False, False, False, False)


class BackendManager:
    """Manager for multiple quantum backends."""
    
    def __init__(self):
        self.backends: Dict[str, BackendInterface] = {}
        self.default_backend = None
    
    def register_backend(self, name: str, backend: BackendInterface) -> bool:
        """Register a backend."""
        try:
            self.backends[name] = backend
            if self.default_backend is None:
                self.default_backend = name
            return True
        except Exception:
            return False
    
    def unregister_backend(self, name: str) -> bool:
        """Unregister a backend."""
        if name in self.backends:
            backend = self.backends[name]
            backend.disconnect()
            del self.backends[name]
            
            if self.default_backend == name:
                self.default_backend = None
                if self.backends:
                    self.default_backend = next(iter(self.backends.keys()))
            
            return True
        return False
    
    def get_backend(self, name: str) -> Optional[BackendInterface]:
        """Get a backend by name."""
        return self.backends.get(name)
    
    def list_backends(self) -> List[str]:
        """List all registered backends."""
        return list(self.backends.keys())
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        available = []
        for name, backend in self.backends.items():
            if backend.is_available():
                available.append(name)
        return available
    
    def execute_on_backend(self, backend_name: str, circuit: Any, 
                          shots: int = 1024, parameters: Dict[str, Any] = None) -> BackendResult:
        """Execute a circuit on a specific backend."""
        backend = self.get_backend(backend_name)
        if backend is None:
            return BackendResult(
                success=False,
                errors=[f"Backend '{backend_name}' not found"]
            )
        
        if not backend.is_available():
            return BackendResult(
                success=False,
                errors=[f"Backend '{backend_name}' is not available"]
            )
        
        return backend.execute_circuit(circuit, shots, parameters)
    
    def execute_on_default(self, circuit: Any, shots: int = 1024, 
                          parameters: Dict[str, Any] = None) -> BackendResult:
        """Execute a circuit on the default backend."""
        if self.default_backend is None:
            return BackendResult(
                success=False,
                errors=["No default backend configured"]
            )
        
        return self.execute_on_backend(self.default_backend, circuit, shots, parameters)
    
    def set_default_backend(self, name: str) -> bool:
        """Set the default backend."""
        if name in self.backends:
            self.default_backend = name
            return True
        return False
    
    def get_backend_status(self, name: str) -> Optional[BackendStatus]:
        """Get the status of a specific backend."""
        backend = self.get_backend(name)
        if backend:
            return backend.get_status()
        return None
    
    def get_all_statuses(self) -> Dict[str, BackendStatus]:
        """Get the status of all backends."""
        statuses = {}
        for name, backend in self.backends.items():
            statuses[name] = backend.get_status()
        return statuses
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect to all backends."""
        results = {}
        for name, backend in self.backends.items():
            results[name] = backend.connect()
        return results
    
    def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect from all backends."""
        results = {}
        for name, backend in self.backends.items():
            results[name] = backend.disconnect()
        return results
    
    def __str__(self) -> str:
        return f"BackendManager with {len(self.backends)} backends"
    
    def __repr__(self) -> str:
        return f"BackendManager(backends={list(self.backends.keys())})"
