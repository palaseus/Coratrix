"""
Simulator Backend

This module provides local quantum simulator backends.
"""

from typing import Dict, Any, List
import numpy as np
import time

from .backend_interface import BackendInterface, BackendConfiguration, BackendType, BackendStatus, BackendCapabilities, BackendResult


class SimulatorBackend(BackendInterface):
    """Backend for local quantum simulators."""
    
    def __init__(self, config: BackendConfiguration):
        super().__init__(config)
        self.simulator = None
    
    def connect(self) -> bool:
        """Connect to the simulator."""
        try:
            simulator_type = self.config.connection_params.get('simulator_type', 'statevector')
            self.simulator = f"{simulator_type} simulator"
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
            
            # Simulate circuit execution
            num_qubits = getattr(circuit, 'num_qubits', 2)
            
            # Create random measurement results for demonstration
            np.random.seed(42)  # For reproducibility
            outcomes = np.random.choice(2**num_qubits, size=shots, p=[1.0/2**num_qubits]*2**num_qubits)
            
            # Count outcomes
            counts = {}
            for outcome in outcomes:
                binary = format(outcome, f'0{num_qubits}b')
                counts[binary] = counts.get(binary, 0) + 1
            
            execution_time = time.time() - start_time
            
            return BackendResult(
                success=True,
                counts=counts,
                execution_time=execution_time,
                metadata={
                    'backend': 'simulator',
                    'simulator': self.simulator,
                    'shots': shots
                }
            )
            
        except Exception as e:
            return BackendResult(
                success=False,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
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
