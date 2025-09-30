"""
Backend Interface

This module provides the base backend interface and management.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Types of quantum backends."""
    SIMULATOR = "simulator"
    HARDWARE = "hardware"
    CLOUD = "cloud"


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
    gate_set: List[str] = None
    noise_models: List[str] = None
    
    def __post_init__(self):
        if self.gate_set is None:
            self.gate_set = []
        if self.noise_models is None:
            self.noise_models = []


@dataclass
class BackendResult:
    """Result from a backend execution."""
    success: bool
    counts: Dict[str, int] = None
    statevector: Optional[List[complex]] = None
    execution_time: float = 0.0
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.counts is None:
            self.counts = {}
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class BackendConfiguration:
    """Configuration for a backend."""
    name: str
    backend_type: BackendType
    connection_params: Dict[str, Any] = None
    capabilities: Optional[BackendCapabilities] = None
    timeout: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if self.connection_params is None:
            self.connection_params = {}


class BackendInterface(ABC):
    """Base interface for quantum backends."""
    
    def __init__(self, config: BackendConfiguration):
        self.config = config
        self.status = BackendStatus.OFFLINE
        self.connection = None
    
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


class BackendManager:
    """Manager for quantum backends."""
    
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
    
    def get_backend_status(self, name: str) -> Optional[BackendStatus]:
        """Get the status of a specific backend."""
        backend = self.get_backend(name)
        if backend:
            return backend.get_status()
        return None
    
    def set_default_backend(self, name: str) -> bool:
        """Set the default backend."""
        if name in self.backends:
            self.default_backend = name
            return True
        return False
