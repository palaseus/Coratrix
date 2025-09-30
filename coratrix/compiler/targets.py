"""
Target Code Generators

This module provides target code generation capabilities.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TargetFormat(Enum):
    """Supported target formats."""
    OPENQASM = "openqasm"
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"


@dataclass
class TargetResult:
    """Result of target code generation."""
    success: bool
    code: str
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class TargetGenerator(ABC):
    """Base class for target code generators."""
    
    def __init__(self, target_format: TargetFormat):
        self.target_format = target_format
    
    @abstractmethod
    def generate(self, ir: 'CoratrixIR') -> TargetResult:
        """Generate target code from IR."""
        pass


class TargetRegistry:
    """Registry for target generators."""
    
    def __init__(self):
        self.generators: Dict[str, TargetGenerator] = {}
    
    def register(self, name: str, generator: TargetGenerator):
        """Register a target generator."""
        self.generators[name] = generator
    
    def get(self, name: str) -> Optional[TargetGenerator]:
        """Get a target generator by name."""
        return self.generators.get(name)
