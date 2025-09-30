"""
Compiler Passes

This module provides the compiler pass system.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .ir import CoratrixIR, IRBuilder, IROptimizer, IROperation


class PassType(Enum):
    """Types of compiler passes."""
    FRONTEND = "frontend"
    OPTIMIZATION = "optimization"
    BACKEND = "backend"


@dataclass
class PassResult:
    """Result of a compiler pass."""
    success: bool
    output: Any
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class CompilerPass(ABC):
    """Base class for compiler passes."""
    
    def __init__(self, name: str, pass_type: PassType):
        self.name = name
        self.pass_type = pass_type
    
    @abstractmethod
    def run(self, input_data: Any) -> PassResult:
        """Run the compiler pass."""
        pass


class PassManager:
    """Manager for compiler passes."""
    
    def __init__(self):
        self.passes: List[CompilerPass] = []
        self.pass_results: Dict[str, PassResult] = {}
    
    def add_pass(self, pass_obj: CompilerPass):
        """Add a pass to the manager."""
        self.passes.append(pass_obj)
    
    def run_passes(self, input_data: Any) -> PassResult:
        """Run all passes in sequence."""
        current_data = input_data
        
        for pass_obj in self.passes:
            result = pass_obj.run(current_data)
            self.pass_results[pass_obj.name] = result
            
            if not result.success:
                return result
            
            current_data = result.output
        
        return PassResult(success=True, output=current_data)


class PassRegistry:
    """Registry for compiler passes."""
    
    def __init__(self):
        self.passes: Dict[str, CompilerPass] = {}
    
    def register(self, name: str, pass_obj: CompilerPass):
        """Register a compiler pass."""
        self.passes[name] = pass_obj
    
    def get(self, name: str) -> Optional[CompilerPass]:
        """Get a compiler pass by name."""
        return self.passes.get(name)
