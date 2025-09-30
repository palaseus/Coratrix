"""
Coratrix 3.1: Modular Quantum Computing SDK

A production-ready, high-performance quantum computing simulation and research platform
with modular architecture, plugin system, and comprehensive developer tools.

Architecture:
- Simulation Core: Quantum state simulation and algorithms
- Compiler Stack: DSL → IR → Passes → Targets
- Backend Manager: Unified interface for execution backends
- Plugin System: Extensible interfaces for custom components
"""

__version__ = "3.1.0"
__author__ = "Coratrix Development Team"
__email__ = "dev@coratrix.org"

# Core simulation modules
from .core import (
    ScalableQuantumState,
    QuantumCircuit,
    QuantumGate,
    QuantumAlgorithm
)

# Compiler stack
from .compiler import (
    DSLParser,
    CoratrixIR,
    CompilerPass,
    TargetGenerator,
    CoratrixCompiler
)

# Backend management
from .backend import (
    BackendInterface,
    BackendManager,
    BackendConfiguration
)

# Plugin system
from .plugins import (
    PluginManager,
    CompilerPassPlugin,
    BackendPlugin,
    DSLExtensionPlugin
)

# CLI interface
from .cli import CoratrixCLI

__all__ = [
    # Core
    'ScalableQuantumState', 'QuantumCircuit', 'QuantumGate', 'QuantumAlgorithm',
    
    # Compiler
    'DSLParser', 'CoratrixIR', 'CompilerPass', 'TargetGenerator', 'CoratrixCompiler',
    
    # Backend
    'BackendInterface', 'BackendManager', 'BackendConfiguration',
    
    # Plugins
    'PluginManager', 'CompilerPassPlugin', 'BackendPlugin', 'DSLExtensionPlugin',
    
    # CLI
    'CoratrixCLI'
]
