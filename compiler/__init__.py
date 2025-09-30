"""
Coratrix 4.0 Quantum OS - Adaptive Compiler Pipeline
====================================================

The Adaptive Compiler Pipeline is the AI-driven brain of Coratrix 4.0's
quantum compilation system. It provides intelligent circuit optimization,
ML-based pattern recognition, and adaptive transpilation capabilities.

This compiler transforms quantum circuits from high-level descriptions
into optimized, backend-specific implementations through:

- AI-driven pattern recognition and optimization
- ML-based circuit analysis and transformation
- Adaptive transpilation with learning capabilities
- Multi-stage optimization pipeline
- Backend-specific code generation
- Performance prediction and optimization

This is the GOD-TIER compiler that makes Coratrix feel alive.
"""

from .adaptive_compiler import AdaptiveCompiler, CompilerConfig
from .ml_optimizer import MLOptimizer, OptimizationModel
from .pattern_recognizer import PatternRecognizer, CircuitPattern
from .transpiler import QuantumTranspiler, TranspilationStrategy
from .optimization_passes import OptimizationPass, PassPipeline
from .backend_generators import BackendGenerator, CodeGenerator

__all__ = [
    'AdaptiveCompiler',
    'CompilerConfig',
    'MLOptimizer',
    'OptimizationModel',
    'PatternRecognizer',
    'CircuitPattern',
    'QuantumTranspiler',
    'TranspilationStrategy',
    'OptimizationPass',
    'PassPipeline',
    'BackendGenerator',
    'CodeGenerator'
]