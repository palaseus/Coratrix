"""
Coratrix 4.0 Quantum OS - Quantum Shader DSL
===========================================

The Quantum Shader DSL extends Coratrix 4.0 with reusable, parameterized
quantum shaders that integrate seamlessly with the adaptive compiler.

This is the GOD-TIER quantum shader system that enables reusable
quantum components and community-driven quantum libraries.

Key Features:
- Reusable quantum shaders
- Parameterized quantum components
- Adaptive compiler integration
- Community library support
- Quantum shader marketplace
- Performance optimization
- Cross-platform compatibility
"""

from .quantum_shader_dsl import QuantumShaderDSL, ShaderType, ShaderConfig
from .shader_compiler import ShaderCompiler, ShaderOptimizer, ShaderValidator
from .shader_library import ShaderLibrary, ShaderRegistry, ShaderMarketplace
from .shader_runtime import ShaderRuntime, ShaderExecutor, ShaderCache
from .shader_analytics import ShaderAnalytics, ShaderProfiler, ShaderMetrics

__all__ = [
    'QuantumShaderDSL',
    'ShaderType',
    'ShaderConfig',
    'ShaderCompiler',
    'ShaderOptimizer',
    'ShaderValidator',
    'ShaderLibrary',
    'ShaderRegistry',
    'ShaderMarketplace',
    'ShaderRuntime',
    'ShaderExecutor',
    'ShaderCache',
    'ShaderAnalytics',
    'ShaderProfiler',
    'ShaderMetrics'
]
