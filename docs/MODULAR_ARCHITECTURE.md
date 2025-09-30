# Coratrix 3.1: Modular Architecture Guide

## Overview

Coratrix 3.1 introduces a completely modular architecture with clear boundaries between simulation core, compiler stack, and backend management. This guide explains the architectural principles, component interactions, and how to extend the system.

## Architecture Principles

### 1. Clear Separation of Concerns
- **Core Simulation**: Quantum state representation and gate operations
- **Compiler Stack**: DSL parsing, IR generation, and target code generation
- **Backend Management**: Unified interface for quantum backends
- **Plugin System**: Extensible interfaces for custom components

### 2. Modular Design
- Each layer operates independently
- Well-defined interfaces between layers
- Pluggable components for extensibility
- Comprehensive testing at each layer

### 3. Production Readiness
- Robust error handling
- Comprehensive logging
- Performance monitoring
- Scalable architecture

## Core Simulation Layer

### ScalableQuantumState
The foundation of quantum simulation with multiple representations:

```python
from coratrix.core import ScalableQuantumState

# Dense representation
state = ScalableQuantumState(3, use_gpu=False, sparse_threshold=8)

# Sparse representation
state = ScalableQuantumState(3, use_sparse=True, sparse_threshold=8)

# GPU acceleration
state = ScalableQuantumState(3, use_gpu=True)
```

### Quantum Gates
Standard quantum gates with optimized implementations:

```python
from coratrix.core.quantum_circuit import HGate, CNOTGate, XGate

# Single qubit gates
h_gate = HGate()
x_gate = XGate()

# Multi-qubit gates
cnot_gate = CNOTGate()

# Apply gates to state
state.apply_gate(h_gate, [0])
state.apply_gate(cnot_gate, [0, 1])
```

### Quantum Circuits
Circuit construction and execution:

```python
from coratrix.core import QuantumCircuit

circuit = QuantumCircuit(3, "bell_state")
circuit.add_gate(HGate(), [0])
circuit.add_gate(CNOTGate(), [0, 1])

# Execute circuit
initial_state = ScalableQuantumState(3)
circuit.execute(initial_state)
```

## Compiler Stack Layer

### Domain-Specific Language (DSL)
High-level quantum programming language:

```python
dsl_source = """
circuit bell_state() {
    h q0;
    cnot q0, q1;
    measure q0, q1;
}
"""
```

### Coratrix Intermediate Representation (IR)
Platform-agnostic quantum circuit representation:

```python
from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode

compiler = CoratrixCompiler()
options = CompilerOptions(
    mode=CompilerMode.COMPILE_ONLY,
    target_format='openqasm',
    optimize=True
)

result = compiler.compile(dsl_source, options)
print(result.target_code)
```

### Compiler Passes
Modular optimization and transformation passes:

```python
from coratrix.compiler.passes import PassManager, CompilerPass

class CustomOptimizationPass(CompilerPass):
    def run(self, ir):
        # Custom optimization logic
        return PassResult(success=True, ir=optimized_ir)

# Register custom pass
pass_manager = PassManager()
pass_manager.register_pass(CustomOptimizationPass())
```

### Target Code Generation
Generate code for multiple quantum frameworks:

```python
# OpenQASM 2.0
options = CompilerOptions(target_format='openqasm')
result = compiler.compile(dsl_source, options)

# Qiskit
options = CompilerOptions(target_format='qiskit')
result = compiler.compile(dsl_source, options)

# PennyLane
options = CompilerOptions(target_format='pennylane')
result = compiler.compile(dsl_source, options)
```

## Backend Management Layer

### Backend Interface
Unified interface for quantum backends:

```python
from coratrix.backend import BackendManager, BackendConfiguration, BackendType

# Configure backend
config = BackendConfiguration(
    name='local_simulator',
    backend_type=BackendType.SIMULATOR,
    connection_params={'shots': 1000}
)

# Register backend
backend_manager = BackendManager()
backend_manager.register_backend('local_simulator', backend)
```

### Simulator Backends
Local quantum simulators:

```python
from coratrix.backend import SimulatorBackend

# Statevector simulator
simulator = SimulatorBackend(config)
result = simulator.execute(circuit, shots=1000)
```

### Hardware Backends
Integration with real quantum hardware:

```python
# IBM Quantum
from coratrix.backend import IBMBackend

ibm_backend = IBMBackend(config)
result = ibm_backend.execute(circuit, shots=1000)
```

## Plugin System

### Plugin Architecture
Extensible plugin system for custom components:

```python
from coratrix.plugins import PluginManager, PluginInfo

plugin_manager = PluginManager()

# Register plugin
plugin = CustomPlugin()
plugin_manager.register_plugin(plugin)
```

### Compiler Pass Plugins
Custom optimization passes:

```python
from coratrix.plugins import CompilerPassPlugin

class CustomOptimizationPlugin(CompilerPassPlugin):
    def get_pass(self):
        return CustomOptimizationPass()
    
    def get_info(self):
        return PluginInfo(
            name='custom_optimization',
            version='1.0.0',
            plugin_type='compiler_pass'
        )
```

### Backend Plugins
Custom quantum backends:

```python
from coratrix.plugins import BackendPlugin

class CustomBackendPlugin(BackendPlugin):
    def get_backend(self):
        return CustomBackend()
    
    def get_info(self):
        return PluginInfo(
            name='custom_backend',
            version='1.0.0',
            plugin_type='backend'
        )
```

### DSL Extension Plugins
Custom language extensions:

```python
from coratrix.plugins import DSLExtensionPlugin

class CustomGatePlugin(DSLExtensionPlugin):
    def extend_dsl(self, parser):
        # Add custom gate syntax
        parser.add_gate('custom_gate', CustomGateHandler())
```

## CLI Integration

### Coratrix Compiler CLI
Command-line interface for compilation and execution:

```bash
# Compile DSL to OpenQASM
coratrixc input.qasm -o output.qasm --target openqasm

# Compile to Qiskit format
coratrixc input.qasm -o output.py --target qiskit

# Execute circuit on backend
coratrixc input.qasm --execute --backend local_simulator --shots 1000

# List available backends
coratrixc --list-backends

# List available plugins
coratrixc --list-plugins
```

### Interactive CLI
Interactive quantum shell:

```bash
# Start interactive shell
coratrixc --interactive

# In the shell:
>>> state = ScalableQuantumState(2)
>>> state.apply_gate(HGate(), [0])
>>> print(state.get_amplitude(0))
```

## Best Practices

### 1. Layer Independence
- Keep layers loosely coupled
- Use well-defined interfaces
- Avoid direct dependencies between layers

### 2. Plugin Development
- Follow plugin interface contracts
- Implement proper error handling
- Provide comprehensive documentation

### 3. Testing
- Test each layer independently
- Use integration tests for layer interactions
- Maintain comprehensive test coverage

### 4. Performance
- Use appropriate state representations
- Optimize for target hardware
- Monitor memory usage and performance

## Migration Guide

### From Coratrix 3.0
1. Update imports to use new modular structure
2. Replace direct state manipulation with circuit-based approach
3. Use new compiler stack for quantum programs
4. Migrate to new backend interface

### Example Migration
```python
# Old (3.0)
from coratrix import QuantumState
state = QuantumState(3)
state.apply_hadamard(0)

# New (3.1)
from coratrix.core import ScalableQuantumState
from coratrix.core.quantum_circuit import HGate
state = ScalableQuantumState(3)
state.apply_gate(HGate(), [0])
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Use absolute imports for plugins
2. **Plugin Loading**: Check plugin registration and initialization
3. **Backend Connection**: Verify backend configuration and connectivity
4. **Performance**: Monitor memory usage and optimization settings

### Debug Mode
Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Roadmap

### Planned Features
- Additional target formats (Cirq, Braket)
- Advanced optimization passes
- Cloud backend integration
- Visual circuit representation
- Performance profiling tools

### Contributing
- Follow architectural principles
- Maintain test coverage
- Document new features
- Submit comprehensive PRs
