# Compiler Stack Guide

## Overview

The Coratrix 3.1 compiler stack provides a complete quantum circuit compilation pipeline from high-level domain-specific language (DSL) to target quantum frameworks. This guide covers the compiler architecture, components, and usage.

## Architecture Overview

```
DSL Source → DSL Parser → Coratrix IR → Compiler Passes → Target Generators → Target Code
```

### Components

1. **DSL Parser**: Parses quantum domain-specific language
2. **Coratrix IR**: Platform-agnostic intermediate representation
3. **Compiler Passes**: Optimization and transformation passes
4. **Target Generators**: Code generation for quantum frameworks
5. **Compiler**: Orchestrates the compilation pipeline

## Domain-Specific Language (DSL)

### Basic Syntax

```python
# Circuit definition
circuit bell_state() {
    h q0;
    cnot q0, q1;
    measure q0, q1;
}

# Parameterized circuit
circuit parameterized_circuit(theta: float) {
    ry(theta) q0;
    cnot q0, q1;
}

# Conditional gates
circuit conditional_circuit(condition: bool) {
    if (condition) {
        h q0;
    }
    cnot q0, q1;
}
```

### Gate Definitions

```python
# Standard gates
h q0;                    # Hadamard gate
x q0;                    # Pauli-X gate
y q0;                    # Pauli-Y gate
z q0;                    # Pauli-Z gate
cnot q0, q1;            # CNOT gate
cz q0, q1;              # Controlled-Z gate
ccx q0, q1, q2;         # Toffoli gate

# Parameterized gates
ry(theta) q0;           # Y-rotation
rz(phi) q0;             # Z-rotation
u3(theta, phi, lambda) q0;  # U3 gate

# Custom gates
gate custom_gate(alpha, beta) q0, q1 {
    ry(alpha) q0;
    cnot q0, q1;
    rz(beta) q1;
}
```

### Control Flow

```python
# Loops
circuit loop_circuit(n: int) {
    for (i in 0..n) {
        h q[i];
    }
}

# Conditional execution
circuit conditional_circuit(flag: bool) {
    if (flag) {
        h q0;
    } else {
        x q0;
    }
}
```

## Coratrix Intermediate Representation (IR)

### IR Structure

```python
from coratrix.compiler.ir import CoratrixIR, IRStatement, IROperation

# IR represents quantum circuits as structured data
ir = CoratrixIR()
ir.add_circuit("bell_state", 2)

# Add statements
statement = IRStatement(
    operation=IROperation.H,
    qubits=[0],
    parameters={}
)
ir.add_statement(statement)
```

### IR Operations

```python
from coratrix.compiler.ir import IROperation

# Single qubit operations
IROperation.H      # Hadamard
IROperation.X      # Pauli-X
IROperation.Y      # Pauli-Y
IROperation.Z      # Pauli-Z

# Multi-qubit operations
IROperation.CNOT   # CNOT gate
IROperation.CZ     # Controlled-Z
IROperation.CCX    # Toffoli gate

# Measurement
IROperation.MEASURE # Measurement

# Arithmetic operations
IROperation.ADD    # Addition
IROperation.SUB    # Subtraction
IROperation.MUL    # Multiplication
IROperation.DIV    # Division
```

### IR Optimization

```python
from coratrix.compiler.ir import IROptimizer

# Create optimizer
optimizer = IROptimizer()

# Apply optimizations
optimized_ir = optimizer.optimize(ir)

# Get optimization statistics
stats = optimizer.get_statistics()
print(f"Gates removed: {stats['gates_removed']}")
print(f"Optimization time: {stats['optimization_time']}")
```

## Compiler Passes

### Built-in Passes

```python
from coratrix.compiler.passes import PassManager, PassType

# Create pass manager
pass_manager = PassManager()

# Register built-in passes
pass_manager.register_pass("gate_merging")
pass_manager.register_pass("redundant_elimination")
pass_manager.register_pass("constant_folding")
```

### Custom Passes

```python
from coratrix.compiler.passes import CompilerPass, PassResult

class CustomOptimizationPass(CompilerPass):
    """Custom optimization pass."""
    
    def run(self, ir: CoratrixIR) -> PassResult:
        """Run the optimization pass."""
        optimized_ir = self._optimize_circuit(ir)
        return PassResult(success=True, ir=optimized_ir)
    
    def _optimize_circuit(self, ir: CoratrixIR):
        """Apply custom optimizations."""
        # Custom optimization logic
        return ir

# Register custom pass
pass_manager.register_pass(CustomOptimizationPass())
```

### Pass Execution

```python
# Run all registered passes
result = pass_manager.run_passes(ir)

if result.success:
    print("All passes completed successfully")
    optimized_ir = result.ir
else:
    print(f"Pass execution failed: {result.errors}")
```

## Target Generators

### OpenQASM Generator

```python
from coratrix.compiler.targets import TargetGenerator

# Generate OpenQASM 2.0
generator = TargetGenerator("openqasm")
result = generator.generate(ir, options)

if result.success:
    print("Generated OpenQASM:")
    print(result.code)
```

### Qiskit Generator

```python
# Generate Qiskit circuit
generator = TargetGenerator("qiskit")
result = generator.generate(ir, options)

if result.success:
    print("Generated Qiskit circuit:")
    print(result.code)
```

### PennyLane Generator

```python
# Generate PennyLane circuit
generator = TargetGenerator("pennylane")
result = generator.generate(ir, options)

if result.success:
    print("Generated PennyLane circuit:")
    print(result.code)
```

## Compiler Usage

### Basic Compilation

```python
from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode

# Create compiler
compiler = CoratrixCompiler()

# Define DSL source
dsl_source = """
circuit bell_state() {
    h q0;
    cnot q0, q1;
}
"""

# Compile options
options = CompilerOptions(
    mode=CompilerMode.COMPILE_ONLY,
    target_format='openqasm',
    optimize=True
)

# Compile
result = compiler.compile(dsl_source, options)

if result.success:
    print("Compilation successful")
    print(f"Target code: {result.target_code}")
else:
    print(f"Compilation failed: {result.errors}")
```

### Compilation with Execution

```python
# Compile and execute
options = CompilerOptions(
    mode=CompilerMode.COMPILE_AND_RUN,
    target_format='openqasm',
    backend_name='local_simulator',
    shots=1000
)

result = compiler.compile(dsl_source, options)

if result.success:
    print("Compilation and execution successful")
    print(f"Execution result: {result.execution_result}")
```

### Advanced Compilation

```python
# Custom compiler passes
options = CompilerOptions(
    mode=CompilerMode.COMPILE_ONLY,
    target_format='qiskit',
    optimize=True,
    passes=['gate_merging', 'redundant_elimination', 'custom_pass']
)

result = compiler.compile(dsl_source, options)
```

## Optimization Strategies

### Gate Merging

```python
# Merge adjacent gates
pass_manager.register_pass("gate_merging")

# Example: H H = I (identity)
# Before: h q0; h q0;
# After: (removed)
```

### Redundant Elimination

```python
# Remove redundant operations
pass_manager.register_pass("redundant_elimination")

# Example: CNOT CNOT = I
# Before: cnot q0, q1; cnot q0, q1;
# After: (removed)
```

### Constant Folding

```python
# Evaluate constant expressions
pass_manager.register_pass("constant_folding")

# Example: ry(0) = I
# Before: ry(0) q0;
# After: (removed)
```

## Error Handling

### Compilation Errors

```python
try:
    result = compiler.compile(invalid_dsl, options)
    if not result.success:
        print(f"Compilation errors: {result.errors}")
        print(f"Warnings: {result.warnings}")
except Exception as e:
    print(f"Compiler error: {e}")
```

### Pass Errors

```python
# Check pass results
for pass_name, pass_result in result.pass_results.items():
    if not pass_result.success:
        print(f"Pass {pass_name} failed: {pass_result.errors}")
```

## Performance Optimization

### Compilation Performance

```python
# Use parallel processing
options = CompilerOptions(
    parallel=True,
    num_workers=4
)

# Profile compilation
options = CompilerOptions(
    profile=True
)
```

### Memory Optimization

```python
# Use sparse representation for large circuits
options = CompilerOptions(
    sparse_threshold=8,
    memory_limit_gb=4
)
```

## Debugging

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Compile with debug info
result = compiler.compile(dsl_source, options)
print(f"Debug info: {result.metadata}")
```

### IR Inspection

```python
# Inspect IR structure
print(f"Circuits: {ir.circuits}")
print(f"Statements: {ir.statements}")
print(f"Variables: {ir.variables}")
```

### Pass Debugging

```python
# Debug specific pass
pass_manager.set_debug_mode(True)
result = pass_manager.run_passes(ir)
```

## Best Practices

### 1. DSL Design
- Use clear, descriptive circuit names
- Group related operations together
- Use meaningful variable names
- Comment complex circuits

### 2. Optimization
- Apply appropriate optimization passes
- Monitor optimization results
- Balance optimization vs. compilation time
- Test optimized circuits

### 3. Error Handling
- Validate input DSL
- Handle compilation errors gracefully
- Provide meaningful error messages
- Log compilation process

### 4. Performance
- Use appropriate target formats
- Optimize for target backends
- Monitor memory usage
- Profile compilation performance

## Examples

### Complete Compilation Example

```python
from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode

# DSL source
dsl_source = """
circuit grover_search(n: int) {
    // Initialize superposition
    for (i in 0..n) {
        h q[i];
    }
    
    // Grover iterations
    for (iter in 0..n) {
        // Oracle
        oracle q[0..n];
        
        // Diffusion operator
        for (i in 0..n) {
            h q[i];
        }
        x q[0..n];
        h q[n-1];
        ccx q[0..n-1], q[n-1];
        h q[n-1];
        x q[0..n];
        for (i in 0..n) {
            h q[i];
        }
    }
}
"""

# Compile options
options = CompilerOptions(
    mode=CompilerMode.COMPILE_ONLY,
    target_format='openqasm',
    optimize=True,
    passes=['gate_merging', 'redundant_elimination']
)

# Compile
compiler = CoratrixCompiler()
result = compiler.compile(dsl_source, options)

if result.success:
    print("Grover search circuit compiled successfully")
    print(f"Target code:\n{result.target_code}")
else:
    print(f"Compilation failed: {result.errors}")
```

### Custom Pass Example

```python
class GateCountPass(CompilerPass):
    """Pass to count gates in circuit."""
    
    def run(self, ir: CoratrixIR) -> PassResult:
        gate_count = 0
        for circuit in ir.circuits:
            gate_count += len(circuit.body.statements)
        
        # Add metadata
        ir.metadata['gate_count'] = gate_count
        
        return PassResult(success=True, ir=ir)

# Register and use
pass_manager.register_pass(GateCountPass())
result = pass_manager.run_passes(ir)
print(f"Gate count: {result.ir.metadata['gate_count']}")
```

## Troubleshooting

### Common Issues

1. **DSL Parsing Errors**
   - Check syntax and grammar
   - Validate gate definitions
   - Verify parameter types

2. **IR Generation Errors**
   - Check circuit structure
   - Validate qubit indices
   - Verify gate parameters

3. **Pass Execution Errors**
   - Check pass implementation
   - Validate IR structure
   - Handle edge cases

4. **Target Generation Errors**
   - Check target format support
   - Validate IR compatibility
   - Handle unsupported features

### Debug Tools

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode
options = CompilerOptions(debug=True)
result = compiler.compile(dsl_source, options)
```

## Contributing

### Adding New Passes

1. Implement `CompilerPass` interface
2. Add pass registration
3. Write comprehensive tests
4. Update documentation

### Adding New Targets

1. Implement `TargetGenerator` interface
2. Add target registration
3. Test with various circuits
4. Update documentation

### Guidelines

- Follow compiler architecture
- Maintain backward compatibility
- Write comprehensive tests
- Document all functionality
