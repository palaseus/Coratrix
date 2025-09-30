# Error Handling Guide for Coratrix 3.1

This guide provides comprehensive examples of error handling in Coratrix 3.1, including common failure scenarios, expected error messages, and troubleshooting solutions.

## Table of Contents

- [Common Error Scenarios](#common-error-scenarios)
- [API Error Handling](#api-error-handling)
- [Backend Error Handling](#backend-error-handling)
- [Troubleshooting FAQ](#troubleshooting-faq)
- [Error Recovery Strategies](#error-recovery-strategies)

## Common Error Scenarios

### 1. Invalid Qubit Indices

#### Error: Qubit Index Out of Range

```python
from coratrix.core import ScalableQuantumState
from coratrix.core.quantum_circuit import HGate

# Create 3-qubit system
state = ScalableQuantumState(3)

# ❌ Error: Invalid qubit index
try:
    state.apply_gate(HGate(), [5])  # Only qubits 0, 1, 2 are valid
except IndexError as e:
    print(f"Error: {e}")
    # Output: Error: Qubit index 5 out of range for 3-qubit system
```

#### Error: Negative Qubit Index

```python
# ❌ Error: Negative qubit index
try:
    state.apply_gate(HGate(), [-1])
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Qubit index must be non-negative
```

#### Error: Duplicate Qubit Indices

```python
from coratrix.core.quantum_circuit import CNOTGate

# ❌ Error: Duplicate qubit indices
try:
    state.apply_gate(CNOTGate(), [0, 0])  # Control and target cannot be the same
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Control and target qubits must be different
```

### 2. Memory and Resource Errors

#### Error: Out of Memory

```python
# ❌ Error: Out of memory for large system
try:
    state = ScalableQuantumState(25, use_gpu=False, use_sparse=False)
except MemoryError as e:
    print(f"Error: {e}")
    # Output: Error: Insufficient memory for 25-qubit dense state (33.6 GB required)
    
    # ✅ Solution: Use sparse representation
    state = ScalableQuantumState(25, use_gpu=True, use_sparse=True)
    print("✅ Success: Using sparse representation with GPU acceleration")
```

#### Error: GPU Memory Exhausted

```python
# ❌ Error: GPU memory exhausted
try:
    state = ScalableQuantumState(30, use_gpu=True, use_sparse=False)
except RuntimeError as e:
    print(f"Error: {e}")
    # Output: Error: GPU memory exhausted (24 GB available, 32 GB required)
    
    # ✅ Solution: Use CPU or reduce system size
    state = ScalableQuantumState(25, use_gpu=False, use_sparse=True)
    print("✅ Success: Using CPU with sparse representation")
```

### 3. Backend Connection Errors

#### Error: Backend Not Available

```python
from coratrix.backend import BackendManager, BackendConfiguration, BackendType

# ❌ Error: Backend not found
backend_manager = BackendManager()
try:
    backend = backend_manager.get_backend("nonexistent_backend")
except KeyError as e:
    print(f"Error: {e}")
    # Output: Error: Backend 'nonexistent_backend' not found
    
    # ✅ Solution: List available backends
    backends = backend_manager.list_backends()
    print(f"Available backends: {backends}")
```

#### Error: Backend Connection Failed

```python
# ❌ Error: Backend connection failed
try:
    # Simulate connection failure
    backend_manager.connect_to_backend("remote_backend")
except ConnectionError as e:
    print(f"Error: {e}")
    # Output: Error: Failed to connect to remote backend (timeout: 30s)
    
    # ✅ Solution: Use local backend
    local_backend = backend_manager.get_backend("local_simulator")
    print("✅ Success: Using local simulator backend")
```

### 4. Compilation Errors

#### Error: Invalid DSL Syntax

```python
from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode

# ❌ Error: Invalid DSL syntax
compiler = CoratrixCompiler()
dsl_source = "circuit invalid_syntax { h q0; invalid_command q1; }"

try:
    result = compiler.compile(dsl_source, CompilerOptions())
except SyntaxError as e:
    print(f"Error: {e}")
    # Output: Error: Invalid DSL syntax at line 1: 'invalid_command' is not recognized
    
    # ✅ Solution: Use valid DSL syntax
    dsl_source = "circuit valid_circuit { h q0; cnot q0, q1; }"
    result = compiler.compile(dsl_source, CompilerOptions())
    print("✅ Success: Valid DSL compiled successfully")
```

#### Error: Target Format Not Supported

```python
# ❌ Error: Unsupported target format
try:
    options = CompilerOptions(target_format="unsupported_format")
    result = compiler.compile(dsl_source, options)
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Target format 'unsupported_format' not supported
    
    # ✅ Solution: Use supported format
    options = CompilerOptions(target_format="openqasm")
    result = compiler.compile(dsl_source, options)
    print("✅ Success: OpenQASM target format supported")
```

## API Error Handling

### 1. Quantum State Errors

```python
from coratrix.core import ScalableQuantumState

def safe_quantum_operation():
    """Demonstrate safe quantum operations with error handling."""
    
    try:
        # Create quantum state
        state = ScalableQuantumState(5, use_gpu=True)
        
        # Apply gates safely
        from coratrix.core.quantum_circuit import HGate, CNOTGate
        
        # Validate qubit indices before applying gates
        qubits = [0, 1, 2, 3, 4]
        for i in qubits:
            if i < state.num_qubits:
                state.apply_gate(HGate(), [i])
            else:
                raise IndexError(f"Qubit {i} out of range")
        
        # Apply CNOT gates safely
        for i in range(len(qubits) - 1):
            if qubits[i] < state.num_qubits and qubits[i+1] < state.num_qubits:
                state.apply_gate(CNOTGate(), [qubits[i], qubits[i+1]])
        
        return state
        
    except (IndexError, ValueError) as e:
        print(f"Quantum operation failed: {e}")
        return None
    except MemoryError as e:
        print(f"Memory error: {e}")
        # Fallback to smaller system
        return ScalableQuantumState(3, use_gpu=False, use_sparse=True)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 2. Circuit Execution Errors

```python
from coratrix.core import QuantumCircuit

def safe_circuit_execution():
    """Demonstrate safe circuit execution with error handling."""
    
    try:
        # Create circuit
        circuit = QuantumCircuit(3, "test_circuit")
        
        # Add gates with validation
        from coratrix.core.quantum_circuit import HGate, CNOTGate, XGate
        
        # Validate gate parameters
        gates = [
            (HGate(), [0]),
            (CNOTGate(), [0, 1]),
            (XGate(), [2])
        ]
        
        for gate, qubits in gates:
            # Validate qubit indices
            for qubit in qubits:
                if qubit >= circuit.num_qubits:
                    raise ValueError(f"Qubit {qubit} out of range for {circuit.num_qubits}-qubit circuit")
            
            circuit.add_gate(gate, qubits)
        
        # Execute circuit safely
        state = ScalableQuantumState(circuit.num_qubits)
        circuit.execute(state)
        
        return circuit, state
        
    except ValueError as e:
        print(f"Circuit validation failed: {e}")
        return None, None
    except Exception as e:
        print(f"Circuit execution failed: {e}")
        return None, None
```

## Backend Error Handling

### 1. Backend Connection Errors

```python
from coratrix.backend import BackendManager, BackendConfiguration, BackendType

def safe_backend_connection():
    """Demonstrate safe backend connection with error handling."""
    
    backend_manager = BackendManager()
    
    # Try different backends in order of preference
    backend_configs = [
        ("local_simulator", BackendType.SIMULATOR),
        ("gpu_simulator", BackendType.SIMULATOR),
        ("remote_backend", BackendType.HARDWARE)
    ]
    
    for backend_name, backend_type in backend_configs:
        try:
            config = BackendConfiguration(
                name=backend_name,
                backend_type=backend_type
            )
            
            # Test connection
            backend = backend_manager.get_backend(backend_name)
            if backend.is_available():
                print(f"✅ Successfully connected to {backend_name}")
                return backend
            else:
                print(f"⚠️  Backend {backend_name} not available")
                
        except KeyError:
            print(f"❌ Backend {backend_name} not found")
        except ConnectionError as e:
            print(f"❌ Connection to {backend_name} failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error with {backend_name}: {e}")
    
    # Fallback to local simulator
    print("🔄 Falling back to local simulator")
    return backend_manager.get_backend("local_simulator")
```

### 2. Backend Execution Errors

```python
def safe_backend_execution(backend, circuit, shots=1000):
    """Demonstrate safe backend execution with error handling."""
    
    try:
        # Validate backend
        if not backend.is_available():
            raise RuntimeError("Backend not available")
        
        # Check backend capabilities
        capabilities = backend.get_capabilities()
        if circuit.num_qubits > capabilities.max_qubits:
            raise ValueError(f"Circuit requires {circuit.num_qubits} qubits, but backend supports only {capabilities.max_qubits}")
        
        # Execute with timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Backend execution timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        try:
            result = backend.execute_circuit(circuit, shots=shots)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            signal.alarm(0)
            raise TimeoutError("Backend execution timed out")
        
    except ValueError as e:
        print(f"Backend validation failed: {e}")
        return None
    except TimeoutError as e:
        print(f"Backend execution timeout: {e}")
        return None
    except Exception as e:
        print(f"Backend execution failed: {e}")
        return None
```

## Troubleshooting FAQ

### Q: How do I handle "Out of Memory" errors?

**A:** Use these strategies:

```python
# 1. Use sparse representation
state = ScalableQuantumState(20, use_sparse=True, sparse_threshold=8)

# 2. Enable GPU acceleration
state = ScalableQuantumState(20, use_gpu=True, use_sparse=True)

# 3. Reduce system size
state = ScalableQuantumState(15, use_gpu=True, use_sparse=True)

# 4. Use memory mapping for very large systems
state = ScalableQuantumState(25, use_memory_mapping=True)
```

### Q: How do I handle backend connection failures?

**A:** Implement fallback strategies:

```python
def robust_backend_execution(circuit, shots=1000):
    """Robust backend execution with fallbacks."""
    
    backends = ["gpu_simulator", "local_simulator", "remote_backend"]
    
    for backend_name in backends:
        try:
            backend = backend_manager.get_backend(backend_name)
            if backend.is_available():
                result = backend.execute_circuit(circuit, shots=shots)
                return result
        except Exception as e:
            print(f"Backend {backend_name} failed: {e}")
            continue
    
    raise RuntimeError("All backends failed")
```

### Q: How do I handle invalid qubit indices?

**A:** Validate indices before use:

```python
def safe_gate_application(state, gate, qubits):
    """Safely apply gate to quantum state."""
    
    # Validate qubit indices
    for qubit in qubits:
        if not isinstance(qubit, int):
            raise TypeError(f"Qubit index must be integer, got {type(qubit)}")
        if qubit < 0:
            raise ValueError(f"Qubit index must be non-negative, got {qubit}")
        if qubit >= state.num_qubits:
            raise IndexError(f"Qubit index {qubit} out of range for {state.num_qubits}-qubit system")
    
    # Apply gate
    state.apply_gate(gate, qubits)
```

### Q: How do I handle DSL compilation errors?

**A:** Use validation and error recovery:

```python
def safe_dsl_compilation(dsl_source):
    """Safely compile DSL with error recovery."""
    
    try:
        # Validate DSL syntax
        if not dsl_source.strip():
            raise ValueError("Empty DSL source")
        
        # Compile with error handling
        compiler = CoratrixCompiler()
        result = compiler.compile(dsl_source, CompilerOptions())
        
        if not result.success:
            print(f"Compilation errors: {result.errors}")
            return None
        
        return result
        
    except SyntaxError as e:
        print(f"DSL syntax error: {e}")
        return None
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None
```

## Error Recovery Strategies

### 1. Graceful Degradation

```python
def robust_quantum_simulation(qubits, use_gpu=True, use_sparse=True):
    """Robust quantum simulation with graceful degradation."""
    
    # Try GPU first
    if use_gpu:
        try:
            return ScalableQuantumState(qubits, use_gpu=True, use_sparse=use_sparse)
        except RuntimeError:
            print("GPU not available, falling back to CPU")
            use_gpu = False
    
    # Try CPU with sparse representation
    if use_sparse:
        try:
            return ScalableQuantumState(qubits, use_gpu=False, use_sparse=True)
        except MemoryError:
            print("Sparse representation failed, reducing system size")
            qubits = min(qubits, 15)
    
    # Final fallback
    try:
        return ScalableQuantumState(qubits, use_gpu=False, use_sparse=False)
    except MemoryError:
        raise RuntimeError(f"System too large: {qubits} qubits not supported")
```

### 2. Retry Mechanisms

```python
import time
import random

def retry_backend_execution(backend, circuit, max_retries=3):
    """Retry backend execution with exponential backoff."""
    
    for attempt in range(max_retries):
        try:
            result = backend.execute_circuit(circuit, shots=1000)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    raise RuntimeError("All retry attempts failed")
```

### 3. Circuit Optimization for Error Recovery

```python
def optimize_circuit_for_execution(circuit):
    """Optimize circuit to reduce execution errors."""
    
    # Remove redundant gates
    optimized_circuit = circuit.optimize()
    
    # Check circuit depth
    if optimized_circuit.depth > 100:
        print("Warning: Circuit depth is high, may cause execution issues")
    
    # Validate gate count
    if len(optimized_circuit.gates) > 1000:
        print("Warning: Many gates, consider circuit optimization")
    
    return optimized_circuit
```

## Best Practices

### 1. Always Validate Inputs

```python
def validate_quantum_parameters(qubits, gates, shots):
    """Validate quantum computation parameters."""
    
    if not isinstance(qubits, int) or qubits <= 0:
        raise ValueError("Qubits must be positive integer")
    
    if qubits > 30:
        raise ValueError("System too large for reliable execution")
    
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("Shots must be positive integer")
    
    if shots > 1000000:
        raise ValueError("Too many shots, may cause memory issues")
```

### 2. Use Context Managers

```python
from contextlib import contextmanager

@contextmanager
def quantum_state_context(qubits, **kwargs):
    """Context manager for quantum state creation."""
    
    state = None
    try:
        state = ScalableQuantumState(qubits, **kwargs)
        yield state
    except Exception as e:
        print(f"Quantum state creation failed: {e}")
        raise
    finally:
        if state:
            # Cleanup if needed
            pass
```

### 3. Implement Comprehensive Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_quantum_operation(operation, *args, **kwargs):
    """Execute quantum operation with logging."""
    
    logger.info(f"Starting quantum operation: {operation}")
    try:
        result = operation(*args, **kwargs)
        logger.info(f"Quantum operation completed successfully")
        return result
    except Exception as e:
        logger.error(f"Quantum operation failed: {e}")
        raise
```

This comprehensive error handling guide provides examples for all common failure scenarios in Coratrix 3.1, helping users understand and resolve issues effectively.
