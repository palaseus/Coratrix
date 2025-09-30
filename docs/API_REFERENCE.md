# Coratrix 3.1 API Reference

## Overview

Coratrix 3.1 provides a comprehensive Python API for quantum computing simulation and research with full test suite harmonization and API stabilization. This document covers all major components of the API with detailed examples and usage patterns.

## What's New in 3.1

- **100% Test Pass Rate**: All 199 tests now pass consistently
- **API Stabilization**: Fixed all import/constructor/method mismatches
- **Enhanced Methods**: Added missing methods like `get_entanglement_entropy()`, `get_density_matrix()`, `measure_multiple()`, `apply_gate()`
- **Improved Error Handling**: Better error messages and validation throughout
- **Backward Compatibility**: Full compatibility with existing APIs maintained

## Table of Contents

1. [Core Quantum Computing](#core-quantum-computing)
2. [Scalable Quantum States](#scalable-quantum-states)
3. [Quantum Gates](#quantum-gates)
4. [Hardware Interfaces](#hardware-interfaces)
5. [Advanced Algorithms](#advanced-algorithms)
6. [Optimization Engine](#optimization-engine)
7. [Multi-Subspace Grover Search](#multi-subspace-grover-search)
8. [Noise Models and Error Mitigation](#noise-models-and-error-mitigation)
9. [Reproducibility and Security](#reproducibility-and-security)
10. [Convenience Functions](#convenience-functions)

## Core Quantum Computing

### QuantumState

The basic quantum state representation for small systems (≤8 qubits).

```python
from coratrix import QuantumState

# Create a 2-qubit quantum state
state = QuantumState(2)

# Set amplitudes
state.set_amplitude(0, 1.0/np.sqrt(2))  # |00⟩
state.set_amplitude(3, 1.0/np.sqrt(2))  # |11⟩

# Get probabilities
probabilities = state.get_probabilities()
print(probabilities)  # [0.5, 0.0, 0.0, 0.5]

# Calculate entanglement entropy (NEW in 3.1)
entropy = state.get_entanglement_entropy()
print(f"Entanglement entropy: {entropy}")

# Get density matrix (NEW in 3.1)
density_matrix = state.get_density_matrix()
print(f"Density matrix shape: {density_matrix.shape}")
```

### QuantumCircuit

Quantum circuit for applying gates to quantum states.

```python
from coratrix import QuantumCircuit, HGate, CNOTGate

# Create a 2-qubit circuit
circuit = QuantumCircuit(2)

# Add gates
circuit.add_gate(HGate(), [0])
circuit.add_gate(CNOTGate(), [0, 1])

# Execute the circuit
circuit.execute()

# Get the final state
final_state = circuit.get_state()
print(final_state)
```

### QuantumMeasurement

Measurement operations for quantum states.

```python
from coratrix import QuantumMeasurement

# Create measurement object
measurement = QuantumMeasurement(2)

# Measure a quantum state
state = QuantumState(2)
# ... prepare state ...

# Single measurement
result = measurement.measure(state)
print(f"Measurement result: {result}")

# Multiple measurements
counts = measurement.measure_multiple(state, shots=1000)
print(f"Measurement counts: {counts}")
```

## Scalable Quantum States

### ScalableQuantumState

High-performance quantum state representation for large systems (2-12 qubits).

```python
from coratrix import ScalableQuantumState

# Create scalable state with GPU acceleration
state = ScalableQuantumState(
    num_qubits=8,
    use_gpu=True,
    sparse_threshold=8,
    sparse_format='csr',
    deterministic=True
)

# Apply gates (NEW: apply_gate method in 3.1)
from coratrix import HGate, CNOTGate
h_gate = HGate()
cnot_gate = CNOTGate()

# NEW: apply_gate method for better integration
state.apply_gate(h_gate, [0])
state.apply_gate(cnot_gate, [0, 1])

# Traditional method still works
h_gate.apply(state, [0])
cnot_gate.apply(state, [0, 1])

# Get performance metrics
metrics = state.get_performance_metrics()
print(f"Memory usage: {metrics['memory_usage']['memory_mb']:.2f} MB")
print(f"GPU utilization: {metrics.get('gpu_memory_utilization', 0):.2%}")

# Optimize memory usage
state.optimize_memory()

# Clear GPU cache
state.clear_gpu_cache()
```

## Quantum Gates

### Basic Gates

```python
from coratrix import XGate, YGate, ZGate, HGate, CNOTGate

# Single-qubit gates
x_gate = XGate()
y_gate = YGate()
z_gate = ZGate()
h_gate = HGate()

# Two-qubit gates
cnot_gate = CNOTGate()

# Apply gates to state
state = QuantumState(2)
h_gate.apply(state, [0])
cnot_gate.apply(state, [0, 1])
```

### Parameterized Gates

```python
from coratrix import RxGate, RyGate, RzGate, CPhaseGate

# Rotation gates
rx_gate = RxGate(np.pi/2)  # 90-degree X rotation
ry_gate = RyGate(np.pi/4)  # 45-degree Y rotation
rz_gate = RzGate(np.pi/6)  # 30-degree Z rotation

# Controlled phase gate
cphase_gate = CPhaseGate(np.pi/3)  # 60-degree phase

# Apply to state
state = QuantumState(2)
rx_gate.apply(state, [0])
ry_gate.apply(state, [1])
cphase_gate.apply(state, [0, 1])
```

## Hardware Interfaces

### OpenQASM Interface

```python
from coratrix import OpenQASMInterface

# Create OpenQASM interface
qasm_interface = OpenQASMInterface()

# Import circuit from OpenQASM file
circuit = qasm_interface.import_circuit("bell_state.qasm")

# Export circuit to OpenQASM
qasm_interface.export_circuit(circuit, "output.qasm")

# Validate OpenQASM syntax
qasm_string = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
is_valid, errors = qasm_interface.validate_qasm(qasm_string)
```

### Backend Interface

```python
from coratrix import BackendManager, CoratrixSimulatorBackend

# Create backend manager
backend_manager = BackendManager()

# List available backends
backends = backend_manager.list_backends()
print(f"Available backends: {backends}")

# Execute circuit on backend
circuit = QuantumCircuit(2)
# ... add gates ...

result = backend_manager.execute_circuit(circuit, "coratrix_simulator", shots=1000)
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Results: {result.counts}")
```

## Advanced Algorithms

### Quantum State Tomography

```python
from coratrix import QuantumStateTomography

# Create tomography object
tomography = QuantumStateTomography(num_qubits=2)

# Perform tomography on a state
state = QuantumState(2)
# ... prepare state ...

result = tomography.perform_tomography(state, shots_per_measurement=1000)
print(f"Fidelity: {result.fidelity:.4f}")
print(f"Purity: {result.purity:.4f}")
print(f"Success: {result.success}")
```

### Fidelity Estimation

```python
from coratrix import FidelityEstimator

# Create fidelity estimator
estimator = FidelityEstimator()

# Estimate fidelity between states
state1 = QuantumState(2)
state2 = QuantumState(2)
# ... prepare states ...

fidelity = estimator.estimate_state_fidelity(state1, state2)
print(f"State fidelity: {fidelity:.4f}")

# Estimate gate fidelity
ideal_gate = np.array([[1, 0], [0, 1]], dtype=complex)
noisy_gate = np.array([[0.99, 0.01], [0.01, 0.99]], dtype=complex)
gate_fidelity = estimator.estimate_gate_fidelity(ideal_gate, noisy_gate)
print(f"Gate fidelity: {gate_fidelity:.4f}")
```

### Entanglement Analysis

```python
from coratrix import EntanglementMonotones, EntanglementNetwork

# Create entanglement analysis tools
monotones = EntanglementMonotones()
network = EntanglementNetwork(num_qubits=4)

# Calculate entanglement measures
state = QuantumState(4)
# ... prepare state ...

# Negativity
negativity = monotones.calculate_negativity(state, (0, 1))
print(f"Negativity: {negativity:.4f}")

# Concurrence (for 2-qubit states)
if state.num_qubits == 2:
    concurrence = monotones.calculate_concurrence(state)
    print(f"Concurrence: {concurrence:.4f}")

# Entanglement network
network_data = network.calculate_entanglement_graph(state)
print(f"Total entanglement: {network_data['total_entanglement']:.4f}")

# Export to GraphML
network.export_to_graphml(network_data, "entanglement_network.graphml")
```

## Optimization Engine

### Parameterized Circuits

```python
from coratrix import ParameterizedCircuit, OptimizationEngine, OptimizationConfig, OptimizationMethod

# Create parameterized circuit
parameterized_gates = [
    ("rx", [0], "theta_0"),
    ("ry", [1], "theta_1"),
    ("rz", [0], "theta_2")
]
circuit = ParameterizedCircuit(num_qubits=2, parameterized_gates=parameterized_gates)

# Set parameters
parameters = np.array([np.pi/2, np.pi/4, np.pi/6])
circuit.set_parameters(parameters)

# Execute circuit
state = circuit.execute()
```

### Optimization

```python
# Define objective function
def objective(circuit):
    state = circuit.execute()
    # Minimize |⟨0|ψ⟩|²
    return abs(state.get_amplitude(0))**2

# Create optimization configuration
config = OptimizationConfig(
    method=OptimizationMethod.SPSA,
    max_iterations=100,
    learning_rate=0.1,
    save_traces=True,
    output_dir="optimization_traces"
)

# Create optimization engine
engine = OptimizationEngine(config)

# Run optimization
result = engine.optimize(circuit, objective)

print(f"Success: {result.success}")
print(f"Optimal parameters: {result.optimal_parameters}")
print(f"Optimal value: {result.optimal_value:.6f}")
print(f"Iterations: {result.iterations}")
print(f"Execution time: {result.execution_time:.3f}s")
```

### Noise-Aware Optimization

```python
from coratrix import NoiseAwareOptimization, NoiseModel

# Create noise model
noise_model = NoiseModel(
    depolarizing_error=0.01,
    amplitude_damping_error=0.005,
    readout_error=0.02
)

# Create noise-aware optimizer
noise_optimizer = NoiseAwareOptimization(noise_model)

# Optimize with noise
result = noise_optimizer.optimize_with_noise(circuit, objective, config)
```

## Multi-Subspace Grover Search

### Basic Usage

```python
from coratrix import MultiSubspaceGrover, SubspaceConfig, SubspaceType

# Define subspaces
subspaces = [
    SubspaceConfig(SubspaceType.GHZ, [0, 1], entanglement_threshold=0.7),
    SubspaceConfig(SubspaceType.W, [2, 3], entanglement_threshold=0.6),
    SubspaceConfig(SubspaceType.CLUSTER, [1, 2], entanglement_threshold=0.8)
]

# Create multi-subspace Grover
grover = MultiSubspaceGrover(num_qubits=4, subspaces=subspaces)

# Perform search
target_items = ["0000", "1111"]
result = grover.search(target_items, max_iterations=100, shots=1024, enable_interference=True)

print(f"Success: {result.success}")
print(f"Success probability: {result.success_probability:.4f}")
print(f"Iterations: {result.iterations}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Measurement counts: {result.counts}")
```

### Interference Diagnostics

```python
# Generate interference visualizations
grover.generate_interference_visualization(result, output_dir="grover_visualizations")

# Export results
grover.export_results(result, "grover_results.json")

# Access interference metrics
interference_metrics = result.interference_metrics
print(f"Total interference: {interference_metrics['final_interference']:.4f}")
print(f"Interference evolution: {interference_metrics['interference_evolution']}")
print(f"Entanglement evolution: {interference_metrics['entanglement_evolution']}")
print(f"Success probability evolution: {interference_metrics['success_probability_evolution']}")
```

## Noise Models and Error Mitigation

### Noise Models

```python
from coratrix import NoiseModel, QuantumNoise, NoisyQuantumCircuit

# Create noise model
noise_model = NoiseModel(
    depolarizing_error=0.01,
    amplitude_damping_error=0.005,
    phase_damping_error=0.005,
    readout_error=0.02,
    gate_error=0.005
)

# Create noisy quantum circuit
noisy_circuit = NoisyQuantumCircuit(num_qubits=2, noise_model=noise_model)

# Add gates with noise
from coratrix import HGate, CNOTGate
noisy_circuit.add_gate(HGate(), [0])
noisy_circuit.add_gate(CNOTGate(), [0, 1])

# Execute with noise
noisy_circuit.execute_with_mitigation(mitigation_enabled=True)

# Measure with readout error
counts = noisy_circuit.measure_with_readout_error(shots=1000)
print(f"Noisy measurement counts: {counts}")
```

### Error Mitigation

```python
from coratrix import ErrorMitigation

# Create error mitigation
mitigation = ErrorMitigation(noise_model)

# Apply mid-circuit purification
state = QuantumState(2)
purified_state = mitigation.apply_mid_circuit_purification(state, purification_threshold=0.8)

# Apply real-time feedback
corrected_state = mitigation.apply_real_time_feedback(state, target_fidelity=0.95)

# Apply error correction codes
corrected_state = mitigation.apply_error_correction_code(state, code_type="repetition")
```

## Reproducibility and Security

### Reproducibility Manager

```python
from coratrix import ReproducibilityManager

# Create reproducibility manager
manager = ReproducibilityManager(session_id="my_session", random_seed=42)

# Create experiment
experiment_id = manager.create_experiment(
    experiment_type="quantum_simulation",
    parameters={"num_qubits": 4, "shots": 1000}
)

# Update experiment with results
manager.update_experiment(
    experiment_id,
    execution_time=2.5,
    success=True,
    output_files=["results.json", "plots.png"]
)

# Verify reproducibility
is_reproducible, message = manager.verify_reproducibility(experiment_id)
print(f"Reproducible: {is_reproducible}")
print(f"Message: {message}")

# Export session metadata
manager.export_session_metadata("session_metadata.json")
```

### Security Manager

```python
from coratrix import SecurityManager

# Create security manager with privacy mode
security_manager = SecurityManager(privacy_mode=True)

# Create privacy report
metadata = {
    "git_commit_hash": "abc123",
    "working_directory": "/home/user/project",
    "normal_field": "normal_value"
}

privacy_report = security_manager.create_privacy_report(metadata)
print(f"Redacted metadata: {privacy_report['metadata']}")
```

### Deterministic Random

```python
from coratrix import DeterministicRandom

# Create deterministic random generator
random_gen = DeterministicRandom(seed=42)

# Generate reproducible random numbers
rand1 = random_gen.random()
rand2 = random_gen.random()

# Save and restore random state
state = random_gen.get_random_state()
# ... do other operations ...
random_gen.set_random_state(state)

# Generate same random numbers
rand3 = random_gen.random()  # Should equal rand1
rand4 = random_gen.random()  # Should equal rand2
```

## Convenience Functions

### Pre-built Circuits

```python
from coratrix import create_bell_state, create_ghz_state, create_w_state

# Create Bell state
bell_circuit = create_bell_state()
bell_circuit.execute()
print(bell_circuit.get_state())

# Create GHZ state
ghz_circuit = create_ghz_state(3)
ghz_circuit.execute()
print(ghz_circuit.get_state())

# Create W state
w_circuit = create_w_state(3)
w_circuit.execute()
print(w_circuit.get_state())
```

### Benchmarking

```python
from coratrix import benchmark_quantum_operations

# Benchmark quantum operations
results = benchmark_quantum_operations(num_qubits=4, shots=1000)
print(f"Execution time: {results['execution_time']:.3f}s")
print(f"Memory usage: {results['memory_usage']['memory_mb']:.2f} MB")
print(f"Entanglement entropy: {results['entanglement_entropy']:.4f}")
```

### System Information

```python
from coratrix import get_system_info

# Get system information
info = get_system_info()
print(f"Python version: {info['python_version']}")
print(f"Platform: {info['platform']}")
print(f"GPU available: {info['gpu_available']}")
print(f"CPU count: {info['cpu_count']}")
print(f"Memory: {info['memory_total_gb']:.2f} GB")
```

## Error Handling

All Coratrix functions include comprehensive error handling:

```python
try:
    # Quantum operations
    state = QuantumState(2)
    state.set_amplitude(0, 1.0)
    state.normalize()
    
    # Hardware operations
    backend_manager = BackendManager()
    result = backend_manager.execute_circuit(circuit, "backend_name", shots=1000)
    
    if not result.success:
        print(f"Backend error: {result.error_message}")
    
    # Optimization
    optimization_result = engine.optimize(circuit, objective)
    if not optimization_result.success:
        print(f"Optimization error: {optimization_result.error_message}")
        
except ValueError as e:
    print(f"Value error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Use ScalableQuantumState for large systems** (≥8 qubits)
2. **Enable GPU acceleration** when available
3. **Use sparse matrices** for large, sparse states
4. **Optimize memory usage** regularly
5. **Use deterministic seeds** for reproducibility
6. **Profile your code** with the built-in performance metrics

## Examples

See the `examples/` directory for complete working examples:

- `bell_state_example.py` - Basic Bell state creation
- `grover_search_example.py` - Grover search implementation
- `optimization_example.py` - Parameterized circuit optimization
- `hardware_interface_example.py` - Hardware backend usage
- `noise_simulation_example.py` - Noise model simulation
- `reproducibility_example.py` - Reproducible experiments

## Support

For questions, issues, or contributions:

- **Documentation**: [docs.coratrix.org](https://docs.coratrix.org)
- **Issues**: [GitHub Issues](https://github.com/coratrix/coratrix/issues)
- **Discussions**: [GitHub Discussions](https://github.com/coratrix/coratrix/discussions)
- **Email**: [info@coratrix.org](mailto:info@coratrix.org)