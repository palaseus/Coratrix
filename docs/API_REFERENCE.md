# Coratrix API Reference

## Core Components

### QuantumState
```python
from core.qubit import QuantumState

# Initialize quantum state
state = QuantumState(num_qubits=3)
print(state)  # 1.000+0.000j|000⟩

# Get probabilities
probabilities = state.get_probabilities()
```

### QuantumExecutor
```python
from vm.executor import QuantumExecutor

# Initialize executor
executor = QuantumExecutor(num_qubits=3)

# Apply gates
executor.apply_gate('H', [0])
executor.apply_gate('CNOT', [0, 1])

# Get current state
state = executor.get_state()
```

### ScalableQuantumState
```python
from core.scalable_quantum_state import ScalableQuantumState

# For large systems
scalable_state = ScalableQuantumState(num_qubits=8)
dense_vector = scalable_state.to_dense()
```

## Quantum Gates

### Basic Gates
- `X`: Pauli-X gate (quantum NOT)
- `Y`: Pauli-Y gate
- `Z`: Pauli-Z gate
- `H`: Hadamard gate (creates superposition)
- `CNOT`: Controlled-NOT gate (creates entanglement)
- `CPhase`: Controlled phase gate

### Usage
```python
executor = QuantumExecutor(2)
executor.apply_gate('H', [0])      # Hadamard on qubit 0
executor.apply_gate('CNOT', [0, 1])  # CNOT with control 0, target 1
```

## Quantum Algorithms

### GHZ State
```python
from algorithms.quantum_algorithms import GHZState

ghz_algorithm = GHZState()
result = ghz_algorithm.execute(executor, {'num_qubits': 3})
```

### W State
```python
from algorithms.quantum_algorithms import WState

w_algorithm = WState()
result = w_algorithm.execute(executor, {'num_qubits': 3})
```

### Grover's Algorithm
```python
from algorithms.quantum_algorithms import GroverAlgorithm

grover_algorithm = GroverAlgorithm()
result = grover_algorithm.execute(executor, {
    'num_qubits': 3,
    'target_state': 5
})
```

### Quantum Teleportation
```python
from algorithms.quantum_algorithms import QuantumTeleportation

teleportation_algorithm = QuantumTeleportation()
result = teleportation_algorithm.execute(executor, {'num_qubits': 3})
```

## Entanglement Analysis

### EntanglementAnalyzer
```python
from core.entanglement_analysis import EntanglementAnalyzer

analyzer = EntanglementAnalyzer()
entanglement_info = analyzer.analyze_entanglement(state)

print(f"Entangled: {entanglement_info['is_entangled']}")
print(f"Entropy: {entanglement_info['entanglement_entropy']}")
print(f"Negativity: {entanglement_info['negativity']}")
print(f"Concurrence: {entanglement_info['concurrence']}")
```

## Visualization

### Probability Heatmap
```python
from visualization.probability_heatmap import ProbabilityHeatmap

heatmap = ProbabilityHeatmap()
heatmap_data = heatmap.generate_heatmap(probabilities, num_qubits=3)
print(heatmap_data)
```

### Circuit Diagram
```python
from visualization.circuit_diagram import CircuitDiagram

circuit = CircuitDiagram()
gate_sequence = [
    {'gate': 'H', 'qubits': [0]},
    {'gate': 'CNOT', 'qubits': [0, 1]}
]
diagram = circuit.generate_diagram(gate_sequence, num_qubits=2)
print(diagram)
```

## Advanced Entanglement Networks

### Advanced 7-Qubit Hybrid Network
```python
from research.advanced_7qubit_hybrid_network import Advanced7QubitHybridNetwork

network = Advanced7QubitHybridNetwork(num_qubits=7)
hybrid_structure = network.create_hybrid_entanglement_structure()
teleportation_cascade = network.execute_advanced_teleportation_cascade()
subspace_search = network.execute_enhanced_subspace_search()
```

### High-Performance 7-Qubit Network
```python
from research.god_tier_7qubit_network import GodTier7QubitNetwork

network = GodTier7QubitNetwork(num_qubits=7)
optimization = network.optimize_entanglement_parameters()
teleportation = network.execute_teleportation_cascade()
grover_search = network.execute_parallel_grover_search()
```

### Corrected Physics Network
```python
from research.corrected_physics_network import CorrectedPhysicsNetwork

network = CorrectedPhysicsNetwork(num_qubits=7)
hybrid_structure = network.create_hybrid_entanglement_structure()
teleportation_cascade = network.execute_teleportation_cascade_with_feedback()
subspace_search = network.execute_enhanced_subspace_search()
```

## Command Line Interface

### Basic Usage
```bash
# Interactive mode
python cli/cli.py --interactive

# Run quantum script
python cli/cli.py --script examples/bell_state.qasm

# Specify number of qubits
python cli/cli.py --qubits 5 --verbose
```

### Advanced Research
```bash
# Run advanced entanglement networks
python research/advanced_7qubit_hybrid_network.py
python research/god_tier_7qubit_network.py
python research/corrected_physics_network.py
```

## Error Handling

### Common Exceptions
- `ValueError`: Invalid qubit indices or gate parameters
- `TypeError`: Incorrect gate types or arguments
- `IndexError`: Qubit index out of range

### Example Error Handling
```python
try:
    executor.apply_gate('H', [5])  # Invalid qubit index
except ValueError as e:
    print(f"Error: {e}")
```
