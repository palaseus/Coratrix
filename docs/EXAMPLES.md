# Coratrix Examples

## Quick Start Examples

### 1. Basic Quantum State Manipulation

```python
from core.qubit import QuantumState
from vm.executor import QuantumExecutor

# Create a 3-qubit system
executor = QuantumExecutor(3)

# Apply Hadamard gate to create superposition
executor.apply_gate('H', [0])
state = executor.get_state()
print(f"After H gate: {state}")

# Apply CNOT to create entanglement
executor.apply_gate('CNOT', [0, 1])
state = executor.get_state()
print(f"After CNOT: {state}")

# Get probability distribution
probabilities = state.get_probabilities()
print(f"Probabilities: {probabilities}")
```

### 2. Bell State Preparation

```python
from vm.executor import QuantumExecutor

# Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
executor = QuantumExecutor(2)
executor.apply_gate('H', [0])      # Create superposition
executor.apply_gate('CNOT', [0, 1])  # Create entanglement

state = executor.get_state()
print(f"Bell state: {state}")
# Output: 0.707|00⟩ + 0.707|11⟩
```

### 3. GHZ State Preparation

```python
from algorithms.quantum_algorithms import GHZState
from vm.executor import QuantumExecutor

executor = QuantumExecutor(3)
ghz_algorithm = GHZState()
result = ghz_algorithm.execute(executor, {'num_qubits': 3})

print(f"GHZ state: {result}")
# Output: 0.707|000⟩ + 0.707|111⟩
```

## Advanced Examples

### 4. Grover's Search Algorithm

```python
from algorithms.quantum_algorithms import GroverAlgorithm
from vm.executor import QuantumExecutor

# Search for target state |101⟩ in 3-qubit system
executor = QuantumExecutor(3)
grover_algorithm = GroverAlgorithm()
result = grover_algorithm.execute(executor, {
    'num_qubits': 3,
    'target_state': 5  # Binary: 101
})

print(f"Grover result: {result}")
# High probability amplitude on |101⟩
```

### 5. Quantum Teleportation

```python
from algorithms.quantum_algorithms import QuantumTeleportation
from vm.executor import QuantumExecutor

executor = QuantumExecutor(3)
teleportation_algorithm = QuantumTeleportation()
result = teleportation_algorithm.execute(executor, {'num_qubits': 3})

print(f"Teleportation result: {result}")
# Shows measurement results and final state
```

### 6. Entanglement Analysis

```python
from core.entanglement_analysis import EntanglementAnalyzer
from vm.executor import QuantumExecutor

# Create entangled state
executor = QuantumExecutor(2)
executor.apply_gate('H', [0])
executor.apply_gate('CNOT', [0, 1])

# Analyze entanglement
analyzer = EntanglementAnalyzer()
state = executor.get_state()
entanglement_info = analyzer.analyze_entanglement(state)

print(f"Entangled: {entanglement_info['is_entangled']}")
print(f"Entropy: {entanglement_info['entanglement_entropy']}")
print(f"Negativity: {entanglement_info['negativity']}")
print(f"Concurrence: {entanglement_info['concurrence']}")
```

## Visualization Examples

### 7. Probability Heatmap

```python
from visualization.probability_heatmap import ProbabilityHeatmap
from vm.executor import QuantumExecutor

# Create quantum state
executor = QuantumExecutor(3)
executor.apply_gate('H', [0])
executor.apply_gate('H', [1])
executor.apply_gate('CNOT', [0, 1])

# Generate heatmap
state = executor.get_state()
probabilities = state.get_probabilities()
heatmap = ProbabilityHeatmap()
heatmap_data = heatmap.generate_heatmap(probabilities, 3)
print(heatmap_data)
```

### 8. Circuit Diagram

```python
from visualization.circuit_diagram import CircuitDiagram

# Define gate sequence
gate_sequence = [
    {'gate': 'H', 'qubits': [0]},
    {'gate': 'CNOT', 'qubits': [0, 1]},
    {'gate': 'X', 'qubits': [2]}
]

# Generate circuit diagram
circuit = CircuitDiagram()
diagram = circuit.generate_diagram(gate_sequence, 3)
print(diagram)
```

## Research Examples

### 9. Advanced 7-Qubit Hybrid Network

```python
from research.advanced_7qubit_hybrid_network import Advanced7QubitHybridNetwork

# Initialize network
network = Advanced7QubitHybridNetwork(num_qubits=7)

# Create hybrid entanglement structure
hybrid_structure = network.create_hybrid_entanglement_structure()
print("Hybrid structure created")

# Execute teleportation cascade
teleportation_cascade = network.execute_advanced_teleportation_cascade()
print("Teleportation cascade completed")

# Execute enhanced subspace search
subspace_search = network.execute_enhanced_subspace_search()
print("Subspace search completed")
```

### 10. High-Performance Entanglement Optimization

```python
from research.quantum_7qubit_network import Quantum7QubitNetwork

# Initialize high-performance network
network = Quantum7QubitNetwork(num_qubits=7)

# Optimize entanglement parameters
optimization = network.optimize_entanglement_parameters()
print(f"Optimization completed: {optimization['best_entropy']:.4f} entropy")

# Execute teleportation cascade
teleportation = network.execute_teleportation_cascade()
print("Teleportation cascade completed")

# Execute parallel Grover search
grover_search = network.execute_parallel_grover_search()
print("Parallel Grover search completed")
```

## Command Line Examples

### 11. Interactive Mode

```bash
# Start interactive quantum programming
python cli/cli.py --interactive

# In interactive mode:
# > H 0
# > CNOT 0 1
# > MEASURE
```

### 12. Script Execution

```bash
# Run quantum script
python cli/cli.py --script examples/bell_state.qasm

# Run with verbose output
python cli/cli.py --script examples/ghz_state.qasm --verbose

# Specify number of qubits
python cli/cli.py --qubits 5 --interactive
```

### 13. Research Exploration

```bash
# Run full quantum exploration
python research_exploration.py --qubits 5 --verbose

# GPU-accelerated exploration
python research_exploration.py --qubits 8 --gpu --verbose

# Sparse matrix representation
python research_exploration.py --qubits 10 --sparse --verbose
```

## Quantum Script Examples

### 14. Bell State Script (bell_state.qasm)

```qasm
# Bell State Preparation
H q0
CNOT q0,q1
MEASURE
```

### 15. GHZ State Script (ghz_state.qasm)

```qasm
# GHZ State Preparation
H q0
CNOT q0,q1
CNOT q1,q2
MEASURE
```

### 16. Quantum Teleportation Script (teleportation.qasm)

```qasm
# Quantum Teleportation Protocol
H q1
CNOT q1,q2
X q0
CNOT q0,q1
H q0
MEASURE q0
MEASURE q1
```

## Performance Examples

### 17. Scalability Testing

```python
import time
from vm.executor import QuantumExecutor

# Test different qubit counts
qubit_counts = [2, 3, 4, 5, 6, 7, 8]

for n in qubit_counts:
    start_time = time.time()
    executor = QuantumExecutor(n)
    executor.apply_gate('H', [0])
    executor.apply_gate('CNOT', [0, 1])
    state = executor.get_state()
    end_time = time.time()
    
    print(f"{n} qubits: {end_time - start_time:.4f}s")
```

### 18. Memory Usage Analysis

```python
import psutil
import os
from vm.executor import QuantumExecutor

# Monitor memory usage
process = psutil.Process(os.getpid())

for n in [6, 7, 8, 9, 10]:
    executor = QuantumExecutor(n)
    executor.apply_gate('H', [0])
    state = executor.get_state()
    
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"{n} qubits: {memory_mb:.1f} MB memory")
```

## Error Handling Examples

### 19. Exception Handling

```python
from vm.executor import QuantumExecutor

executor = QuantumExecutor(3)

try:
    # Invalid qubit index
    executor.apply_gate('H', [5])
except ValueError as e:
    print(f"Error: {e}")

try:
    # Invalid gate name
    executor.apply_gate('INVALID', [0])
except ValueError as e:
    print(f"Error: {e}")
```

### 20. Robust Algorithm Execution

```python
from algorithms.quantum_algorithms import GroverAlgorithm
from vm.executor import QuantumExecutor

def safe_grover_search(num_qubits, target_state):
    try:
        executor = QuantumExecutor(num_qubits)
        grover_algorithm = GroverAlgorithm()
        result = grover_algorithm.execute(executor, {
            'num_qubits': num_qubits,
            'target_state': target_state
        })
        return result
    except Exception as e:
        print(f"Grover search failed: {e}")
        return None

# Safe execution
result = safe_grover_search(3, 5)
if result:
    print("Grover search successful")
```

## Integration Examples

### 21. Custom Quantum Algorithm

```python
from vm.executor import QuantumExecutor

class CustomAlgorithm:
    def __init__(self):
        self.name = "Custom Quantum Algorithm"
    
    def execute(self, executor, params):
        num_qubits = params.get('num_qubits', 2)
        
        # Custom quantum circuit
        for i in range(num_qubits):
            executor.apply_gate('H', [i])
        
        for i in range(num_qubits - 1):
            executor.apply_gate('CNOT', [i, i + 1])
        
        return executor.get_state()

# Usage
executor = QuantumExecutor(4)
custom_algorithm = CustomAlgorithm()
result = custom_algorithm.execute(executor, {'num_qubits': 4})
print(f"Custom algorithm result: {result}")
```

### 22. Batch Processing

```python
from algorithms.quantum_algorithms import GHZState
from vm.executor import QuantumExecutor

def batch_ghz_preparation(qubit_counts):
    results = []
    
    for n in qubit_counts:
        executor = QuantumExecutor(n)
        ghz_algorithm = GHZState()
        result = ghz_algorithm.execute(executor, {'num_qubits': n})
        results.append({
            'qubits': n,
            'state': str(result),
            'success': True
        })
    
    return results

# Batch process multiple qubit counts
qubit_counts = [2, 3, 4, 5]
results = batch_ghz_preparation(qubit_counts)
for result in results:
    print(f"{result['qubits']} qubits: {result['success']}")
```
