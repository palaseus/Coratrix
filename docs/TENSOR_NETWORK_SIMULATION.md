# Tensor Network Simulation Layer

## Overview

The Tensor Network Simulation Layer is a revolutionary component of Coratrix 4.0 that provides hybrid sparse-tensor simulation capabilities. This layer seamlessly switches between sparse state vectors and tensor network methods based on circuit characteristics, providing unmatched performance for circuits with limited depth but large width.

## Key Features

### 🧠 Hybrid Sparse-Tensor Simulation
- **Dynamic Method Selection**: Automatically chooses between sparse and tensor network methods based on circuit characteristics
- **Memory-Efficient Operations**: Prevents 4TB+ memory allocation issues through intelligent sparsity management
- **Real-Time Sparsity Tracking**: Monitors and maintains sparsity under entangling gates
- **Performance Optimization**: Optimized for both wide, shallow circuits and deep, narrow circuits

### 🔧 Tensor Network Operations
- **Contraction Optimization**: Uses Cotengra integration for optimal tensor network contraction paths
- **Bond Dimension Management**: Automatic bond dimension tracking and optimization
- **Memory Management**: Efficient memory usage with configurable limits
- **Performance Metrics**: Real-time performance monitoring and optimization

## Architecture

### Core Components

#### `TensorNetworkSimulator`
The main simulator class that handles tensor network operations:

```python
from core.tensor_network_simulation import TensorNetworkSimulator, TensorNetworkConfig

# Configure tensor network simulation
config = TensorNetworkConfig(
    max_bond_dimension=32,
    contraction_optimization='greedy',
    memory_limit_gb=8.0,
    sparsity_threshold=0.1
)

# Initialize simulator
simulator = TensorNetworkSimulator(config)
simulator.initialize_circuit(4)  # 4-qubit system
```

#### `HybridSparseTensorSimulator`
Hybrid simulator that switches between methods:

```python
from core.tensor_network_simulation import HybridSparseTensorSimulator

# Create hybrid simulator
hybrid_sim = HybridSparseTensorSimulator(4, config)

# Apply gates (automatically chooses optimal method)
result = hybrid_sim.apply_gate(gate_matrix, qubit_indices)
```

### Configuration Options

#### `TensorNetworkConfig`
```python
@dataclass
class TensorNetworkConfig:
    max_bond_dimension: int = 32          # Maximum bond dimension
    contraction_optimization: str = 'greedy'  # 'greedy', 'optimal', 'random'
    use_cotengra: bool = True             # Enable Cotengra optimization
    memory_limit_gb: float = 8.0          # Memory limit in GB
    sparsity_threshold: float = 0.1       # Sparsity threshold for switching
```

## Usage Examples

### Basic Tensor Network Simulation

```python
import numpy as np
from core.tensor_network_simulation import TensorNetworkSimulator, TensorNetworkConfig

# Configure simulation
config = TensorNetworkConfig(
    max_bond_dimension=16,
    contraction_optimization='greedy',
    memory_limit_gb=4.0
)

# Initialize simulator
simulator = TensorNetworkSimulator(config)
simulator.initialize_circuit(4)

# Apply Hadamard gate
h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
simulator.apply_gate(h_gate, [0])

# Get performance metrics
metrics = simulator.get_performance_metrics()
print(f"Execution time: {metrics['execution_time']:.4f}s")
print(f"Memory usage: {metrics['memory_usage']:.2f} MB")
print(f"Sparsity ratio: {metrics['sparsity_ratio']:.2f}")
```

### Hybrid Simulation

```python
from core.tensor_network_simulation import HybridSparseTensorSimulator

# Create hybrid simulator
hybrid_sim = HybridSparseTensorSimulator(5, config)

# Apply multiple gates
h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
cnot_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)

# Apply gates (method selection is automatic)
hybrid_sim.apply_gate(h_gate, [0])
hybrid_sim.apply_gate(cnot_gate, [0, 1])
hybrid_sim.apply_gate(h_gate, [1])

# Get state vector
state_vector = hybrid_sim.get_state_vector()
```

### Advanced Configuration

```python
# High-performance configuration
config = TensorNetworkConfig(
    max_bond_dimension=64,
    contraction_optimization='optimal',
    use_cotengra=True,
    memory_limit_gb=16.0,
    sparsity_threshold=0.05
)

# Memory-constrained configuration
config = TensorNetworkConfig(
    max_bond_dimension=16,
    contraction_optimization='greedy',
    use_cotengra=False,
    memory_limit_gb=2.0,
    sparsity_threshold=0.2
)
```

## Performance Characteristics

### Memory Usage
- **Sparse State Vectors**: O(2^n) for n qubits, but only stores non-zero elements
- **Tensor Networks**: O(bond_dimension^2) for contraction, much more efficient for wide circuits
- **Hybrid Switching**: Automatically chooses method based on sparsity and circuit structure

### Execution Time
- **Single-Qubit Gates**: ~1ms for small systems, scales with bond dimension
- **Two-Qubit Gates**: ~5ms for small systems, depends on entanglement
- **Contraction**: O(bond_dimension^3) for optimal paths, O(bond_dimension^2) for greedy

### Scalability
- **Wide Circuits**: Excellent performance for circuits with many qubits but shallow depth
- **Deep Circuits**: Good performance for circuits with few qubits but many gates
- **Mixed Circuits**: Hybrid approach provides optimal performance for both cases

## Integration with Other Components

### Sparse Gate Operations
```python
from core.sparse_gate_operations import SparseGateOperator
from core.tensor_network_simulation import HybridSparseTensorSimulator

# Hybrid approach automatically integrates sparse operations
hybrid_sim = HybridSparseTensorSimulator(4, config)
# Automatically uses sparse operations when appropriate
```

### Performance Optimization
```python
from core.performance_optimization_suite import ComprehensivePerformanceOptimizer

# Tensor networks integrate with performance optimization
optimizer = ComprehensivePerformanceOptimizer()
# Can optimize tensor network contraction paths
```

## Best Practices

### When to Use Tensor Networks
- **Wide Circuits**: Circuits with many qubits (10+) but shallow depth
- **Limited Entanglement**: Circuits with controlled entanglement patterns
- **Memory Constraints**: When sparse state vectors become too large

### When to Use Sparse State Vectors
- **Deep Circuits**: Circuits with many gates but few qubits
- **High Entanglement**: Circuits with complex entanglement patterns
- **Real-Time Requirements**: When fast single-gate operations are needed

### Optimization Tips
1. **Configure Bond Dimensions**: Set appropriate limits based on available memory
2. **Choose Contraction Strategy**: Use 'optimal' for best results, 'greedy' for speed
3. **Monitor Sparsity**: Track sparsity ratios to optimize switching thresholds
4. **Memory Management**: Set appropriate memory limits to prevent crashes

## Troubleshooting

### Common Issues

#### Memory Errors
```python
# Reduce bond dimension limit
config = TensorNetworkConfig(max_bond_dimension=16)

# Increase memory limit
config = TensorNetworkConfig(memory_limit_gb=16.0)
```

#### Slow Performance
```python
# Use greedy contraction for speed
config = TensorNetworkConfig(contraction_optimization='greedy')

# Disable Cotengra if causing issues
config = TensorNetworkConfig(use_cotengra=False)
```

#### Sparsity Issues
```python
# Adjust sparsity threshold
config = TensorNetworkConfig(sparsity_threshold=0.2)

# Force tensor network mode
hybrid_sim.use_tensor_network = True
```

## API Reference

### `TensorNetworkSimulator`

#### Methods
- `initialize_circuit(num_qubits, initial_state=None)`: Initialize tensor network
- `apply_gate(gate_matrix, qubit_indices)`: Apply quantum gate
- `get_state_vector()`: Convert to state vector
- `get_entanglement_entropy()`: Calculate entanglement entropy
- `get_performance_metrics()`: Get performance statistics
- `cleanup()`: Clean up resources

#### Properties
- `num_qubits`: Number of qubits in the system
- `tensors`: List of tensor network tensors
- `bond_dimensions`: Current bond dimensions
- `contraction_history`: History of contractions performed

### `HybridSparseTensorSimulator`

#### Methods
- `apply_gate(gate_matrix, qubit_indices)`: Apply gate with automatic method selection
- `get_state_vector()`: Get state vector from current method
- `get_performance_metrics()`: Get metrics from current method
- `cleanup()`: Clean up all resources

#### Properties
- `use_tensor_network`: Whether to use tensor network mode
- `sparsity_threshold`: Threshold for method switching

## Future Enhancements

- **Advanced Contraction Algorithms**: Integration with more sophisticated contraction libraries
- **GPU Acceleration**: CUDA support for tensor network operations
- **Distributed Computing**: Multi-node tensor network simulation
- **Machine Learning**: AI-driven contraction path optimization
- **Visualization**: Tensor network structure visualization tools
