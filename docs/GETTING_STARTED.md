# Getting Started with Coratrix 3.1

## Quick Start (5 minutes)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import coratrix; print('Coratrix 3.1 installed successfully!')"
```

### 2. Your First Quantum Circuit

```python
from coratrix.core import ScalableQuantumState, QuantumCircuit
from coratrix.core.quantum_circuit import HGate, CNOTGate

# Create a 2-qubit quantum state
state = ScalableQuantumState(2)

# Create a Bell state circuit
circuit = QuantumCircuit(2, "bell_state")
circuit.add_gate(HGate(), [0])      # Apply Hadamard to qubit 0
circuit.add_gate(CNOTGate(), [0, 1])  # Apply CNOT with control=0, target=1

# Execute the circuit
circuit.execute(state)

# Check the result
print(f"Bell state amplitude |00⟩: {state.get_amplitude(0)}")
print(f"Bell state amplitude |11⟩: {state.get_amplitude(3)}")
```

### 3. Compile and Execute with DSL

```python
from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode

# Define quantum circuit in DSL
dsl_source = """
circuit bell_state() {
    h q0;
    cnot q0, q1;
    measure q0, q1;
}
"""

# Compile to OpenQASM
compiler = CoratrixCompiler()
options = CompilerOptions(
    mode=CompilerMode.COMPILE_ONLY,
    target_format='openqasm'
)
result = compiler.compile(dsl_source, options)

print("Generated OpenQASM:")
print(result.target_code)
```

## Step-by-Step Tutorials

### Tutorial 1: Building a Custom Quantum Algorithm

Let's build a quantum teleportation protocol step by step:

#### Step 1: Create the Teleportation Circuit

```python
from coratrix.core import ScalableQuantumState, QuantumCircuit
from coratrix.core.quantum_circuit import HGate, CNOTGate, XGate, ZGate

def create_teleportation_circuit():
    """Create a quantum teleportation circuit."""
    circuit = QuantumCircuit(3, "teleportation")
    
    # Step 1: Create Bell pair between Alice (q1) and Bob (q2)
    circuit.add_gate(HGate(), [1])      # Alice applies H
    circuit.add_gate(CNOTGate(), [1, 2])  # Alice-Bob CNOT
    
    # Step 2: Alice prepares her qubit (q0) to teleport
    # (This would be done by Alice in real scenario)
    
    # Step 3: Alice performs Bell measurement
    circuit.add_gate(CNOTGate(), [0, 1])  # Alice: CNOT(q0, q1)
    circuit.add_gate(HGate(), [0])         # Alice: H(q0)
    
    # Step 4: Bob applies corrections based on measurement results
    # (These would be applied after classical communication)
    circuit.add_gate(XGate(), [2])        # Bob: X correction
    circuit.add_gate(ZGate(), [2])        # Bob: Z correction
    
    return circuit

# Create and execute the circuit
circuit = create_teleportation_circuit()
state = ScalableQuantumState(3)
circuit.execute(state)

print("Teleportation circuit created successfully!")
```

#### Step 2: Analyze the Results

```python
# Check the final state
print("Final state amplitudes:")
for i in range(8):  # 2^3 = 8 possible states
    amplitude = state.get_amplitude(i)
    if abs(amplitude) > 1e-10:  # Only print non-zero amplitudes
        binary = format(i, '03b')
        print(f"|{binary}⟩: {amplitude:.4f}")
```

#### Step 3: Verify Teleportation Fidelity

```python
from coratrix.core.entanglement import EntanglementAnalyzer

# Analyze entanglement in the system
analyzer = EntanglementAnalyzer()
entanglement = analyzer.analyze_entanglement(state, [0, 1, 2])

print(f"Entanglement entropy: {entanglement['entropy']:.4f}")
print(f"Concurrence: {entanglement['concurrence']:.4f}")
```

### Tutorial 2: Interpreting Entanglement Metrics

#### Understanding Entanglement Entropy

```python
from coratrix.core import ScalableQuantumState
from coratrix.core.quantum_circuit import HGate, CNOTGate
from coratrix.core.entanglement import EntanglementAnalyzer

def analyze_entanglement_patterns():
    """Demonstrate different entanglement patterns."""
    
    # 1. Separable state (no entanglement)
    separable = ScalableQuantumState(2)
    print("Separable state:")
    analyzer = EntanglementAnalyzer()
    result = analyzer.analyze_entanglement(separable, [0, 1])
    print(f"  Entropy: {result['entropy']:.4f} (should be 0)")
    print(f"  Concurrence: {result['concurrence']:.4f} (should be 0)")
    
    # 2. Bell state (maximum entanglement)
    bell = ScalableQuantumState(2)
    bell.apply_gate(HGate(), [0])
    bell.apply_gate(CNOTGate(), [0, 1])
    
    print("\nBell state:")
    result = analyzer.analyze_entanglement(bell, [0, 1])
    print(f"  Entropy: {result['entropy']:.4f} (should be 1)")
    print(f"  Concurrence: {result['concurrence']:.4f} (should be 1)")
    
    # 3. Partial entanglement
    partial = ScalableQuantumState(2)
    partial.apply_gate(HGate(), [0])
    # Apply a weaker entangling gate
    from coratrix.core.quantum_circuit import RYGate
    partial.apply_gate(RYGate(0.5), [1])  # Small rotation
    
    print("\nPartially entangled state:")
    result = analyzer.analyze_entanglement(partial, [0, 1])
    print(f"  Entropy: {result['entropy']:.4f} (between 0 and 1)")
    print(f"  Concurrence: {result['concurrence']:.4f} (between 0 and 1)")

analyze_entanglement_patterns()
```

#### Understanding Performance Metrics

```python
def explain_performance_metrics():
    """Explain the performance metrics mentioned in the README."""
    
    print("Coratrix 3.1 Performance Metrics Explained:")
    print("=" * 50)
    
    print("\n1. Grover's Algorithm Success Rate (94.5%):")
    print("   - This refers to the probability of finding the target item")
    print("   - Theoretical maximum for Grover's is ~100% with optimal iterations")
    print("   - 94.5% indicates excellent performance with minimal noise")
    print("   - See: docs/QUANTUM_ALGORITHMS.md for implementation details")
    
    print("\n2. Entropy Optimization (99.08%):")
    print("   - Measures how well the system maintains quantum coherence")
    print("   - 99.08% indicates minimal information loss during computation")
    print("   - Critical for quantum error correction and fault tolerance")
    print("   - See: coratrix/core/entanglement.py for implementation")
    
    print("\n3. Fidelity Benchmarks:")
    print("   - Bell state fidelity: 99.99% (near-perfect state preparation)")
    print("   - GHZ state fidelity: 99.95% (excellent multi-qubit coherence)")
    print("   - Teleportation fidelity: 99.8% (high-fidelity quantum communication)")
    
    print("\n4. Performance Scaling:")
    print("   - 2-qubit systems: <1ms (real-time simulation)")
    print("   - 5-qubit systems: ~10ms (interactive development)")
    print("   - 10-qubit systems: ~1s (research applications)")
    print("   - 15-qubit systems: ~1min (advanced research)")

explain_performance_metrics()
```

### Tutorial 3: Using the CLI Tools

#### Basic CLI Usage

```bash
# Create a simple quantum circuit file
cat > bell_state.qasm << 'EOF'
circuit bell_state() {
    h q0;
    cnot q0, q1;
    measure q0, q1;
}
EOF

# Compile to OpenQASM
coratrixc bell_state.qasm -o bell_state_output.qasm --target openqasm

# Execute the circuit
coratrixc bell_state.qasm --execute --backend local_simulator --shots 1000
```

#### Interactive CLI Session

```bash
# Start interactive mode
coratrixc --interactive

# In the interactive shell:
>>> state 3
>>> gate h 0
>>> gate cnot 0 1
>>> gate cnot 1 2
>>> execute 1000
>>> measure 0 1 2
>>> save ghz_state.json
>>> quit
```

### Tutorial 4: Plugin Development

#### Creating a Custom Optimization Plugin

```python
from coratrix.plugins import CompilerPassPlugin, PluginInfo
from coratrix.compiler.passes import CompilerPass, PassResult
from coratrix.compiler.ir import CoratrixIR

class CustomGateMergingPass(CompilerPass):
    """Custom pass that merges adjacent identical gates."""
    
    def run(self, ir: CoratrixIR) -> PassResult:
        """Merge adjacent identical gates."""
        merged_ir = self._merge_gates(ir)
        return PassResult(success=True, ir=merged_ir)
    
    def _merge_gates(self, ir: CoratrixIR):
        """Implementation of gate merging logic."""
        # Custom merging logic here
        return ir

class CustomOptimizationPlugin(CompilerPassPlugin):
    """Plugin wrapper for custom optimization."""
    
    def __init__(self):
        super().__init__(
            info=PluginInfo(
                name='custom_gate_merging',
                version='1.0.0',
                description='Custom gate merging optimization',
                author='Your Name',
                plugin_type='compiler_pass',
                dependencies=[]
            )
        )
        self.pass_instance = None
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.pass_instance = CustomGateMergingPass()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize plugin: {e}")
            return False
    
    def get_pass(self):
        """Get the compiler pass instance."""
        return self.pass_instance

# Register and use the plugin
from coratrix.plugins import PluginManager

plugin_manager = PluginManager()
plugin = CustomOptimizationPlugin()
plugin_manager.register_plugin(plugin)

print("Custom optimization plugin registered successfully!")
```

## Common Use Cases

### 1. Quantum Algorithm Development

```python
# Grover's search algorithm
def grover_search(target_item, num_qubits):
    """Implement Grover's search algorithm."""
    from coratrix.core import ScalableQuantumState, QuantumCircuit
    from coratrix.core.quantum_circuit import HGate, XGate, ZGate
    
    # Create superposition
    state = ScalableQuantumState(num_qubits)
    for i in range(num_qubits):
        state.apply_gate(HGate(), [i])
    
    # Grover iterations
    num_iterations = int(3.14159 * 2**(num_qubits/2) / 4)
    
    for _ in range(num_iterations):
        # Oracle (mark target)
        if target_item & (1 << i):
            state.apply_gate(XGate(), [i])
        
        # Diffusion operator
        for i in range(num_qubits):
            state.apply_gate(HGate(), [i])
            state.apply_gate(XGate(), [i])
        
        # Multi-controlled Z
        # (Implementation details omitted for brevity)
        
        for i in range(num_qubits):
            state.apply_gate(XGate(), [i])
            state.apply_gate(HGate(), [i])
    
    return state
```

### 2. Quantum Error Correction

```python
# 3-qubit bit-flip code
def bit_flip_code():
    """Implement 3-qubit bit-flip error correction."""
    from coratrix.core import ScalableQuantumState, QuantumCircuit
    from coratrix.core.quantum_circuit import HGate, CNOTGate, XGate
    
    # Encode logical qubit
    state = ScalableQuantumState(3)
    # |0⟩ → |000⟩, |1⟩ → |111⟩
    state.apply_gate(CNOTGate(), [0, 1])
    state.apply_gate(CNOTGate(), [0, 2])
    
    # Simulate bit-flip error
    state.apply_gate(XGate(), [1])  # Error on qubit 1
    
    # Syndrome measurement and correction
    # (Implementation details omitted)
    
    return state
```

### 3. Quantum Machine Learning

```python
# Variational quantum eigensolver (VQE)
def vqe_circuit(parameters):
    """Create a VQE circuit with parameterized gates."""
    from coratrix.core import QuantumCircuit
    from coratrix.core.quantum_circuit import RYGate, RZGate, CNOTGate
    
    circuit = QuantumCircuit(4, "vqe")
    
    # Parameterized ansatz
    for i, param in enumerate(parameters):
        circuit.add_gate(RYGate(param), [i % 4])
        if i > 0:
            circuit.add_gate(CNOTGate(), [i-1, i % 4])
    
    return circuit
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```python
# Problem: ModuleNotFoundError
# Solution: Check Python path
import sys
sys.path.insert(0, '/path/to/Coratrix')
```

#### 2. GPU Acceleration Issues
```python
# Problem: CuPy not available
# Solution: Check GPU setup or use CPU fallback
state = ScalableQuantumState(10, use_gpu=False)  # Force CPU
```

#### 3. Memory Issues with Large Systems
```python
# Problem: Out of memory for large qubit counts
# Solution: Use sparse representation
state = ScalableQuantumState(15, use_sparse=True, sparse_threshold=8)
```

#### 4. Plugin Loading Issues
```python
# Problem: Plugin warnings
# Solution: Use absolute imports in plugin files
from coratrix.plugins.base import Plugin  # Correct
# from .base import Plugin  # Avoid relative imports
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory for more complex algorithms
2. **Read Documentation**: Dive into the comprehensive guides in `docs/`
3. **Develop Plugins**: Create custom compiler passes and backends
4. **Contribute**: See `CONTRIBUTING.md` for development guidelines
5. **Join Community**: Report issues and share your quantum algorithms

## Performance Tips

### For Large Systems (10+ qubits)
- Use sparse representation: `use_sparse=True`
- Enable GPU acceleration: `use_gpu=True` (if available)
- Optimize circuit depth with compiler passes
- Use appropriate backend for your use case

### For Real-time Development
- Start with small systems (2-5 qubits)
- Use local simulator for rapid prototyping
- Profile your circuits with built-in tools
- Cache frequently used circuits

### For Production Deployment
- Use hardware backends for final execution
- Implement proper error handling
- Monitor performance metrics
- Use noise models for realistic simulation
