# Installation Guide for Coratrix 3.1

This guide provides comprehensive installation instructions for the Coratrix 3.1 modular quantum computing SDK.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Detailed Installation](#detailed-installation)
- [GPU Acceleration Setup](#gpu-acceleration-setup)
- [Development Setup](#development-setup)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher (recommended: 3.10+)
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 2 GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **Python**: 3.10 or higher
- **RAM**: 16 GB or more
- **Storage**: 10 GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **CPU**: Multi-core processor (8+ cores recommended)

### Platform Support

| Platform | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|----------|-------------|-------------|--------------|--------------|--------------|
| Linux    | ✅          | ✅          | ✅           | ✅           | ✅           |
| macOS    | ✅          | ✅          | ✅           | ✅           | ✅           |
| Windows  | ✅          | ✅          | ✅           | ✅           | ✅           |

## Quick Installation

### Option 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix

# Create virtual environment
python -m venv coratrix_env
source coratrix_env/bin/activate  # On Windows: coratrix_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python -c "import coratrix; print('Coratrix 3.1 installed successfully!')"
```

### Option 2: Minimal Installation

```bash
# Install only core dependencies
pip install numpy scipy matplotlib psutil

# Clone and install Coratrix
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix
pip install -e .
```

## Detailed Installation

### Step 1: Python Environment Setup

#### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv coratrix_env

# Activate virtual environment
# On Linux/macOS:
source coratrix_env/bin/activate

# On Windows:
coratrix_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Using Conda (Alternative)

```bash
# Create conda environment
conda create -n coratrix python=3.10
conda activate coratrix

# Install basic dependencies
conda install numpy scipy matplotlib
```

### Step 2: Install Dependencies

#### Core Dependencies (Required)

```bash
# Essential packages
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0
pip install psutil>=5.8.0
```

#### Optional Dependencies

```bash
# Visualization
pip install plotly>=5.0.0
pip install seaborn>=0.11.0

# Quantum framework integration
pip install qiskit>=0.45.0
pip install pennylane>=0.32.0

# Optimization
pip install scikit-optimize>=0.9.0
pip install optuna>=3.0.0
```

### Step 3: Install Coratrix

```bash
# Clone repository
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## GPU Acceleration Setup

### CUDA Installation

#### For NVIDIA GPUs

1. **Install CUDA Toolkit**
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   
   # Verify CUDA installation
   nvcc --version
   ```

2. **Install CuPy**
   ```bash
   # For CUDA 11.x
   pip install cupy-cuda11x
   
   # For CUDA 12.x
   pip install cupy-cuda12x
   
   # Verify CuPy installation
   python -c "import cupy; print('CuPy installed successfully!')"
   ```

#### For Apple Silicon (M1/M2)

```bash
# CuPy is not available for ARM64, but you can use other optimizations
pip install numpy scipy matplotlib
# Coratrix will automatically use optimized CPU implementations
```

### GPU Verification

```python
# Test GPU acceleration
from coratrix.core import ScalableQuantumState

# Test GPU availability
state = ScalableQuantumState(10, use_gpu=True)
print(f"GPU acceleration: {state.gpu_available}")

# Test performance
import time
start = time.time()
state = ScalableQuantumState(15, use_gpu=True)
end = time.time()
print(f"GPU setup time: {end - start:.3f} seconds")
```

## Development Setup

### Development Dependencies

```bash
# Install development dependencies
pip install -r requirements.txt

# Additional development tools
pip install black flake8 mypy isort
pip install pytest pytest-cov pytest-xdist
pip install pre-commit
```

### Code Quality Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run code formatting
black coratrix/ tests/

# Run linting
flake8 coratrix/ tests/

# Run type checking
mypy coratrix/
```

### Testing Setup

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=coratrix --cov-report=html

# Run specific test categories
python -m pytest tests/test_core/ -v
python -m pytest tests/test_compiler/ -v
python -m pytest tests/test_backend/ -v
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'coratrix'`

**Solution**:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/Coratrix"

# Or install in development mode
pip install -e .
```

#### 2. GPU Issues

**Problem**: `ImportError: No module named 'cupy'`

**Solution**:
```bash
# Install CuPy for your CUDA version
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x

# Or disable GPU acceleration
state = ScalableQuantumState(10, use_gpu=False)
```

#### 3. Memory Issues

**Problem**: `MemoryError` for large systems

**Solution**:
```bash
# Use sparse representation
state = ScalableQuantumState(15, use_sparse=True, sparse_threshold=8)

# Or reduce system size
state = ScalableQuantumState(10, use_gpu=False)
```

#### 4. Plugin Loading Issues

**Problem**: Plugin loading warnings

**Solution**:
```bash
# Check plugin directory
ls -la ~/.coratrix/plugins/

# Verify plugin files
python -c "from coratrix.plugins import PluginManager; pm = PluginManager(); print(pm.list_plugins())"
```

### Platform-Specific Issues

#### Windows

```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Windows-specific dependencies
pip install pywin32
```

#### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install macOS-specific dependencies
pip install pyobjc
```

#### Linux

```bash
# Install development headers
sudo apt-get install python3-dev build-essential

# Install system dependencies
sudo apt-get install libopenblas-dev liblapack-dev
```

### Performance Issues

#### Slow Execution

```python
# Check system resources
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Use sparse representation for large systems
state = ScalableQuantumState(15, use_sparse=True)

# Enable GPU acceleration if available
state = ScalableQuantumState(15, use_gpu=True)
```

#### Memory Usage

```python
# Monitor memory usage
import tracemalloc
tracemalloc.start()

# Your code here
state = ScalableQuantumState(15)

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## Verification

### Basic Verification

```python
# Test core functionality
from coratrix.core import ScalableQuantumState, QuantumCircuit
from coratrix.core.quantum_circuit import HGate, CNOTGate

# Create and test quantum state
state = ScalableQuantumState(2)
print(f"Initial state: {state.get_amplitude(0)}")

# Test quantum gates
state.apply_gate(HGate(), [0])
print(f"After H gate: {state.get_amplitude(0)}")

# Test quantum circuit
circuit = QuantumCircuit(2, "test")
circuit.add_gate(HGate(), [0])
circuit.add_gate(CNOTGate(), [0, 1])
circuit.execute(state)
print("Circuit executed successfully!")
```

### Advanced Verification

```python
# Test compiler stack
from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode

compiler = CoratrixCompiler()
dsl_source = "circuit test() { h q0; cnot q0, q1; }"
options = CompilerOptions(mode=CompilerMode.COMPILE_ONLY, target_format='openqasm')
result = compiler.compile(dsl_source, options)
print(f"Compilation successful: {result.success}")

# Test backend management
from coratrix.backend import BackendManager, BackendConfiguration, BackendType, SimulatorBackend

backend_manager = BackendManager()
config = BackendConfiguration(name='test', backend_type=BackendType.SIMULATOR)
backend = SimulatorBackend(config)
success = backend_manager.register_backend('test', backend)
print(f"Backend registration: {success}")

# Test plugin system
from coratrix.plugins import PluginManager
from coratrix.plugins.example_optimization_pass import ExampleOptimizationPlugin

plugin_manager = PluginManager()
plugin = ExampleOptimizationPlugin()
success = plugin_manager.register_plugin(plugin)
print(f"Plugin registration: {success}")
```

### Performance Benchmarking

```python
# Benchmark quantum state creation
import time

def benchmark_state_creation(qubits, use_gpu=False, use_sparse=False):
    start = time.time()
    state = ScalableQuantumState(qubits, use_gpu=use_gpu, use_sparse=use_sparse)
    end = time.time()
    return end - start

# Test different configurations
configs = [
    (10, False, False),  # CPU, dense
    (10, False, True),   # CPU, sparse
    (10, True, False),  # GPU, dense
    (10, True, True),   # GPU, sparse
]

for qubits, gpu, sparse in configs:
    time_taken = benchmark_state_creation(qubits, gpu, sparse)
    print(f"Qubits: {qubits}, GPU: {gpu}, Sparse: {sparse}, Time: {time_taken:.3f}s")
```

### Complete System Test

```python
# Run comprehensive system test
def test_complete_system():
    print("🧪 Testing Coratrix 3.1 Complete System...")
    
    # Test 1: Core simulation
    from coratrix.core import ScalableQuantumState, QuantumCircuit
    from coratrix.core.quantum_circuit import HGate, CNOTGate
    
    state = ScalableQuantumState(2)
    circuit = QuantumCircuit(2, "test")
    circuit.add_gate(HGate(), [0])
    circuit.add_gate(CNOTGate(), [0, 1])
    circuit.execute(state)
    print("✅ Core simulation working")
    
    # Test 2: Compiler stack
    from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode
    
    compiler = CoratrixCompiler()
    dsl_source = "circuit test() { h q0; cnot q0, q1; }"
    options = CompilerOptions(mode=CompilerMode.COMPILE_ONLY, target_format='openqasm')
    result = compiler.compile(dsl_source, options)
    print(f"✅ Compiler stack working: {result.success}")
    
    # Test 3: Backend management
    from coratrix.backend import BackendManager, BackendConfiguration, BackendType, SimulatorBackend
    
    backend_manager = BackendManager()
    config = BackendConfiguration(name='test', backend_type=BackendType.SIMULATOR)
    backend = SimulatorBackend(config)
    success = backend_manager.register_backend('test', backend)
    print(f"✅ Backend management working: {success}")
    
    # Test 4: Plugin system
    from coratrix.plugins import PluginManager
    from coratrix.plugins.example_optimization_pass import ExampleOptimizationPlugin
    
    plugin_manager = PluginManager()
    plugin = ExampleOptimizationPlugin()
    success = plugin_manager.register_plugin(plugin)
    print(f"✅ Plugin system working: {success}")
    
    # Test 5: CLI integration
    from coratrix.cli import CoratrixCompilerCLI
    
    cli = CoratrixCompilerCLI()
    parser = cli.create_parser()
    print("✅ CLI integration working")
    
    print("\n🎉 Coratrix 3.1 installation verified successfully!")
    print("   All systems operational and ready for quantum computing!")

# Run the test
test_complete_system()
```

## Next Steps

After successful installation:

1. **Read the Getting Started Guide**: `docs/GETTING_STARTED.md`
2. **Explore Examples**: Check the `examples/` directory
3. **Read Documentation**: Browse the comprehensive guides in `docs/`
4. **Join the Community**: Report issues and share your quantum algorithms
5. **Contribute**: See `CONTRIBUTING.md` for development guidelines

## Support

If you encounter issues:

1. **Check Documentation**: Review this guide and other docs
2. **Search Issues**: Look for similar problems in GitHub issues
3. **Create Issue**: Report new issues with detailed information
4. **Community**: Join discussions and get help from the community

Welcome to Coratrix 3.1! 🚀