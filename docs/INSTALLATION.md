# Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM (8GB recommended for 8+ qubits)
- **Storage**: 1GB free space

### Recommended Requirements
- **Python**: 3.11+
- **Memory**: 8GB RAM or more
- **CPU**: Multi-core processor
- **GPU**: NVIDIA GPU with CUDA support (optional, for CuPy acceleration)

## Installation Methods

### Method 1: Clone and Install

1. **Clone the repository**:
```bash
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix
```

2. **Create virtual environment** (recommended):
```bash
python -m venv coratrix_env
source coratrix_env/bin/activate  # On Windows: coratrix_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Method 2: Development Installation

1. **Clone and setup development environment**:
```bash
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix
python -m venv coratrix_dev
source coratrix_dev/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dependencies

### Core Dependencies
- **NumPy** (≥1.21.0): Numerical computing
- **SciPy** (≥1.7.0): Scientific computing
- **Matplotlib** (≥3.5.0): Visualization

### Optional Dependencies
- **CuPy** (≥10.0.0): GPU acceleration (requires CUDA)
- **pytest** (≥6.0.0): Testing framework
- **pytest-cov** (≥2.0.0): Coverage testing
- **Sphinx** (≥4.0.0): Documentation generation

### GPU Acceleration Setup

For GPU acceleration with CuPy:

1. **Install CUDA toolkit** (if not already installed):
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA website
```

2. **Install CuPy**:
```bash
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

3. **Verify GPU support**:
```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Should show number of GPUs
```

## Verification

### Basic Installation Test
```bash
python -c "from core.qubit import QuantumState; print('Installation successful!')"
```

### Full System Test
```bash
python -c "
from vm.executor import QuantumExecutor
from algorithms.quantum_algorithms import GHZState

executor = QuantumExecutor(3)
ghz_algorithm = GHZState()
result = ghz_algorithm.execute(executor, {'num_qubits': 3})
print('Full system test passed!')
"
```

### Run Test Suite
```bash
python -m pytest tests/ -v
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install Coratrix
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix
python3.11 -m venv coratrix_env
source coratrix_env/bin/activate
pip install -r requirements.txt
```

### macOS
```bash
# Install Python via Homebrew
brew install python@3.11

# Install Coratrix
git clone https://github.com/palaseus/Coratrix.git
cd Coratrix
python3.11 -m venv coratrix_env
source coratrix_env/bin/activate
pip install -r requirements.txt
```

### Windows
```cmd
# Install Python from python.org
# Then in Command Prompt or PowerShell:

git clone https://github.com/palaseus/Coratrix.git
cd Coratrix
python -m venv coratrix_env
coratrix_env\Scripts\activate
pip install -r requirements.txt
```

## Docker Installation (Optional)

### Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### Build and Run
```bash
docker build -t coratrix .
docker run -it coratrix
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'core'**
   - Solution: Ensure you're in the Coratrix directory
   - Run: `export PYTHONPATH=$PYTHONPATH:$(pwd)`

2. **CuPy installation fails**
   - Solution: Install CUDA toolkit first
   - Alternative: Use CPU-only version (remove CuPy from requirements)

3. **Memory errors with large qubit counts**
   - Solution: Use sparse matrices or reduce qubit count
   - For 8+ qubits, consider GPU acceleration

4. **Permission errors on Linux/macOS**
   - Solution: Use virtual environment
   - Avoid `sudo pip install`

### Performance Optimization

1. **Enable GPU acceleration**:
```python
# In your code
import cupy as cp
# Use cp.array() instead of np.array() for GPU operations
```

2. **Use sparse matrices for large systems**:
```python
from core.scalable_quantum_state import ScalableQuantumState
# Automatically uses sparse representation for large systems
```

3. **Memory management**:
```python
# For very large systems, process in batches
# or use memory-mapped arrays
```

## Development Setup

### Install Development Dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-cov sphinx sphinx-rtd-theme
```

### Run Tests
```bash
python -m pytest tests/ -v --cov=.
```

### Generate Documentation
```bash
cd docs
sphinx-build -b html . _build/html
```

### Code Formatting
```bash
# Install black for code formatting
pip install black
black .
```

## Uninstallation

### Remove Virtual Environment
```bash
deactivate  # Exit virtual environment
rm -rf coratrix_env  # Remove virtual environment
```

### Remove System Installation
```bash
pip uninstall coratrix
```

## Support

For installation issues:
1. Check the troubleshooting section above
2. Review the GitHub issues page
3. Create a new issue with system details and error messages

## Next Steps

After successful installation:
1. Read the [API Reference](API_REFERENCE.md)
2. Explore [Quantum Algorithms](QUANTUM_ALGORITHMS.md)
3. Try the examples in the `examples/` directory
4. Run the research exploration scripts
