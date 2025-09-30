# Contributing to Coratrix

Thank you for your interest in contributing to Coratrix! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Code Style and Standards](#code-style-and-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Issue Reporting](#issue-reporting)
10. [Community Guidelines](#community-guidelines)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html). By participating, you agree to uphold this code.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of quantum computing concepts
- Familiarity with Python development

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/coratrix.git
   cd coratrix
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/coratrix/coratrix.git
   ```

## Development Setup

### 1. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n coratrix python=3.10
conda activate coratrix
```

### 2. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements.txt

# Install additional development tools
pip install pytest pytest-cov black flake8 mypy sphinx
```

### 3. Install in Development Mode

```bash
pip install -e .
```

### 4. Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=coratrix --cov-report=html

# Run specific test file
python -m pytest tests/test_quantum_state.py
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Fix existing issues
2. **Feature Additions**: Add new functionality
3. **Documentation**: Improve documentation
4. **Tests**: Add or improve tests
5. **Performance**: Optimize existing code
6. **Examples**: Add usage examples
7. **Tutorials**: Create educational content

### Contribution Process

1. **Check Existing Issues**: Look for existing issues or discussions
2. **Create an Issue**: If proposing a new feature, create an issue first
3. **Fork and Branch**: Create a feature branch from `main`
4. **Develop**: Implement your changes
5. **Test**: Ensure all tests pass
6. **Document**: Update documentation as needed
7. **Submit**: Create a pull request

### Branch Naming

Use descriptive branch names:

- `feature/quantum-algorithm-x`
- `bugfix/fix-measurement-issue`
- `docs/update-api-reference`
- `test/add-unit-tests`
- `perf/optimize-gpu-operations`

## Code Style and Standards

### Python Style

We follow PEP 8 with some modifications:

```python
# Use 4 spaces for indentation
def function_name(parameter1: int, parameter2: str) -> bool:
    """Function docstring."""
    if parameter1 > 0:
        return True
    return False

# Use type hints
from typing import List, Dict, Optional, Union

def process_quantum_state(
    state: QuantumState,
    gates: List[QuantumGate],
    parameters: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Process quantum state with gates."""
    pass
```

### Code Formatting

We use Black for code formatting:

```bash
# Format code
black coratrix/ tests/

# Check formatting
black --check coratrix/ tests/
```

### Linting

We use flake8 for linting:

```bash
# Run linter
flake8 coratrix/ tests/

# With configuration
flake8 --config=.flake8 coratrix/ tests/
```

### Type Checking

We use mypy for type checking:

```bash
# Run type checker
mypy coratrix/
```

## Testing

### Test Structure

```
tests/
├── test_quantum_state.py
├── test_quantum_gates.py
├── test_circuit.py
├── test_scalable_quantum_state.py
├── test_noise_models.py
├── test_optimization_engine.py
├── test_multi_subspace_grover.py
├── test_reproducibility.py
├── test_hardware_interface.py
├── test_advanced_algorithms.py
├── test_unitary_consistency.py
├── test_property_based.py
├── test_circuit_fidelity.py
└── test_correctness_suite.py
```

### Writing Tests

```python
import unittest
import numpy as np
from coratrix import QuantumState, HGate, CNOTGate

class TestQuantumState(unittest.TestCase):
    """Test cases for QuantumState."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = QuantumState(2)
    
    def test_initialization(self):
        """Test quantum state initialization."""
        self.assertEqual(self.state.num_qubits, 2)
        self.assertEqual(self.state.dimension, 4)
    
    def test_bell_state_creation(self):
        """Test creating a Bell state."""
        # Create Bell state
        self.state.set_amplitude(0, 1.0/np.sqrt(2))
        self.state.set_amplitude(3, 1.0/np.sqrt(2))
        self.state.normalize()
        
        # Check normalization
        probabilities = self.state.get_probabilities()
        self.assertAlmostEqual(sum(probabilities), 1.0, places=10)
        
        # Check Bell state properties
        self.assertAlmostEqual(probabilities[0], 0.5, places=10)
        self.assertAlmostEqual(probabilities[3], 0.5, places=10)
```

### Test Requirements

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Property-Based Tests**: Use Hypothesis for random testing
4. **Performance Tests**: Benchmark critical operations
5. **Regression Tests**: Prevent previously fixed bugs

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_quantum_state.py::TestQuantumState::test_initialization

# Run with coverage
python -m pytest tests/ --cov=coratrix --cov-report=html

# Run property-based tests
python -m pytest tests/test_property_based.py
```

## Documentation

### Documentation Standards

1. **Docstrings**: Use Google-style docstrings
2. **Type Hints**: Include type annotations
3. **Examples**: Provide usage examples
4. **API Reference**: Document all public APIs
5. **Tutorials**: Create educational content

### Docstring Format

```python
def quantum_algorithm(
    state: QuantumState,
    parameters: Dict[str, float],
    iterations: int = 100
) -> Dict[str, Any]:
    """
    Apply quantum algorithm to state.
    
    Args:
        state: Quantum state to process
        parameters: Algorithm parameters
        iterations: Number of iterations to run
        
    Returns:
        Dictionary containing results and metrics
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If algorithm fails to converge
        
    Example:
        >>> state = QuantumState(2)
        >>> params = {"theta": 0.5, "phi": 1.0}
        >>> result = quantum_algorithm(state, params, iterations=50)
        >>> print(f"Success: {result['success']}")
    """
    pass
```

### Building Documentation

```bash
# Build Sphinx documentation
cd docs/
sphinx-build -b html . _build/html

# View documentation
open _build/html/index.html
```

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Update relevant documentation
2. **Add Tests**: Add tests for new functionality
3. **Run Tests**: Ensure all tests pass
4. **Check Style**: Run code formatting and linting
5. **Update Changelog**: Add entry to CHANGELOG.md

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Coverage maintained

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves PR
5. **Merge**: PR is merged to main branch

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed reproduction steps
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: System information
6. **Code Sample**: Minimal code to reproduce

### Feature Requests

When requesting features, include:

1. **Use Case**: Why this feature is needed
2. **Proposed Solution**: How you envision it working
3. **Alternatives**: Other approaches considered
4. **Additional Context**: Any other relevant information

### Issue Templates

Use the provided issue templates:
- Bug report template
- Feature request template
- Question template

## Community Guidelines

### Communication

1. **Be Respectful**: Treat everyone with respect
2. **Be Constructive**: Provide helpful feedback
3. **Be Patient**: Allow time for responses
4. **Be Clear**: Communicate clearly and concisely

### Getting Help

1. **Documentation**: Check existing documentation
2. **Issues**: Search existing issues
3. **Discussions**: Use GitHub Discussions
4. **Email**: Contact maintainers directly

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Community acknowledgments

## Development Workflow

### Daily Workflow

1. **Sync**: Pull latest changes from upstream
2. **Branch**: Create feature branch
3. **Develop**: Implement changes
4. **Test**: Run tests locally
5. **Commit**: Commit changes with clear messages
6. **Push**: Push to your fork
7. **PR**: Create pull request

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add multi-subspace Grover search
fix: resolve measurement normalization issue
docs: update API reference for optimization engine
test: add property-based tests for quantum gates
perf: optimize GPU memory usage in scalable state
```

### Release Process

1. **Version Bump**: Update version numbers
2. **Changelog**: Update CHANGELOG.md
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update documentation
5. **Release**: Create GitHub release
6. **Distribution**: Publish to PyPI

## Advanced Topics

### Performance Optimization

1. **Profiling**: Use profiling tools
2. **Benchmarking**: Create benchmarks
3. **Memory Usage**: Monitor memory consumption
4. **GPU Utilization**: Optimize GPU usage
5. **Parallel Processing**: Use multiprocessing

### Security Considerations

1. **Input Validation**: Validate all inputs
2. **Error Handling**: Handle errors gracefully
3. **Data Privacy**: Protect sensitive data
4. **Reproducibility**: Ensure deterministic behavior
5. **Audit Trail**: Maintain operation logs

### Internationalization

1. **Documentation**: Support multiple languages
2. **Error Messages**: Localize error messages
3. **User Interface**: Support different locales
4. **Cultural Sensitivity**: Consider cultural differences

## Resources

### Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Installation Guide](docs/INSTALLATION.md)
- [Examples](examples/)
- [Tutorials](tutorials/)

### External Resources

- [Quantum Computing Concepts](https://qiskit.org/textbook/)
- [Python Development](https://docs.python.org/3/)
- [Git Workflow](https://git-scm.com/docs)
- [Testing Best Practices](https://docs.pytest.org/)

### Community

- [GitHub Discussions](https://github.com/coratrix/coratrix/discussions)
- [Discord Server](https://discord.gg/coratrix)
- [Mailing List](https://groups.google.com/forum/#!forum/coratrix)
- [Twitter](https://twitter.com/coratrix)

## Contact

For questions about contributing:

- **Email**: [contributors@coratrix.org](mailto:contributors@coratrix.org)
- **GitHub**: [@coratrix/coratrix](https://github.com/coratrix/coratrix)
- **Discord**: [Coratrix Community](https://discord.gg/coratrix)

Thank you for contributing to Coratrix! 🚀
