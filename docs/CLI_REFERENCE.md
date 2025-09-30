# CLI Reference Guide

## Overview

Coratrix 3.1 provides comprehensive command-line tools for quantum circuit compilation, execution, and management. This guide covers all CLI commands, options, and usage patterns.

## Coratrix Compiler CLI (`coratrixc`)

### Basic Usage

```bash
coratrixc [OPTIONS] INPUT_FILE [OUTPUT_FILE]
```

### Global Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message and exit |
| `-v, --version` | Show version information |
| `--verbose` | Enable verbose output |
| `--debug` | Enable debug mode |
| `--config FILE` | Use configuration file |

### Compilation Options

| Option | Description |
|--------|-------------|
| `-t, --target FORMAT` | Target format (openqasm, qiskit, pennylane) |
| `-o, --output FILE` | Output file path |
| `--optimize` | Enable optimization passes |
| `--no-optimize` | Disable optimization passes |
| `--passes LIST` | Specify compiler passes to run |

### Execution Options

| Option | Description |
|--------|-------------|
| `-e, --execute` | Execute circuit after compilation |
| `-b, --backend BACKEND` | Backend for execution |
| `-s, --shots COUNT` | Number of shots for execution |
| `--noise-model FILE` | Noise model configuration |
| `--seed SEED` | Random seed for execution |

### Backend Management

| Option | Description |
|--------|-------------|
| `--list-backends` | List available backends |
| `--backend-info BACKEND` | Show backend information |
| `--register-backend CONFIG` | Register new backend |
| `--unregister-backend BACKEND` | Unregister backend |

### Plugin Management

| Option | Description |
|--------|-------------|
| `--list-plugins` | List available plugins |
| `--plugin-info PLUGIN` | Show plugin information |
| `--load-plugin FILE` | Load plugin from file |
| `--plugin-dir DIR` | Plugin directory path |

## Usage Examples

### Basic Compilation

```bash
# Compile DSL to OpenQASM
coratrixc input.qasm -o output.qasm --target openqasm

# Compile to Qiskit format
coratrixc input.qasm -o output.py --target qiskit

# Compile with optimization
coratrixc input.qasm -o output.qasm --target openqasm --optimize
```

### Circuit Execution

```bash
# Execute circuit on local simulator
coratrixc input.qasm --execute --backend local_simulator --shots 1000

# Execute with specific noise model
coratrixc input.qasm --execute --backend local_simulator --noise-model noise.json

# Execute with custom seed
coratrixc input.qasm --execute --backend local_simulator --seed 42
```

### Backend Management

```bash
# List available backends
coratrixc --list-backends

# Show backend information
coratrixc --backend-info local_simulator

# Register custom backend
coratrixc --register-backend backend_config.json
```

### Plugin Management

```bash
# List available plugins
coratrixc --list-plugins

# Show plugin information
coratrixc --plugin-info custom_optimization

# Load plugin from file
coratrixc --load-plugin my_plugin.py
```

### Advanced Usage

```bash
# Compile with specific passes
coratrixc input.qasm --passes "gate_merging,redundant_elimination" --target openqasm

# Execute with custom configuration
coratrixc input.qasm --execute --backend local_simulator --config custom_config.json

# Debug mode
coratrixc input.qasm --debug --target openqasm
```

## Interactive CLI

### Starting Interactive Mode

```bash
# Start interactive shell
coratrixc --interactive

# Start with specific backend
coratrixc --interactive --backend local_simulator
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `state <qubits>` | Create quantum state |
| `gate <gate> <qubits>` | Apply quantum gate |
| `circuit <name>` | Create quantum circuit |
| `execute [shots]` | Execute current circuit |
| `measure <qubits>` | Measure qubits |
| `reset` | Reset to initial state |
| `save <file>` | Save current state |
| `load <file>` | Load quantum state |
| `quit` | Exit interactive mode |

### Interactive Examples

```bash
# Start interactive shell
coratrixc --interactive

# In the shell:
>>> state 3
>>> gate h 0
>>> gate cnot 0 1
>>> execute 1000
>>> measure 0 1
>>> save bell_state.json
>>> quit
```

## Configuration Files

### Global Configuration

```json
{
  "default_backend": "local_simulator",
  "default_shots": 1000,
  "optimization_level": 2,
  "plugin_directories": [
    "./plugins",
    "~/.coratrix/plugins"
  ],
  "backend_configs": {
    "local_simulator": {
      "type": "simulator",
      "max_qubits": 20,
      "supports_noise": true
    }
  }
}
```

### Backend Configuration

```json
{
  "name": "custom_backend",
  "type": "simulator",
  "connection_params": {
    "host": "localhost",
    "port": 8080,
    "timeout": 30
  },
  "capabilities": {
    "max_qubits": 20,
    "max_shots": 1000000,
    "supports_noise": true
  }
}
```

### Noise Model Configuration

```json
{
  "depolarizing_error": 0.01,
  "amplitude_damping_error": 0.005,
  "phase_damping_error": 0.005,
  "readout_error": 0.02,
  "gate_errors": {
    "h": 0.001,
    "cnot": 0.01,
    "measure": 0.02
  }
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CORATRIX_CONFIG` | Path to configuration file |
| `CORATRIX_PLUGIN_DIR` | Plugin directory path |
| `CORATRIX_BACKEND_DIR` | Backend directory path |
| `CORATRIX_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CORATRIX_CACHE_DIR` | Cache directory path |

## Error Handling

### Common Errors

1. **File Not Found**
   ```
   Error: Input file 'input.qasm' not found
   ```
   Solution: Check file path and permissions

2. **Invalid Target Format**
   ```
   Error: Invalid target format 'invalid_format'
   ```
   Solution: Use supported formats (openqasm, qiskit, pennylane)

3. **Backend Not Available**
   ```
   Error: Backend 'custom_backend' not found
   ```
   Solution: Check backend registration and availability

4. **Plugin Loading Failed**
   ```
   Error: Failed to load plugin 'my_plugin'
   ```
   Solution: Check plugin implementation and dependencies

### Debug Mode

```bash
# Enable debug mode
coratrixc input.qasm --debug --target openqasm

# Set log level
export CORATRIX_LOG_LEVEL=DEBUG
coratrixc input.qasm --target openqasm
```

## Performance Optimization

### Compilation Performance

```bash
# Use optimization passes
coratrixc input.qasm --optimize --target openqasm

# Specify optimization level
coratrixc input.qasm --optimize-level 3 --target openqasm

# Use parallel processing
coratrixc input.qasm --parallel --target openqasm
```

### Execution Performance

```bash
# Use GPU acceleration
coratrixc input.qasm --execute --backend gpu_simulator

# Optimize for specific backend
coratrixc input.qasm --execute --backend local_simulator --optimize-backend

# Use sparse representation
coratrixc input.qasm --execute --backend local_simulator --sparse
```

## Integration Examples

### Python Integration

```python
import subprocess

# Compile DSL to OpenQASM
result = subprocess.run([
    'coratrixc', 'input.qasm', '-o', 'output.qasm', 
    '--target', 'openqasm'
], capture_output=True, text=True)

if result.returncode == 0:
    print("Compilation successful")
    print(result.stdout)
else:
    print("Compilation failed")
    print(result.stderr)
```

### Shell Script Integration

```bash
#!/bin/bash

# Compile and execute circuit
coratrixc input.qasm --execute --backend local_simulator --shots 1000

# Check execution result
if [ $? -eq 0 ]; then
    echo "Execution successful"
else
    echo "Execution failed"
    exit 1
fi
```

### Makefile Integration

```makefile
# Makefile
.PHONY: compile execute clean

compile:
	coratrixc input.qasm -o output.qasm --target openqasm

execute: compile
	coratrixc output.qasm --execute --backend local_simulator

clean:
	rm -f output.qasm
```

## Troubleshooting

### Debug Information

```bash
# Show version and build information
coratrixc --version

# Show configuration
coratrixc --show-config

# Show available backends and plugins
coratrixc --list-backends
coratrixc --list-plugins
```

### Log Files

```bash
# Enable logging to file
coratrixc input.qasm --log-file coratrix.log --target openqasm

# View log file
tail -f coratrix.log
```

### Performance Profiling

```bash
# Profile compilation
coratrixc input.qasm --profile --target openqasm

# Profile execution
coratrixc input.qasm --execute --profile --backend local_simulator
```

## Best Practices

### 1. File Organization
- Use descriptive file names
- Organize circuits by project
- Keep configuration files in version control

### 2. Error Handling
- Always check return codes
- Use verbose mode for debugging
- Implement proper error recovery

### 3. Performance
- Use appropriate optimization levels
- Choose suitable backends
- Monitor resource usage

### 4. Security
- Validate input files
- Use secure configuration files
- Limit plugin permissions

## Advanced Features

### Custom Passes

```bash
# Use custom optimization passes
coratrixc input.qasm --passes "custom_pass1,custom_pass2" --target openqasm
```

### Plugin Development

```bash
# Load development plugin
coratrixc --load-plugin dev_plugin.py --debug
```

### Backend Development

```bash
# Register development backend
coratrixc --register-backend dev_backend.json --debug
```

## Support and Resources

### Documentation
- Architecture Guide: `docs/MODULAR_ARCHITECTURE.md`
- Plugin Development: `docs/PLUGIN_DEVELOPMENT.md`
- API Reference: `docs/API_REFERENCE.md`

### Examples
- Basic examples: `examples/`
- Advanced examples: `examples/advanced/`
- Plugin examples: `coratrix/plugins/example_*.py`

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Contributing: Submit pull requests and improvements
