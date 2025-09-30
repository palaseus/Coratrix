# Plugin Development Guide

## Overview

Coratrix 3.1 features a comprehensive plugin system that allows developers to extend the quantum computing platform with custom components. This guide covers plugin development, registration, and best practices.

## Plugin Types

### 1. Compiler Pass Plugins
Custom optimization and transformation passes for the compiler stack.

### 2. Backend Plugins
Custom quantum backends for simulation and hardware execution.

### 3. DSL Extension Plugins
Extensions to the domain-specific language for custom gates and syntax.

### 4. Target Generator Plugins
Custom target code generators for new quantum frameworks.

## Plugin Architecture

### Base Plugin Class
All plugins inherit from the base `Plugin` class:

```python
from coratrix.plugins import Plugin, PluginInfo

class MyPlugin(Plugin):
    def __init__(self):
        super().__init__(
            info=PluginInfo(
                name='my_plugin',
                version='1.0.0',
                description='My custom plugin',
                author='Your Name',
                plugin_type='compiler_pass',
                dependencies=[]
            )
        )
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        return True
    
    def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        return True
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self.initialized
```

## Compiler Pass Plugins

### Creating a Compiler Pass Plugin

```python
from coratrix.plugins import CompilerPassPlugin
from coratrix.compiler.passes import CompilerPass, PassResult
from coratrix.compiler.ir import CoratrixIR, IRStatement, IROperation

class CustomOptimizationPass(CompilerPass):
    """Custom optimization pass."""
    
    def run(self, ir: CoratrixIR) -> PassResult:
        """Run the optimization pass."""
        optimized_ir = self._optimize_circuit(ir)
        return PassResult(success=True, ir=optimized_ir)
    
    def _optimize_circuit(self, ir: CoratrixIR):
        """Apply custom optimizations."""
        # Custom optimization logic
        return ir

class CustomOptimizationPlugin(CompilerPassPlugin):
    """Plugin wrapper for custom optimization pass."""
    
    def __init__(self):
        super().__init__(
            info=PluginInfo(
                name='custom_optimization',
                version='1.0.0',
                description='Custom circuit optimization',
                author='Your Name',
                plugin_type='compiler_pass',
                dependencies=[]
            )
        )
        self.pass_instance = None
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.pass_instance = CustomOptimizationPass()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize custom optimization plugin: {e}")
            return False
    
    def get_pass(self):
        """Get the compiler pass instance."""
        return self.pass_instance
    
    def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.pass_instance = None
            self.initialized = False
            return True
        except Exception as e:
            print(f"Failed to cleanup custom optimization plugin: {e}")
            return False
```

### Registering Compiler Pass Plugins

```python
from coratrix.plugins import PluginManager

# Create plugin manager
plugin_manager = PluginManager()

# Create and register plugin
plugin = CustomOptimizationPlugin()
success = plugin_manager.register_plugin(plugin)

if success:
    print("Plugin registered successfully")
else:
    print("Failed to register plugin")
```

## Backend Plugins

### Creating a Backend Plugin

```python
from coratrix.plugins import BackendPlugin
from coratrix.backend import BackendInterface, BackendConfiguration, BackendType, BackendResult
import numpy as np

class CustomSimulatorBackend(BackendInterface):
    """Custom simulator backend."""
    
    def __init__(self, config: BackendConfiguration):
        super().__init__(config)
        self.name = config.name
        self.capabilities = {
            'max_qubits': 20,
            'max_shots': 1000000,
            'supports_noise': True
        }
    
    def execute(self, circuit, shots=1000, **kwargs):
        """Execute quantum circuit."""
        # Custom simulation logic
        result = {
            'success': True,
            'counts': {'00': shots//2, '11': shots//2},
            'execution_time': 0.001,
            'backend': self.name
        }
        return BackendResult(**result)
    
    def get_capabilities(self):
        """Get backend capabilities."""
        return self.capabilities

class CustomBackendPlugin(BackendPlugin):
    """Plugin wrapper for custom backend."""
    
    def __init__(self):
        super().__init__(
            info=PluginInfo(
                name='custom_simulator',
                version='1.0.0',
                description='Custom quantum simulator',
                author='Your Name',
                plugin_type='backend',
                dependencies=[]
            )
        )
        self.backend_instance = None
        self.backend_config = None
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.backend_config = BackendConfiguration(
                name='custom_simulator',
                backend_type=BackendType.SIMULATOR,
                connection_params={'simulator_type': 'custom'}
            )
            self.backend_instance = CustomSimulatorBackend(self.backend_config)
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize custom backend plugin: {e}")
            return False
    
    def get_backend(self):
        """Get the backend instance."""
        return self.backend_instance
    
    def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.backend_instance = None
            self.backend_config = None
            self.initialized = False
            return True
        except Exception as e:
            print(f"Failed to cleanup custom backend plugin: {e}")
            return False
```

## DSL Extension Plugins

### Creating a DSL Extension Plugin

```python
from coratrix.plugins import DSLExtensionPlugin

class CustomGatePlugin(DSLExtensionPlugin):
    """Plugin for custom quantum gates."""
    
    def __init__(self):
        super().__init__(
            info=PluginInfo(
                name='custom_gates',
                version='1.0.0',
                description='Custom quantum gates',
                author='Your Name',
                plugin_type='dsl_extension',
                dependencies=[]
            )
        )
        self.custom_gates = {}
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            # Register custom gates
            self.custom_gates = {
                'custom_rotation': self._create_custom_rotation_gate,
                'custom_entangling': self._create_custom_entangling_gate
            }
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize custom gates plugin: {e}")
            return False
    
    def extend_dsl(self, parser):
        """Extend the DSL parser with custom gates."""
        for gate_name, gate_factory in self.custom_gates.items():
            parser.add_gate(gate_name, gate_factory)
    
    def _create_custom_rotation_gate(self, params):
        """Create custom rotation gate."""
        # Custom gate implementation
        pass
    
    def _create_custom_entangling_gate(self, params):
        """Create custom entangling gate."""
        # Custom gate implementation
        pass
```

## Target Generator Plugins

### Creating a Target Generator Plugin

```python
from coratrix.plugins import TargetGeneratorPlugin
from coratrix.compiler.targets import TargetGenerator, TargetResult

class CustomTargetGenerator(TargetGenerator):
    """Custom target code generator."""
    
    def generate(self, ir, options):
        """Generate target code from IR."""
        # Custom code generation logic
        code = self._generate_custom_code(ir)
        return TargetResult(success=True, code=code)
    
    def _generate_custom_code(self, ir):
        """Generate custom target code."""
        # Custom code generation implementation
        return "// Custom target code"

class CustomTargetPlugin(TargetGeneratorPlugin):
    """Plugin wrapper for custom target generator."""
    
    def __init__(self):
        super().__init__(
            info=PluginInfo(
                name='custom_target',
                version='1.0.0',
                description='Custom target code generator',
                author='Your Name',
                plugin_type='target_generator',
                dependencies=[]
            )
        )
        self.generator_instance = None
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.generator_instance = CustomTargetGenerator()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize custom target plugin: {e}")
            return False
    
    def get_generator(self):
        """Get the target generator instance."""
        return self.generator_instance
```

## Plugin Registration and Management

### Plugin Manager Usage

```python
from coratrix.plugins import PluginManager

# Create plugin manager
plugin_manager = PluginManager()

# Register plugins
plugins = [
    CustomOptimizationPlugin(),
    CustomBackendPlugin(),
    CustomGatePlugin(),
    CustomTargetPlugin()
]

for plugin in plugins:
    success = plugin_manager.register_plugin(plugin)
    if success:
        print(f"Registered {plugin.info.name}")
    else:
        print(f"Failed to register {plugin.info.name}")

# List registered plugins
print("Registered plugins:", plugin_manager.list_plugins())

# Get plugins by type
compiler_plugins = plugin_manager.get_plugins_by_type('compiler_pass')
backend_plugins = plugin_manager.get_plugins_by_type('backend')

# Unregister plugin
plugin_manager.unregister_plugin('custom_optimization')
```

### Plugin Discovery

```python
# Load plugins from directory
loaded_count = plugin_manager.load_plugins_from_directory('./plugins')
print(f"Loaded {loaded_count} plugins")

# Load all plugins from configured directories
total_loaded = plugin_manager.load_all_plugins()
print(f"Total loaded: {total_loaded} plugins")
```

## Best Practices

### 1. Plugin Design
- Keep plugins focused and single-purpose
- Use clear, descriptive names
- Implement proper error handling
- Provide comprehensive documentation

### 2. Dependencies
- Minimize external dependencies
- Use version pinning for critical dependencies
- Document all dependencies clearly

### 3. Testing
- Write unit tests for plugin functionality
- Test plugin registration and initialization
- Test plugin cleanup and resource management

### 4. Error Handling
- Implement robust error handling
- Provide meaningful error messages
- Handle initialization failures gracefully

### 5. Performance
- Optimize plugin performance
- Monitor resource usage
- Implement efficient algorithms

## Plugin Packaging

### Directory Structure
```
my_plugin/
├── __init__.py
├── plugin.py
├── tests/
│   └── test_plugin.py
├── requirements.txt
└── README.md
```

### Plugin Entry Point
```python
# __init__.py
from .plugin import MyPlugin

__all__ = ['MyPlugin']
```

### Installation
```python
# Install plugin
pip install my_plugin

# Register plugin
from my_plugin import MyPlugin
plugin_manager.register_plugin(MyPlugin())
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Use absolute imports
   - Check Python path
   - Verify module structure

2. **Plugin Registration Failures**
   - Check plugin initialization
   - Verify plugin info
   - Check for naming conflicts

3. **Runtime Errors**
   - Implement proper error handling
   - Check plugin dependencies
   - Monitor resource usage

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable plugin debug logging
plugin_manager.debug = True
```

## Examples

### Complete Plugin Example
See `coratrix/plugins/example_optimization_pass.py` and `coratrix/plugins/example_custom_backend.py` for complete working examples.

### Plugin Testing
```python
import unittest
from coratrix.plugins import PluginManager

class TestMyPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin_manager = PluginManager()
        self.plugin = MyPlugin()
    
    def test_plugin_initialization(self):
        self.assertTrue(self.plugin.initialize())
        self.assertTrue(self.plugin.is_enabled())
    
    def test_plugin_registration(self):
        success = self.plugin_manager.register_plugin(self.plugin)
        self.assertTrue(success)
    
    def test_plugin_cleanup(self):
        self.plugin.initialize()
        self.assertTrue(self.plugin.cleanup())
```

## Contributing

### Guidelines
- Follow plugin interface contracts
- Maintain backward compatibility
- Write comprehensive tests
- Document all functionality
- Submit clean, well-documented code

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Implement plugin
4. Write tests
5. Update documentation
6. Submit pull request
