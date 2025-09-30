"""
Visual Plugin Editor for Coratrix 4.0

This module provides a web-based and CLI-driven interface for creating,
testing, and sharing custom compiler passes, gates, and backends.
"""

import json
import os
import sys
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod
import subprocess
import webbrowser
from pathlib import Path

# Web framework imports
try:
    from flask import Flask, render_template, request, jsonify, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    render_template = None
    request = None
    jsonify = None
    send_file = None
    CORS = None

# Template engine imports
try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    jinja2 = None

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins that can be created."""
    COMPILER_PASS = "compiler_pass"
    QUANTUM_GATE = "quantum_gate"
    BACKEND = "backend"
    VISUALIZATION = "visualization"
    NOISE_MODEL = "noise_model"
    OPTIMIZER = "optimizer"


class PluginTemplate(Enum):
    """Available plugin templates."""
    BASIC_GATE = "basic_gate"
    PARAMETERIZED_GATE = "parameterized_gate"
    OPTIMIZATION_PASS = "optimization_pass"
    CUSTOM_BACKEND = "custom_backend"
    NOISE_CHANNEL = "noise_channel"
    VISUALIZATION_COMPONENT = "visualization_component"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    tags: List[str]
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None


@dataclass
class PluginTemplate:
    """Template for plugin generation."""
    name: str
    description: str
    plugin_type: PluginType
    template_files: List[str]
    required_fields: List[str]
    optional_fields: List[str]


class PluginEditor:
    """
    Main plugin editor class for creating and managing plugins.
    """
    
    def __init__(self, output_dir: str = "plugins"):
        """
        Initialize plugin editor.
        
        Args:
            output_dir: Directory to save generated plugins
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.templates = self._load_templates()
        self.plugins = self._load_existing_plugins()
        
        # Initialize web server if Flask is available
        if FLASK_AVAILABLE:
            self.app = self._create_web_app()
        else:
            self.app = None
            logger.warning("Flask not available, web interface disabled")
    
    def _load_templates(self) -> Dict[str, PluginTemplate]:
        """Load available plugin templates."""
        templates = {}
        
        # Basic gate template
        templates["basic_gate"] = PluginTemplate(
            name="Basic Gate",
            description="Template for creating a basic quantum gate",
            plugin_type=PluginType.QUANTUM_GATE,
            template_files=["gate.py", "test_gate.py", "README.md"],
            required_fields=["gate_name", "gate_matrix"],
            optional_fields=["gate_description", "gate_parameters"]
        )
        
        # Parameterized gate template
        templates["parameterized_gate"] = PluginTemplate(
            name="Parameterized Gate",
            description="Template for creating a parameterized quantum gate",
            plugin_type=PluginType.QUANTUM_GATE,
            template_files=["gate.py", "test_gate.py", "README.md"],
            required_fields=["gate_name", "gate_function", "parameters"],
            optional_fields=["gate_description", "parameter_bounds"]
        )
        
        # Optimization pass template
        templates["optimization_pass"] = PluginTemplate(
            name="Optimization Pass",
            description="Template for creating a compiler optimization pass",
            plugin_type=PluginType.COMPILER_PASS,
            template_files=["pass.py", "test_pass.py", "README.md"],
            required_fields=["pass_name", "optimization_function"],
            optional_fields=["pass_description", "optimization_level"]
        )
        
        # Custom backend template
        templates["custom_backend"] = PluginTemplate(
            name="Custom Backend",
            description="Template for creating a custom quantum backend",
            plugin_type=PluginType.BACKEND,
            template_files=["backend.py", "test_backend.py", "README.md"],
            required_fields=["backend_name", "backend_type", "execution_function"],
            optional_fields=["backend_description", "capabilities"]
        )
        
        # Noise channel template
        templates["noise_channel"] = PluginTemplate(
            name="Noise Channel",
            description="Template for creating a custom noise channel",
            plugin_type=PluginType.NOISE_MODEL,
            template_files=["noise.py", "test_noise.py", "README.md"],
            required_fields=["channel_name", "channel_matrix"],
            optional_fields=["channel_description", "error_rate"]
        )
        
        # Visualization component template
        templates["visualization_component"] = PluginTemplate(
            name="Visualization Component",
            description="Template for creating a visualization component",
            plugin_type=PluginType.VISUALIZATION,
            template_files=["visualization.py", "test_visualization.py", "README.md"],
            required_fields=["component_name", "visualization_function"],
            optional_fields=["component_description", "dependencies"]
        )
        
        return templates
    
    def _load_existing_plugins(self) -> Dict[str, PluginMetadata]:
        """Load existing plugins from the output directory."""
        plugins = {}
        
        for plugin_dir in self.output_dir.iterdir():
            if plugin_dir.is_dir():
                metadata_file = plugin_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                            metadata = PluginMetadata(**metadata_dict)
                            plugins[metadata.name] = metadata
                    except Exception as e:
                        logger.warning(f"Failed to load plugin metadata from {plugin_dir}: {e}")
        
        return plugins
    
    def create_plugin(self, template_name: str, plugin_name: str, 
                     metadata: PluginMetadata, custom_fields: Dict[str, Any] = None) -> str:
        """
        Create a new plugin from template.
        
        Args:
            template_name: Name of the template to use
            plugin_name: Name for the new plugin
            metadata: Plugin metadata
            custom_fields: Custom fields for template generation
            
        Returns:
            Path to created plugin directory
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        plugin_dir = self.output_dir / plugin_name
        plugin_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_file = plugin_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Generate files from template
        for template_file in template.template_files:
            self._generate_file_from_template(
                template, template_file, plugin_dir, 
                metadata, custom_fields or {}
            )
        
        # Add to plugins registry
        self.plugins[plugin_name] = metadata
        
        logger.info(f"Created plugin '{plugin_name}' from template '{template_name}'")
        return str(plugin_dir)
    
    def _generate_file_from_template(self, template: PluginTemplate, 
                                   template_file: str, plugin_dir: Path,
                                   metadata: PluginMetadata, custom_fields: Dict[str, Any]):
        """Generate a file from template."""
        # Get template content
        template_content = self._get_template_content(template.plugin_type, template_file)
        
        # Replace placeholders
        content = self._replace_template_placeholders(
            template_content, metadata, custom_fields
        )
        
        # Write file
        output_file = plugin_dir / template_file
        with open(output_file, 'w') as f:
            f.write(content)
    
    def _get_template_content(self, plugin_type: PluginType, template_file: str) -> str:
        """Get template content for a specific file."""
        templates = {
            PluginType.QUANTUM_GATE: {
                "gate.py": '''"""
{gate_name} - Custom Quantum Gate

{gate_description}
"""

import numpy as np
from typing import List, Optional, Union
from core.gates import QuantumGate


class {gate_name}Gate(QuantumGate):
    """
    {gate_name} quantum gate implementation.
    """
    
    def __init__(self, {gate_parameters}):
        """
        Initialize {gate_name} gate.
        
        Args:
            {gate_parameters_doc}
        """
        super().__init__()
        {gate_initialization}
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the matrix representation of the gate.
        
        Returns:
            Gate matrix
        """
        {gate_matrix_code}
    
    def get_name(self) -> str:
        """Get gate name."""
        return "{gate_name}"
    
    def get_qubits(self) -> List[int]:
        """Get qubits this gate acts on."""
        return {gate_qubits}
    
    def is_parameterized(self) -> bool:
        """Check if gate is parameterized."""
        return {is_parameterized}
    
    def get_parameters(self) -> List[float]:
        """Get gate parameters."""
        return {gate_parameters_list}
''',
                "test_gate.py": '''"""
Tests for {gate_name} gate.
"""

import pytest
import numpy as np
from {gate_name.lower()}_gate import {gate_name}Gate


class Test{gate_name}Gate:
    """Test cases for {gate_name} gate."""
    
    def test_gate_initialization(self):
        """Test gate initialization."""
        gate = {gate_name}Gate({test_parameters})
        assert gate.get_name() == "{gate_name}"
        assert gate.is_parameterized() == {is_parameterized}
    
    def test_gate_matrix(self):
        """Test gate matrix."""
        gate = {gate_name}Gate({test_parameters})
        matrix = gate.get_matrix()
        
        # Check matrix properties
        assert matrix.shape == {matrix_shape}
        assert np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]))
    
    def test_gate_parameters(self):
        """Test gate parameters."""
        gate = {gate_name}Gate({test_parameters})
        parameters = gate.get_parameters()
        assert len(parameters) == {num_parameters}
    
    def test_gate_qubits(self):
        """Test gate qubits."""
        gate = {gate_name}Gate({test_parameters})
        qubits = gate.get_qubits()
        assert len(qubits) == {num_qubits}


if __name__ == "__main__":
    pytest.main([__file__])
''',
                "README.md": '''# {gate_name} Gate Plugin

{gate_description}

## Installation

```bash
pip install -e .
```

## Usage

```python
from {gate_name.lower()}_gate import {gate_name}Gate

# Create gate instance
gate = {gate_name}Gate({usage_example})

# Use in quantum circuit
from core.circuit import QuantumCircuit
circuit = QuantumCircuit(2)
circuit.add_gate(gate, [0, 1])
```

## API Reference

### {gate_name}Gate

{gate_description}

#### Parameters

{parameters_doc}

#### Methods

- `get_matrix()`: Get gate matrix representation
- `get_name()`: Get gate name
- `get_qubits()`: Get qubits this gate acts on
- `is_parameterized()`: Check if gate is parameterized
- `get_parameters()`: Get gate parameters

## Testing

```bash
python -m pytest test_{gate_name.lower()}_gate.py
```

## License

{license}
'''
            },
            PluginType.COMPILER_PASS: {
                "pass.py": '''"""
{pass_name} - Custom Compiler Pass

{pass_description}
"""

from typing import List, Dict, Any, Optional
from compiler.passes import CompilerPass, PassResult
from compiler.ir import CoratrixIR, IRStatement


class {pass_name}Pass(CompilerPass):
    """
    {pass_name} compiler pass implementation.
    """
    
    def __init__(self, {pass_parameters}):
        """
        Initialize {pass_name} pass.
        
        Args:
            {pass_parameters_doc}
        """
        super().__init__()
        {pass_initialization}
    
    def apply(self, ir: CoratrixIR) -> PassResult:
        """
        Apply the pass to the IR.
        
        Args:
            ir: Coratrix IR to process
            
        Returns:
            Pass result
        """
        {pass_implementation}
    
    def get_name(self) -> str:
        """Get pass name."""
        return "{pass_name}"
    
    def get_description(self) -> str:
        """Get pass description."""
        return "{pass_description}"
    
    def get_optimization_level(self) -> int:
        """Get optimization level."""
        return {optimization_level}
''',
                "test_pass.py": '''"""
Tests for {pass_name} pass.
"""

import pytest
from compiler.ir import CoratrixIR, IRStatement
from {pass_name.lower()}_pass import {pass_name}Pass


class Test{pass_name}Pass:
    """Test cases for {pass_name} pass."""
    
    def test_pass_initialization(self):
        """Test pass initialization."""
        pass_instance = {pass_name}Pass({test_parameters})
        assert pass_instance.get_name() == "{pass_name}"
        assert pass_instance.get_optimization_level() == {optimization_level}
    
    def test_pass_application(self):
        """Test pass application."""
        pass_instance = {pass_name}Pass({test_parameters})
        
        # Create test IR
        ir = CoratrixIR()
        {test_ir_creation}
        
        # Apply pass
        result = pass_instance.apply(ir)
        
        # Check result
        assert result.success
        assert result.modified
        {test_assertions}
    
    def test_pass_optimization(self):
        """Test pass optimization."""
        pass_instance = {pass_name}Pass({test_parameters})
        
        # Test optimization level
        assert pass_instance.get_optimization_level() >= 0
        assert pass_instance.get_optimization_level() <= 3


if __name__ == "__main__":
    pytest.main([__file__])
''',
                "README.md": '''# {pass_name} Pass Plugin

{pass_description}

## Installation

```bash
pip install -e .
```

## Usage

```python
from {pass_name.lower()}_pass import {pass_name}Pass

# Create pass instance
pass_instance = {pass_name}Pass({usage_example})

# Use in compiler
from compiler.compiler import CoratrixCompiler
compiler = CoratrixCompiler()
compiler.add_pass(pass_instance)
```

## API Reference

### {pass_name}Pass

{pass_description}

#### Parameters

{parameters_doc}

#### Methods

- `apply(ir)`: Apply pass to IR
- `get_name()`: Get pass name
- `get_description()`: Get pass description
- `get_optimization_level()`: Get optimization level

## Testing

```bash
python -m pytest test_{pass_name.lower()}_pass.py
```

## License

{license}
'''
            }
        }
        
        return templates.get(plugin_type, {}).get(template_file, "")
    
    def _replace_template_placeholders(self, content: str, metadata: PluginMetadata, 
                                    custom_fields: Dict[str, Any]) -> str:
        """Replace placeholders in template content."""
        # Replace metadata placeholders
        content = content.replace("{gate_name}", metadata.name)
        content = content.replace("{pass_name}", metadata.name)
        content = content.replace("{gate_description}", metadata.description)
        content = content.replace("{pass_description}", metadata.description)
        content = content.replace("{license}", metadata.license)
        
        # Replace custom fields
        for key, value in custom_fields.items():
            placeholder = "{" + key + "}"
            content = content.replace(placeholder, str(value))
        
        return content
    
    def _create_web_app(self) -> Flask:
        """Create Flask web application."""
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/')
        def index():
            return render_template('plugin_editor.html')
        
        @app.route('/api/templates')
        def get_templates():
            return jsonify({
                'templates': {name: {
                    'name': template.name,
                    'description': template.description,
                    'plugin_type': template.plugin_type.value,
                    'required_fields': template.required_fields,
                    'optional_fields': template.optional_fields
                } for name, template in self.templates.items()}
            })
        
        @app.route('/api/plugins')
        def get_plugins():
            return jsonify({
                'plugins': {name: asdict(metadata) for name, metadata in self.plugins.items()}
            })
        
        @app.route('/api/create-plugin', methods=['POST'])
        def create_plugin():
            data = request.json
            
            # Extract data
            template_name = data['template']
            plugin_name = data['name']
            metadata = PluginMetadata(**data['metadata'])
            custom_fields = data.get('custom_fields', {})
            
            try:
                plugin_path = self.create_plugin(template_name, plugin_name, metadata, custom_fields)
                return jsonify({'success': True, 'path': plugin_path})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/test-plugin', methods=['POST'])
        def test_plugin():
            data = request.json
            plugin_name = data['name']
            
            try:
                result = self.test_plugin(plugin_name)
                return jsonify({'success': True, 'result': result})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        return app
    
    def test_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Test a plugin.
        
        Args:
            plugin_name: Name of plugin to test
            
        Returns:
            Test results
        """
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        
        plugin_dir = self.output_dir / plugin_name
        
        # Run tests
        test_results = {
            'plugin_name': plugin_name,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_output': '',
            'success': False
        }
        
        try:
            # Find test files
            test_files = list(plugin_dir.glob('test_*.py'))
            
            if not test_files:
                test_results['test_output'] = "No test files found"
                return test_results
            
            # Run pytest
            cmd = ['python', '-m', 'pytest', str(plugin_dir), '-v']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            test_results['test_output'] = result.stdout + result.stderr
            test_results['success'] = result.returncode == 0
            
            # Parse test results
            if 'passed' in result.stdout:
                test_results['tests_passed'] = result.stdout.count('PASSED')
            if 'failed' in result.stdout:
                test_results['tests_failed'] = result.stdout.count('FAILED')
            
        except Exception as e:
            test_results['test_output'] = f"Error running tests: {e}"
            test_results['success'] = False
        
        return test_results
    
    def start_web_server(self, host: str = 'localhost', port: int = 5000, 
                        debug: bool = False) -> None:
        """Start the web server."""
        if not self.app:
            raise RuntimeError("Web interface not available (Flask not installed)")
        
        print(f"Starting plugin editor web server at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def open_web_interface(self, host: str = 'localhost', port: int = 5000) -> None:
        """Open web interface in browser."""
        url = f"http://{host}:{port}"
        webbrowser.open(url)
        print(f"Opening web interface at {url}")


class CLIPluginEditor:
    """
    Command-line interface for plugin editor.
    """
    
    def __init__(self, editor: PluginEditor):
        """
        Initialize CLI plugin editor.
        
        Args:
            editor: Plugin editor instance
        """
        self.editor = editor
    
    def run_interactive(self):
        """Run interactive CLI."""
        print("=== Coratrix Plugin Editor ===")
        print("Create custom plugins for Coratrix quantum computing platform")
        print()
        
        while True:
            print("\nOptions:")
            print("1. Create new plugin")
            print("2. List available templates")
            print("3. List existing plugins")
            print("4. Test plugin")
            print("5. Start web interface")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self._create_plugin_interactive()
            elif choice == '2':
                self._list_templates()
            elif choice == '3':
                self._list_plugins()
            elif choice == '4':
                self._test_plugin_interactive()
            elif choice == '5':
                self._start_web_interface()
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _create_plugin_interactive(self):
        """Create plugin interactively."""
        print("\n=== Create New Plugin ===")
        
        # Select template
        print("\nAvailable templates:")
        for i, (name, template) in enumerate(self.editor.templates.items(), 1):
            print(f"{i}. {template.name} - {template.description}")
        
        try:
            template_choice = int(input("\nSelect template (number): ")) - 1
            template_names = list(self.editor.templates.keys())
            template_name = template_names[template_choice]
        except (ValueError, IndexError):
            print("Invalid template selection")
            return
        
        template = self.editor.templates[template_name]
        
        # Get plugin metadata
        print(f"\nCreating {template.name} plugin...")
        
        plugin_name = input("Plugin name: ").strip()
        if not plugin_name:
            print("Plugin name is required")
            return
        
        description = input("Description: ").strip()
        author = input("Author: ").strip()
        license_type = input("License (default: MIT): ").strip() or "MIT"
        
        # Get custom fields
        custom_fields = {}
        print(f"\nRequired fields for {template.name}:")
        for field in template.required_fields:
            value = input(f"{field}: ").strip()
            if value:
                custom_fields[field] = value
        
        print(f"\nOptional fields for {template.name}:")
        for field in template.optional_fields:
            value = input(f"{field} (optional): ").strip()
            if value:
                custom_fields[field] = value
        
        # Create metadata
        metadata = PluginMetadata(
            name=plugin_name,
            version="1.0.0",
            description=description,
            author=author,
            plugin_type=template.plugin_type,
            dependencies=[],
            tags=[],
            license=license_type
        )
        
        try:
            plugin_path = self.editor.create_plugin(template_name, plugin_name, metadata, custom_fields)
            print(f"\nPlugin created successfully at: {plugin_path}")
            
            # Ask if user wants to test the plugin
            test_choice = input("\nTest the plugin now? (y/n): ").strip().lower()
            if test_choice == 'y':
                self._test_plugin(plugin_name)
                
        except Exception as e:
            print(f"Error creating plugin: {e}")
    
    def _list_templates(self):
        """List available templates."""
        print("\n=== Available Templates ===")
        
        for name, template in self.editor.templates.items():
            print(f"\n{template.name} ({name})")
            print(f"  Type: {template.plugin_type.value}")
            print(f"  Description: {template.description}")
            print(f"  Required fields: {', '.join(template.required_fields)}")
            print(f"  Optional fields: {', '.join(template.optional_fields)}")
    
    def _list_plugins(self):
        """List existing plugins."""
        print("\n=== Existing Plugins ===")
        
        if not self.editor.plugins:
            print("No plugins found")
            return
        
        for name, metadata in self.editor.plugins.items():
            print(f"\n{metadata.name} (v{metadata.version})")
            print(f"  Type: {metadata.plugin_type.value}")
            print(f"  Author: {metadata.author}")
            print(f"  Description: {metadata.description}")
            print(f"  License: {metadata.license}")
    
    def _test_plugin_interactive(self):
        """Test plugin interactively."""
        print("\n=== Test Plugin ===")
        
        if not self.editor.plugins:
            print("No plugins available for testing")
            return
        
        print("\nAvailable plugins:")
        for i, (name, metadata) in enumerate(self.editor.plugins.items(), 1):
            print(f"{i}. {metadata.name} - {metadata.description}")
        
        try:
            plugin_choice = int(input("\nSelect plugin to test (number): ")) - 1
            plugin_names = list(self.editor.plugins.keys())
            plugin_name = plugin_names[plugin_choice]
        except (ValueError, IndexError):
            print("Invalid plugin selection")
            return
        
        self._test_plugin(plugin_name)
    
    def _test_plugin(self, plugin_name: str):
        """Test a specific plugin."""
        print(f"\nTesting plugin: {plugin_name}")
        
        try:
            result = self.editor.test_plugin(plugin_name)
            
            print(f"\nTest Results:")
            print(f"  Success: {result['success']}")
            print(f"  Tests passed: {result['tests_passed']}")
            print(f"  Tests failed: {result['tests_failed']}")
            
            if result['test_output']:
                print(f"\nTest output:")
                print(result['test_output'])
                
        except Exception as e:
            print(f"Error testing plugin: {e}")
    
    def _start_web_interface(self):
        """Start web interface."""
        if not self.editor.app:
            print("Web interface not available (Flask not installed)")
            return
        
        print("Starting web interface...")
        print("Press Ctrl+C to stop the server")
        
        try:
            self.editor.start_web_server()
        except KeyboardInterrupt:
            print("\nWeb server stopped")


# Main functions
def create_plugin_editor(output_dir: str = "plugins") -> PluginEditor:
    """Create a plugin editor instance."""
    return PluginEditor(output_dir)


def run_cli_editor(output_dir: str = "plugins"):
    """Run the CLI plugin editor."""
    editor = create_plugin_editor(output_dir)
    cli = CLIPluginEditor(editor)
    cli.run_interactive()


def run_web_editor(output_dir: str = "plugins", host: str = 'localhost', port: int = 5000):
    """Run the web plugin editor."""
    editor = create_plugin_editor(output_dir)
    
    if not editor.app:
        print("Web interface not available (Flask not installed)")
        print("Please install Flask: pip install flask flask-cors")
        return
    
    print("Starting web interface...")
    editor.start_web_server(host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coratrix Plugin Editor")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", 
                      help="Editor mode (cli or web)")
    parser.add_argument("--output-dir", default="plugins", 
                      help="Output directory for plugins")
    parser.add_argument("--host", default="localhost", 
                      help="Host for web interface")
    parser.add_argument("--port", type=int, default=5000, 
                      help="Port for web interface")
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        run_cli_editor(args.output_dir)
    elif args.mode == "web":
        run_web_editor(args.output_dir, args.host, args.port)
