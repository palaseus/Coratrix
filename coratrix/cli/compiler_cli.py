"""
Coratrix Compiler CLI (coratrixc)

Command-line interface for compiling DSL to target formats and executing on backends.
"""

import argparse
import sys
import os
from typing import List, Optional
from pathlib import Path

from ..compiler import CoratrixCompiler, CompilerOptions, CompilerMode
from ..backend import BackendManager, BackendConfiguration, BackendType
from ..plugins import PluginManager


class CoratrixCompilerCLI:
    """Command-line interface for the Coratrix compiler."""
    
    def __init__(self):
        self.compiler = CoratrixCompiler()
        self.backend_manager = BackendManager()
        self.plugin_manager = PluginManager()
        
        # Load plugins
        self.plugin_manager.load_all_plugins()
        
        # Register plugin backends
        self._register_plugin_backends()
    
    def _register_plugin_backends(self):
        """Register backends from plugins."""
        backend_plugins = self.plugin_manager.get_plugins_by_type('backend')
        for plugin in backend_plugins:
            if hasattr(plugin, 'get_backend_config'):
                config = plugin.get_backend_config()
                if config:
                    self.backend_manager.register_backend(config.name, plugin)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog='coratrixc',
            description='Coratrix Quantum Compiler - Compile DSL to target formats',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  coratrixc input.qasm -o output.py --target qiskit
  coratrixc input.qasm --execute --backend local_simulator
  coratrixc input.qasm --optimize --target openqasm
  coratrixc --list-backends
  coratrixc --list-plugins
            """
        )
        
        # Input/Output
        parser.add_argument('input', nargs='?', help='Input DSL file')
        parser.add_argument('-o', '--output', help='Output file')
        parser.add_argument('--target', choices=['openqasm', 'qiskit', 'pennylane', 'cirq'], 
                          default='openqasm', help='Target format')
        
        # Compilation options
        parser.add_argument('--optimize', action='store_true', 
                          help='Enable optimization passes')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        
        # Execution options
        parser.add_argument('--execute', action='store_true',
                          help='Execute the compiled circuit')
        parser.add_argument('--backend', help='Backend for execution')
        parser.add_argument('--shots', type=int, default=1024,
                          help='Number of shots for execution')
        parser.add_argument('--parameters', nargs='*', 
                          help='Parameters for the circuit')
        
        # Information options
        parser.add_argument('--list-backends', action='store_true',
                          help='List available backends')
        parser.add_argument('--list-plugins', action='store_true',
                          help='List available plugins')
        parser.add_argument('--backend-info', help='Show backend information')
        parser.add_argument('--plugin-info', help='Show plugin information')
        
        # Plugin management
        parser.add_argument('--load-plugin', help='Load a plugin')
        parser.add_argument('--unload-plugin', help='Unload a plugin')
        parser.add_argument('--plugin-dir', help='Plugin directory')
        
        return parser
    
    def list_backends(self):
        """List available backends."""
        print("Available Backends:")
        print("=" * 50)
        
        backends = self.backend_manager.list_backends()
        if not backends:
            print("No backends available.")
            return
        
        for backend_name in backends:
            status = self.backend_manager.get_backend_status(backend_name)
            backend = self.backend_manager.get_backend(backend_name)
            
            print(f"  {backend_name}")
            print(f"    Status: {status}")
            if backend:
                capabilities = backend.get_capabilities()
                print(f"    Max Qubits: {capabilities.max_qubits}")
                print(f"    Max Shots: {capabilities.max_shots}")
                print(f"    Supports Noise: {capabilities.supports_noise}")
            print()
    
    def list_plugins(self):
        """List available plugins."""
        print("Available Plugins:")
        print("=" * 50)
        
        plugins = self.plugin_manager.list_plugins()
        if not plugins:
            print("No plugins available.")
            return
        
        for plugin_name in plugins:
            plugin = self.plugin_manager.get_plugin(plugin_name)
            if plugin:
                print(f"  {plugin_name}")
                print(f"    Type: {plugin.info.plugin_type}")
                print(f"    Version: {plugin.info.version}")
                print(f"    Description: {plugin.info.description}")
                print(f"    Enabled: {plugin.is_enabled()}")
                print()
    
    def show_backend_info(self, backend_name: str):
        """Show detailed backend information."""
        backend = self.backend_manager.get_backend(backend_name)
        if not backend:
            print(f"Backend '{backend_name}' not found.")
            return
        
        print(f"Backend Information: {backend_name}")
        print("=" * 50)
        
        status = self.backend_manager.get_backend_status(backend_name)
        print(f"Status: {status}")
        
        capabilities = backend.get_capabilities()
        print(f"Max Qubits: {capabilities.max_qubits}")
        print(f"Max Shots: {capabilities.max_shots}")
        print(f"Supports Noise: {capabilities.supports_noise}")
        print(f"Supports Parametric Circuits: {capabilities.supports_parametric_circuits}")
        print(f"Supports Measurement: {capabilities.supports_measurement}")
        print(f"Supports Conditional Operations: {capabilities.supports_conditional_operations}")
        
        if capabilities.gate_set:
            print(f"Gate Set: {', '.join(capabilities.gate_set)}")
        
        if capabilities.noise_models:
            print(f"Noise Models: {', '.join(capabilities.noise_models)}")
    
    def show_plugin_info(self, plugin_name: str):
        """Show detailed plugin information."""
        plugin = self.plugin_manager.get_plugin(plugin_name)
        if not plugin:
            print(f"Plugin '{plugin_name}' not found.")
            return
        
        print(f"Plugin Information: {plugin_name}")
        print("=" * 50)
        print(f"Type: {plugin.info.plugin_type}")
        print(f"Version: {plugin.info.version}")
        print(f"Author: {plugin.info.author}")
        print(f"Description: {plugin.info.description}")
        print(f"Enabled: {plugin.is_enabled()}")
        
        if plugin.info.dependencies:
            print(f"Dependencies: {', '.join(plugin.info.dependencies)}")
    
    def compile_file(self, input_file: str, output_file: Optional[str], 
                    target_format: str, optimize: bool, debug: bool) -> bool:
        """Compile a DSL file to target format."""
        try:
            # Read input file
            with open(input_file, 'r') as f:
                source = f.read()
            
            # Set up compilation options
            options = CompilerOptions(
                mode=CompilerMode.COMPILE_ONLY,
                target_format=target_format,
                optimize=optimize,
                debug=debug
            )
            
            # Compile
            result = self.compiler.compile(source, options)
            
            if not result.success:
                print("Compilation failed:")
                for error in result.errors:
                    print(f"  Error: {error}")
                return False
            
            # Write output
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(result.target_code)
                print(f"Compiled to {output_file}")
            else:
                print(result.target_code)
            
            return True
            
        except Exception as e:
            print(f"Compilation error: {e}")
            return False
    
    def execute_circuit(self, input_file: str, backend_name: str, 
                       shots: int, parameters: List[str]) -> bool:
        """Execute a circuit on a backend."""
        try:
            # Read input file
            with open(input_file, 'r') as f:
                source = f.read()
            
            # Parse parameters
            param_dict = {}
            for param in parameters:
                if '=' in param:
                    key, value = param.split('=', 1)
                    try:
                        param_dict[key] = float(value)
                    except ValueError:
                        param_dict[key] = value
            
            # Set up compilation options
            options = CompilerOptions(
                mode=CompilerMode.COMPILE_AND_RUN,
                target_format='openqasm',
                backend_name=backend_name,
                shots=shots,
                parameters=param_dict
            )
            
            # Compile and execute
            result = self.compiler.compile(source, options)
            
            if not result.success:
                print("Compilation failed:")
                for error in result.errors:
                    print(f"  Error: {error}")
                return False
            
            if result.execution_result:
                exec_result = result.execution_result
                if exec_result.success:
                    print("Execution successful!")
                    print(f"Execution time: {exec_result.execution_time:.4f} seconds")
                    print(f"Measurement counts: {exec_result.counts}")
                else:
                    print("Execution failed:")
                    for error in exec_result.errors:
                        print(f"  Error: {error}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Execution error: {e}")
            return False
    
    def run(self, args: List[str] = None):
        """Run the CLI with arguments."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Handle information requests
        if parsed_args.list_backends:
            self.list_backends()
            return
        
        if parsed_args.list_plugins:
            self.list_plugins()
            return
        
        if parsed_args.backend_info:
            self.show_backend_info(parsed_args.backend_info)
            return
        
        if parsed_args.plugin_info:
            self.show_plugin_info(parsed_args.plugin_info)
            return
        
        # Handle plugin management
        if parsed_args.load_plugin:
            # Load plugin logic would go here
            print(f"Loading plugin: {parsed_args.load_plugin}")
            return
        
        if parsed_args.unload_plugin:
            # Unload plugin logic would go here
            print(f"Unloading plugin: {parsed_args.unload_plugin}")
            return
        
        # Handle compilation/execution
        if not parsed_args.input:
            parser.print_help()
            return
        
        if not os.path.exists(parsed_args.input):
            print(f"Input file '{parsed_args.input}' not found.")
            return
        
        if parsed_args.execute:
            success = self.execute_circuit(
                parsed_args.input,
                parsed_args.backend or 'local_simulator',
                parsed_args.shots,
                parsed_args.parameters or []
            )
        else:
            success = self.compile_file(
                parsed_args.input,
                parsed_args.output,
                parsed_args.target,
                parsed_args.optimize,
                parsed_args.debug
            )
        
        sys.exit(0 if success else 1)


def main():
    """Main entry point for coratrixc."""
    cli = CoratrixCompilerCLI()
    cli.run()


if __name__ == '__main__':
    main()
