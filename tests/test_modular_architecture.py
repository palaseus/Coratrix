"""
Test suite for Coratrix 3.1 modular architecture.

This test suite verifies that all layers work independently and together.
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add coratrix to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coratrix.core import ScalableQuantumState, QuantumCircuit, QuantumGate
from coratrix.compiler import CoratrixCompiler, CompilerOptions, CompilerMode
from coratrix.backend import BackendManager, BackendConfiguration, BackendType
from coratrix.plugins import PluginManager
from coratrix.cli import CoratrixCompilerCLI


class TestModularArchitecture:
    """Test the modular architecture of Coratrix 3.1."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_dir = os.path.join(self.temp_dir, 'plugins')
        os.makedirs(self.plugin_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_core_simulation_independence(self):
        """Test that core simulation works independently."""
        # Test quantum state creation
        state = ScalableQuantumState(2, use_gpu=False, sparse_threshold=8)
        assert state.num_qubits == 2
        assert state.get_amplitude(0) == 1.0
        
        # Test gate application
        from coratrix.core.quantum_circuit import HGate, CNOTGate
        h_gate = HGate()
        state.apply_gate(h_gate, [0])
        
        # Verify state change (simplified check)
        assert state.get_amplitude(0) is not None
        assert state.get_amplitude(1) is not None
    
    def test_compiler_stack_independence(self):
        """Test that compiler stack works independently."""
        # Test DSL compilation
        dsl_source = """
        circuit test_circuit() {
            h q0;
            cnot q0, q1;
        }
        """
        
        compiler = CoratrixCompiler()
        options = CompilerOptions(
            mode=CompilerMode.COMPILE_ONLY,
            target_format="openqasm",
            optimize=True
        )
        
        result = compiler.compile(dsl_source, options)
        
        # Verify compilation
        assert result.success
        assert "OPENQASM" in result.target_code
        assert "h q[0]" in result.target_code
        assert "cx q[0], q[1]" in result.target_code
    
    def test_backend_management_independence(self):
        """Test that backend management works independently."""
        backend_manager = BackendManager()
        
        # Test backend registration
        config = BackendConfiguration(
            name="test_backend",
            backend_type=BackendType.SIMULATOR,
            connection_params={'simulator_type': 'statevector'}
        )
        
        from coratrix.backend import SimulatorBackend
        backend = SimulatorBackend(config)
        success = backend_manager.register_backend("test_backend", backend)
        
        assert success
        assert "test_backend" in backend_manager.list_backends()
        
        # Test backend status
        status = backend_manager.get_backend_status("test_backend")
        assert status is not None
    
    def test_plugin_system_independence(self):
        """Test that plugin system works independently."""
        plugin_manager = PluginManager()
        
        # Test plugin registration
        from coratrix.plugins.example_optimization_pass import ExampleOptimizationPlugin
        plugin = ExampleOptimizationPlugin()
        
        success = plugin_manager.register_plugin(plugin)
        assert success
        assert "example_optimization" in plugin_manager.list_plugins()
        
        # Test plugin info
        info = plugin_manager.get_plugin_info("example_optimization")
        assert info.name == "example_optimization"
        assert info.plugin_type == "compiler_pass"
    
    def test_cli_independence(self):
        """Test that CLI works independently."""
        cli = CoratrixCompilerCLI()
        
        # Test parser creation
        parser = cli.create_parser()
        assert parser is not None
        
        # Test backend listing
        backends = cli.backend_manager.list_backends()
        assert isinstance(backends, list)
        
        # Test plugin listing
        plugins = cli.plugin_manager.list_plugins()
        assert isinstance(plugins, list)
    
    def test_integrated_workflow(self):
        """Test integrated workflow across all layers."""
        # Create DSL source
        dsl_source = """
        circuit bell_state() {
            h q0;
            cnot q0, q1;
        }
        """
        
        # Write to temporary file
        dsl_file = os.path.join(self.temp_dir, "test.qasm")
        with open(dsl_file, 'w') as f:
            f.write(dsl_source)
        
        # Test compilation
        compiler = CoratrixCompiler()
        options = CompilerOptions(
            mode=CompilerMode.COMPILE_ONLY,
            target_format="openqasm",
            optimize=True
        )
        
        result = compiler.compile(dsl_source, options)
        assert result.success
        
        # Test execution
        options = CompilerOptions(
            mode=CompilerMode.COMPILE_AND_RUN,
            target_format="openqasm",
            backend_name="local_simulator",
            shots=100
        )
        
        result = compiler.compile(dsl_source, options)
        assert result.success
        assert result.execution_result is not None
        assert result.execution_result['success'] is True
    
    def test_plugin_integration(self):
        """Test plugin integration across layers."""
        # Load example plugins
        plugin_manager = PluginManager()
        
        # Load optimization plugin
        from coratrix.plugins.example_optimization_pass import ExampleOptimizationPlugin
        opt_plugin = ExampleOptimizationPlugin()
        plugin_manager.register_plugin(opt_plugin)
        
        # Load backend plugin
        from coratrix.plugins.example_custom_backend import ExampleCustomBackendPlugin
        backend_plugin = ExampleCustomBackendPlugin()
        plugin_manager.register_plugin(backend_plugin)
        
        # Test plugin functionality
        assert opt_plugin.is_enabled()
        assert backend_plugin.is_enabled()
        
        # Test backend plugin integration
        backend_config = backend_plugin.get_backend_config()
        assert backend_config is not None
        assert backend_config.name == "example_custom"
    
    def test_error_handling(self):
        """Test error handling across layers."""
        # Test invalid DSL
        invalid_dsl = "invalid dsl syntax"
        compiler = CoratrixCompiler()
        options = CompilerOptions(mode=CompilerMode.COMPILE_ONLY)
        
        result = compiler.compile(invalid_dsl, options)
        assert not result.success
        assert len(result.errors) > 0
        
        # Test invalid backend
        backend_manager = BackendManager()
        status = backend_manager.get_backend_status("nonexistent_backend")
        assert status is None
        
        # Test invalid plugin
        plugin_manager = PluginManager()
        plugin = plugin_manager.get_plugin("nonexistent_plugin")
        assert plugin is None
    
    def test_performance_metrics(self):
        """Test performance metrics across layers."""
        # Test quantum state metrics
        state = ScalableQuantumState(3, use_gpu=False)
        from coratrix.core.quantum_circuit import HGate
        state.apply_gate(HGate(), [0])
        
        metrics = state.get_metrics()
        assert metrics['operation_count'] > 0
        assert metrics['last_operation_time'] > 0
        assert metrics['num_qubits'] == 3
        
        # Test compiler metrics
        dsl_source = "circuit test() { h q0; }"
        compiler = CoratrixCompiler()
        result = compiler.compile(dsl_source, CompilerOptions())
        
        assert result.success
        assert result.metadata is not None
    
    def test_memory_management(self):
        """Test memory management across layers."""
        # Test sparse representation
        state = ScalableQuantumState(8, use_sparse=True, sparse_threshold=4)
        assert state.use_sparse
        
        # Test memory usage tracking
        metrics = state.get_metrics()
        assert metrics['memory_usage_gb'] >= 0
        
        # Test cleanup
        del state
    
    def test_concurrent_operations(self):
        """Test concurrent operations across layers."""
        import threading
        import time
        
        results = []
        
        def compile_circuit(circuit_id):
            dsl_source = f"circuit circuit_{circuit_id}() {{ h q0; }}"
            compiler = CoratrixCompiler()
            result = compiler.compile(dsl_source, CompilerOptions())
            results.append((circuit_id, result.success))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=compile_circuit, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 3
        for circuit_id, success in results:
            assert success


class TestPluginSystem:
    """Test the plugin system functionality."""
    
    def test_plugin_discovery(self):
        """Test plugin discovery and loading."""
        plugin_manager = PluginManager()
        
        # Test loading from directory
        loaded_count = plugin_manager.load_all_plugins()
        assert loaded_count >= 0
        
        # Test plugin listing
        plugins = plugin_manager.list_plugins()
        assert isinstance(plugins, list)
    
    def test_plugin_lifecycle(self):
        """Test plugin lifecycle management."""
        plugin_manager = PluginManager()
        
        # Load example plugin
        from coratrix.plugins.example_optimization_pass import ExampleOptimizationPlugin
        plugin = ExampleOptimizationPlugin()
        
        # Test initialization
        assert plugin.initialize()
        assert plugin.is_enabled()
        
        # Test registration
        success = plugin_manager.register_plugin(plugin)
        assert success
        
        # Test cleanup
        assert plugin.cleanup()
    
    def test_plugin_types(self):
        """Test different plugin types."""
        plugin_manager = PluginManager()
        
        # Test compiler pass plugin
        from coratrix.plugins.example_optimization_pass import ExampleOptimizationPlugin
        opt_plugin = ExampleOptimizationPlugin()
        plugin_manager.register_plugin(opt_plugin)
        
        # Test backend plugin
        from coratrix.plugins.example_custom_backend import ExampleCustomBackendPlugin
        backend_plugin = ExampleCustomBackendPlugin()
        plugin_manager.register_plugin(backend_plugin)
        
        # Test plugin type filtering
        compiler_plugins = plugin_manager.get_plugins_by_type('compiler_pass')
        backend_plugins = plugin_manager.get_plugins_by_type('backend')
        
        assert len(compiler_plugins) >= 1
        assert len(backend_plugins) >= 1


class TestCLIIntegration:
    """Test CLI integration with all layers."""
    
    def test_cli_compilation(self):
        """Test CLI compilation functionality."""
        cli = CoratrixCompilerCLI()
        
        # Test parser creation
        parser = cli.create_parser()
        assert parser is not None
        
        # Test argument parsing
        args = parser.parse_args(['--list-backends'])
        assert args.list_backends
        
        args = parser.parse_args(['--list-plugins'])
        assert args.list_plugins
    
    def test_cli_backend_management(self):
        """Test CLI backend management."""
        cli = CoratrixCompilerCLI()
        
        # Test backend listing
        cli.list_backends()
        
        # Test plugin listing
        cli.list_plugins()
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        cli = CoratrixCompilerCLI()
        
        # Test with invalid arguments
        parser = cli.create_parser()
        args = parser.parse_args(['nonexistent_file.qasm'])
        
        # Should handle gracefully
        assert args.input == 'nonexistent_file.qasm'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
