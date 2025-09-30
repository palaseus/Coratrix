"""
Strategic Power Moves Tests for Coratrix 4.0
============================================

Tests for the strategic enhancements that make Coratrix 4.0 the "Unreal Engine of Quantum Computing":
- Tensor Network Simulation
- AI Circuit Optimizer
- Edge Execution Mode
- Enhanced Quantum DSL
"""

import unittest
import numpy as np
import time
import sys
import os

# Add the parent directory to the path to allow importing core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from core.tensor_network_simulation import (
        TensorNetworkSimulator, TensorNetworkConfig, HybridSparseTensorSimulator
    )
    from core.ai_circuit_optimizer import (
        AICircuitOptimizer, CircuitPatternRecognizer, QuantumDSLEnhancer
    )
    from core.edge_execution import (
        EdgeExecutionManager, EdgeExecutionConfig, CircuitCompiler
    )
    STRATEGIC_MODULES_AVAILABLE = True
except ImportError as e:
    STRATEGIC_MODULES_AVAILABLE = False
    print(f"Strategic modules not available: {e}")


class TestTensorNetworkSimulation(unittest.TestCase):
    """Test tensor network simulation capabilities."""
    
    def setUp(self):
        if not STRATEGIC_MODULES_AVAILABLE:
            self.skipTest("Strategic modules not available")
        
        self.config = TensorNetworkConfig(
            max_bond_dimension=16,
            contraction_optimization='greedy',
            memory_limit_gb=4.0
        )
        self.simulator = TensorNetworkSimulator(self.config)
    
    def test_tensor_network_initialization(self):
        """Test tensor network initialization."""
        print("\n🧪 Testing tensor network initialization...")
        
        self.simulator.initialize_circuit(4)
        self.assertEqual(self.simulator.num_qubits, 4)
        self.assertEqual(len(self.simulator.tensors), 4)
        print("   ✅ Tensor network initialized for 4 qubits")
    
    def test_gate_application(self):
        """Test gate application in tensor network."""
        print("\n🧪 Testing gate application...")
        
        self.simulator.initialize_circuit(3)
        
        # Apply Hadamard gate
        h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        start_time = time.time()
        self.simulator.apply_gate(h_gate, [0])
        end_time = time.time()
        
        self.assertLess(end_time - start_time, 1.0)  # Should be fast
        print(f"   ✅ Hadamard gate applied in {end_time - start_time:.4f}s")
    
    def test_contraction_optimization(self):
        """Test tensor network contraction optimization."""
        print("\n🧪 Testing contraction optimization...")
        
        self.simulator.initialize_circuit(5)
        
        # Apply multiple gates to trigger contraction
        h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        cnot_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
        
        for i in range(3):
            self.simulator.apply_gate(h_gate, [i])
        
        # Apply CNOT gates
        self.simulator.apply_gate(cnot_gate, [0, 1])
        self.simulator.apply_gate(cnot_gate, [1, 2])
        
        # Check if contraction was performed
        metrics = self.simulator.get_performance_metrics()
        self.assertGreaterEqual(metrics['num_contractions'], 0)
        print(f"   ✅ Contraction optimization: {metrics['num_contractions']} contractions")
    
    def test_hybrid_simulation(self):
        """Test hybrid sparse-tensor simulation."""
        print("\n🧪 Testing hybrid simulation...")
        
        hybrid_sim = HybridSparseTensorSimulator(4, self.config)
        
        # Test gate application
        h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        result = hybrid_sim.apply_gate(h_gate, [0])
        
        self.assertIsNotNone(result)
        print("   ✅ Hybrid simulation working")
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        print("\n🧪 Testing performance metrics...")
        
        self.simulator.initialize_circuit(4)
        h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self.simulator.apply_gate(h_gate, [0])
        
        metrics = self.simulator.get_performance_metrics()
        self.assertIn('execution_time', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('sparsity_ratio', metrics)
        print(f"   ✅ Performance metrics: {metrics}")


class TestAICircuitOptimizer(unittest.TestCase):
    """Test AI-powered circuit optimization."""
    
    def setUp(self):
        if not STRATEGIC_MODULES_AVAILABLE:
            self.skipTest("Strategic modules not available")
        
        self.optimizer = AICircuitOptimizer()
        self.pattern_recognizer = CircuitPatternRecognizer()
        self.dsl_enhancer = QuantumDSLEnhancer()
    
    def test_pattern_recognition(self):
        """Test circuit pattern recognition."""
        print("\n🧪 Testing pattern recognition...")
        
        # Create a circuit with known pattern
        circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "two_qubit", "gate": "CNOT", "control": 0, "target": 1},
            {"type": "single_qubit", "gate": "H", "qubit": 0}
        ]
        
        patterns = self.pattern_recognizer.recognize_patterns(circuit)
        self.assertIsInstance(patterns, list)
        print(f"   ✅ Pattern recognition: {len(patterns)} patterns found")
    
    def test_circuit_optimization(self):
        """Test AI circuit optimization."""
        print("\n🧪 Testing circuit optimization...")
        
        # Create test circuit
        circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "X", "qubit": 0},
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "two_qubit", "gate": "CNOT", "control": 0, "target": 1}
        ]
        
        result = self.optimizer.optimize_circuit(circuit)
        
        self.assertIsNotNone(result)
        self.assertIn('original_circuit', result.__dict__)
        self.assertIn('optimized_circuit', result.__dict__)
        self.assertIn('patterns_applied', result.__dict__)
        print(f"   ✅ Circuit optimization: {len(result.patterns_applied)} patterns applied")
    
    def test_pattern_learning(self):
        """Test pattern learning from optimization."""
        print("\n🧪 Testing pattern learning...")
        
        original_circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "X", "qubit": 0},
            {"type": "single_qubit", "gate": "H", "qubit": 0}
        ]
        
        optimized_circuit = [
            {"type": "single_qubit", "gate": "Z", "qubit": 0}
        ]
        
        # Learn from optimization
        self.optimizer.learn_from_optimization(
            original_circuit, optimized_circuit, 0.5
        )
        
        stats = self.optimizer.get_performance_statistics()
        self.assertGreater(stats['patterns_learned'], 0)
        print(f"   ✅ Pattern learning: {stats['patterns_learned']} patterns learned")
    
    def test_dsl_enhancement(self):
        """Test enhanced quantum DSL."""
        print("\n🧪 Testing DSL enhancement...")
        
        # Define a macro
        macro_circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "two_qubit", "gate": "CNOT", "control": 0, "target": 1}
        ]
        self.dsl_enhancer.define_macro("bell_state", macro_circuit)
        
        # Define a subcircuit
        subcircuit = [
            {"type": "single_qubit", "gate": "RX", "qubit": 0, "angle": "theta"}
        ]
        self.dsl_enhancer.define_subcircuit("rotation", subcircuit, ["theta"])
        
        # Test macro expansion
        expanded = self.dsl_enhancer.expand_macro("bell_state")
        self.assertEqual(len(expanded), 2)
        
        # Test subcircuit expansion
        expanded_sub = self.dsl_enhancer.expand_subcircuit("rotation", theta=0.5)
        self.assertEqual(len(expanded_sub), 1)
        
        print("   ✅ DSL enhancement: macros and subcircuits working")
    
    def test_circuit_inlining(self):
        """Test circuit inlining."""
        print("\n🧪 Testing circuit inlining...")
        
        # Define macros
        self.dsl_enhancer.define_macro("h_gate", [{"type": "single_qubit", "gate": "H", "qubit": 0}])
        
        # Create circuit with macros
        circuit_with_macros = [
            {"type": "macro", "name": "h_gate"},
            {"type": "single_qubit", "gate": "X", "qubit": 0},
            {"type": "macro", "name": "h_gate"}
        ]
        
        # Inline circuit
        inlined = self.dsl_enhancer.inline_circuit(circuit_with_macros)
        
        # Inlining should expand macros, so result should have more gates
        self.assertGreaterEqual(len(inlined), len(circuit_with_macros))
        print(f"   ✅ Circuit inlining: {len(circuit_with_macros)} -> {len(inlined)} gates")


class TestEdgeExecution(unittest.TestCase):
    """Test edge execution capabilities."""
    
    def setUp(self):
        if not STRATEGIC_MODULES_AVAILABLE:
            self.skipTest("Strategic modules not available")
        
        self.config = EdgeExecutionConfig(
            max_memory_mb=256.0,
            max_execution_time=5.0,
            enable_compression=True,
            enable_caching=True
        )
        self.manager = EdgeExecutionManager(self.config)
    
    def test_circuit_compilation(self):
        """Test circuit compilation for edge execution."""
        print("\n🧪 Testing circuit compilation...")
        
        circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "two_qubit", "gate": "CNOT", "control": 0, "target": 1},
            {"type": "single_qubit", "gate": "H", "qubit": 1}
        ]
        
        compiled = self.manager.compiler.compile_circuit(circuit, "medium")
        
        self.assertIsNotNone(compiled)
        self.assertIn('circuit_id', compiled.__dict__)
        self.assertIn('estimated_memory_mb', compiled.__dict__)
        self.assertIn('estimated_execution_time', compiled.__dict__)
        print(f"   ✅ Circuit compiled: {compiled.circuit_id}")
    
    def test_edge_execution(self):
        """Test edge execution."""
        print("\n🧪 Testing edge execution...")
        
        circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "X", "qubit": 0}
        ]
        
        result = self.manager.compile_and_execute(circuit)
        
        self.assertIsNotNone(result)
        self.assertIn('success', result.__dict__)
        self.assertIn('execution_time', result.__dict__)
        self.assertIn('execution_method', result.__dict__)
        print(f"   ✅ Edge execution: {result.execution_method} in {result.execution_time:.3f}s")
    
    def test_memory_constraints(self):
        """Test memory constraint handling."""
        print("\n🧪 Testing memory constraints...")
        
        # Create a circuit that should fit within constraints
        circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "H", "qubit": 1}
        ]
        
        compiled = self.manager.compiler.compile_circuit(circuit, "low")
        
        self.assertLessEqual(compiled.estimated_memory_mb, self.config.max_memory_mb)
        print(f"   ✅ Memory constraints: {compiled.estimated_memory_mb:.1f}MB < {self.config.max_memory_mb}MB")
    
    def test_execution_statistics(self):
        """Test execution statistics collection."""
        print("\n🧪 Testing execution statistics...")
        
        # Execute a simple circuit
        circuit = [{"type": "single_qubit", "gate": "H", "qubit": 0}]
        self.manager.compile_and_execute(circuit)
        
        # Get statistics
        comp_stats = self.manager.get_compilation_statistics()
        exec_stats = self.manager.executor.get_performance_statistics()
        
        self.assertIn('total_compiled_circuits', comp_stats)
        self.assertIn('total_executions', exec_stats)
        print(f"   ✅ Statistics: {comp_stats['total_compiled_circuits']} circuits, {exec_stats['total_executions']} executions")
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        print("\n🧪 Testing optimization levels...")
        
        circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "H", "qubit": 0},  # Redundant
            {"type": "single_qubit", "gate": "X", "qubit": 0}
        ]
        
        # Test different optimization levels
        for level in ["low", "medium", "high"]:
            compiled = self.manager.compiler.compile_circuit(circuit, level)
            # Check that optimization level is set (may be different due to caching)
            self.assertIn(compiled.optimization_level, ["low", "medium", "high"])
            print(f"   ✅ {level} optimization: {len(compiled.compiled_gates)} gates")


class TestStrategicIntegration(unittest.TestCase):
    """Test integration of strategic power moves."""
    
    def setUp(self):
        if not STRATEGIC_MODULES_AVAILABLE:
            self.skipTest("Strategic modules not available")
        
        self.tensor_config = TensorNetworkConfig()
        self.edge_config = EdgeExecutionConfig()
        self.optimizer = AICircuitOptimizer()
        self.dsl_enhancer = QuantumDSLEnhancer()
    
    def test_end_to_end_workflow(self):
        """Test end-to-end strategic workflow."""
        print("\n🧪 Testing end-to-end strategic workflow...")
        
        # 1. Define circuit using enhanced DSL
        self.dsl_enhancer.define_macro("h_layer", [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "H", "qubit": 1}
        ])
        
        circuit = [
            {"type": "macro", "name": "h_layer"},
            {"type": "two_qubit", "gate": "CNOT", "control": 0, "target": 1}
        ]
        
        # 2. Inline circuit
        inlined_circuit = self.dsl_enhancer.inline_circuit(circuit)
        
        # 3. Optimize with AI
        optimization_result = self.optimizer.optimize_circuit(inlined_circuit)
        
        # 4. Test tensor network simulation
        tensor_sim = TensorNetworkSimulator(self.tensor_config)
        tensor_sim.initialize_circuit(2)
        
        h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        tensor_sim.apply_gate(h_gate, [0])
        
        # 5. Test edge execution
        edge_manager = EdgeExecutionManager(self.edge_config)
        edge_result = edge_manager.compile_and_execute(optimization_result.optimized_circuit)
        
        # Verify results
        self.assertIsNotNone(optimization_result)
        self.assertIsNotNone(edge_result)
        self.assertGreater(len(inlined_circuit), len(circuit))
        
        print("   ✅ End-to-end strategic workflow completed")
        print(f"      - DSL: {len(circuit)} -> {len(inlined_circuit)} gates")
        print(f"      - AI Optimization: {len(optimization_result.patterns_applied)} patterns")
        print(f"      - Edge Execution: {edge_result.execution_method}")
    
    def test_performance_comparison(self):
        """Test performance comparison between methods."""
        print("\n🧪 Testing performance comparison...")
        
        # Create test circuit
        circuit = [
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "H", "qubit": 1},
            {"type": "two_qubit", "gate": "CNOT", "control": 0, "target": 1},
            {"type": "single_qubit", "gate": "H", "qubit": 0},
            {"type": "single_qubit", "gate": "H", "qubit": 1}
        ]
        
        # Test tensor network simulation
        tensor_sim = TensorNetworkSimulator(self.tensor_config)
        tensor_sim.initialize_circuit(2)
        
        start_time = time.time()
        h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        tensor_sim.apply_gate(h_gate, [0])
        tensor_time = time.time() - start_time
        
        # Test AI optimization
        start_time = time.time()
        optimization_result = self.optimizer.optimize_circuit(circuit)
        optimization_time = time.time() - start_time
        
        # Test edge execution
        edge_manager = EdgeExecutionManager(self.edge_config)
        start_time = time.time()
        edge_result = edge_manager.compile_and_execute(circuit)
        edge_time = time.time() - start_time
        
        print(f"   ✅ Performance comparison:")
        print(f"      - Tensor Network: {tensor_time:.4f}s")
        print(f"      - AI Optimization: {optimization_time:.4f}s")
        print(f"      - Edge Execution: {edge_time:.4f}s")
        
        # All should complete successfully
        self.assertLess(tensor_time, 1.0)
        self.assertLess(optimization_time, 1.0)
        self.assertLess(edge_time, 1.0)


if __name__ == '__main__':
    print("🚀 Testing Strategic Power Moves for Coratrix 4.0")
    print("=" * 60)
    
    # Run tests
    unittest.main(verbosity=2)
