"""
Tests for optimization engine and parameterized circuits.

This module tests optimization algorithms for parameterized quantum circuits.
"""

import unittest
import sys
import os
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimization_engine import (
    ParameterizedCircuit, OptimizationEngine, OptimizationConfig,
    OptimizationMethod, OptimizationResult, NoiseAwareOptimization,
    ConstrainedOptimization
)
from core.qubit import QuantumState
from core.gates import HGate, CNOTGate


class TestParameterizedCircuit(unittest.TestCase):
    """Test cases for parameterized quantum circuits."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 2
        self.parameterized_gates = [
            ("rx", [0], "theta_0"),
            ("ry", [1], "theta_1"),
            ("rz", [0], "theta_2")
        ]
        self.circuit = ParameterizedCircuit(self.num_qubits, self.parameterized_gates)
    
    def test_circuit_initialization(self):
        """Test parameterized circuit initialization."""
        self.assertEqual(self.circuit.num_qubits, 2)
        self.assertEqual(self.circuit.num_parameters, 3)
        self.assertEqual(len(self.circuit.parameter_names), 3)
        self.assertEqual(self.circuit.parameter_names, ["theta_0", "theta_1", "theta_2"])
    
    def test_set_parameters(self):
        """Test setting parameters."""
        parameters = np.array([np.pi/2, np.pi/4, np.pi/6])
        self.circuit.set_parameters(parameters)
        
        np.testing.assert_array_almost_equal(self.circuit.parameters, parameters)
        self.assertEqual(len(self.circuit.circuit.gates), 3)
    
    def test_set_parameters_invalid_length(self):
        """Test setting parameters with invalid length."""
        parameters = np.array([np.pi/2, np.pi/4])  # Wrong length
        
        with self.assertRaises(ValueError):
            self.circuit.set_parameters(parameters)
    
    def test_execute_circuit(self):
        """Test executing parameterized circuit."""
        parameters = np.array([np.pi/2, np.pi/4, np.pi/6])
        self.circuit.set_parameters(parameters)
        
        state = self.circuit.execute()
        
        # Check that state is normalized
        norm = np.sum(np.abs(state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_get_parameter_bounds(self):
        """Test getting parameter bounds."""
        bounds = self.circuit.get_parameter_bounds()
        
        self.assertEqual(len(bounds), 3)
        for lower, upper in bounds:
            self.assertEqual(lower, 0.0)
            self.assertEqual(upper, 2 * np.pi)
    
    def test_different_gate_types(self):
        """Test circuit with different gate types."""
        gates = [
            ("rx", [0], "theta_0"),
            ("ry", [1], "theta_1"),
            ("rz", [0], "theta_2"),
            ("cphase", [0, 1], "phi_0")
        ]
        circuit = ParameterizedCircuit(2, gates)
        
        self.assertEqual(circuit.num_parameters, 4)
        self.assertEqual(len(circuit.parameter_names), 4)
        
        # Test setting parameters
        parameters = np.array([np.pi/2, np.pi/4, np.pi/6, np.pi/3])
        circuit.set_parameters(parameters)
        
        # Test execution
        state = circuit.execute()
        norm = np.sum(np.abs(state.state_vector)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)


class TestOptimizationEngine(unittest.TestCase):
    """Test cases for optimization engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = OptimizationConfig(
            method=OptimizationMethod.SPSA,
            max_iterations=50,
            learning_rate=0.1
        )
        self.engine = OptimizationEngine(self.config)
        
        # Create test circuit
        self.circuit = ParameterizedCircuit(2, [
            ("rx", [0], "theta_0"),
            ("ry", [1], "theta_1")
        ])
    
    def test_engine_initialization(self):
        """Test optimization engine initialization."""
        self.assertEqual(self.engine.config.method, OptimizationMethod.SPSA)
        self.assertEqual(self.engine.config.max_iterations, 50)
        self.assertEqual(self.engine.config.learning_rate, 0.1)
    
    def test_simple_objective_function(self):
        """Test optimization with simple objective function."""
        def objective(circuit):
            # Simple objective: minimize |⟨0|ψ⟩|²
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        # Run optimization
        result = self.engine.optimize(self.circuit, objective)
        
        # Check results
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.optimal_parameters, np.ndarray)
        self.assertIsInstance(result.optimal_value, float)
        self.assertIsInstance(result.iterations, int)
        self.assertIsInstance(result.execution_time, float)
        self.assertIsInstance(result.convergence_history, list)
        self.assertEqual(result.method, "spsa")
    
    def test_optimization_with_initial_parameters(self):
        """Test optimization with custom initial parameters."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        initial_parameters = np.array([np.pi/4, np.pi/3])
        result = self.engine.optimize(self.circuit, objective, initial_parameters)
        
        # Check that optimization completed
        self.assertIsInstance(result, OptimizationResult)
        self.assertGreaterEqual(result.iterations, 1)
    
    def test_different_optimization_methods(self):
        """Test different optimization methods."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        methods = [
            OptimizationMethod.SPSA,
            OptimizationMethod.NELDER_MEAD,
            OptimizationMethod.LBFGS
        ]
        
        for method in methods:
            with self.subTest(method=method):
                config = OptimizationConfig(method=method, max_iterations=20)
                engine = OptimizationEngine(config)
                
                result = engine.optimize(self.circuit, objective)
                
                self.assertIsInstance(result, OptimizationResult)
                self.assertEqual(result.method, method.value)
    
    def test_convergence_tracking(self):
        """Test convergence history tracking."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        result = self.engine.optimize(self.circuit, objective)
        
        # Check convergence history
        self.assertIsInstance(result.convergence_history, list)
        self.assertGreater(len(result.convergence_history), 0)
        self.assertEqual(len(result.convergence_history), result.iterations)
    
    def test_optimization_with_bounds(self):
        """Test optimization with parameter bounds."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        # Test that parameters stay within bounds
        result = self.engine.optimize(self.circuit, objective)
        
        bounds = self.circuit.get_parameter_bounds()
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(result.optimal_parameters[i], lower)
            self.assertLessEqual(result.optimal_parameters[i], upper)


class TestNoiseAwareOptimization(unittest.TestCase):
    """Test cases for noise-aware optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.noise_model = {'depolarizing_error': 0.01}
        self.noise_optimizer = NoiseAwareOptimization(self.noise_model)
        
        self.circuit = ParameterizedCircuit(2, [
            ("rx", [0], "theta_0"),
            ("ry", [1], "theta_1")
        ])
    
    def test_noise_optimizer_initialization(self):
        """Test noise-aware optimizer initialization."""
        self.assertEqual(self.noise_optimizer.noise_model, self.noise_model)
    
    def test_optimize_with_noise(self):
        """Test optimization with noise."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        config = OptimizationConfig(method=OptimizationMethod.SPSA, max_iterations=20)
        result = self.noise_optimizer.optimize_with_noise(self.circuit, objective, config)
        
        # Check that optimization completed
        self.assertIsInstance(result, OptimizationResult)
        self.assertGreaterEqual(result.iterations, 1)


class TestConstrainedOptimization(unittest.TestCase):
    """Test cases for constrained optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define constraint: sum of parameters must be less than π
        def constraint(params):
            return np.sum(params) < np.pi
        
        self.constraints = [constraint]
        self.constrained_optimizer = ConstrainedOptimization(self.constraints)
        
        self.circuit = ParameterizedCircuit(2, [
            ("rx", [0], "theta_0"),
            ("ry", [1], "theta_1")
        ])
    
    def test_constrained_optimizer_initialization(self):
        """Test constrained optimizer initialization."""
        self.assertEqual(len(self.constrained_optimizer.constraints), 1)
    
    def test_optimize_with_constraints(self):
        """Test optimization with constraints."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        config = OptimizationConfig(method=OptimizationMethod.SPSA, max_iterations=20)
        result = self.constrained_optimizer.optimize_with_constraints(
            self.circuit, objective, config
        )
        
        # Check that optimization completed
        self.assertIsInstance(result, OptimizationResult)
        self.assertGreaterEqual(result.iterations, 1)
        
        # Check that constraints are satisfied
        if result.success:
            constraint_satisfied = self.constraints[0](result.optimal_parameters)
            self.assertTrue(constraint_satisfied)


class TestOptimizationIntegration(unittest.TestCase):
    """Integration tests for optimization engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create circuit for testing
        self.circuit = ParameterizedCircuit(2, [
            ("rx", [0], "theta_0"),
            ("ry", [1], "theta_1"),
            ("rz", [0], "theta_2")
        ])
        
        # Create optimization engine
        config = OptimizationConfig(method=OptimizationMethod.SPSA, max_iterations=10)
        self.engine = OptimizationEngine(config)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_optimization_with_trace_saving(self):
        """Test optimization with trace saving."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        config = OptimizationConfig(
            method=OptimizationMethod.SPSA,
            max_iterations=20,
            save_traces=True,
            output_dir=self.temp_dir
        )
        engine = OptimizationEngine(config)
        
        result = engine.optimize(self.circuit, objective)
        
        # Check that traces were saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'convergence_history.json')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'optimization_data.csv')))
    
    def test_optimization_result_properties(self):
        """Test optimization result properties."""
        def objective(circuit):
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        result = self.engine.optimize(self.circuit, objective)
        
        # Check all required properties
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.optimal_parameters, np.ndarray)
        self.assertIsInstance(result.optimal_value, float)
        self.assertIsInstance(result.iterations, int)
        self.assertIsInstance(result.execution_time, float)
        self.assertIsInstance(result.convergence_history, list)
        self.assertIsInstance(result.method, str)
        
        # Check array shapes
        self.assertEqual(len(result.optimal_parameters), self.circuit.num_parameters)
        self.assertEqual(len(result.convergence_history), result.iterations)
    
    def test_optimization_with_different_objectives(self):
        """Test optimization with different objective functions."""
        objectives = [
            lambda circuit: abs(circuit.execute().get_amplitude(0))**2,
            lambda circuit: abs(circuit.execute().get_amplitude(3))**2,
            lambda circuit: circuit.execute().get_entanglement_entropy()
        ]
        
        for i, objective in enumerate(objectives):
            with self.subTest(objective=i):
                result = self.engine.optimize(self.circuit, objective)
                
                self.assertIsInstance(result, OptimizationResult)
                self.assertIsInstance(result.optimal_value, float)
                self.assertGreaterEqual(result.iterations, 1)
    
    def test_optimization_error_handling(self):
        """Test optimization error handling."""
        def objective(circuit):
            # This objective might cause issues
            state = circuit.execute()
            return abs(state.get_amplitude(0))**2
        
        # Test with invalid configuration
        invalid_config = OptimizationConfig(
            method=OptimizationMethod.SPSA,
            max_iterations=0  # Invalid
        )
        engine = OptimizationEngine(invalid_config)
        
        result = engine.optimize(self.circuit, objective)
        
        # Should still return a result (possibly unsuccessful)
        self.assertIsInstance(result, OptimizationResult)


if __name__ == '__main__':
    unittest.main()
