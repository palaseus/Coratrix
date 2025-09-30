"""
Optimization engine for parameterized quantum circuits.

This module provides optimization algorithms for parameterized quantum circuits,
including SPSA, Nelder-Mead, and gradient-free methods.
"""

import numpy as np
import scipy.optimize as opt
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
import json
import matplotlib.pyplot as plt
import csv
import os

from core.circuit import QuantumCircuit
from core.advanced_gates import RxGate, RyGate, RzGate, CPhaseGate
from core.qubit import QuantumState
from core.scalable_quantum_state import ScalableQuantumState


class OptimizationMethod(Enum):
    """Optimization method enumeration."""
    SPSA = "spsa"
    NELDER_MEAD = "nelder_mead"
    LBFGS = "lbfgs"
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"


@dataclass
class OptimizationResult:
    """Result from optimization process."""
    success: bool
    optimal_parameters: np.ndarray
    optimal_value: float
    iterations: int
    execution_time: float
    convergence_history: List[float]
    method: str
    error_message: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    method: OptimizationMethod = OptimizationMethod.SPSA
    max_iterations: int = 100
    tolerance: float = 1e-6
    learning_rate: float = 0.01
    noise_factor: float = 0.1
    convergence_threshold: float = 1e-8
    save_traces: bool = True
    output_dir: str = "optimization_traces"


class ParameterizedCircuit:
    """Parameterized quantum circuit for optimization."""
    
    def __init__(self, num_qubits: int, parameterized_gates: List[Tuple[str, List[int], str]]):
        """
        Initialize parameterized circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            parameterized_gates: List of (gate_type, target_qubits, parameter_name) tuples
        """
        self.num_qubits = num_qubits
        self.parameterized_gates = parameterized_gates
        self.num_parameters = len(parameterized_gates)
        self.parameter_names = [gate[2] for gate in parameterized_gates]
        
        # Initialize parameter values
        self.parameters = np.zeros(self.num_parameters)
        
        # Create circuit
        self.circuit = QuantumCircuit(num_qubits)
    
    def set_parameters(self, parameters: np.ndarray):
        """Set parameter values and update circuit."""
        if len(parameters) != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} parameters, got {len(parameters)}")
        
        self.parameters = parameters.copy()
        self._update_circuit()
    
    def _update_circuit(self):
        """Update circuit with current parameter values."""
        self.circuit = QuantumCircuit(self.num_qubits)
        
        for i, (gate_type, target_qubits, param_name) in enumerate(self.parameterized_gates):
            param_value = self.parameters[i]
            
            if gate_type == "rx":
                gate = RxGate(param_value)
            elif gate_type == "ry":
                gate = RyGate(param_value)
            elif gate_type == "rz":
                gate = RzGate(param_value)
            elif gate_type == "cphase":
                gate = CPhaseGate(param_value)
            else:
                raise ValueError(f"Unknown gate type: {gate_type}")
            
            self.circuit.add_gate(gate, target_qubits)
    
    def execute(self) -> QuantumState:
        """Execute the parameterized circuit."""
        self.circuit.execute()
        return self.circuit.get_state()
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for gate_type, _, _ in self.parameterized_gates:
            if gate_type in ["rx", "ry", "rz", "cphase"]:
                bounds.append((0.0, 2 * np.pi))  # Standard rotation bounds
            else:
                bounds.append((-np.pi, np.pi))  # Default bounds
        return bounds


class OptimizationEngine:
    """Main optimization engine for parameterized circuits."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.convergence_history = []
        self.parameter_history = []
        self.function_evaluations = 0
    
    def optimize(self, circuit: ParameterizedCircuit, objective_function: Callable,
                 initial_parameters: Optional[np.ndarray] = None) -> OptimizationResult:
        """Optimize parameterized circuit."""
        start_time = time.time()
        
        # Initialize parameters
        if initial_parameters is None:
            initial_parameters = np.random.uniform(0, 2*np.pi, circuit.num_parameters)
        
        # Set up optimization
        self.convergence_history = []
        self.parameter_history = []
        self.function_evaluations = 0
        
        # Choose optimization method
        if self.config.method == OptimizationMethod.SPSA:
            result = self._optimize_spsa(circuit, objective_function, initial_parameters)
        elif self.config.method == OptimizationMethod.NELDER_MEAD:
            result = self._optimize_nelder_mead(circuit, objective_function, initial_parameters)
        elif self.config.method == OptimizationMethod.LBFGS:
            result = self._optimize_lbfgs(circuit, objective_function, initial_parameters)
        elif self.config.method == OptimizationMethod.GRADIENT_DESCENT:
            result = self._optimize_gradient_descent(circuit, objective_function, initial_parameters)
        elif self.config.method == OptimizationMethod.ADAM:
            result = self._optimize_adam(circuit, objective_function, initial_parameters)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")
        
        # Add execution time
        result.execution_time = time.time() - start_time
        result.convergence_history = self.convergence_history.copy()
        result.method = self.config.method.value
        
        # Save traces if requested
        if self.config.save_traces:
            self._save_optimization_traces(circuit, result)
        
        return result
    
    def optimize_with_constraints(self, circuit: ParameterizedCircuit, objective_function: Callable, 
                                 constraints: List[Callable]) -> OptimizationResult:
        """Optimize circuit with constraints."""
        # Get initial parameters
        initial_parameters = np.random.random(circuit.num_parameters) * 2 * np.pi
        
        # Run SPSA optimization with constraint checking
        return self._optimize_spsa_with_constraints(circuit, objective_function, initial_parameters, constraints)
    
    def _optimize_spsa_with_constraints(self, circuit: ParameterizedCircuit, objective_function: Callable,
                                      initial_parameters: np.ndarray, constraints: List[Callable]) -> OptimizationResult:
        """Optimize using SPSA with constraints."""
        parameters = initial_parameters.copy()
        best_parameters = parameters.copy()
        best_value = float('inf')
        actual_iterations = 0
        
        for iteration in range(self.config.max_iterations):
            actual_iterations += 1
            
            # Check constraints before optimization
            for constraint in constraints:
                if not constraint(parameters):
                    return OptimizationResult(
                        success=False,
                        optimal_parameters=parameters,
                        optimal_value=float('inf'),
                        iterations=actual_iterations,
                        execution_time=0.0,
                        convergence_history=self.convergence_history.copy(),
                        method="spsa_constrained",
                        error_message="Constraint violation"
                    )
            
            # Generate perturbation
            perturbation = np.random.choice([-1, 1], size=len(parameters)).astype(np.float64)
            perturbation *= self.config.noise_factor
            
            # Evaluate objective at current point and perturbed point
            circuit.set_parameters(parameters)
            value_plus = objective_function(circuit)
            
            circuit.set_parameters(parameters + perturbation)
            value_minus = objective_function(circuit)
            
            # Calculate gradient estimate
            gradient = (value_plus - value_minus) / (2 * self.config.noise_factor) * perturbation
            
            # Update parameters
            parameters -= self.config.learning_rate * gradient
            
            # Keep parameters in bounds
            bounds = circuit.get_parameter_bounds()
            for i, (lower, upper) in enumerate(bounds):
                parameters[i] = np.clip(parameters[i], lower, upper)
            
            # Track convergence
            current_value = min(value_plus, value_minus)
            self.convergence_history.append(current_value)
            self.parameter_history.append(parameters.copy())
            
            # Update best result
            if current_value < best_value:
                best_value = current_value
                best_parameters = parameters.copy()
            
            # Check convergence
            if len(self.convergence_history) > 10:
                recent_improvement = abs(self.convergence_history[-1] - self.convergence_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    break
        
        return OptimizationResult(
            success=True,
            optimal_parameters=best_parameters,
            optimal_value=best_value,
            iterations=actual_iterations,
            execution_time=0.0,  # Will be set by caller
            convergence_history=self.convergence_history.copy(),
            method="spsa_constrained"
        )
    
    def _optimize_spsa(self, circuit: ParameterizedCircuit, objective_function: Callable,
                      initial_parameters: np.ndarray) -> OptimizationResult:
        """Optimize using Simultaneous Perturbation Stochastic Approximation (SPSA)."""
        parameters = initial_parameters.copy()
        best_parameters = parameters.copy()
        best_value = float('inf')
        actual_iterations = 0
        
        for iteration in range(self.config.max_iterations):
            actual_iterations += 1
            # Generate perturbation
            perturbation = np.random.choice([-1, 1], size=len(parameters)).astype(np.float64)
            perturbation *= self.config.noise_factor
            
            # Evaluate objective at current point and perturbed point
            circuit.set_parameters(parameters)
            value_plus = objective_function(circuit)
            
            circuit.set_parameters(parameters + perturbation)
            value_minus = objective_function(circuit)
            
            # Calculate gradient estimate
            gradient = (value_plus - value_minus) / (2 * self.config.noise_factor) * perturbation
            
            # Update parameters
            parameters -= self.config.learning_rate * gradient
            
            # Keep parameters in bounds
            bounds = circuit.get_parameter_bounds()
            for i, (lower, upper) in enumerate(bounds):
                parameters[i] = np.clip(parameters[i], lower, upper)
            
            # Track convergence
            current_value = min(value_plus, value_minus)
            self.convergence_history.append(current_value)
            self.parameter_history.append(parameters.copy())
            
            # Update best result
            if current_value < best_value:
                best_value = current_value
                best_parameters = parameters.copy()
            
            # Check convergence
            if len(self.convergence_history) > 10:
                recent_improvement = abs(self.convergence_history[-1] - self.convergence_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    break
        
        return OptimizationResult(
            success=True,
            optimal_parameters=best_parameters,
            optimal_value=best_value,
            iterations=actual_iterations,
            execution_time=0.0,  # Will be set by caller
            convergence_history=self.convergence_history.copy(),
            method="spsa"
        )
    
    def _optimize_nelder_mead(self, circuit: ParameterizedCircuit, objective_function: Callable,
                             initial_parameters: np.ndarray) -> OptimizationResult:
        """Optimize using Nelder-Mead simplex method."""
        def objective_wrapper(params):
            circuit.set_parameters(params)
            return objective_function(circuit)
        
        # Set up bounds
        bounds = circuit.get_parameter_bounds()
        
        # Run optimization
        result = opt.minimize(
            objective_wrapper,
            initial_parameters,
            method='Nelder-Mead',
            options={'maxiter': self.config.max_iterations, 'xatol': self.config.tolerance}
        )
        
        # Track convergence (simplified)
        self.convergence_history = [result.fun] * (result.nit + 1)
        
        return OptimizationResult(
            success=result.success,
            optimal_parameters=result.x,
            optimal_value=result.fun,
            iterations=result.nit,
            execution_time=0.0,  # Will be set by caller
            convergence_history=self.convergence_history.copy(),
            method="nelder_mead",
            error_message=result.message if not result.success else None
        )
    
    def _optimize_lbfgs(self, circuit: ParameterizedCircuit, objective_function: Callable,
                       initial_parameters: np.ndarray) -> OptimizationResult:
        """Optimize using L-BFGS-B method."""
        def objective_wrapper(params):
            circuit.set_parameters(params)
            return objective_function(circuit)
        
        # Set up bounds
        bounds = circuit.get_parameter_bounds()
        
        # Run optimization
        result = opt.minimize(
            objective_wrapper,
            initial_parameters,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        # Track convergence (simplified)
        self.convergence_history = [result.fun] * (result.nit + 1)
        
        return OptimizationResult(
            success=result.success,
            optimal_parameters=result.x,
            optimal_value=result.fun,
            iterations=result.nit,
            execution_time=0.0,  # Will be set by caller
            convergence_history=self.convergence_history.copy(),
            method="lbfgs",
            error_message=result.message if not result.success else None
        )
    
    def _optimize_gradient_descent(self, circuit: ParameterizedCircuit, objective_function: Callable,
                                  initial_parameters: np.ndarray) -> OptimizationResult:
        """Optimize using gradient descent."""
        parameters = initial_parameters.copy()
        best_parameters = parameters.copy()
        best_value = float('inf')
        
        for iteration in range(self.config.max_iterations):
            # Calculate gradient using finite differences
            gradient = self._calculate_gradient(circuit, objective_function, parameters)
            
            # Update parameters
            parameters -= self.config.learning_rate * gradient
            
            # Keep parameters in bounds
            bounds = circuit.get_parameter_bounds()
            for i, (lower, upper) in enumerate(bounds):
                parameters[i] = np.clip(parameters[i], lower, upper)
            
            # Evaluate objective
            circuit.set_parameters(parameters)
            current_value = objective_function(circuit)
            
            # Track convergence
            self.convergence_history.append(current_value)
            self.parameter_history.append(parameters.copy())
            
            # Update best result
            if current_value < best_value:
                best_value = current_value
                best_parameters = parameters.copy()
            
            # Check convergence
            if len(self.convergence_history) > 1:
                improvement = abs(self.convergence_history[-1] - self.convergence_history[-2])
                if improvement < self.config.convergence_threshold:
                    break
        
        return OptimizationResult(
            success=True,
            optimal_parameters=best_parameters,
            optimal_value=best_value,
            iterations=actual_iterations,
            execution_time=0.0,  # Will be set by caller
            convergence_history=self.convergence_history.copy(),
            method="gradient_descent"
        )
    
    def _optimize_adam(self, circuit: ParameterizedCircuit, objective_function: Callable,
                      initial_parameters: np.ndarray) -> OptimizationResult:
        """Optimize using Adam optimizer."""
        parameters = initial_parameters.copy()
        best_parameters = parameters.copy()
        best_value = float('inf')
        
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        # Initialize moments
        m = np.zeros_like(parameters)
        v = np.zeros_like(parameters)
        
        for iteration in range(self.config.max_iterations):
            # Calculate gradient
            gradient = self._calculate_gradient(circuit, objective_function, parameters)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** (iteration + 1))
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** (iteration + 1))
            
            # Update parameters
            parameters -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Keep parameters in bounds
            bounds = circuit.get_parameter_bounds()
            for i, (lower, upper) in enumerate(bounds):
                parameters[i] = np.clip(parameters[i], lower, upper)
            
            # Evaluate objective
            circuit.set_parameters(parameters)
            current_value = objective_function(circuit)
            
            # Track convergence
            self.convergence_history.append(current_value)
            self.parameter_history.append(parameters.copy())
            
            # Update best result
            if current_value < best_value:
                best_value = current_value
                best_parameters = parameters.copy()
            
            # Check convergence
            if len(self.convergence_history) > 10:
                recent_improvement = abs(self.convergence_history[-1] - self.convergence_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    break
        
        return OptimizationResult(
            success=True,
            optimal_parameters=best_parameters,
            optimal_value=best_value,
            iterations=actual_iterations,
            execution_time=0.0,  # Will be set by caller
            convergence_history=self.convergence_history.copy(),
            method="adam"
        )
    
    def _calculate_gradient(self, circuit: ParameterizedCircuit, objective_function: Callable,
                           parameters: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Calculate gradient using finite differences."""
        gradient = np.zeros_like(parameters)
        
        for i in range(len(parameters)):
            # Forward difference
            params_plus = parameters.copy()
            params_plus[i] += epsilon
            circuit.set_parameters(params_plus)
            value_plus = objective_function(circuit)
            
            # Backward difference
            params_minus = parameters.copy()
            params_minus[i] -= epsilon
            circuit.set_parameters(params_minus)
            value_minus = objective_function(circuit)
            
            # Calculate gradient
            gradient[i] = (value_plus - value_minus) / (2 * epsilon)
        
        return gradient
    
    def _save_optimization_traces(self, circuit: ParameterizedCircuit, result: OptimizationResult):
        """Save optimization traces and convergence plots."""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save convergence history
        with open(os.path.join(self.config.output_dir, 'convergence_history.json'), 'w') as f:
            json.dump({
                'convergence_history': result.convergence_history,
                'parameter_history': [params.tolist() for params in self.parameter_history],
                'method': result.method,
                'iterations': result.iterations,
                'optimal_value': result.optimal_value,
                'success': result.success
            }, f, indent=2)
        
        # Save convergence plot
        if len(result.convergence_history) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(result.convergence_history)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title(f'Optimization Convergence - {result.method.upper()}')
            plt.grid(True)
            plt.savefig(os.path.join(self.config.output_dir, 'convergence_plot.png'))
            plt.close()
        
        # Save parameter evolution
        if len(self.parameter_history) > 1:
            plt.figure(figsize=(12, 8))
            for i in range(circuit.num_parameters):
                param_values = [params[i] for params in self.parameter_history]
                plt.plot(param_values, label=f'Parameter {i} ({circuit.parameter_names[i]})')
            plt.xlabel('Iteration')
            plt.ylabel('Parameter Value')
            plt.title(f'Parameter Evolution - {result.method.upper()}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.config.output_dir, 'parameter_evolution.png'))
            plt.close()
        
        # Save CSV data
        with open(os.path.join(self.config.output_dir, 'optimization_data.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'objective_value'] + [f'param_{i}' for i in range(circuit.num_parameters)])
            
            for i, (obj_val, params) in enumerate(zip(result.convergence_history, self.parameter_history)):
                writer.writerow([i, obj_val] + params.tolist())


class NoiseAwareOptimization:
    """Noise-aware optimization for quantum circuits."""
    
    def __init__(self, noise_model: Optional[Dict[str, Any]] = None):
        self.noise_model = noise_model or {'depolarizing_error': 0.01}
    
    def optimize_with_noise(self, circuit: ParameterizedCircuit, objective_function: Callable,
                           config: OptimizationConfig) -> OptimizationResult:
        """Optimize circuit with noise-aware objective function."""
        # Create noise-aware objective function
        def noisy_objective(circuit_instance):
            # Apply noise to circuit
            noisy_circuit = self._apply_noise_to_circuit(circuit_instance)
            return objective_function(noisy_circuit)
        
        # Run optimization
        engine = OptimizationEngine(config)
        return engine.optimize(circuit, noisy_objective)
    
    def _apply_noise_to_circuit(self, circuit: ParameterizedCircuit) -> ParameterizedCircuit:
        """Apply noise model to circuit."""
        # This is a simplified noise application
        # In practice, this would involve more sophisticated noise models
        noisy_circuit = circuit
        return noisy_circuit


class ConstrainedOptimization:
    """Constrained optimization for quantum circuits."""
    
    def __init__(self, constraints: List[Callable]):
        self.constraints = constraints
    
    def optimize_with_constraints(self, circuit: ParameterizedCircuit, objective_function: Callable,
                                 config: OptimizationConfig) -> OptimizationResult:
        """Optimize circuit with constraints."""
        def constrained_objective(circuit_instance, parameters=None):
            # Check constraints if parameters are provided
            if parameters is not None:
                for constraint in self.constraints:
                    if not constraint(parameters):
                        return float('inf')  # Penalty for constraint violation
            
            return objective_function(circuit_instance)
        
        # Run optimization with constraint checking
        engine = OptimizationEngine(config)
        return engine.optimize_with_constraints(circuit, constrained_objective, self.constraints)
