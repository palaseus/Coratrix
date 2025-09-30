"""
Edge Execution Mode for Coratrix 4.0
====================================

Enables lightweight "compiled circuit packages" that can run partially offline
(precompiled + classical orchestration) for deployment on edge GPUs or low-power clusters.
"""

import numpy as np
import json
import pickle
import gzip
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class CompiledCircuit:
    """Represents a compiled quantum circuit package."""
    circuit_id: str
    circuit_hash: str
    compiled_gates: List[Dict[str, Any]]
    optimization_level: str
    target_qubits: int
    estimated_memory_mb: float
    estimated_execution_time: float
    dependencies: List[str]
    metadata: Dict[str, Any]

@dataclass
class EdgeExecutionConfig:
    """Configuration for edge execution."""
    max_memory_mb: float = 512.0
    max_execution_time: float = 10.0
    enable_compression: bool = True
    enable_caching: bool = True
    cache_size_mb: float = 100.0
    fallback_to_cloud: bool = True
    cloud_endpoint: Optional[str] = None

@dataclass
class EdgeExecutionResult:
    """Result of edge execution."""
    success: bool
    result_data: Optional[np.ndarray]
    execution_time: float
    memory_used: float
    execution_method: str  # 'edge', 'cloud', 'hybrid'
    error_message: Optional[str] = None

class CircuitCompiler:
    """
    Compiles quantum circuits into optimized packages for edge execution.
    """
    
    def __init__(self, config: EdgeExecutionConfig):
        self.config = config
        self.compiled_circuits: Dict[str, CompiledCircuit] = {}
        self.compilation_cache: Dict[str, str] = {}  # hash -> circuit_id mapping
    
    def compile_circuit(self, circuit: List[Dict[str, Any]], 
                       optimization_level: str = "medium") -> CompiledCircuit:
        """Compile a quantum circuit for edge execution."""
        start_time = time.time()
        
        # Create circuit hash
        circuit_hash = self._hash_circuit(circuit)
        
        # Check if already compiled
        if circuit_hash in self.compilation_cache:
            circuit_id = self.compilation_cache[circuit_hash]
            return self.compiled_circuits[circuit_id]
        
        # Create circuit ID
        circuit_id = f"circuit_{len(self.compiled_circuits)}"
        
        # Optimize circuit for edge execution
        optimized_circuit = self._optimize_for_edge(circuit, optimization_level)
        
        # Estimate resource requirements
        memory_estimate = self._estimate_memory_usage(optimized_circuit)
        time_estimate = self._estimate_execution_time(optimized_circuit)
        
        # Check if circuit fits within edge constraints
        if memory_estimate > self.config.max_memory_mb:
            logger.warning(f"Circuit requires {memory_estimate:.1f}MB, exceeds edge limit")
            if not self.config.fallback_to_cloud:
                raise RuntimeError("Circuit too large for edge execution")
        
        if time_estimate > self.config.max_execution_time:
            logger.warning(f"Circuit estimated {time_estimate:.1f}s, exceeds edge limit")
            if not self.config.fallback_to_cloud:
                raise RuntimeError("Circuit too slow for edge execution")
        
        # Create compiled circuit
        compiled_circuit = CompiledCircuit(
            circuit_id=circuit_id,
            circuit_hash=circuit_hash,
            compiled_gates=optimized_circuit,
            optimization_level=optimization_level,
            target_qubits=self._count_qubits(optimized_circuit),
            estimated_memory_mb=memory_estimate,
            estimated_execution_time=time_estimate,
            dependencies=self._extract_dependencies(optimized_circuit),
            metadata={
                'compilation_time': time.time() - start_time,
                'optimization_applied': True,
                'edge_compatible': memory_estimate <= self.config.max_memory_mb
            }
        )
        
        # Store compiled circuit
        self.compiled_circuits[circuit_id] = compiled_circuit
        self.compilation_cache[circuit_hash] = circuit_id
        
        logger.info(f"Circuit compiled: {circuit_id}, {memory_estimate:.1f}MB, {time_estimate:.1f}s")
        
        return compiled_circuit
    
    def _optimize_for_edge(self, circuit: List[Dict[str, Any]], 
                          optimization_level: str) -> List[Dict[str, Any]]:
        """Optimize circuit for edge execution."""
        optimized = circuit.copy()
        
        if optimization_level == "high":
            # Apply aggressive optimizations
            optimized = self._apply_aggressive_optimizations(optimized)
        elif optimization_level == "medium":
            # Apply moderate optimizations
            optimized = self._apply_moderate_optimizations(optimized)
        else:  # "low"
            # Apply basic optimizations
            optimized = self._apply_basic_optimizations(optimized)
        
        return optimized
    
    def _apply_basic_optimizations(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply basic optimizations for edge execution."""
        # Remove redundant gates
        optimized = []
        for i, gate in enumerate(circuit):
            if i == 0 or not self._is_redundant(gate, circuit[i-1]):
                optimized.append(gate)
        
        return optimized
    
    def _apply_moderate_optimizations(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply moderate optimizations for edge execution."""
        optimized = self._apply_basic_optimizations(circuit)
        
        # Merge adjacent single-qubit gates
        optimized = self._merge_single_qubit_gates(optimized)
        
        return optimized
    
    def _apply_aggressive_optimizations(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply aggressive optimizations for edge execution."""
        optimized = self._apply_moderate_optimizations(circuit)
        
        # Apply circuit-specific optimizations
        optimized = self._apply_circuit_specific_optimizations(optimized)
        
        return optimized
    
    def _is_redundant(self, gate1: Dict[str, Any], gate2: Dict[str, Any]) -> bool:
        """Check if two gates are redundant."""
        # Simple redundancy check - in practice, you'd implement more sophisticated logic
        return (gate1.get('type') == gate2.get('type') and
                gate1.get('gate') == gate2.get('gate') and
                gate1.get('qubit') == gate2.get('qubit'))
    
    def _merge_single_qubit_gates(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge adjacent single-qubit gates on the same qubit."""
        if len(circuit) < 2:
            return circuit
        
        merged = []
        i = 0
        
        while i < len(circuit):
            current_gate = circuit[i]
            
            # Check if next gate can be merged
            if (i + 1 < len(circuit) and
                current_gate.get('type') == 'single_qubit' and
                circuit[i + 1].get('type') == 'single_qubit' and
                current_gate.get('qubit') == circuit[i + 1].get('qubit')):
                
                # Merge gates (simplified)
                merged_gate = current_gate.copy()
                merged_gate['merged'] = True
                merged.append(merged_gate)
                i += 2  # Skip next gate
            else:
                merged.append(current_gate)
                i += 1
        
        return merged
    
    def _apply_circuit_specific_optimizations(self, circuit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply circuit-specific optimizations."""
        # This would implement domain-specific optimizations
        # For now, just return the circuit as-is
        return circuit
    
    def _estimate_memory_usage(self, circuit: List[Dict[str, Any]]) -> float:
        """Estimate memory usage for circuit execution."""
        num_qubits = self._count_qubits(circuit)
        
        # Estimate based on qubit count and circuit complexity
        base_memory = 2 ** num_qubits * 16 / (1024 * 1024)  # Complex128 state vector
        complexity_factor = len(circuit) / 10.0  # Circuit complexity
        
        return base_memory * complexity_factor
    
    def _estimate_execution_time(self, circuit: List[Dict[str, Any]]) -> float:
        """Estimate execution time for circuit."""
        # Simple estimation based on gate count and types
        total_time = 0.0
        
        for gate in circuit:
            if gate.get('type') == 'single_qubit':
                total_time += 0.001  # 1ms per single-qubit gate
            elif gate.get('type') == 'two_qubit':
                total_time += 0.005  # 5ms per two-qubit gate
            else:
                total_time += 0.01   # 10ms for other gates
        
        return total_time
    
    def _count_qubits(self, circuit: List[Dict[str, Any]]) -> int:
        """Count qubits used in circuit."""
        qubits = set()
        for gate in circuit:
            if 'qubit' in gate:
                qubits.add(gate['qubit'])
            if 'control' in gate:
                qubits.add(gate['control'])
            if 'target' in gate:
                qubits.add(gate['target'])
        return len(qubits)
    
    def _extract_dependencies(self, circuit: List[Dict[str, Any]]) -> List[str]:
        """Extract dependencies for circuit execution."""
        dependencies = []
        
        # Check for specific gate dependencies
        for gate in circuit:
            gate_type = gate.get('gate', '')
            if gate_type in ['RX', 'RY', 'RZ']:
                dependencies.append('rotation_gates')
            elif gate_type == 'CNOT':
                dependencies.append('entangling_gates')
            elif gate_type in ['H', 'X', 'Y', 'Z']:
                dependencies.append('pauli_gates')
        
        return list(set(dependencies))
    
    def _hash_circuit(self, circuit: List[Dict[str, Any]]) -> str:
        """Create hash for circuit identification."""
        circuit_str = json.dumps(circuit, sort_keys=True)
        return hashlib.sha256(circuit_str.encode()).hexdigest()
    
    def get_compiled_circuit(self, circuit_id: str) -> Optional[CompiledCircuit]:
        """Get a compiled circuit by ID."""
        return self.compiled_circuits.get(circuit_id)
    
    def save_compiled_circuit(self, circuit_id: str, filepath: str):
        """Save compiled circuit to file."""
        if circuit_id not in self.compiled_circuits:
            raise ValueError(f"Circuit {circuit_id} not found")
        
        circuit = self.compiled_circuits[circuit_id]
        
        # Serialize circuit data
        circuit_data = {
            'circuit_id': circuit.circuit_id,
            'circuit_hash': circuit.circuit_hash,
            'compiled_gates': circuit.compiled_gates,
            'optimization_level': circuit.optimization_level,
            'target_qubits': circuit.target_qubits,
            'estimated_memory_mb': circuit.estimated_memory_mb,
            'estimated_execution_time': circuit.estimated_execution_time,
            'dependencies': circuit.dependencies,
            'metadata': circuit.metadata
        }
        
        # Save with compression if enabled
        if self.config.enable_compression:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(circuit_data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(circuit_data, f)
        
        logger.info(f"Compiled circuit saved to {filepath}")
    
    def load_compiled_circuit(self, filepath: str) -> CompiledCircuit:
        """Load compiled circuit from file."""
        try:
            if self.config.enable_compression:
                with gzip.open(filepath, 'rb') as f:
                    circuit_data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    circuit_data = pickle.load(f)
            
            # Reconstruct CompiledCircuit
            circuit = CompiledCircuit(
                circuit_id=circuit_data['circuit_id'],
                circuit_hash=circuit_data['circuit_hash'],
                compiled_gates=circuit_data['compiled_gates'],
                optimization_level=circuit_data['optimization_level'],
                target_qubits=circuit_data['target_qubits'],
                estimated_memory_mb=circuit_data['estimated_memory_mb'],
                estimated_execution_time=circuit_data['estimated_execution_time'],
                dependencies=circuit_data['dependencies'],
                metadata=circuit_data['metadata']
            )
            
            # Add to cache
            self.compiled_circuits[circuit.circuit_id] = circuit
            self.compilation_cache[circuit.circuit_hash] = circuit.circuit_id
            
            logger.info(f"Compiled circuit loaded from {filepath}")
            return circuit
            
        except Exception as e:
            logger.error(f"Error loading compiled circuit: {e}")
            raise


class EdgeExecutor:
    """
    Executes compiled circuits on edge devices with fallback to cloud.
    """
    
    def __init__(self, config: EdgeExecutionConfig, compiler: CircuitCompiler):
        self.config = config
        self.compiler = compiler
        self.execution_cache: Dict[str, EdgeExecutionResult] = {}
        self.performance_stats = {
            'edge_executions': 0,
            'cloud_executions': 0,
            'hybrid_executions': 0,
            'total_execution_time': 0.0,
            'average_memory_usage': 0.0
        }
    
    def execute_circuit(self, circuit_id: str, input_data: Optional[np.ndarray] = None) -> EdgeExecutionResult:
        """Execute a compiled circuit on edge or cloud."""
        start_time = time.time()
        
        # Get compiled circuit
        compiled_circuit = self.compiler.get_compiled_circuit(circuit_id)
        if not compiled_circuit:
            return EdgeExecutionResult(
                success=False,
                result_data=None,
                execution_time=0.0,
                memory_used=0.0,
                execution_method='error',
                error_message=f"Circuit {circuit_id} not found"
            )
        
        # Check if circuit is suitable for edge execution
        if (compiled_circuit.estimated_memory_mb > self.config.max_memory_mb or
            compiled_circuit.estimated_execution_time > self.config.max_execution_time):
            
            if self.config.fallback_to_cloud:
                return self._execute_on_cloud(compiled_circuit, input_data)
            else:
                return EdgeExecutionResult(
                    success=False,
                    result_data=None,
                    execution_time=0.0,
                    memory_used=0.0,
                    execution_method='error',
                    error_message="Circuit too large for edge execution"
                )
        
        # Execute on edge
        return self._execute_on_edge(compiled_circuit, input_data)
    
    def _execute_on_edge(self, compiled_circuit: CompiledCircuit, 
                        input_data: Optional[np.ndarray]) -> EdgeExecutionResult:
        """Execute circuit on edge device."""
        start_time = time.time()
        
        try:
            # Simulate edge execution
            result_data = self._simulate_circuit_execution(compiled_circuit, input_data)
            
            execution_time = time.time() - start_time
            memory_used = compiled_circuit.estimated_memory_mb
            
            result = EdgeExecutionResult(
                success=True,
                result_data=result_data,
                execution_time=execution_time,
                memory_used=memory_used,
                execution_method='edge'
            )
            
            # Update statistics
            self.performance_stats['edge_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['average_memory_usage'] = (
                (self.performance_stats['average_memory_usage'] * 
                 (self.performance_stats['edge_executions'] - 1) + memory_used) /
                self.performance_stats['edge_executions']
            )
            
            logger.info(f"Circuit executed on edge: {execution_time:.3f}s, {memory_used:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Edge execution failed: {e}")
            return EdgeExecutionResult(
                success=False,
                result_data=None,
                execution_time=time.time() - start_time,
                memory_used=0.0,
                execution_method='edge',
                error_message=str(e)
            )
    
    def _execute_on_cloud(self, compiled_circuit: CompiledCircuit, 
                          input_data: Optional[np.ndarray]) -> EdgeExecutionResult:
        """Execute circuit on cloud with hybrid orchestration."""
        start_time = time.time()
        
        try:
            # Simulate cloud execution with hybrid orchestration
            result_data = self._simulate_hybrid_execution(compiled_circuit, input_data)
            
            execution_time = time.time() - start_time
            memory_used = compiled_circuit.estimated_memory_mb
            
            result = EdgeExecutionResult(
                success=True,
                result_data=result_data,
                execution_time=execution_time,
                memory_used=memory_used,
                execution_method='hybrid'
            )
            
            # Update statistics
            self.performance_stats['hybrid_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            logger.info(f"Circuit executed on cloud: {execution_time:.3f}s, {memory_used:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Cloud execution failed: {e}")
            return EdgeExecutionResult(
                success=False,
                result_data=None,
                execution_time=time.time() - start_time,
                memory_used=0.0,
                execution_method='cloud',
                error_message=str(e)
            )
    
    def _simulate_circuit_execution(self, compiled_circuit: CompiledCircuit, 
                                   input_data: Optional[np.ndarray]) -> np.ndarray:
        """Simulate circuit execution (placeholder for actual implementation)."""
        # This would integrate with the actual quantum simulation
        # For now, return a simulated result
        num_qubits = compiled_circuit.target_qubits
        result_size = 2 ** num_qubits
        
        if input_data is not None:
            return input_data.copy()
        else:
            # Return a simulated quantum state
            return np.random.rand(result_size) + 1j * np.random.rand(result_size)
    
    def _simulate_hybrid_execution(self, compiled_circuit: CompiledCircuit, 
                                  input_data: Optional[np.ndarray]) -> np.ndarray:
        """Simulate hybrid cloud execution."""
        # This would implement actual hybrid cloud execution
        # For now, return a simulated result
        return self._simulate_circuit_execution(compiled_circuit, input_data)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for edge execution."""
        total_executions = (self.performance_stats['edge_executions'] + 
                           self.performance_stats['cloud_executions'] + 
                           self.performance_stats['hybrid_executions'])
        
        return {
            **self.performance_stats,
            'total_executions': total_executions,
            'average_execution_time': (
                self.performance_stats['total_execution_time'] / max(total_executions, 1)
            )
        }
    
    def cleanup(self):
        """Clean up edge executor resources."""
        self.execution_cache.clear()
        logger.info("Edge executor cleaned up")


class EdgeExecutionManager:
    """
    Manages edge execution with caching, optimization, and fallback strategies.
    """
    
    def __init__(self, config: EdgeExecutionConfig):
        self.config = config
        self.compiler = CircuitCompiler(config)
        self.executor = EdgeExecutor(config, self.compiler)
        self.cache_dir = tempfile.mkdtemp(prefix="coratrix_edge_")
        
        logger.info(f"Edge execution manager initialized with cache dir: {self.cache_dir}")
    
    def compile_and_execute(self, circuit: List[Dict[str, Any]], 
                          input_data: Optional[np.ndarray] = None,
                          optimization_level: str = "medium") -> EdgeExecutionResult:
        """Compile and execute a circuit with edge optimization."""
        # Compile circuit
        compiled_circuit = self.compiler.compile_circuit(circuit, optimization_level)
        
        # Execute circuit
        result = self.executor.execute_circuit(compiled_circuit.circuit_id, input_data)
        
        return result
    
    def get_compilation_statistics(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        return {
            'total_compiled_circuits': len(self.compiler.compiled_circuits),
            'cache_hit_rate': len(self.compiler.compilation_cache) / max(len(self.compiler.compiled_circuits), 1),
            'average_memory_estimate': np.mean([
                c.estimated_memory_mb for c in self.compiler.compiled_circuits.values()
            ]) if self.compiler.compiled_circuits else 0.0
        }
    
    def cleanup(self):
        """Clean up edge execution manager."""
        self.executor.cleanup()
        # Clean up cache directory
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        logger.info("Edge execution manager cleaned up")
