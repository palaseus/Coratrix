"""
Multi-subspace Grover search with interference diagnostics.

This module implements concurrent Grover search across multiple subspaces
with interference analysis and success probability reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict

from core.qubit import QuantumState
from core.scalable_quantum_state import ScalableQuantumState
from core.gates import HGate, XGate, ZGate, CNOTGate
from core.circuit import QuantumCircuit
from core.advanced_algorithms import EntanglementMonotones, EntanglementNetwork


class SubspaceType(Enum):
    """Types of quantum subspaces."""
    GHZ = "ghz"
    W = "w"
    CLUSTER = "cluster"
    BELL = "bell"
    CUSTOM = "custom"


@dataclass
class SubspaceConfig:
    """Configuration for quantum subspace."""
    subspace_type: SubspaceType
    qubit_indices: List[int]
    target_state: Optional[str] = None
    entanglement_threshold: float = 0.7
    interference_weight: float = 1.0


@dataclass
class GroverResult:
    """Result from Grover search."""
    success: bool
    success_probability: float
    iterations: int
    final_state: np.ndarray
    measurement_counts: Dict[str, int]
    interference_metrics: Dict[str, float]
    execution_time: float
    subspace_results: List[Dict[str, Any]]


@dataclass
class InterferenceDiagnostics:
    """Interference diagnostics for multi-subspace Grover."""
    interference_matrix: np.ndarray
    coherence_metrics: Dict[str, float]
    entanglement_evolution: List[float]
    success_probability_evolution: List[float]
    interference_heatmap_data: Dict[str, Any]


class MultiSubspaceGrover:
    """Multi-subspace Grover search implementation."""
    
    def __init__(self, num_qubits: int, subspaces: List[SubspaceConfig]):
        self.num_qubits = num_qubits
        self.subspaces = subspaces
        self.num_subspaces = len(subspaces)
        self.dimension = 2 ** num_qubits
        
        # Initialize entanglement analysis
        self.entanglement_monotones = EntanglementMonotones()
        self.entanglement_network = EntanglementNetwork(num_qubits)
        
        # Interference tracking
        self.interference_history = []
        self.entanglement_history = []
        self.success_probability_history = []
    
    def search(self, target_items: List[str], max_iterations: int = 100, 
               shots: int = 1024, enable_interference: bool = True) -> GroverResult:
        """Perform multi-subspace Grover search."""
        start_time = time.time()
        
        # Initialize quantum state
        state = ScalableQuantumState(self.num_qubits, use_gpu=False, sparse_threshold=8)
        
        # Create superposition
        self._create_superposition(state)
        
        # Initialize tracking
        self.interference_history = []
        self.entanglement_history = []
        self.success_probability_history = []
        
        # Perform Grover iterations
        best_iteration = 0
        best_success_prob = 0.0
        
        for iteration in range(max_iterations):
            # Apply Grover operator to each subspace
            subspace_results = []
            for i, subspace in enumerate(self.subspaces):
                result = self._apply_subspace_grover(state, subspace, target_items, iteration)
                subspace_results.append(result)
            
            # Calculate interference between subspaces
            if enable_interference:
                interference_metrics = self._calculate_interference(state, subspace_results)
                self.interference_history.append(interference_metrics)
            
            # Track entanglement evolution
            entanglement = state.get_entanglement_entropy()
            self.entanglement_history.append(entanglement)
            
            # Calculate success probability
            success_prob = self._calculate_success_probability(state, target_items)
            self.success_probability_history.append(success_prob)
            
            # Update best result
            if success_prob > best_success_prob:
                best_success_prob = success_prob
                best_iteration = iteration
            
            # Check for convergence
            if success_prob > 0.95:  # 95% success threshold
                break
        
        # Perform final measurement
        measurement_counts = self._measure_state(state, shots)
        
        # Calculate final metrics
        execution_time = time.time() - start_time
        interference_metrics = self._calculate_final_interference_metrics()
        
        return GroverResult(
            success=best_success_prob > 0.5,
            success_probability=best_success_prob,
            iterations=len(self.entanglement_history),
            final_state=state.to_dense(),
            measurement_counts=measurement_counts,
            interference_metrics=interference_metrics,
            execution_time=execution_time,
            subspace_results=subspace_results
        )
    
    def _create_superposition(self, state: ScalableQuantumState):
        """Create initial superposition state."""
        # Apply Hadamard gates to all qubits
        h_gate = HGate()
        for i in range(self.num_qubits):
            h_gate.apply(state, [i])
    
    def _apply_subspace_grover(self, state: ScalableQuantumState, subspace: SubspaceConfig,
                              target_items: List[str], iteration: int) -> Dict[str, Any]:
        """Apply Grover operator to specific subspace."""
        # Create oracle for this subspace
        oracle = self._create_subspace_oracle(subspace, target_items)
        
        # Apply oracle
        self._apply_oracle(state, oracle, subspace.qubit_indices)
        
        # Apply diffusion operator
        self._apply_diffusion_operator(state, subspace.qubit_indices)
        
        # Calculate subspace metrics
        subspace_entanglement = self._calculate_subspace_entanglement(state, subspace)
        subspace_success_prob = self._calculate_subspace_success_probability(
            state, subspace, target_items
        )
        
        return {
            'subspace_type': subspace.subspace_type.value,
            'entanglement': subspace_entanglement,
            'success_probability': subspace_success_prob,
            'iteration': iteration,
            'qubit_indices': subspace.qubit_indices
        }
    
    def _create_subspace_oracle(self, subspace: SubspaceConfig, target_items: List[str]) -> Dict[str, Any]:
        """Create oracle for specific subspace."""
        oracle = {
            'target_items': target_items,
            'subspace_type': subspace.subspace_type.value,
            'qubit_indices': subspace.qubit_indices,
            'entanglement_threshold': subspace.entanglement_threshold
        }
        
        # Add subspace-specific oracle logic
        if subspace.subspace_type == SubspaceType.GHZ:
            oracle['ghz_signature'] = self._create_ghz_oracle_signature(subspace)
        elif subspace.subspace_type == SubspaceType.W:
            oracle['w_signature'] = self._create_w_oracle_signature(subspace)
        elif subspace.subspace_type == SubspaceType.CLUSTER:
            oracle['cluster_signature'] = self._create_cluster_oracle_signature(subspace)
        
        return oracle
    
    def _create_ghz_oracle_signature(self, subspace: SubspaceConfig) -> List[int]:
        """Create GHZ state oracle signature."""
        # GHZ state: |000...0⟩ + |111...1⟩
        signature = []
        for i in range(2**len(subspace.qubit_indices)):
            binary = format(i, f'0{len(subspace.qubit_indices)}b')
            if all(bit == '0' for bit in binary) or all(bit == '1' for bit in binary):
                signature.append(i)
        return signature
    
    def _create_w_oracle_signature(self, subspace: SubspaceConfig) -> List[int]:
        """Create W state oracle signature."""
        # W state: equal superposition of all states with exactly one |1⟩
        signature = []
        for i in range(2**len(subspace.qubit_indices)):
            binary = format(i, f'0{len(subspace.qubit_indices)}b')
            if binary.count('1') == 1:
                signature.append(i)
        return signature
    
    def _create_cluster_oracle_signature(self, subspace: SubspaceConfig) -> List[int]:
        """Create cluster state oracle signature."""
        # Cluster state: specific entanglement pattern
        signature = []
        for i in range(2**len(subspace.qubit_indices)):
            binary = format(i, f'0{len(subspace.qubit_indices)}b')
            # Check for cluster state pattern
            if self._is_cluster_state_pattern(binary):
                signature.append(i)
        return signature
    
    def _is_cluster_state_pattern(self, binary: str) -> bool:
        """Check if binary string matches cluster state pattern."""
        # Simplified cluster state pattern detection
        # In practice, this would be more sophisticated
        return len(binary) >= 2 and binary.count('1') >= 1
    
    def _apply_oracle(self, state: ScalableQuantumState, oracle: Dict[str, Any], 
                     qubit_indices: List[int]):
        """Apply oracle to quantum state."""
        # Mark target states with phase flip
        for i in range(self.dimension):
            # Check if state matches oracle criteria
            if self._state_matches_oracle(i, oracle, qubit_indices):
                # Apply phase flip
                amplitude = state.get_amplitude(i)
                state.set_amplitude(i, -amplitude)
    
    def _state_matches_oracle(self, state_index: int, oracle: Dict[str, Any], 
                             qubit_indices: List[int]) -> bool:
        """Check if state matches oracle criteria."""
        # Convert state index to binary
        binary = format(state_index, f'0{self.num_qubits}b')
        
        # Extract relevant qubit states (qubit indices are 0-based from LSB)
        # Reverse the binary string to get LSB first, then extract qubits
        binary_reversed = binary[::-1]
        relevant_bits = ''.join(binary_reversed[i] for i in qubit_indices)
        relevant_index = int(relevant_bits, 2)
        
        # Check against oracle signatures
        if 'ghz_signature' in oracle:
            return relevant_index in oracle['ghz_signature']
        elif 'w_signature' in oracle:
            return relevant_index in oracle['w_signature']
        elif 'cluster_signature' in oracle:
            return relevant_index in oracle['cluster_signature']
        
        return False
    
    def _apply_diffusion_operator(self, state: ScalableQuantumState, qubit_indices: List[int]):
        """Apply diffusion operator to specific qubits."""
        # Apply Hadamard gates to create superposition
        h_gate = HGate()
        for i in qubit_indices:
            h_gate.apply(state, [i])
        
        # Apply phase flip to |0...0⟩ state (all zeros)
        # This is the key part of the diffusion operator
        zero_state_index = 0  # |0...0⟩ state
        zero_amplitude = state.get_amplitude(zero_state_index)
        state.set_amplitude(zero_state_index, -zero_amplitude)
        
        # Apply Hadamard gates again
        for i in qubit_indices:
            h_gate.apply(state, [i])
    
    def _calculate_subspace_entanglement(self, state: ScalableQuantumState, 
                                        subspace: SubspaceConfig) -> float:
        """Calculate entanglement within subspace."""
        if len(subspace.qubit_indices) < 2:
            return 0.0
        
        # Calculate entanglement between first two qubits in subspace
        return state.get_entanglement_entropy()
    
    def _calculate_subspace_success_probability(self, state: ScalableQuantumState,
                                              subspace: SubspaceConfig, 
                                              target_items: List[str]) -> float:
        """Calculate success probability for subspace."""
        # Simplified success probability calculation
        # In practice, this would be more sophisticated
        probabilities = state.get_probabilities()
        
        # Sum probabilities of target states
        success_prob = 0.0
        for target in target_items:
            # Convert target to state index
            target_index = self._target_to_state_index(target)
            if target_index < len(probabilities):
                success_prob += probabilities[target_index]
        
        return float(success_prob)
    
    def _target_to_state_index(self, target: str) -> int:
        """Convert target string to state index."""
        # Simplified conversion
        # In practice, this would depend on the specific target format
        try:
            return int(target, 2)
        except ValueError:
            return 0
    
    def _calculate_interference(self, state: ScalableQuantumState, 
                              subspace_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate interference between subspaces."""
        # Calculate interference matrix
        interference_matrix = np.zeros((self.num_subspaces, self.num_subspaces))
        
        for i in range(self.num_subspaces):
            for j in range(self.num_subspaces):
                if i != j:
                    # Calculate interference between subspaces i and j
                    interference = self._calculate_pairwise_interference(
                        state, subspace_results[i], subspace_results[j]
                    )
                    interference_matrix[i, j] = interference
        
        # Calculate coherence metrics
        coherence_metrics = self._calculate_coherence_metrics(state, interference_matrix)
        
        return {
            'interference_matrix': interference_matrix.tolist(),
            'coherence_metrics': coherence_metrics,
            'total_interference': float(np.sum(interference_matrix))
        }
    
    def _calculate_pairwise_interference(self, state: ScalableQuantumState,
                                        result1: Dict[str, Any], 
                                        result2: Dict[str, Any]) -> float:
        """Calculate interference between two subspaces."""
        # Simplified interference calculation
        # In practice, this would involve more sophisticated quantum interference analysis
        
        # Calculate overlap between subspace states
        entanglement1 = result1.get('entanglement', 0.0)
        entanglement2 = result2.get('entanglement', 0.0)
        
        # Interference is related to entanglement correlation
        interference = abs(entanglement1 - entanglement2) / max(entanglement1 + entanglement2, 1e-10)
        
        return float(interference)
    
    def _calculate_coherence_metrics(self, state: ScalableQuantumState, 
                                   interference_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate coherence metrics for interference."""
        # Calculate various coherence measures
        total_interference = np.sum(interference_matrix)
        max_interference = np.max(interference_matrix)
        mean_interference = np.mean(interference_matrix)
        
        # Calculate coherence length (simplified)
        coherence_length = 1.0 / (1.0 + total_interference)
        
        return {
            'total_interference': float(total_interference),
            'max_interference': float(max_interference),
            'mean_interference': float(mean_interference),
            'coherence_length': float(coherence_length)
        }
    
    def _calculate_success_probability(self, state: ScalableQuantumState, 
                                     target_items: List[str]) -> float:
        """Calculate overall success probability."""
        probabilities = state.get_probabilities()
        
        success_prob = 0.0
        for target in target_items:
            target_index = self._target_to_state_index(target)
            if target_index < len(probabilities):
                success_prob += probabilities[target_index]
        
        return float(success_prob)
    
    def _measure_state(self, state: ScalableQuantumState, shots: int) -> Dict[str, int]:
        """Measure quantum state and return counts."""
        from core.measurement import Measurement
        
        measurement = Measurement(state)
        counts = measurement.measure_multiple(state, shots)
        
        return counts
    
    def _calculate_final_interference_metrics(self) -> Dict[str, Any]:
        """Calculate final interference metrics."""
        if not self.interference_history:
            return {}
        
        # Calculate interference evolution (limit to actual iterations)
        interference_evolution = [metrics.get('total_interference', 0.0) 
                                for metrics in self.interference_history[:len(self.entanglement_history)]]
        
        # Calculate interference heatmap data
        heatmap_data = self._create_interference_heatmap_data()
        
        return {
            'interference_evolution': interference_evolution,
            'entanglement_evolution': self.entanglement_history,
            'success_probability_evolution': self.success_probability_history,
            'heatmap_data': heatmap_data,
            'final_interference': interference_evolution[-1] if interference_evolution else 0.0
        }
    
    def _create_interference_heatmap_data(self) -> Dict[str, Any]:
        """Create interference heatmap data."""
        if not self.interference_history:
            return {}
        
        # Extract interference matrices
        interference_matrices = [metrics.get('interference_matrix', [])
                               for metrics in self.interference_history]
        
        # Create heatmap data
        heatmap_data = {
            'matrices': interference_matrices,
            'time_points': list(range(len(interference_matrices))),
            'subspace_labels': [f'Subspace {i}' for i in range(self.num_subspaces)]
        }
        
        return heatmap_data
    
    def generate_interference_visualization(self, result: GroverResult, 
                                          output_dir: str = "grover_visualizations"):
        """Generate interference visualization plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create interference heatmap
        self._create_interference_heatmap(result, output_dir)
        
        # Create time evolution plots
        self._create_time_evolution_plots(result, output_dir)
        
        # Create entanglement network visualization
        self._create_entanglement_network_plot(result, output_dir)
    
    def _create_interference_heatmap(self, result: GroverResult, output_dir: str):
        """Create interference heatmap visualization."""
        if 'heatmap_data' not in result.interference_metrics:
            return
        
        heatmap_data = result.interference_metrics['heatmap_data']
        matrices = heatmap_data.get('matrices', [])
        
        if not matrices:
            return
        
        # Create heatmap for final interference matrix
        final_matrix = np.array(matrices[-1])
        
        plt.figure(figsize=(10, 8))
        plt.imshow(final_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Interference Strength')
        plt.title('Multi-Subspace Grover Interference Heatmap')
        plt.xlabel('Subspace Index')
        plt.ylabel('Subspace Index')
        
        # Add subspace labels
        subspace_labels = heatmap_data.get('subspace_labels', [])
        if subspace_labels:
            plt.xticks(range(len(subspace_labels)), subspace_labels)
            plt.yticks(range(len(subspace_labels)), subspace_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interference_heatmap.png'), dpi=300)
        plt.close()
    
    def _create_time_evolution_plots(self, result: GroverResult, output_dir: str):
        """Create time evolution plots."""
        metrics = result.interference_metrics
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Interference evolution
        if 'interference_evolution' in metrics:
            axes[0, 0].plot(metrics['interference_evolution'])
            axes[0, 0].set_title('Interference Evolution')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Total Interference')
            axes[0, 0].grid(True)
        
        # Entanglement evolution
        if 'entanglement_evolution' in metrics:
            axes[0, 1].plot(metrics['entanglement_evolution'])
            axes[0, 1].set_title('Entanglement Evolution')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Entanglement Entropy')
            axes[0, 1].grid(True)
        
        # Success probability evolution
        if 'success_probability_evolution' in metrics:
            axes[1, 0].plot(metrics['success_probability_evolution'])
            axes[1, 0].set_title('Success Probability Evolution')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Success Probability')
            axes[1, 0].grid(True)
        
        # Combined plot
        if all(key in metrics for key in ['interference_evolution', 'entanglement_evolution', 
                                       'success_probability_evolution']):
            axes[1, 1].plot(metrics['interference_evolution'], label='Interference')
            axes[1, 1].plot(metrics['entanglement_evolution'], label='Entanglement')
            axes[1, 1].plot(metrics['success_probability_evolution'], label='Success Probability')
            axes[1, 1].set_title('Combined Evolution')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Normalized Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_evolution.png'), dpi=300)
        plt.close()
    
    def _create_entanglement_network_plot(self, result: GroverResult, output_dir: str):
        """Create entanglement network visualization."""
        # Create entanglement network analysis
        network_data = self.entanglement_network.calculate_entanglement_graph(
            ScalableQuantumState(self.num_qubits, use_gpu=False, sparse_threshold=8)
        )
        
        # Create network plot
        plt.figure(figsize=(12, 8))
        
        # Plot entanglement matrix
        entanglement_matrix = np.array(network_data['entanglement_matrix'])
        
        plt.imshow(entanglement_matrix, cmap='plasma', interpolation='nearest')
        plt.colorbar(label='Entanglement Strength')
        plt.title('Entanglement Network')
        plt.xlabel('Qubit Index')
        plt.ylabel('Qubit Index')
        
        # Add qubit labels
        qubit_labels = [f'Q{i}' for i in range(self.num_qubits)]
        plt.xticks(range(len(qubit_labels)), qubit_labels)
        plt.yticks(range(len(qubit_labels)), qubit_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entanglement_network.png'), dpi=300)
        plt.close()
    
    def export_results(self, result: GroverResult, filename: str):
        """Export Grover search results to JSON file."""
        # Prepare data for export
        export_data = {
            'success': result.success,
            'success_probability': result.success_probability,
            'iterations': result.iterations,
            'execution_time': result.execution_time,
            'measurement_counts': result.measurement_counts,
            'interference_metrics': result.interference_metrics,
            'subspace_results': result.subspace_results,
            'num_qubits': self.num_qubits,
            'num_subspaces': self.num_subspaces,
            'subspace_configs': [
                {
                    'subspace_type': subspace.subspace_type.value,
                    'qubit_indices': subspace.qubit_indices,
                    'entanglement_threshold': subspace.entanglement_threshold,
                    'interference_weight': subspace.interference_weight
                }
                for subspace in self.subspaces
            ]
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
