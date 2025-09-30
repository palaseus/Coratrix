"""
Advanced visualization engine for quantum exploration.

This module provides comprehensive visualization capabilities
including ASCII circuit diagrams, probability heatmaps,
and dynamic state evolution visualization.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from core.qubit import QuantumState
from visualization.circuit_diagram import CircuitDiagram
from visualization.probability_heatmap import ProbabilityHeatmap
from visualization.quantum_state_plotter import QuantumStatePlotter


class VisualizationEngine:
    """
    Advanced visualization engine for quantum exploration.
    
    Provides comprehensive visualization capabilities including
    circuit diagrams, probability heatmaps, and state evolution.
    """
    
    def __init__(self, width: int = 80):
        """
        Initialize the visualization engine.
        
        Args:
            width: Maximum width for visualizations
        """
        self.width = width
        self.circuit_diagram = CircuitDiagram(width)
        self.probability_heatmap = ProbabilityHeatmap(width)
        self.state_plotter = QuantumStatePlotter()
        
        # Visualization data storage
        self.visualization_history = []
        self.circuit_diagrams = {}
        self.probability_heatmaps = {}
        self.state_plots = {}
    
    def visualize_algorithm_execution(self, algorithm_name: str, 
                                   states: List[QuantumState],
                                   gates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Visualize complete algorithm execution.
        
        Args:
            algorithm_name: Name of the algorithm
            states: List of quantum states during execution
            gates: List of gates applied during execution
        
        Returns:
            Comprehensive visualization data
        """
        visualization_data = {
            'algorithm': algorithm_name,
            'num_states': len(states),
            'num_gates': len(gates),
            'circuit_diagram': None,
            'state_evolution': [],
            'probability_evolution': [],
            'entanglement_evolution': []
        }
        
        # Generate circuit diagram
        if gates:
            circuit_diagram = self.circuit_diagram.generate_diagram(gates, states[0].num_qubits)
            visualization_data['circuit_diagram'] = circuit_diagram
            self.circuit_diagrams[algorithm_name] = circuit_diagram
        
        # Visualize state evolution
        for i, state in enumerate(states):
            state_visualization = self._visualize_single_state(state, f"Step {i+1}")
            visualization_data['state_evolution'].append(state_visualization)
        
        # Generate probability evolution
        probability_evolution = self._generate_probability_evolution(states)
        visualization_data['probability_evolution'] = probability_evolution
        
        # Store visualization data
        self.visualization_history.append(visualization_data)
        
        return visualization_data
    
    def _visualize_single_state(self, state: QuantumState, title: str) -> Dict[str, Any]:
        """Visualize a single quantum state."""
        probabilities = state.get_probabilities()
        
        # Generate probability heatmap
        heatmap = self.probability_heatmap.generate_heatmap(probabilities, state.num_qubits)
        
        # Generate state plot
        state_plot = self.state_plotter.plot_probability_distribution(probabilities, state.num_qubits)
        
        visualization = {
            'title': title,
            'state': str(state),
            'probabilities': probabilities.tolist(),
            'heatmap': heatmap,
            'state_plot': state_plot,
            'num_qubits': state.num_qubits
        }
        
        return visualization
    
    def _generate_probability_evolution(self, states: List[QuantumState]) -> List[Dict[str, Any]]:
        """Generate probability evolution visualization."""
        evolution = []
        
        for i, state in enumerate(states):
            probabilities = state.get_probabilities()
            
            # Create evolution entry
            evolution_entry = {
                'step': i,
                'probabilities': probabilities.tolist(),
                'max_probability': float(np.max(probabilities)),
                'min_probability': float(np.min(probabilities)),
                'entropy': self._calculate_probability_entropy(probabilities),
                'concentration': self._calculate_probability_concentration(probabilities)
            }
            
            evolution.append(evolution_entry)
        
        return evolution
    
    def _calculate_probability_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of probability distribution."""
        # Remove zero probabilities to avoid log(0)
        non_zero_probs = probabilities[probabilities > 0]
        if len(non_zero_probs) == 0:
            return 0.0
        
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        return float(entropy)
    
    def _calculate_probability_concentration(self, probabilities: np.ndarray) -> float:
        """Calculate concentration of probability distribution."""
        # Measure how concentrated the probability is
        max_prob = np.max(probabilities)
        concentration = max_prob / np.sum(probabilities) if np.sum(probabilities) > 0 else 0.0
        return float(concentration)
    
    def generate_comprehensive_visualization(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive visualization for entire exploration."""
        print("\n🎨 GENERATING COMPREHENSIVE VISUALIZATION")
        print("=" * 60)
        
        visualization_summary = {
            'exploration_visualization': {},
            'algorithm_visualizations': {},
            'entanglement_visualizations': {},
            'comparative_analysis': {}
        }
        
        # Visualize each algorithm
        for algo_name, algo_data in exploration_data.get('algorithms', {}).items():
            print(f"   Visualizing {algo_name}...")
            
            # Generate algorithm-specific visualization
            algo_viz = self._visualize_algorithm_data(algo_name, algo_data)
            visualization_summary['algorithm_visualizations'][algo_name] = algo_viz
        
        # Generate entanglement visualization
        entanglement_viz = self._visualize_entanglement_evolution(exploration_data)
        visualization_summary['entanglement_visualizations'] = entanglement_viz
        
        # Generate comparative analysis
        comparative_viz = self._generate_comparative_analysis(exploration_data)
        visualization_summary['comparative_analysis'] = comparative_viz
        
        print("✅ Comprehensive visualization generated")
        
        return visualization_summary
    
    def _visualize_algorithm_data(self, algo_name: str, algo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize data for a specific algorithm."""
        visualization = {
            'algorithm': algo_name,
            'visualizations': {}
        }
        
        # Visualize final state if available
        if 'final_state' in algo_data:
            # This would need the actual QuantumState object, simplified for now
            visualization['final_state'] = algo_data.get('final_state', 'N/A')
        
        # Visualize probabilities if available
        if 'probabilities' in algo_data:
            probabilities = np.array(algo_data['probabilities'])
            heatmap = self.probability_heatmap.generate_heatmap(probabilities, self._infer_num_qubits(probabilities))
            visualization['probability_heatmap'] = heatmap
        
        # Visualize entanglement if available
        if 'entanglement_analysis' in algo_data:
            entanglement_viz = self._visualize_entanglement_metrics(algo_data['entanglement_analysis'])
            visualization['entanglement_visualization'] = entanglement_viz
        
        return visualization
    
    def _infer_num_qubits(self, probabilities: np.ndarray) -> int:
        """Infer number of qubits from probability array length."""
        return int(np.log2(len(probabilities)))
    
    def _visualize_entanglement_metrics(self, entanglement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize entanglement metrics."""
        visualization = {
            'entanglement_entropy': entanglement_data.get('entanglement_entropy', 0.0),
            'concurrence': entanglement_data.get('concurrence', 0.0),
            'negativity': entanglement_data.get('negativity', 0.0),
            'is_entangled': entanglement_data.get('is_entangled', False),
            'bell_state_type': entanglement_data.get('bell_state_type'),
            'entanglement_rank': entanglement_data.get('entanglement_rank', 0)
        }
        
        # Generate visual representation
        if visualization['is_entangled']:
            if visualization['bell_state_type']:
                visualization['visual_summary'] = f"Bell State: {visualization['bell_state_type']}"
            else:
                visualization['visual_summary'] = "Entangled State"
        else:
            visualization['visual_summary'] = "Separable State"
        
        return visualization
    
    def _visualize_entanglement_evolution(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize entanglement evolution across algorithms."""
        entanglement_evolution = {
            'algorithms': {},
            'overall_trends': {},
            'entanglement_classification': {}
        }
        
        # Analyze entanglement for each algorithm
        for algo_name, algo_data in exploration_data.get('algorithms', {}).items():
            if 'entanglement_analysis' in algo_data:
                entanglement_info = algo_data['entanglement_analysis']
                
                entanglement_evolution['algorithms'][algo_name] = {
                    'entanglement_entropy': entanglement_info.get('entanglement_entropy', 0.0),
                    'is_entangled': entanglement_info.get('is_entangled', False),
                    'bell_state_type': entanglement_info.get('bell_state_type'),
                    'classification': self._classify_entanglement_level(entanglement_info)
                }
        
        # Calculate overall trends
        entropies = [data.get('entanglement_entropy', 0.0) 
                    for data in entanglement_evolution['algorithms'].values()]
        
        if entropies:
            entanglement_evolution['overall_trends'] = {
                'average_entropy': float(np.mean(entropies)),
                'max_entropy': float(np.max(entropies)),
                'min_entropy': float(np.min(entropies)),
                'entropy_variance': float(np.var(entropies))
            }
        
        return entanglement_evolution
    
    def _classify_entanglement_level(self, entanglement_data: Dict[str, Any]) -> str:
        """Classify entanglement level."""
        entropy = entanglement_data.get('entanglement_entropy', 0.0)
        
        if entropy > 0.8:
            return "High Entanglement"
        elif entropy > 0.4:
            return "Medium Entanglement"
        elif entropy > 0.1:
            return "Low Entanglement"
        else:
            return "No Entanglement"
    
    def _generate_comparative_analysis(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis visualization."""
        comparative = {
            'algorithm_comparison': {},
            'entanglement_comparison': {},
            'performance_comparison': {}
        }
        
        # Compare algorithms
        algorithms = exploration_data.get('algorithms', {})
        
        for algo_name, algo_data in algorithms.items():
            comparative['algorithm_comparison'][algo_name] = {
                'execution_time': algo_data.get('execution_time', 0.0),
                'entanglement_entropy': algo_data.get('entanglement_analysis', {}).get('entanglement_entropy', 0.0),
                'success_metrics': self._extract_success_metrics(algo_data)
            }
        
        return comparative
    
    def _extract_success_metrics(self, algo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract success metrics from algorithm data."""
        metrics = {}
        
        # Extract success probability if available
        if 'success_probability' in algo_data:
            metrics['success_probability'] = algo_data['success_probability']
        
        # Extract fidelity if available
        if 'fidelity' in algo_data:
            metrics['fidelity'] = algo_data['fidelity']
        
        # Extract measurement results if available
        if 'measurement_results' in algo_data:
            metrics['measurement_results'] = algo_data['measurement_results']
        
        return metrics
    
    def generate_ascii_summary(self, exploration_data: Dict[str, Any]) -> str:
        """Generate ASCII summary of the entire exploration."""
        summary_lines = []
        
        summary_lines.append("🔬 QUANTUM EXPLORATION SUMMARY")
        summary_lines.append("=" * 50)
        
        # Add exploration metadata
        metadata = exploration_data.get('exploration_metadata', {})
        summary_lines.append(f"Qubits: {metadata.get('num_qubits', 'N/A')}")
        summary_lines.append(f"GPU: {'Enabled' if metadata.get('use_gpu', False) else 'Disabled'}")
        summary_lines.append(f"Sparse: {'Enabled' if metadata.get('use_sparse', False) else 'Disabled'}")
        summary_lines.append("")
        
        # Add algorithm results
        algorithms = exploration_data.get('algorithm_results', {})
        summary_lines.append("ALGORITHM RESULTS:")
        summary_lines.append("-" * 30)
        
        for algo_name, algo_data in algorithms.items():
            summary_lines.append(f"  {algo_name}:")
            if 'final_state' in algo_data:
                summary_lines.append(f"    State: {algo_data['final_state']}")
            if 'entanglement_analysis' in algo_data:
                ent = algo_data['entanglement_analysis']
                summary_lines.append(f"    Entangled: {ent.get('is_entangled', False)}")
                summary_lines.append(f"    Entropy: {ent.get('entanglement_entropy', 0.0):.4f}")
            summary_lines.append("")
        
        # Add summary metrics
        metrics = exploration_data.get('summary_metrics', {})
        if metrics:
            summary_lines.append("SUMMARY METRICS:")
            summary_lines.append("-" * 30)
            summary_lines.append(f"  Successful algorithms: {metrics.get('successful_algorithms', 0)}")
            summary_lines.append(f"  Average execution time: {metrics.get('average_execution_time', 0.0):.4f}s")
            summary_lines.append(f"  Average entanglement entropy: {metrics.get('average_entanglement_entropy', 0.0):.4f}")
            summary_lines.append("")
        
        # Add recommendations
        recommendations = exploration_data.get('recommendations', [])
        if recommendations:
            summary_lines.append("RECOMMENDATIONS:")
            summary_lines.append("-" * 30)
            for rec in recommendations:
                summary_lines.append(f"  • {rec}")
        
        return "\n".join(summary_lines)
