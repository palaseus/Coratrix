"""
Advanced entanglement tracking and analysis system.

This module provides comprehensive entanglement tracking
throughout quantum algorithm execution.
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from core.entanglement_analysis import EntanglementAnalyzer
from core.qubit import QuantumState
from core.scalable_quantum_state import ScalableQuantumState


class EntanglementTracker:
    """
    Advanced entanglement tracking system.
    
    Tracks entanglement evolution through quantum algorithms
    with detailed metrics and analysis.
    """
    
    def __init__(self):
        """Initialize the entanglement tracker."""
        self.analyzer = EntanglementAnalyzer()
        self.tracking_data = []
        self.entanglement_evolution = []
        
    def track_entanglement_evolution(self, states: List[QuantumState], 
                                   algorithm_name: str) -> Dict[str, Any]:
        """
        Track entanglement evolution through a sequence of states.
        
        Args:
            states: List of quantum states to analyze
            algorithm_name: Name of the algorithm being executed
        
        Returns:
            Comprehensive entanglement evolution data
        """
        evolution_data = {
            'algorithm': algorithm_name,
            'num_states': len(states),
            'entanglement_metrics': [],
            'entanglement_transitions': [],
            'max_entanglement': 0.0,
            'min_entanglement': 1.0,
            'entanglement_variance': 0.0
        }
        
        previous_entanglement = None
        
        for i, state in enumerate(states):
            # Analyze current state
            analysis = self.analyzer.analyze_entanglement(state)
            
            # Calculate additional metrics
            metrics = self._calculate_advanced_metrics(state, analysis)
            
            # Store metrics
            evolution_data['entanglement_metrics'].append({
                'step': i,
                'entanglement_entropy': analysis['entanglement_entropy'],
                'concurrence': analysis.get('concurrence', 0.0),
                'negativity': analysis.get('negativity', 0.0),
                'is_entangled': analysis['is_entangled'],
                'is_bell_state': analysis.get('is_bell_state', False),
                'bell_state_type': analysis.get('bell_state_type'),
                'entanglement_rank': analysis.get('entanglement_rank', 0),
                'advanced_metrics': metrics
            })
            
            # Track transitions
            if previous_entanglement is not None:
                transition = self._analyze_entanglement_transition(
                    previous_entanglement, analysis
                )
                evolution_data['entanglement_transitions'].append(transition)
            
            previous_entanglement = analysis
            
            # Update min/max
            entropy = analysis['entanglement_entropy']
            evolution_data['max_entanglement'] = max(evolution_data['max_entanglement'], entropy)
            evolution_data['min_entanglement'] = min(evolution_data['min_entanglement'], entropy)
        
        # Calculate variance
        entropies = [m['entanglement_entropy'] for m in evolution_data['entanglement_metrics']]
        evolution_data['entanglement_variance'] = float(np.var(entropies))
        
        # Store in tracking data
        self.tracking_data.append(evolution_data)
        self.entanglement_evolution.append(evolution_data)
        
        return evolution_data
    
    def _calculate_advanced_metrics(self, state: QuantumState, 
                                  analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate advanced entanglement metrics."""
        metrics = {}
        
        # Get state vector
        if hasattr(state, 'to_dense'):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        # Calculate Schmidt rank (simplified)
        metrics['schmidt_rank'] = self._calculate_schmidt_rank(state_vector, state.num_qubits)
        
        # Calculate entanglement of formation (simplified)
        metrics['entanglement_of_formation'] = self._calculate_entanglement_of_formation(
            analysis.get('concurrence', 0.0)
        )
        
        # Calculate relative entropy of entanglement (simplified)
        metrics['relative_entropy'] = self._calculate_relative_entropy(state_vector)
        
        # Calculate entanglement robustness
        metrics['entanglement_robustness'] = self._calculate_entanglement_robustness(
            state_vector, state.num_qubits
        )
        
        return metrics
    
    def _calculate_schmidt_rank(self, state_vector: np.ndarray, num_qubits: int) -> int:
        """Calculate Schmidt rank of the quantum state."""
        if num_qubits < 2:
            return 1
        
        # Simplified Schmidt rank calculation
        # For demonstration, we'll use a heuristic based on non-zero amplitudes
        non_zero_amplitudes = np.sum(np.abs(state_vector) > 1e-10)
        return min(non_zero_amplitudes, 2**(num_qubits-1))
    
    def _calculate_entanglement_of_formation(self, concurrence: float) -> float:
        """Calculate entanglement of formation from concurrence."""
        if concurrence <= 0:
            return 0.0
        
        # Entanglement of formation formula
        c = concurrence
        if c >= 1.0:
            return 1.0
        
        # Simplified calculation
        return -c * math.log2(c) - (1 - c) * math.log2(1 - c)
    
    def _calculate_relative_entropy(self, state_vector: np.ndarray) -> float:
        """Calculate relative entropy of entanglement."""
        # Simplified relative entropy calculation
        probabilities = np.abs(state_vector)**2
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        if len(probabilities) == 0:
            return 0.0
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _calculate_entanglement_robustness(self, state_vector: np.ndarray, 
                                         num_qubits: int) -> float:
        """Calculate entanglement robustness."""
        # Simplified robustness calculation
        # Based on the state's resistance to decoherence
        
        # Calculate purity
        purity = np.sum(np.abs(state_vector)**4)
        
        # Calculate robustness as function of purity and qubit number
        robustness = purity * (1.0 / num_qubits)
        return float(robustness)
    
    def _analyze_entanglement_transition(self, previous: Dict[str, Any], 
                                      current: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transition between entanglement states."""
        transition = {
            'entropy_change': current['entanglement_entropy'] - previous['entanglement_entropy'],
            'concurrence_change': current.get('concurrence', 0) - previous.get('concurrence', 0),
            'negativity_change': current.get('negativity', 0) - previous.get('negativity', 0),
            'entanglement_gained': current['is_entangled'] and not previous['is_entangled'],
            'entanglement_lost': not current['is_entangled'] and previous['is_entangled'],
            'bell_state_transition': self._analyze_bell_state_transition(previous, current)
        }
        
        return transition
    
    def _analyze_bell_state_transition(self, previous: Dict[str, Any], 
                                     current: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Bell state transitions."""
        prev_bell = previous.get('bell_state_type')
        curr_bell = current.get('bell_state_type')
        
        return {
            'previous_bell_state': prev_bell,
            'current_bell_state': curr_bell,
            'bell_state_changed': prev_bell != curr_bell,
            'became_bell_state': prev_bell is None and curr_bell is not None,
            'lost_bell_state': prev_bell is not None and curr_bell is None
        }
    
    def generate_entanglement_report(self) -> Dict[str, Any]:
        """Generate comprehensive entanglement report."""
        if not self.tracking_data:
            return {'error': 'No entanglement data available'}
        
        # Calculate overall statistics
        all_entropies = []
        all_concurrences = []
        all_negativities = []
        
        for evolution in self.tracking_data:
            for metrics in evolution['entanglement_metrics']:
                all_entropies.append(metrics['entanglement_entropy'])
                all_concurrences.append(metrics['concurrence'])
                all_negativities.append(metrics['negativity'])
        
        # Calculate statistics
        report = {
            'entanglement_statistics': {
                'total_measurements': len(all_entropies),
                'average_entropy': float(np.mean(all_entropies)) if all_entropies else 0.0,
                'max_entropy': float(np.max(all_entropies)) if all_entropies else 0.0,
                'min_entropy': float(np.min(all_entropies)) if all_entropies else 0.0,
                'entropy_std': float(np.std(all_entropies)) if all_entropies else 0.0,
                'average_concurrence': float(np.mean(all_concurrences)) if all_concurrences else 0.0,
                'max_concurrence': float(np.max(all_concurrences)) if all_concurrences else 0.0,
                'average_negativity': float(np.mean(all_negativities)) if all_negativities else 0.0,
                'max_negativity': float(np.max(all_negativities)) if all_negativities else 0.0
            },
            'algorithm_analysis': self.tracking_data,
            'entanglement_evolution': self.entanglement_evolution,
            'recommendations': self._generate_entanglement_recommendations()
        }
        
        return report
    
    def _generate_entanglement_recommendations(self) -> List[str]:
        """Generate recommendations based on entanglement analysis."""
        recommendations = []
        
        if not self.tracking_data:
            return ["No entanglement data available for recommendations"]
        
        # Analyze entanglement patterns
        max_entropy = 0.0
        entangled_count = 0
        bell_state_count = 0
        
        for evolution in self.tracking_data:
            for metrics in evolution['entanglement_metrics']:
                max_entropy = max(max_entropy, metrics['entanglement_entropy'])
                if metrics['is_entangled']:
                    entangled_count += 1
                if metrics['is_bell_state']:
                    bell_state_count += 1
        
        # Generate recommendations based on patterns
        if max_entropy > 0.8:
            recommendations.append("High entanglement detected - consider entanglement-based quantum algorithms")
        
        if bell_state_count > 0:
            recommendations.append("Bell states detected - suitable for quantum communication protocols")
        
        if entangled_count > len(self.tracking_data) * 0.5:
            recommendations.append("Frequent entanglement - consider entanglement-based quantum computing")
        else:
            recommendations.append("Limited entanglement - consider separable state algorithms")
        
        if not recommendations:
            recommendations.append("Entanglement patterns are within normal ranges")
        
        return recommendations
