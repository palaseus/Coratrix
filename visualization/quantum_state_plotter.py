"""
Quantum state plotter for visualization.

This module provides plotting capabilities for quantum states
and probability distributions.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class QuantumStatePlotter:
    """
    Quantum state plotter for visualization.
    
    Provides plotting capabilities for quantum states,
    probability distributions, and entanglement analysis.
    """
    
    def __init__(self):
        """Initialize the quantum state plotter."""
        pass
    
    def plot_probability_distribution(self, probabilities: np.ndarray, num_qubits: int) -> str:
        """
        Plot probability distribution as ASCII art.
        
        Args:
            probabilities: Array of probabilities
            num_qubits: Number of qubits
        
        Returns:
            ASCII plot representation
        """
        lines = []
        lines.append("Probability Distribution Plot")
        lines.append("=" * 30)
        lines.append("")
        
        # Find maximum probability for scaling
        max_prob = np.max(probabilities)
        
        # Generate plot
        for i, prob in enumerate(probabilities):
            binary = format(i, f'0{num_qubits}b')
            height = int(prob / max_prob * 20) if max_prob > 0 else 0
            bar = '█' * height
            lines.append(f"|{binary}⟩: {bar} {prob:.4f}")
        
        return "\n".join(lines)
    
    def plot_entanglement_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Plot entanglement analysis results.
        
        Args:
            analysis: Entanglement analysis results
        
        Returns:
            ASCII plot representation
        """
        lines = []
        lines.append("Entanglement Analysis Plot")
        lines.append("=" * 30)
        lines.append("")
        
        # Plot entanglement entropy
        entropy = analysis.get('entanglement_entropy', 0.0)
        entropy_bar = '█' * int(entropy * 20)
        lines.append(f"Entanglement Entropy: {entropy_bar} {entropy:.4f}")
        
        # Plot concurrence
        concurrence = analysis.get('concurrence', 0.0)
        concurrence_bar = '█' * int(concurrence * 20)
        lines.append(f"Concurrence: {concurrence_bar} {concurrence:.4f}")
        
        # Plot negativity
        negativity = analysis.get('negativity', 0.0)
        negativity_bar = '█' * int(negativity * 20)
        lines.append(f"Negativity: {negativity_bar} {negativity:.4f}")
        
        return "\n".join(lines)
