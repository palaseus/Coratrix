"""
Probability heatmap visualization for quantum states.

This module provides probability heatmap visualization
for quantum state probability distributions.
"""

import numpy as np
from typing import List, Dict, Any


class ProbabilityHeatmap:
    """
    Probability heatmap visualizer for quantum states.
    
    Provides visualization of probability distributions
    over computational basis states.
    """
    
    def __init__(self, width: int = 80):
        """
        Initialize the probability heatmap visualizer.
        
        Args:
            width: Maximum width of the heatmap
        """
        self.width = width
    
    def generate_heatmap(self, probabilities: np.ndarray, num_qubits: int) -> str:
        """
        Generate ASCII heatmap of probability distribution.
        
        Args:
            probabilities: Array of probabilities
            num_qubits: Number of qubits
        
        Returns:
            ASCII heatmap representation
        """
        lines = []
        lines.append("Probability Distribution Heatmap")
        lines.append("=" * 40)
        lines.append("")
        
        # Generate heatmap for each state
        for i, prob in enumerate(probabilities):
            binary = format(i, f'0{num_qubits}b')
            bar = self._generate_probability_bar(prob)
            lines.append(f"|{binary}⟩: {bar} {prob:.4f}")
        
        return "\n".join(lines)
    
    def _generate_probability_bar(self, probability: float) -> str:
        """Generate ASCII bar for probability value."""
        # Scale probability to bar length
        bar_length = int(probability * self.width)
        
        # Choose character based on probability
        if probability > 0.8:
            char = '█'
        elif probability > 0.6:
            char = '▓'
        elif probability > 0.4:
            char = '▒'
        elif probability > 0.2:
            char = '░'
        else:
            char = '·'
        
        return char * bar_length
