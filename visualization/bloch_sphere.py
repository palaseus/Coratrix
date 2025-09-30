"""
Bloch sphere visualization for single qubits.

This module provides Bloch sphere visualization capabilities
for single-qubit quantum states.
"""

import numpy as np
from typing import Tuple, Optional


class BlochSphereVisualizer:
    """
    Bloch sphere visualizer for single-qubit quantum states.
    
    Provides visualization of single-qubit states on the Bloch sphere.
    """
    
    def __init__(self):
        """Initialize the Bloch sphere visualizer."""
        pass
    
    def visualize_state(self, state_vector: np.ndarray) -> str:
        """
        Generate ASCII representation of Bloch sphere.
        
        Args:
            state_vector: 2-element complex state vector
        
        Returns:
            ASCII representation of the Bloch sphere
        """
        # Calculate Bloch sphere coordinates
        alpha, beta = state_vector[0], state_vector[1]
        
        # Convert to spherical coordinates
        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta) - np.angle(alpha)
        
        # Generate ASCII Bloch sphere
        return self._generate_ascii_sphere(theta, phi)
    
    def _generate_ascii_sphere(self, theta: float, phi: float) -> str:
        """Generate ASCII representation of Bloch sphere."""
        # Simple ASCII representation
        lines = []
        lines.append("    Bloch Sphere")
        lines.append("    ============")
        lines.append("")
        lines.append("    θ (polar): {:.3f} rad".format(theta))
        lines.append("    φ (azimuthal): {:.3f} rad".format(phi))
        lines.append("")
        lines.append("    State on Bloch sphere:")
        lines.append("    (x, y, z) = ({:.3f}, {:.3f}, {:.3f})".format(
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ))
        
        return "\n".join(lines)
