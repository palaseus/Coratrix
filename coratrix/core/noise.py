"""
Noise Models

This module provides noise models and error channels.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class NoiseChannel(ABC):
    """Base class for noise channels."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self, state: 'ScalableQuantumState') -> 'ScalableQuantumState':
        """Apply the noise channel to a state."""
        pass


class NoiseModel:
    """Noise model for quantum circuits."""
    
    def __init__(self):
        self.channels: List[NoiseChannel] = []
    
    def add_channel(self, channel: NoiseChannel):
        """Add a noise channel to the model."""
        self.channels.append(channel)
    
    def apply(self, state: 'ScalableQuantumState') -> 'ScalableQuantumState':
        """Apply all noise channels to a state."""
        for channel in self.channels:
            state = channel.apply(state)
        return state
