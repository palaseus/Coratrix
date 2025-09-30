"""
Backend Management

This module provides backend management capabilities.
"""

from .backend_interface import (
    BackendInterface, BackendManager, BackendConfiguration, BackendType,
    BackendStatus, BackendCapabilities, BackendResult
)
from .simulator_backend import SimulatorBackend

__all__ = [
    'BackendInterface',
    'BackendManager', 
    'BackendConfiguration',
    'BackendType',
    'BackendStatus',
    'BackendCapabilities',
    'BackendResult',
    'SimulatorBackend'
]
