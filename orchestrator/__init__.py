"""
Coratrix 4.0 Quantum OS - Dynamic Backend Orchestrator
======================================================

The Dynamic Backend Orchestrator is the brain of Coratrix 4.0's Quantum OS layer.
It intelligently routes quantum circuits to optimal execution backends based on
real-time performance metrics, cost analysis, and resource availability.

This orchestrator transforms Coratrix from a high-performance engine into a
self-optimizing, distributed Quantum OS that feels alive.

Key Features:
- Latency and cost-aware routing
- Hot-swap mid-circuit execution
- Real-time backend performance monitoring
- Adaptive resource allocation
- Multi-backend orchestration
"""

from .backend_orchestrator import DynamicBackendOrchestrator, OrchestrationConfig
from .backend_router import BackendRouter, RoutingStrategy
from .performance_monitor import PerformanceMonitor, TelemetryCollector
from .hot_swap_executor import HotSwapExecutor, CircuitPartitioner
from .cost_analyzer import CostAnalyzer

__all__ = [
    'DynamicBackendOrchestrator',
    'OrchestrationConfig', 
    'BackendRouter',
    'RoutingStrategy',
    'PerformanceMonitor',
    'TelemetryCollector',
    'HotSwapExecutor',
    'CircuitPartitioner',
    'CostAnalyzer',
]
