"""
Quantum Research Engine for Coratrix 4.0

This module provides autonomous quantum algorithm generation, evaluation,
and refinement capabilities. The Quantum Research Engine can invent entirely
new quantum algorithms, hybrid methods, and optimization paradigms.

Key Components:
- QuantumAlgorithmGenerator: Generates novel quantum algorithms
- AutonomousExperimenter: Tests algorithms across all backends
- SelfEvolvingOptimizer: Continuously improves algorithms
- QuantumStrategyAdvisor: Provides strategic recommendations
- KnowledgeExpander: Documents and maintains discoveries
- ContinuousEvolver: Enables endless iteration and improvement
"""

from .quantum_algorithm_generator import QuantumAlgorithmGenerator
from .autonomous_experimenter import AutonomousExperimenter
from .self_evolving_optimizer import SelfEvolvingOptimizer
from .quantum_strategy_advisor import QuantumStrategyAdvisor
from .knowledge_expander import KnowledgeExpander
from .continuous_evolver import ContinuousEvolver
from .quantum_research_engine import QuantumResearchEngine

__all__ = [
    'QuantumAlgorithmGenerator',
    'AutonomousExperimenter', 
    'SelfEvolvingOptimizer',
    'QuantumStrategyAdvisor',
    'KnowledgeExpander',
    'ContinuousEvolver',
    'QuantumResearchEngine'
]
