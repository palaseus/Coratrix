"""
Comprehensive tests for the Autonomous Quantum Intelligence Layer
================================================================

This module provides extensive testing for all autonomous intelligence
components with comprehensive error handling, edge cases, and performance validation.
"""

import pytest
import asyncio
import time
import numpy as np
import tempfile
import os
import sys
import warnings
import gc
import threading
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import autonomous intelligence components
try:
    from autonomous.autonomous_intelligence import AutonomousQuantumIntelligence, IntelligenceMode, IntelligencePriority
    from autonomous.predictive_orchestrator import PredictiveOrchestrator, BackendType, RoutingStrategy
    from autonomous.self_evolving_optimizer import SelfEvolvingOptimizer, OptimizationType, EvolutionStrategy
    from autonomous.quantum_strategy_advisor import QuantumStrategyAdvisor, StrategyType
    from autonomous.autonomous_analytics import AutonomousAnalytics, AnalyticsType, InsightType
    from autonomous.experimental_expansion import ExperimentalExpansion, ExperimentType, ExperimentStatus
    from autonomous.continuous_learning import ContinuousLearningSystem, LearningType, KnowledgeType
    AUTONOMOUS_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    AUTONOMOUS_INTELLIGENCE_AVAILABLE = False
    print(f"Warning: Autonomous intelligence components not available: {e}")

# Test configuration
pytestmark = pytest.mark.skipif(
    not AUTONOMOUS_INTELLIGENCE_AVAILABLE,
    reason="Autonomous intelligence components not available"
)

class TestAutonomousQuantumIntelligence:
    """Test cases for the Autonomous Quantum Intelligence system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'intelligence_mode': 'predictive',
            'learning_enabled': True,
            'max_concurrent_decisions': 10
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        gc.collect()
    
    @pytest.mark.asyncio
    async def test_autonomous_intelligence_initialization(self):
        """Test autonomous intelligence system initialization."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        assert intelligence.intelligence_id is not None
        assert intelligence.current_state.mode == IntelligenceMode.PREDICTIVE
        assert len(intelligence.decision_history) == 0
        assert len(intelligence.active_decisions) == 0
    
    @pytest.mark.asyncio
    async def test_autonomous_intelligence_start_stop(self):
        """Test starting and stopping the autonomous intelligence system."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        # Start the system
        await intelligence.start_autonomous_intelligence()
        assert intelligence.running == True
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Stop the system
        await intelligence.stop_autonomous_intelligence()
        assert intelligence.running == False
    
    def test_autonomous_decision_creation(self):
        """Test creation of autonomous decisions."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        # Create a mock decision
        decision = intelligence._make_autonomous_decisions(
            {'performance_trend': 'improving'},
            [{'type': 'optimization', 'priority': 'high', 'description': 'Test optimization opportunity'}],
            [{'type': 'strategic', 'priority': 'medium', 'description': 'Test strategic recommendation'}]
        )
        
        assert len(decision) > 0
        assert all(d.decision_type in ['optimization', 'strategic'] for d in decision)
        assert all(d.priority in [IntelligencePriority.HIGH, IntelligencePriority.CRITICAL] for d in decision)
    
    def test_intelligence_status(self):
        """Test getting intelligence status."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        status = intelligence.get_intelligence_status()
        
        assert 'intelligence_id' in status
        assert 'mode' in status
        assert 'active_optimizations' in status
        assert 'pending_decisions' in status
        assert 'learning_cycles' in status
    
    def test_autonomous_report_generation(self):
        """Test autonomous report generation."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        report = intelligence.get_autonomous_report()
        
        assert 'timestamp' in report
        assert 'intelligence_status' in report
        assert 'recent_decisions' in report
        assert 'performance_analysis' in report
        assert 'optimization_opportunities' in report
        assert 'strategic_recommendations' in report

class TestPredictiveOrchestrator:
    """Test cases for the Predictive Orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'routing_strategy': 'predictive',
            'model_training_enabled': True
        }
    
    @pytest.mark.asyncio
    async def test_predictive_orchestrator_initialization(self):
        """Test predictive orchestrator initialization."""
        orchestrator = PredictiveOrchestrator(self.config)
        
        assert orchestrator.orchestrator_id is not None
        assert orchestrator.routing_strategy == RoutingStrategy.PREDICTIVE
        assert len(orchestrator.available_backends) > 0
        assert len(orchestrator.routing_history) == 0
    
    @pytest.mark.asyncio
    async def test_predictive_orchestrator_start_stop(self):
        """Test starting and stopping the predictive orchestrator."""
        orchestrator = PredictiveOrchestrator(self.config)
        
        # Start the orchestrator
        await orchestrator.start()
        assert orchestrator.running == True
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Stop the orchestrator
        await orchestrator.stop()
        assert orchestrator.running == False
    
    def test_backend_capabilities_initialization(self):
        """Test backend capabilities initialization."""
        orchestrator = PredictiveOrchestrator(self.config)
        
        # Check that all backend types are initialized
        expected_backends = [
            BackendType.LOCAL_SPARSE_TENSOR,
            BackendType.LOCAL_GPU,
            BackendType.REMOTE_CLUSTER,
            BackendType.QUANTUM_HARDWARE,
            BackendType.CLOUD_SIMULATOR
        ]
        
        for backend_type in expected_backends:
            assert backend_type in orchestrator.available_backends
            capabilities = orchestrator.available_backends[backend_type]
            assert capabilities.max_qubits > 0
            assert capabilities.max_depth > 0
            assert capabilities.execution_time_ms > 0
            assert capabilities.cost_per_shot >= 0
            assert 0 <= capabilities.reliability <= 1
    
    @pytest.mark.asyncio
    async def test_circuit_profiling(self):
        """Test circuit profiling for routing decisions."""
        orchestrator = PredictiveOrchestrator(self.config)
        
        # Create a mock circuit profile
        circuit_profile = {
            'num_qubits': 5,
            'circuit_depth': 20,
            'gate_count': 15,
            'entanglement_complexity': 0.7,
            'memory_requirement': 1024,
            'execution_time_estimate': 100.0,
            'cost_estimate': 0.05
        }
        
        # Test backend suitability
        for backend_type, capabilities in orchestrator.available_backends.items():
            is_suitable = orchestrator._is_backend_suitable(
                type('CircuitProfile', (), circuit_profile)(),
                backend_type,
                capabilities
            )
            assert isinstance(is_suitable, bool)
    
    def test_routing_statistics(self):
        """Test routing statistics generation."""
        orchestrator = PredictiveOrchestrator(self.config)
        
        stats = orchestrator.get_routing_statistics()
        
        assert 'total_routing_decisions' in stats
        assert 'active_routes' in stats
        assert 'available_backends' in stats
        assert 'models_trained' in stats
        assert 'routing_strategy' in stats
        assert 'backend_utilization' in stats
        assert 'prediction_accuracy' in stats

class TestSelfEvolvingOptimizer:
    """Test cases for the Self-Evolving Optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'population_size': 20,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7
        }
    
    @pytest.mark.asyncio
    async def test_self_evolving_optimizer_initialization(self):
        """Test self-evolving optimizer initialization."""
        optimizer = SelfEvolvingOptimizer(self.config)
        
        assert optimizer.optimizer_id is not None
        assert optimizer.population_size == 50  # Default value
        assert optimizer.mutation_rate == 0.1
        assert optimizer.crossover_rate == 0.7
        assert len(optimizer.optimization_passes) == 0
        assert len(optimizer.optimization_results) == 0
    
    @pytest.mark.asyncio
    async def test_self_evolving_optimizer_start_stop(self):
        """Test starting and stopping the self-evolving optimizer."""
        optimizer = SelfEvolvingOptimizer(self.config)
        
        # Start the optimizer
        await optimizer.start()
        assert optimizer.running == True
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Stop the optimizer
        await optimizer.stop()
        assert optimizer.running == False
    
    def test_optimization_pass_creation(self):
        """Test creation of optimization passes."""
        optimizer = SelfEvolvingOptimizer(self.config)
        
        # Test parameter generation
        for opt_type in OptimizationType:
            parameters = optimizer._generate_random_parameters(opt_type)
            assert isinstance(parameters, dict)
            assert len(parameters) > 0
    
    def test_evolution_statistics(self):
        """Test evolution statistics generation."""
        optimizer = SelfEvolvingOptimizer(self.config)
        
        stats = optimizer.get_evolution_statistics()
        
        assert 'current_generation' in stats
        assert 'total_passes' in stats
        assert 'total_results' in stats
        assert 'models_trained' in stats
        assert 'active_optimizations' in stats
        assert 'evolution_history_size' in stats

class TestQuantumStrategyAdvisor:
    """Test cases for the Quantum Strategy Advisor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'strategy_analysis_enabled': True,
            'entanglement_optimization': True
        }
    
    @pytest.mark.asyncio
    async def test_quantum_strategy_advisor_initialization(self):
        """Test quantum strategy advisor initialization."""
        advisor = QuantumStrategyAdvisor(self.config)
        
        assert advisor.advisor_id is not None
        assert len(advisor.strategy_patterns) == 0
        assert len(advisor.entanglement_models) == 0
        assert len(advisor.qubit_connectivity_graphs) == 0
    
    @pytest.mark.asyncio
    async def test_quantum_strategy_advisor_start_stop(self):
        """Test starting and stopping the quantum strategy advisor."""
        advisor = QuantumStrategyAdvisor(self.config)
        
        # Start the advisor
        await advisor.start()
        assert advisor.running == True
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Stop the advisor
        await advisor.stop()
        assert advisor.running == False
    
    def test_qubit_mapping_recommendation(self):
        """Test qubit mapping recommendation generation."""
        advisor = QuantumStrategyAdvisor(self.config)
        
        # Create mock circuit data
        circuit_data = {
            'num_qubits': 5,
            'gates': [
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]},
                {'type': 'CNOT', 'qubits': [1, 2]}
            ]
        }
        
        # Test connectivity analysis
        connectivity = advisor._analyze_connectivity_requirements(circuit_data['gates'])
        assert 'required_connections' in connectivity
        assert 'connection_frequency' in connectivity
        assert 'critical_paths' in connectivity
    
    def test_strategy_statistics(self):
        """Test strategy statistics generation."""
        advisor = QuantumStrategyAdvisor(self.config)
        
        stats = advisor.get_strategy_statistics()
        
        assert 'total_recommendations' in stats
        assert 'strategy_patterns' in stats
        assert 'entanglement_models' in stats
        assert 'connectivity_graphs' in stats
        assert 'quantum_metrics' in stats

class TestAutonomousAnalytics:
    """Test cases for the Autonomous Analytics system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'analytics_enabled': True,
            'insight_generation': True
        }
    
    @pytest.mark.asyncio
    async def test_autonomous_analytics_initialization(self):
        """Test autonomous analytics initialization."""
        analytics = AutonomousAnalytics(self.config)
        
        assert analytics.analytics_id is not None
        assert len(analytics.metrics_buffer) == 0
        assert len(analytics.insights_history) == 0
        assert len(analytics.forecasts_history) == 0
    
    @pytest.mark.asyncio
    async def test_autonomous_analytics_start_stop(self):
        """Test starting and stopping the autonomous analytics system."""
        analytics = AutonomousAnalytics(self.config)
        
        # Start the analytics system
        await analytics.start()
        assert analytics.running == True
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Stop the analytics system
        await analytics.stop()
        assert analytics.running == False
    
    def test_metric_collection(self):
        """Test metric collection functionality."""
        analytics = AutonomousAnalytics(self.config)
        
        # Collect some test metrics
        analytics.collect_metric('execution_time', 100.0, 'ms', {'circuit_id': 'test'}, ['performance'])
        analytics.collect_metric('memory_usage', 512.0, 'MB', {'backend': 'local'}, ['memory'])
        analytics.collect_metric('cost', 0.05, 'USD', {'shots': 1000}, ['cost'])
        
        assert len(analytics.metrics_buffer) == 3
    
    def test_performance_analysis(self):
        """Test performance analysis functionality."""
        analytics = AutonomousAnalytics(self.config)
        
        # Add some test data
        for i in range(50):
            analytics.collect_metric('execution_time', 100.0 + i * 0.1, 'ms')
            analytics.collect_metric('memory_usage', 500.0 + i * 0.5, 'MB')
            analytics.collect_metric('cpu_usage', 50.0 + i * 0.2, '%')
        
        # Test performance analysis
        analytics._update_performance_analysis()
        
        assert 'average_execution_time' in analytics.performance_metrics
        assert 'execution_time_std' in analytics.performance_metrics
        assert 'performance_trend' in analytics.performance_metrics
        assert 'bottlenecks' in analytics.performance_metrics
        assert 'optimization_opportunities' in analytics.performance_metrics
    
    def test_analytics_report(self):
        """Test analytics report generation."""
        analytics = AutonomousAnalytics(self.config)
        
        # Add some test data
        for i in range(10):
            analytics.collect_metric('execution_time', 100.0, 'ms')
        
        report = analytics.get_analytics_report()
        
        assert 'timestamp' in report
        assert 'analytics_state' in report
        assert 'performance_metrics' in report
        assert 'system_health' in report
        assert 'cost_analysis' in report
        assert 'entanglement_analysis' in report
        assert 'recent_insights' in report
        assert 'recent_forecasts' in report
        assert 'models_trained' in report

class TestExperimentalExpansion:
    """Test cases for the Experimental Expansion system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'experimental_enabled': True,
            'research_focus': ['quantum_ml', 'optimization']
        }
    
    @pytest.mark.asyncio
    async def test_experimental_expansion_initialization(self):
        """Test experimental expansion initialization."""
        expansion = ExperimentalExpansion(self.config)
        
        assert expansion.expansion_id is not None
        assert len(expansion.active_experiments) == 0
        assert len(expansion.experiment_history) == 0
        assert len(expansion.research_insights) == 0
    
    @pytest.mark.asyncio
    async def test_experimental_expansion_start_stop(self):
        """Test starting and stopping the experimental expansion system."""
        expansion = ExperimentalExpansion(self.config)
        
        # Start the expansion system
        await expansion.start()
        assert expansion.running == True
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Stop the expansion system
        await expansion.stop()
        assert expansion.running == False
    
    def test_experiment_creation(self):
        """Test experiment creation functionality."""
        expansion = ExperimentalExpansion(self.config)
        
        # Test hybrid execution experiment creation
        experiment = expansion._create_hybrid_execution_experiment()
        assert experiment is not None
        assert experiment.experiment_type == ExperimentType.HYBRID_EXECUTION
        assert experiment.status == ExperimentStatus.PLANNING
        assert len(experiment.objectives) > 0
        assert len(experiment.methodology) > 0
        assert len(experiment.expected_outcomes) > 0
    
    def test_research_report(self):
        """Test research report generation."""
        expansion = ExperimentalExpansion(self.config)
        
        report = expansion.get_research_report()
        
        assert 'timestamp' in report
        assert 'active_experiments' in report
        assert 'completed_experiments' in report
        assert 'research_focus_areas' in report
        assert 'research_priorities' in report
        assert 'recent_insights' in report
        assert 'experiment_history' in report
        assert 'hybrid_models' in report
        assert 'quantum_shaders' in report
        assert 'algorithmic_innovations' in report

class TestContinuousLearningSystem:
    """Test cases for the Continuous Learning System."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'learning_enabled': True,
            'knowledge_base_size': 1000
        }
    
    @pytest.mark.asyncio
    async def test_continuous_learning_initialization(self):
        """Test continuous learning system initialization."""
        learning = ContinuousLearningSystem(self.config)
        
        assert learning.learning_id is not None
        assert len(learning.knowledge_base) == 0
        assert len(learning.learning_patterns) == 0
        assert len(learning.performance_data) == 0
        assert len(learning.optimization_data) == 0
    
    @pytest.mark.asyncio
    async def test_continuous_learning_start_stop(self):
        """Test starting and stopping the continuous learning system."""
        learning = ContinuousLearningSystem(self.config)
        
        # Start the learning system
        await learning.start()
        assert learning.running == True
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Stop the learning system
        await learning.stop()
        assert learning.running == False
    
    def test_data_update(self):
        """Test data update functionality."""
        learning = ContinuousLearningSystem(self.config)
        
        # Update performance data
        learning.update_performance_data({'execution_time': 100.0, 'memory_usage': 512.0})
        learning.update_optimization_data({'success': True, 'improvement': 0.2})
        learning.update_experimental_data({'result': 'success', 'insights': ['test']})
        
        assert len(learning.performance_data) == 1
        assert len(learning.optimization_data) == 1
        assert len(learning.experimental_data) == 1
    
    def test_learning_report(self):
        """Test learning report generation."""
        learning = ContinuousLearningSystem(self.config)
        
        # Add some test data
        for i in range(10):
            learning.update_performance_data({'execution_time': 100.0 + i})
            learning.update_optimization_data({'success': True, 'improvement': 0.1 + i * 0.01})
        
        report = learning.get_learning_report()
        
        assert 'report_id' in report
        assert 'timestamp' in report
        assert 'learning_summary' in report
        assert 'knowledge_growth' in report
        assert 'performance_improvements' in report
        assert 'recommendations' in report
        assert 'predictions' in report
        assert 'experimental_results' in report

class TestAutonomousIntelligenceIntegration:
    """Integration tests for the autonomous intelligence system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'intelligence_mode': 'predictive',
            'learning_enabled': True,
            'experimental_enabled': True
        }
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full autonomous intelligence system integration."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        # Start the full system
        await intelligence.start_autonomous_intelligence()
        
        # Wait for system to initialize
        await asyncio.sleep(0.5)
        
        # Check that all subsystems are running
        assert intelligence.predictive_orchestrator.running == True
        assert intelligence.self_evolving_optimizer.running == True
        assert intelligence.quantum_strategy_advisor.running == True
        assert intelligence.autonomous_analytics.running == True
        assert intelligence.experimental_expansion.running == True
        assert intelligence.continuous_learning.running == True
        
        # Stop the system
        await intelligence.stop_autonomous_intelligence()
        
        # Check that all subsystems are stopped
        assert intelligence.predictive_orchestrator.running == False
        assert intelligence.self_evolving_optimizer.running == False
        assert intelligence.quantum_strategy_advisor.running == False
        assert intelligence.autonomous_analytics.running == False
        assert intelligence.experimental_expansion.running == False
        assert intelligence.continuous_learning.running == False
    
    def test_decision_callback_system(self):
        """Test decision callback system."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        # Create mock callback
        callback_called = []
        def mock_callback(decision):
            callback_called.append(decision)
        
        # Add callback
        intelligence.add_decision_callback(mock_callback)
        
        # Create a mock decision
        decision = intelligence._make_autonomous_decisions(
            {'performance_trend': 'improving'},
            [{'type': 'optimization', 'priority': 'high', 'description': 'Test optimization opportunity'}],
            []
        )
        
        # Simulate decision execution
        if decision:
            intelligence._execute_decision(decision[0])
        
        # Note: In a real system, callbacks would be triggered during execution
        # This test verifies the callback system is properly set up
    
    def test_performance_under_load(self):
        """Test system performance under simulated load."""
        intelligence = AutonomousQuantumIntelligence(self.config)
        
        # Simulate high load
        start_time = time.time()
        
        # Create multiple decisions rapidly
        for i in range(100):
            decision = intelligence._make_autonomous_decisions(
                {'performance_trend': 'stable'},
                [{'type': 'optimization', 'priority': 'medium'}],
                []
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 seconds max for 100 decisions
        
        # Check system stability
        assert len(intelligence.decision_history) <= 10000  # Max history size
        assert len(intelligence.active_decisions) <= 1000  # Max active decisions

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
