"""
Tests for Quantum Research Engine

This module provides comprehensive tests for the Quantum Research Engine,
including all components: algorithm generator, experimenter, optimizer,
strategy advisor, knowledge expander, and continuous evolver.

Author: Quantum Research Engine - Coratrix 4.0
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# Import the quantum research engine components
from quantum_research.quantum_research_engine import QuantumResearchEngine, ResearchConfig, ResearchMode
from quantum_research.quantum_algorithm_generator import QuantumAlgorithmGenerator, AlgorithmType, InnovationLevel
from quantum_research.autonomous_experimenter import AutonomousExperimenter, ExperimentType, BackendType
from quantum_research.self_evolving_optimizer import SelfEvolvingOptimizer, OptimizationStrategy
from quantum_research.quantum_strategy_advisor import QuantumStrategyAdvisor, StrategyType, UseCase
from quantum_research.knowledge_expander import KnowledgeExpander, KnowledgeType
from quantum_research.continuous_evolver import ContinuousEvolver, EvolutionPhase, EvolutionStrategy

class TestQuantumResearchEngine:
    """Test cases for the Quantum Research Engine."""
    
    @pytest.fixture
    def research_engine(self):
        """Create a quantum research engine for testing."""
        config = ResearchConfig(
            research_mode=ResearchMode.EXPLORATION,
            enable_algorithm_generation=True,
            enable_autonomous_experimentation=True,
            enable_self_evolving_optimization=True,
            enable_strategy_advice=True,
            enable_knowledge_expansion=True,
            enable_continuous_evolution=True,
            max_concurrent_research=5,
            research_timeout=60.0,
            innovation_threshold=0.8,
            performance_threshold=0.7
        )
        return QuantumResearchEngine(config)
    
    @pytest.mark.asyncio
    async def test_research_engine_initialization(self, research_engine):
        """Test research engine initialization."""
        assert research_engine.engine_id is not None
        assert research_engine.running == False
        assert research_engine.config.research_mode == ResearchMode.EXPLORATION
        assert research_engine.algorithm_generator is not None
        assert research_engine.experimenter is not None
        assert research_engine.optimizer is not None
        assert research_engine.strategy_advisor is not None
        assert research_engine.knowledge_expander is not None
        assert research_engine.continuous_evolver is not None
    
    @pytest.mark.asyncio
    async def test_research_engine_start_stop(self, research_engine):
        """Test research engine start and stop."""
        # Test start
        await research_engine.start()
        assert research_engine.running == True
        
        # Test stop
        await research_engine.stop()
        assert research_engine.running == False
    
    @pytest.mark.asyncio
    async def test_research_engine_statistics(self, research_engine):
        """Test research engine statistics."""
        stats = research_engine.get_research_statistics()
        
        assert 'engine_id' in stats
        assert 'running' in stats
        assert 'research_mode' in stats
        assert 'total_research_results' in stats
        assert 'active_research' in stats
        assert 'queued_research' in stats
        assert 'breakthrough_detections' in stats
        assert 'trend_analysis' in stats
        assert 'research_statistics' in stats
    
    @pytest.mark.asyncio
    async def test_research_engine_breakthrough_detection(self, research_engine):
        """Test breakthrough detection."""
        # Start the engine
        await research_engine.start()
        
        # Wait for breakthrough detection
        await asyncio.sleep(2.0)
        
        # Check breakthrough detections
        breakthroughs = research_engine.get_breakthrough_detections()
        assert isinstance(breakthroughs, list)
        
        # Stop the engine
        await research_engine.stop()
    
    @pytest.mark.asyncio
    async def test_research_engine_trend_analysis(self, research_engine):
        """Test trend analysis."""
        # Start the engine
        await research_engine.start()
        
        # Wait for trend analysis
        await asyncio.sleep(2.0)
        
        # Check trend analysis
        trends = research_engine.get_trend_analysis()
        assert isinstance(trends, dict)
        
        # Stop the engine
        await research_engine.stop()

class TestQuantumAlgorithmGenerator:
    """Test cases for the Quantum Algorithm Generator."""
    
    @pytest.fixture
    def algorithm_generator(self):
        """Create an algorithm generator for testing."""
        return QuantumAlgorithmGenerator()
    
    @pytest.mark.asyncio
    async def test_algorithm_generator_initialization(self, algorithm_generator):
        """Test algorithm generator initialization."""
        assert algorithm_generator.generator_id is not None
        assert algorithm_generator.running == False
        assert algorithm_generator.generated_algorithms == []
        assert algorithm_generator.algorithm_templates is not None
        assert algorithm_generator.entanglement_patterns is not None
        assert algorithm_generator.state_encodings is not None
        assert algorithm_generator.error_mitigation_methods is not None
    
    @pytest.mark.asyncio
    async def test_algorithm_generator_start_stop(self, algorithm_generator):
        """Test algorithm generator start and stop."""
        # Test start
        await algorithm_generator.start()
        assert algorithm_generator.running == True
        
        # Test stop
        await algorithm_generator.stop()
        assert algorithm_generator.running == False
    
    @pytest.mark.asyncio
    async def test_algorithm_generation(self, algorithm_generator):
        """Test algorithm generation."""
        # Generate algorithms
        algorithms = await algorithm_generator.generate_algorithms(
            num_algorithms=3,
            focus_innovation=True
        )
        
        assert len(algorithms) == 3
        assert all(hasattr(alg, 'algorithm_id') for alg in algorithms)
        assert all(hasattr(alg, 'name') for alg in algorithms)
        assert all(hasattr(alg, 'algorithm_type') for alg in algorithms)
        assert all(hasattr(alg, 'innovation_level') for alg in algorithms)
        assert all(hasattr(alg, 'complexity') for alg in algorithms)
        assert all(hasattr(alg, 'description') for alg in algorithms)
        assert all(hasattr(alg, 'quantum_circuit') for alg in algorithms)
    
    @pytest.mark.asyncio
    async def test_algorithm_generator_statistics(self, algorithm_generator):
        """Test algorithm generator statistics."""
        stats = algorithm_generator.get_generation_statistics()
        
        assert 'generator_id' in stats
        assert 'running' in stats
        assert 'total_algorithms' in stats
        assert 'recent_performance' in stats
        assert 'algorithm_types' in stats
        assert 'innovation_levels' in stats
        assert 'complexity_levels' in stats
    
    @pytest.mark.asyncio
    async def test_algorithm_recommendations(self, algorithm_generator):
        """Test algorithm recommendations."""
        # Generate some algorithms first
        await algorithm_generator.generate_algorithms(num_algorithms=5)
        
        # Get recommendations
        recommendations = algorithm_generator.get_algorithm_recommendations({
            'algorithm_type': AlgorithmType.QUANTUM_OPTIMIZATION,
            'innovation_level': InnovationLevel.BREAKTHROUGH,
            'novelty_threshold': 0.7,
            'practical_threshold': 0.6
        })
        
        assert isinstance(recommendations, list)

class TestAutonomousExperimenter:
    """Test cases for the Autonomous Experimenter."""
    
    @pytest.fixture
    def experimenter(self):
        """Create an experimenter for testing."""
        return AutonomousExperimenter()
    
    @pytest.mark.asyncio
    async def test_experimenter_initialization(self, experimenter):
        """Test experimenter initialization."""
        assert experimenter.experimenter_id is not None
        assert experimenter.running == False
        assert experimenter.active_experiments == {}
        assert experimenter.completed_experiments == []
        assert experimenter.failed_experiments == []
        assert experimenter.experiment_queue == deque()
        assert experimenter.backend_capabilities is not None
        assert experimenter.experiment_templates is not None
    
    @pytest.mark.asyncio
    async def test_experimenter_start_stop(self, experimenter):
        """Test experimenter start and stop."""
        # Test start
        await experimenter.start()
        assert experimenter.running == True
        
        # Test stop
        await experimenter.stop()
        assert experimenter.running == False
    
    @pytest.mark.asyncio
    async def test_experiment_execution(self, experimenter):
        """Test experiment execution."""
        # Run an experiment
        experiment_id = await experimenter.run_experiment(
            algorithm_id="test_algorithm",
            experiment_type=ExperimentType.PERFORMANCE_BENCHMARK,
            backend_type=BackendType.LOCAL_SIMULATOR
        )
        
        assert experiment_id is not None
        assert experiment_id.startswith("exp_")
    
    @pytest.mark.asyncio
    async def test_experimenter_statistics(self, experimenter):
        """Test experimenter statistics."""
        stats = experimenter.get_experiment_statistics()
        
        assert 'experimenter_id' in stats
        assert 'running' in stats
        assert 'total_experiments' in stats
        assert 'completed_experiments' in stats
        assert 'failed_experiments' in stats
        assert 'active_experiments' in stats
        assert 'queued_experiments' in stats
        assert 'success_rate' in stats
        assert 'backend_performance' in stats
        assert 'experiment_type_performance' in stats
        assert 'average_confidence' in stats
    
    @pytest.mark.asyncio
    async def test_algorithm_recommendations(self, experimenter):
        """Test algorithm recommendations."""
        recommendations = experimenter.get_algorithm_recommendations("test_algorithm")
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_promising_candidates(self, experimenter):
        """Test promising candidates."""
        candidates = experimenter.get_promising_candidates(min_confidence=0.8)
        assert isinstance(candidates, list)
    
    @pytest.mark.asyncio
    async def test_backend_recommendations(self, experimenter):
        """Test backend recommendations."""
        requirements = {
            'max_qubits': 10,
            'min_fidelity': 0.95,
            'execution_speed': 'fast',
            'noise_model': 'ideal'
        }
        
        recommendations = experimenter.get_backend_recommendations(requirements)
        assert isinstance(recommendations, list)

class TestSelfEvolvingOptimizer:
    """Test cases for the Self-Evolving Optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create an optimizer for testing."""
        return SelfEvolvingOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.optimizer_id is not None
        assert optimizer.running == False
        assert optimizer.active_optimizations == {}
        assert optimizer.completed_optimizations == []
        assert optimizer.failed_optimizations == []
        assert optimizer.optimization_queue == deque()
        assert optimizer.evolution_population == []
        assert optimizer.retired_algorithms == []
    
    @pytest.mark.asyncio
    async def test_optimizer_start_stop(self, optimizer):
        """Test optimizer start and stop."""
        # Test start
        await optimizer.start()
        assert optimizer.running == True
        
        # Test stop
        await optimizer.stop()
        assert optimizer.running == False
    
    @pytest.mark.asyncio
    async def test_algorithm_optimization(self, optimizer):
        """Test algorithm optimization."""
        # Optimize an algorithm
        optimization_id = await optimizer.optimize_algorithm(
            algorithm_id="test_algorithm",
            target_metrics=['execution_time', 'accuracy', 'scalability'],
            target_values={'execution_time': 0.1, 'accuracy': 0.95, 'scalability': 0.9},
            strategy=OptimizationStrategy.GENETIC_ALGORITHM
        )
        
        assert optimization_id is not None
        assert optimization_id.startswith("opt_")
    
    @pytest.mark.asyncio
    async def test_optimizer_statistics(self, optimizer):
        """Test optimizer statistics."""
        stats = optimizer.get_optimization_statistics()
        
        assert 'optimizer_id' in stats
        assert 'running' in stats
        assert 'current_generation' in stats
        assert 'total_optimizations' in stats
        assert 'completed_optimizations' in stats
        assert 'failed_optimizations' in stats
        assert 'active_optimizations' in stats
        assert 'queued_optimizations' in stats
        assert 'success_rate' in stats
        assert 'strategy_performance' in stats
        assert 'evolution_statistics' in stats
        assert 'population_size' in stats
        assert 'retired_algorithms' in stats
        assert 'average_confidence' in stats
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, optimizer):
        """Test optimization recommendations."""
        recommendations = optimizer.get_optimization_recommendations("test_algorithm")
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_evolution_insights(self, optimizer):
        """Test evolution insights."""
        insights = optimizer.get_evolution_insights()
        assert isinstance(insights, dict)
        assert 'current_generation' in insights
        assert 'population_size' in insights
        assert 'best_individuals' in insights
        assert 'innovation_pool' in insights

class TestQuantumStrategyAdvisor:
    """Test cases for the Quantum Strategy Advisor."""
    
    @pytest.fixture
    def strategy_advisor(self):
        """Create a strategy advisor for testing."""
        return QuantumStrategyAdvisor()
    
    @pytest.mark.asyncio
    async def test_strategy_advisor_initialization(self, strategy_advisor):
        """Test strategy advisor initialization."""
        assert strategy_advisor.advisor_id is not None
        assert strategy_advisor.running == False
        assert strategy_advisor.recommendations == []
        assert strategy_advisor.algorithm_analysis == {}
        assert strategy_advisor.use_case_patterns is not None
        assert strategy_advisor.backend_capabilities is not None
        assert strategy_advisor.strategy_templates is not None
    
    @pytest.mark.asyncio
    async def test_strategy_advisor_start_stop(self, strategy_advisor):
        """Test strategy advisor start and stop."""
        # Test start
        await strategy_advisor.start()
        assert strategy_advisor.running == True
        
        # Test stop
        await strategy_advisor.stop()
        assert strategy_advisor.running == False
    
    @pytest.mark.asyncio
    async def test_algorithm_analysis(self, strategy_advisor):
        """Test algorithm analysis."""
        algorithm = {
            'algorithm_id': 'test_algorithm',
            'algorithm_type': 'quantum_optimization',
            'content': 'Novel quantum optimization algorithm',
            'performance_metrics': {
                'execution_time': 0.1,
                'accuracy': 0.95,
                'scalability': 0.9
            }
        }
        
        recommendations = await strategy_advisor.analyze_algorithm(algorithm)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for recommendation in recommendations:
            assert hasattr(recommendation, 'recommendation_id')
            assert hasattr(recommendation, 'algorithm_id')
            assert hasattr(recommendation, 'strategy_type')
            assert hasattr(recommendation, 'use_case')
            assert hasattr(recommendation, 'confidence_level')
            assert hasattr(recommendation, 'confidence_score')
            assert hasattr(recommendation, 'description')
            assert hasattr(recommendation, 'implementation_guidance')
            assert hasattr(recommendation, 'expected_benefits')
            assert hasattr(recommendation, 'potential_risks')
            assert hasattr(recommendation, 'resource_requirements')
            assert hasattr(recommendation, 'performance_predictions')
            assert hasattr(recommendation, 'backend_recommendations')
            assert hasattr(recommendation, 'partitioning_suggestions')
            assert hasattr(recommendation, 'execution_strategies')
            assert hasattr(recommendation, 'optimization_opportunities')
            assert hasattr(recommendation, 'error_mitigation_strategies')
    
    @pytest.mark.asyncio
    async def test_strategy_advisor_statistics(self, strategy_advisor):
        """Test strategy advisor statistics."""
        stats = strategy_advisor.get_advisory_statistics()
        
        assert 'advisor_id' in stats
        assert 'running' in stats
        assert 'total_recommendations' in stats
        assert 'use_case_distribution' in stats
        assert 'confidence_distribution' in stats
        assert 'strategy_distribution' in stats
        assert 'average_confidence' in stats
        assert 'high_confidence_recommendations' in stats
        assert 'algorithm_analyses' in stats
    
    @pytest.mark.asyncio
    async def test_algorithm_recommendations(self, strategy_advisor):
        """Test algorithm recommendations."""
        recommendations = strategy_advisor.get_algorithm_recommendations("test_algorithm")
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_high_confidence_recommendations(self, strategy_advisor):
        """Test high confidence recommendations."""
        recommendations = strategy_advisor.get_high_confidence_recommendations(min_confidence=0.8)
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_use_case_recommendations(self, strategy_advisor):
        """Test use case recommendations."""
        recommendations = strategy_advisor.get_use_case_recommendations(UseCase.OPTIMIZATION)
        assert isinstance(recommendations, list)

class TestKnowledgeExpander:
    """Test cases for the Knowledge Expander."""
    
    @pytest.fixture
    def knowledge_expander(self):
        """Create a knowledge expander for testing."""
        return KnowledgeExpander()
    
    @pytest.mark.asyncio
    async def test_knowledge_expander_initialization(self, knowledge_expander):
        """Test knowledge expander initialization."""
        assert knowledge_expander.expander_id is not None
        assert knowledge_expander.running == False
        assert knowledge_expander.knowledge_base == {}
        assert knowledge_expander.research_directions == []
        assert knowledge_expander.insight_patterns == defaultdict(list)
        assert knowledge_expander.knowledge_graph is not None
        assert knowledge_expander.vectorizer is not None
        assert knowledge_expander.clustering_model is not None
    
    @pytest.mark.asyncio
    async def test_knowledge_expander_start_stop(self, knowledge_expander):
        """Test knowledge expander start and stop."""
        # Test start
        await knowledge_expander.start()
        assert knowledge_expander.running == True
        
        # Test stop
        await knowledge_expander.stop()
        assert knowledge_expander.running == False
    
    @pytest.mark.asyncio
    async def test_discovery_documentation(self, knowledge_expander):
        """Test discovery documentation."""
        discovery = {
            'title': 'Novel Quantum Algorithm',
            'content': 'A breakthrough quantum algorithm for optimization',
            'algorithm_type': 'quantum_optimization',
            'performance_metrics': {
                'execution_time': 0.1,
                'accuracy': 0.95,
                'scalability': 0.9
            }
        }
        
        entry_id = await knowledge_expander.document_discovery(discovery)
        assert entry_id is not None
        assert entry_id.startswith("entry_")
        assert entry_id in knowledge_expander.knowledge_base
    
    @pytest.mark.asyncio
    async def test_knowledge_expander_statistics(self, knowledge_expander):
        """Test knowledge expander statistics."""
        stats = knowledge_expander.get_knowledge_statistics()
        
        assert 'expander_id' in stats
        assert 'running' in stats
        assert 'total_entries' in stats
        assert 'type_distribution' in stats
        assert 'research_directions' in stats
        assert 'insight_patterns' in stats
        assert 'knowledge_graph_nodes' in stats
        assert 'knowledge_graph_edges' in stats
        assert 'average_confidence' in stats
        assert 'average_novelty' in stats
        assert 'average_practical_value' in stats
    
    @pytest.mark.asyncio
    async def test_knowledge_entries(self, knowledge_expander):
        """Test knowledge entries."""
        entries = knowledge_expander.get_knowledge_entries()
        assert isinstance(entries, list)
        
        # Test filtering by knowledge type
        entries = knowledge_expander.get_knowledge_entries(KnowledgeType.ALGORITHM_DISCOVERY)
        assert isinstance(entries, list)
    
    @pytest.mark.asyncio
    async def test_research_directions(self, knowledge_expander):
        """Test research directions."""
        directions = knowledge_expander.get_research_directions()
        assert isinstance(directions, list)
    
    @pytest.mark.asyncio
    async def test_insight_patterns(self, knowledge_expander):
        """Test insight patterns."""
        patterns = knowledge_expander.get_insight_patterns()
        assert isinstance(patterns, dict)
    
    @pytest.mark.asyncio
    async def test_knowledge_graph(self, knowledge_expander):
        """Test knowledge graph."""
        graph = knowledge_expander.get_knowledge_graph()
        assert graph is not None

class TestContinuousEvolver:
    """Test cases for the Continuous Evolver."""
    
    @pytest.fixture
    def continuous_evolver(self):
        """Create a continuous evolver for testing."""
        return ContinuousEvolver()
    
    @pytest.mark.asyncio
    async def test_continuous_evolver_initialization(self, continuous_evolver):
        """Test continuous evolver initialization."""
        assert continuous_evolver.evolver_id is not None
        assert continuous_evolver.running == False
        assert continuous_evolver.active_evolutions == {}
        assert continuous_evolver.completed_evolutions == []
        assert continuous_evolver.failed_evolutions == []
        assert continuous_evolver.evolution_queue == deque()
        assert continuous_evolver.experiment_proposals == []
        assert continuous_evolver.evolution_population == []
        assert continuous_evolver.best_individuals == []
        assert continuous_evolver.innovation_pool == []
    
    @pytest.mark.asyncio
    async def test_continuous_evolver_start_stop(self, continuous_evolver):
        """Test continuous evolver start and stop."""
        # Test start
        await continuous_evolver.start()
        assert continuous_evolver.running == True
        
        # Test stop
        await continuous_evolver.stop()
        assert continuous_evolver.running == False
    
    @pytest.mark.asyncio
    async def test_continuous_evolver_statistics(self, continuous_evolver):
        """Test continuous evolver statistics."""
        stats = continuous_evolver.get_evolution_statistics()
        
        assert 'evolver_id' in stats
        assert 'running' in stats
        assert 'current_phase' in stats
        assert 'total_evolutions' in stats
        assert 'completed_evolutions' in stats
        assert 'failed_evolutions' in stats
        assert 'active_evolutions' in stats
        assert 'queued_evolutions' in stats
        assert 'success_rate' in stats
        assert 'phase_distribution' in stats
        assert 'strategy_distribution' in stats
        assert 'population_size' in stats
        assert 'best_individuals' in stats
        assert 'innovation_pool' in stats
        assert 'experiment_proposals' in stats
    
    @pytest.mark.asyncio
    async def test_experiment_proposals(self, continuous_evolver):
        """Test experiment proposals."""
        proposals = continuous_evolver.get_experiment_proposals()
        assert isinstance(proposals, list)
    
    @pytest.mark.asyncio
    async def test_evolution_insights(self, continuous_evolver):
        """Test evolution insights."""
        insights = continuous_evolver.get_evolution_insights()
        assert isinstance(insights, dict)
        assert 'current_phase' in insights
        assert 'population_size' in insights
        assert 'best_individuals' in insights
        assert 'innovation_pool' in insights
        assert 'adaptation_history' in insights
        assert 'research_trends' in insights
        assert 'hardware_trends' in insights
        assert 'algorithm_trends' in insights

class TestQuantumResearchEngineIntegration:
    """Integration tests for the Quantum Research Engine."""
    
    @pytest.fixture
    def research_engine(self):
        """Create a research engine for integration testing."""
        config = ResearchConfig(
            research_mode=ResearchMode.EXPLORATION,
            enable_algorithm_generation=True,
            enable_autonomous_experimentation=True,
            enable_self_evolving_optimization=True,
            enable_strategy_advice=True,
            enable_knowledge_expansion=True,
            enable_continuous_evolution=True,
            max_concurrent_research=3,
            research_timeout=30.0,
            innovation_threshold=0.7,
            performance_threshold=0.6
        )
        return QuantumResearchEngine(config)
    
    @pytest.mark.asyncio
    async def test_full_research_workflow(self, research_engine):
        """Test full research workflow."""
        # Start the research engine
        await research_engine.start()
        
        # Wait for some research activities
        await asyncio.sleep(5.0)
        
        # Check that research is happening
        stats = research_engine.get_research_statistics()
        assert stats['running'] == True
        
        # Check breakthrough detections
        breakthroughs = research_engine.get_breakthrough_detections()
        assert isinstance(breakthroughs, list)
        
        # Check trend analysis
        trends = research_engine.get_trend_analysis()
        assert isinstance(trends, dict)
        
        # Check research results
        results = research_engine.get_research_results()
        assert isinstance(results, list)
        
        # Stop the research engine
        await research_engine.stop()
        assert research_engine.running == False
    
    @pytest.mark.asyncio
    async def test_research_mode_switching(self, research_engine):
        """Test research mode switching."""
        # Test different research modes
        modes = [
            ResearchMode.EXPLORATION,
            ResearchMode.EXPLOITATION,
            ResearchMode.INNOVATION,
            ResearchMode.OPTIMIZATION,
            ResearchMode.VALIDATION,
            ResearchMode.INTEGRATION
        ]
        
        for mode in modes:
            research_engine.config.research_mode = mode
            assert research_engine.config.research_mode == mode
    
    @pytest.mark.asyncio
    async def test_component_integration(self, research_engine):
        """Test component integration."""
        # Start the research engine
        await research_engine.start()
        
        # Wait for components to initialize
        await asyncio.sleep(2.0)
        
        # Check that all components are running
        assert research_engine.algorithm_generator.running == True
        assert research_engine.experimenter.running == True
        assert research_engine.optimizer.running == True
        assert research_engine.strategy_advisor.running == True
        assert research_engine.knowledge_expander.running == True
        assert research_engine.continuous_evolver.running == True
        
        # Stop the research engine
        await research_engine.stop()
        
        # Check that all components are stopped
        assert research_engine.algorithm_generator.running == False
        assert research_engine.experimenter.running == False
        assert research_engine.optimizer.running == False
        assert research_engine.strategy_advisor.running == False
        assert research_engine.knowledge_expander.running == False
        assert research_engine.continuous_evolver.running == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
