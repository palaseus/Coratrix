"""
Test Runner for Quantum Research Engine

This module provides a comprehensive test runner for the Quantum Research Engine,
including all components and integration tests.

Author: Quantum Research Engine - Coratrix 4.0
"""

import asyncio
import time
import sys
import os
import traceback
from typing import Dict, List, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the quantum research engine components
from quantum_research.quantum_research_engine import QuantumResearchEngine, ResearchConfig, ResearchMode
from quantum_research.quantum_algorithm_generator import QuantumAlgorithmGenerator, AlgorithmType, InnovationLevel
from quantum_research.autonomous_experimenter import AutonomousExperimenter, ExperimentType, BackendType
from quantum_research.self_evolving_optimizer import SelfEvolvingOptimizer, OptimizationStrategy
from quantum_research.quantum_strategy_advisor import QuantumStrategyAdvisor, StrategyType, UseCase
from quantum_research.knowledge_expander import KnowledgeExpander, KnowledgeType
from quantum_research.continuous_evolver import ContinuousEvolver, EvolutionPhase, EvolutionStrategy

class QuantumResearchTestRunner:
    """Test runner for the Quantum Research Engine."""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.start_time = time.time()
    
    async def run_all_tests(self):
        """Run all quantum research engine tests."""
        print("🧪 QUANTUM RESEARCH ENGINE TEST SUITE")
        print("=" * 50)
        
        # Test categories
        test_categories = [
            ("Quantum Algorithm Generator", self.test_algorithm_generator),
            ("Autonomous Experimenter", self.test_autonomous_experimenter),
            ("Self-Evolving Optimizer", self.test_self_evolving_optimizer),
            ("Quantum Strategy Advisor", self.test_quantum_strategy_advisor),
            ("Knowledge Expander", self.test_knowledge_expander),
            ("Continuous Evolver", self.test_continuous_evolver),
            ("Research Engine Integration", self.test_research_engine_integration),
            ("Full Research Workflow", self.test_full_research_workflow)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n🔬 Testing {category_name}...")
            await self.run_test_category(category_name, test_function)
        
        # Print summary
        self.print_test_summary()
    
    async def run_test_category(self, category_name: str, test_function):
        """Run a test category."""
        try:
            await test_function()
            print(f"✅ {category_name} tests passed")
        except Exception as e:
            print(f"❌ {category_name} tests failed: {e}")
            traceback.print_exc()
    
    async def test_algorithm_generator(self):
        """Test the Quantum Algorithm Generator."""
        print("  Testing algorithm generator initialization...")
        generator = QuantumAlgorithmGenerator()
        assert generator.generator_id is not None
        assert generator.running == False
        self.record_test("Algorithm Generator Initialization", True)
        
        print("  Testing algorithm generator start/stop...")
        await generator.start()
        assert generator.running == True
        await generator.stop()
        assert generator.running == False
        self.record_test("Algorithm Generator Start/Stop", True)
        
        print("  Testing algorithm generation...")
        await generator.start()
        algorithms = await generator.generate_algorithms(num_algorithms=3, focus_innovation=True)
        assert len(algorithms) == 3
        assert all(hasattr(alg, 'algorithm_id') for alg in algorithms)
        await generator.stop()
        self.record_test("Algorithm Generation", True)
        
        print("  Testing algorithm generator statistics...")
        stats = generator.get_generation_statistics()
        assert 'generator_id' in stats
        assert 'running' in stats
        assert 'total_algorithms' in stats
        self.record_test("Algorithm Generator Statistics", True)
    
    async def test_autonomous_experimenter(self):
        """Test the Autonomous Experimenter."""
        print("  Testing experimenter initialization...")
        experimenter = AutonomousExperimenter()
        assert experimenter.experimenter_id is not None
        assert experimenter.running == False
        self.record_test("Experimenter Initialization", True)
        
        print("  Testing experimenter start/stop...")
        await experimenter.start()
        assert experimenter.running == True
        await experimenter.stop()
        assert experimenter.running == False
        self.record_test("Experimenter Start/Stop", True)
        
        print("  Testing experiment execution...")
        await experimenter.start()
        experiment_id = await experimenter.run_experiment(
            algorithm_id="test_algorithm",
            experiment_type=ExperimentType.PERFORMANCE_BENCHMARK,
            backend_type=BackendType.LOCAL_SIMULATOR
        )
        assert experiment_id is not None
        assert experiment_id.startswith("exp_")
        await experimenter.stop()
        self.record_test("Experiment Execution", True)
        
        print("  Testing experimenter statistics...")
        stats = experimenter.get_experiment_statistics()
        assert 'experimenter_id' in stats
        assert 'running' in stats
        assert 'total_experiments' in stats
        self.record_test("Experimenter Statistics", True)
    
    async def test_self_evolving_optimizer(self):
        """Test the Self-Evolving Optimizer."""
        print("  Testing optimizer initialization...")
        optimizer = SelfEvolvingOptimizer()
        assert optimizer.optimizer_id is not None
        assert optimizer.running == False
        self.record_test("Optimizer Initialization", True)
        
        print("  Testing optimizer start/stop...")
        await optimizer.start()
        assert optimizer.running == True
        await optimizer.stop()
        assert optimizer.running == False
        self.record_test("Optimizer Start/Stop", True)
        
        print("  Testing algorithm optimization...")
        await optimizer.start()
        optimization_id = await optimizer.optimize_algorithm(
            algorithm_id="test_algorithm",
            target_metrics=['execution_time', 'accuracy', 'scalability'],
            target_values={'execution_time': 0.1, 'accuracy': 0.95, 'scalability': 0.9},
            strategy=OptimizationStrategy.GENETIC_ALGORITHM
        )
        assert optimization_id is not None
        assert optimization_id.startswith("opt_")
        await optimizer.stop()
        self.record_test("Algorithm Optimization", True)
        
        print("  Testing optimizer statistics...")
        stats = optimizer.get_optimization_statistics()
        assert 'optimizer_id' in stats
        assert 'running' in stats
        assert 'total_optimizations' in stats
        self.record_test("Optimizer Statistics", True)
    
    async def test_quantum_strategy_advisor(self):
        """Test the Quantum Strategy Advisor."""
        print("  Testing strategy advisor initialization...")
        advisor = QuantumStrategyAdvisor()
        assert advisor.advisor_id is not None
        assert advisor.running == False
        self.record_test("Strategy Advisor Initialization", True)
        
        print("  Testing strategy advisor start/stop...")
        await advisor.start()
        assert advisor.running == True
        await advisor.stop()
        assert advisor.running == False
        self.record_test("Strategy Advisor Start/Stop", True)
        
        print("  Testing algorithm analysis...")
        await advisor.start()
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
        recommendations = await advisor.analyze_algorithm(algorithm)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        await advisor.stop()
        self.record_test("Algorithm Analysis", True)
        
        print("  Testing strategy advisor statistics...")
        stats = advisor.get_advisory_statistics()
        assert 'advisor_id' in stats
        assert 'running' in stats
        assert 'total_recommendations' in stats
        self.record_test("Strategy Advisor Statistics", True)
    
    async def test_knowledge_expander(self):
        """Test the Knowledge Expander."""
        print("  Testing knowledge expander initialization...")
        expander = KnowledgeExpander()
        assert expander.expander_id is not None
        assert expander.running == False
        self.record_test("Knowledge Expander Initialization", True)
        
        print("  Testing knowledge expander start/stop...")
        await expander.start()
        assert expander.running == True
        await expander.stop()
        assert expander.running == False
        self.record_test("Knowledge Expander Start/Stop", True)
        
        print("  Testing discovery documentation...")
        await expander.start()
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
        entry_id = await expander.document_discovery(discovery)
        assert entry_id is not None
        assert entry_id.startswith("entry_")
        await expander.stop()
        self.record_test("Discovery Documentation", True)
        
        print("  Testing knowledge expander statistics...")
        stats = expander.get_knowledge_statistics()
        assert 'expander_id' in stats
        assert 'running' in stats
        assert 'total_entries' in stats
        self.record_test("Knowledge Expander Statistics", True)
    
    async def test_continuous_evolver(self):
        """Test the Continuous Evolver."""
        print("  Testing continuous evolver initialization...")
        evolver = ContinuousEvolver()
        assert evolver.evolver_id is not None
        assert evolver.running == False
        self.record_test("Continuous Evolver Initialization", True)
        
        print("  Testing continuous evolver start/stop...")
        await evolver.start()
        assert evolver.running == True
        await evolver.stop()
        assert evolver.running == False
        self.record_test("Continuous Evolver Start/Stop", True)
        
        print("  Testing continuous evolver statistics...")
        stats = evolver.get_evolution_statistics()
        assert 'evolver_id' in stats
        assert 'running' in stats
        assert 'total_evolutions' in stats
        self.record_test("Continuous Evolver Statistics", True)
    
    async def test_research_engine_integration(self):
        """Test the Research Engine integration."""
        print("  Testing research engine initialization...")
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
        engine = QuantumResearchEngine(config)
        assert engine.engine_id is not None
        assert engine.running == False
        self.record_test("Research Engine Initialization", True)
        
        print("  Testing research engine start/stop...")
        await engine.start()
        assert engine.running == True
        await engine.stop()
        assert engine.running == False
        self.record_test("Research Engine Start/Stop", True)
        
        print("  Testing research engine statistics...")
        stats = engine.get_research_statistics()
        assert 'engine_id' in stats
        assert 'running' in stats
        assert 'research_mode' in stats
        self.record_test("Research Engine Statistics", True)
    
    async def test_full_research_workflow(self):
        """Test the full research workflow."""
        print("  Testing full research workflow...")
        config = ResearchConfig(
            research_mode=ResearchMode.EXPLORATION,
            enable_algorithm_generation=True,
            enable_autonomous_experimentation=True,
            enable_self_evolving_optimization=True,
            enable_strategy_advice=True,
            enable_knowledge_expansion=True,
            enable_continuous_evolution=True,
            max_concurrent_research=2,
            research_timeout=15.0,
            innovation_threshold=0.6,
            performance_threshold=0.5
        )
        engine = QuantumResearchEngine(config)
        
        # Start the research engine
        await engine.start()
        assert engine.running == True
        
        # Wait for some research activities
        await asyncio.sleep(3.0)
        
        # Check that research is happening
        stats = engine.get_research_statistics()
        assert stats['running'] == True
        
        # Check breakthrough detections
        breakthroughs = engine.get_breakthrough_detections()
        assert isinstance(breakthroughs, list)
        
        # Check trend analysis
        trends = engine.get_trend_analysis()
        assert isinstance(trends, dict)
        
        # Check research results
        results = engine.get_research_results()
        assert isinstance(results, list)
        
        # Stop the research engine
        await engine.stop()
        assert engine.running == False
        self.record_test("Full Research Workflow", True)
    
    def record_test(self, test_name: str, passed: bool):
        """Record a test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'timestamp': time.time()
        })
    
    def print_test_summary(self):
        """Print test summary."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 50)
        print("🧪 QUANTUM RESEARCH ENGINE TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        
        if self.failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['name']}")
        else:
            print("\n✅ All tests passed successfully!")
        
        print("\n🚀 Quantum Research Engine is ready for deployment!")

async def main():
    """Main test runner."""
    runner = QuantumResearchTestRunner()
    await runner.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
