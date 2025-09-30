"""
Test Suite for Execution Graph Optimizer
========================================

Comprehensive test suite for the Execution Graph Optimizer module.
"""

import unittest
import asyncio
import time
import logging
from typing import Dict, List, Any

# Import the optimizer modules
from optimizer.execution_graph_optimizer import (
    ExecutionGraphOptimizer, OptimizationStrategy, PartitioningAlgorithm
)
from optimizer.quantum_optimizer import (
    QuantumOptimizer, OptimizationType, OptimizationAlgorithm, OptimizationTarget
)
from optimizer.optimization_manager import (
    OptimizationManager, OptimizationConfig, OptimizationPriority
)

class TestExecutionGraphOptimizer(unittest.TestCase):
    """Test cases for Execution Graph Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.optimizer = ExecutionGraphOptimizer()
        self.quantum_optimizer = QuantumOptimizer()
        self.optimization_config = OptimizationConfig(max_concurrent_optimizations=2)
        self.optimization_manager = OptimizationManager(self.optimization_config)
        
        # Test circuit data
        self.circuit_data = {
            'name': 'Test Circuit',
            'num_qubits': 4,
            'gates': [
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]},
                {'type': 'H', 'qubits': [2]},
                {'type': 'CNOT', 'qubits': [2, 3]},
                {'type': 'CNOT', 'qubits': [1, 2]}
            ]
        }
    
    def test_execution_graph_optimizer_initialization(self):
        """Test execution graph optimizer initialization."""
        print("\n🎨 Testing Execution Graph Optimizer Initialization...")
        
        # Test optimizer creation
        self.assertIsNotNone(self.optimizer)
        self.assertIsInstance(self.optimizer, ExecutionGraphOptimizer)
        
        # Test optimization strategies
        self.assertGreater(len(self.optimizer.optimization_strategies), 0)
        self.assertIn(OptimizationStrategy.MINIMIZE_ENTANGLEMENT_CUTS, self.optimizer.optimization_strategies)
        self.assertIn(OptimizationStrategy.MINIMIZE_COMMUNICATION, self.optimizer.optimization_strategies)
        self.assertIn(OptimizationStrategy.BALANCE_LOAD, self.optimizer.optimization_strategies)
        self.assertIn(OptimizationStrategy.MAXIMIZE_PARALLELISM, self.optimizer.optimization_strategies)
        self.assertIn(OptimizationStrategy.MINIMIZE_LATENCY, self.optimizer.optimization_strategies)
        
        # Test partitioning algorithms
        self.assertGreater(len(self.optimizer.partitioning_algorithms), 0)
        self.assertIn(PartitioningAlgorithm.METIS, self.optimizer.partitioning_algorithms)
        self.assertIn(PartitioningAlgorithm.SPECTRAL, self.optimizer.partitioning_algorithms)
        self.assertIn(PartitioningAlgorithm.KERNIGHAN_LIN, self.optimizer.partitioning_algorithms)
        self.assertIn(PartitioningAlgorithm.GENETIC, self.optimizer.partitioning_algorithms)
        self.assertIn(PartitioningAlgorithm.HYBRID, self.optimizer.partitioning_algorithms)
        
        print("  ✅ Execution Graph Optimizer initialization successful")
    
    def test_quantum_optimizer_initialization(self):
        """Test quantum optimizer initialization."""
        print("\n🎨 Testing Quantum Optimizer Initialization...")
        
        # Test optimizer creation
        self.assertIsNotNone(self.quantum_optimizer)
        self.assertIsInstance(self.quantum_optimizer, QuantumOptimizer)
        
        # Test optimization algorithms
        self.assertGreater(len(self.quantum_optimizer.optimization_algorithms), 0)
        self.assertIn(OptimizationAlgorithm.GENETIC, self.quantum_optimizer.optimization_algorithms)
        self.assertIn(OptimizationAlgorithm.SIMULATED_ANNEALING, self.quantum_optimizer.optimization_algorithms)
        self.assertIn(OptimizationAlgorithm.PARTICLE_SWARM, self.quantum_optimizer.optimization_algorithms)
        self.assertIn(OptimizationAlgorithm.REINFORCEMENT_LEARNING, self.quantum_optimizer.optimization_algorithms)
        self.assertIn(OptimizationAlgorithm.GRADIENT_DESCENT, self.quantum_optimizer.optimization_algorithms)
        
        print("  ✅ Quantum Optimizer initialization successful")
    
    def test_optimization_manager_initialization(self):
        """Test optimization manager initialization."""
        print("\n🎨 Testing Optimization Manager Initialization...")
        
        # Test manager creation
        self.assertIsNotNone(self.optimization_manager)
        self.assertIsInstance(self.optimization_manager, OptimizationManager)
        
        # Test configuration
        self.assertEqual(self.optimization_manager.config.max_concurrent_optimizations, 2)
        self.assertEqual(self.optimization_manager.config.optimization_timeout, 300.0)
        self.assertEqual(self.optimization_manager.config.retry_attempts, 3)
        self.assertTrue(self.optimization_manager.config.enable_parallel_optimization)
        self.assertEqual(self.optimization_manager.config.optimization_queue_size, 1000)
        
        # Test worker threads
        self.assertGreater(len(self.optimization_manager.optimization_workers), 0)
        
        print("  ✅ Optimization Manager initialization successful")
    
    def test_execution_graph_optimization(self):
        """Test execution graph optimization."""
        print("\n🎨 Testing Execution Graph Optimization...")
        
        # Test optimization with different strategies
        strategies = [
            OptimizationStrategy.MINIMIZE_ENTANGLEMENT_CUTS,
            OptimizationStrategy.MINIMIZE_COMMUNICATION,
            OptimizationStrategy.BALANCE_LOAD,
            OptimizationStrategy.MAXIMIZE_PARALLELISM,
            OptimizationStrategy.MINIMIZE_LATENCY
        ]
        
        for strategy in strategies:
            result = asyncio.run(self.optimizer.optimize_execution_graph(
                self.circuit_data, num_nodes=2, strategy=strategy
            ))
            
            self.assertIsNotNone(result)
            self.assertTrue(result.success)
            self.assertIsInstance(result.optimized_graph, dict)
            self.assertIsInstance(result.partitioning, dict)
            self.assertGreaterEqual(result.performance_improvement, 0.0)
            self.assertGreater(result.optimization_time, 0.0)
            self.assertIsInstance(result.recommendations, list)
            self.assertIsInstance(result.metrics, dict)
            
            print(f"  ✅ {strategy.value} optimization successful: {result.performance_improvement:.2%} improvement")
    
    def test_quantum_circuit_optimization(self):
        """Test quantum circuit optimization."""
        print("\n🎨 Testing Quantum Circuit Optimization...")
        
        # Create optimization targets
        targets = [
            OptimizationTarget(OptimizationType.GATE_REDUCTION, 3.0, 1.0),
            OptimizationTarget(OptimizationType.DEPTH_REDUCTION, 4.0, 0.8),
            OptimizationTarget(OptimizationType.FIDELITY_IMPROVEMENT, 0.95, 0.6)
        ]
        
        # Test optimization with different algorithms
        algorithms = [
            OptimizationAlgorithm.GENETIC,
            OptimizationAlgorithm.SIMULATED_ANNEALING,
            OptimizationAlgorithm.PARTICLE_SWARM,
            OptimizationAlgorithm.REINFORCEMENT_LEARNING,
            OptimizationAlgorithm.GRADIENT_DESCENT
        ]
        
        for algorithm in algorithms:
            result = asyncio.run(self.quantum_optimizer.optimize_circuit(
                self.circuit_data, targets, algorithm
            ))
            
            self.assertIsNotNone(result)
            self.assertTrue(result.success)
            self.assertIsInstance(result.optimized_circuit, dict)
            self.assertIsInstance(result.optimization_metrics, dict)
            self.assertGreaterEqual(result.performance_improvement, 0.0)
            self.assertGreater(result.optimization_time, 0.0)
            self.assertIsInstance(result.recommendations, list)
            
            print(f"  ✅ {algorithm.value} optimization successful: {result.performance_improvement:.2%} improvement")
    
    def test_optimization_task_management(self):
        """Test optimization task management."""
        print("\n🎨 Testing Optimization Task Management...")
        
        # Submit optimization tasks
        task_ids = []
        for i in range(3):
            task_id = asyncio.run(self.optimization_manager.submit_optimization_task(
                self.circuit_data, f"optimization_{i}", OptimizationPriority.MEDIUM
            ))
            task_ids.append(task_id)
            self.assertIsNotNone(task_id)
            self.assertIsInstance(task_id, str)
        
        print(f"  ✅ Submitted {len(task_ids)} optimization tasks")
        
        # Check task status
        for task_id in task_ids:
            status = asyncio.run(self.optimization_manager.get_optimization_status(task_id))
            if status is not None:
                self.assertIn('status', status)
                self.assertIn('optimization_type', status)
                self.assertIn('priority', status)
        
        print("  ✅ Task status checking successful")
        
        # Wait for tasks to complete
        await_time = 0
        max_wait = 10
        while await_time < max_wait:
            all_completed = True
            for task_id in task_ids:
                status = asyncio.run(self.optimization_manager.get_optimization_status(task_id))
                if status and status['status'] not in ['completed', 'failed', 'cancelled']:
                    all_completed = False
                    break
            
            if all_completed:
                break
            
            await_time += 0.1
            time.sleep(0.1)
        
        print("  ✅ Task completion monitoring successful")
        
        # Get optimization results
        for task_id in task_ids:
            result = asyncio.run(self.optimization_manager.get_optimization_results(task_id))
            if result:
                self.assertIsInstance(result, dict)
                self.assertIn('task_id', result)
                self.assertIn('optimization_type', result)
                self.assertIn('success', result)
                self.assertIn('performance_improvement', result)
                self.assertIn('optimization_time', result)
                self.assertIn('recommendations', result)
        
        print("  ✅ Optimization results retrieval successful")
    
    def test_optimization_statistics(self):
        """Test optimization statistics."""
        print("\n🎨 Testing Optimization Statistics...")
        
        # Get execution graph optimizer statistics
        stats = self.optimizer.get_optimization_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('optimization_stats', stats)
        self.assertIn('optimization_history_size', stats)
        self.assertIn('available_strategies', stats)
        self.assertIn('available_algorithms', stats)
        
        print("  ✅ Execution Graph Optimizer statistics successful")
        
        # Get quantum optimizer statistics
        stats = self.quantum_optimizer.get_optimization_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('optimization_stats', stats)
        self.assertIn('optimization_history_size', stats)
        self.assertIn('available_algorithms', stats)
        
        print("  ✅ Quantum Optimizer statistics successful")
        
        # Get optimization manager statistics
        stats = asyncio.run(self.optimization_manager.get_optimization_statistics())
        self.assertIsInstance(stats, dict)
        self.assertIn('optimization_stats', stats)
        self.assertIn('active_optimizations', stats)
        self.assertIn('queue_size', stats)
        self.assertIn('completed_optimizations', stats)
        self.assertIn('worker_threads', stats)
        
        print("  ✅ Optimization Manager statistics successful")
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations."""
        print("\n🎨 Testing Optimization Recommendations...")
        
        # Get execution graph optimizer recommendations
        recommendations = self.optimizer.get_optimization_recommendations(self.circuit_data)
        self.assertIsInstance(recommendations, list)
        
        print("  ✅ Execution Graph Optimizer recommendations successful")
        
        # Get quantum optimizer recommendations
        recommendations = self.quantum_optimizer.get_optimization_recommendations(self.circuit_data)
        self.assertIsInstance(recommendations, list)
        
        print("  ✅ Quantum Optimizer recommendations successful")
        
        # Get optimization manager recommendations
        recommendations = asyncio.run(self.optimization_manager.get_optimization_recommendations(self.circuit_data))
        self.assertIsInstance(recommendations, list)
        
        print("  ✅ Optimization Manager recommendations successful")
    
    async def _async_test_execution_graph_optimization(self):
        """Test execution graph optimization asynchronously."""
        print("\n🔄 Testing Async Execution Graph Optimization...")
        
        # Test optimization with different partitioning algorithms
        algorithms = [
            PartitioningAlgorithm.METIS,
            PartitioningAlgorithm.SPECTRAL,
            PartitioningAlgorithm.KERNIGHAN_LIN,
            PartitioningAlgorithm.GENETIC,
            PartitioningAlgorithm.HYBRID
        ]
        
        for algorithm in algorithms:
            result = await self.optimizer.optimize_execution_graph(
                self.circuit_data, num_nodes=2, 
                strategy=OptimizationStrategy.MINIMIZE_ENTANGLEMENT_CUTS,
                algorithm=algorithm
            )
            
            self.assertIsNotNone(result)
            self.assertTrue(result.success)
            self.assertIsInstance(result.optimized_graph, dict)
            self.assertIsInstance(result.partitioning, dict)
            self.assertGreaterEqual(result.performance_improvement, 0.0)
            self.assertGreater(result.optimization_time, 0.0)
            
            print(f"  ✅ {algorithm.value} partitioning successful: {result.performance_improvement:.2%} improvement")
    
    async def _async_test_quantum_circuit_optimization(self):
        """Test quantum circuit optimization asynchronously."""
        print("\n🔄 Testing Async Quantum Circuit Optimization...")
        
        # Create optimization targets
        targets = [
            OptimizationTarget(OptimizationType.GATE_REDUCTION, 3.0, 1.0),
            OptimizationTarget(OptimizationType.DEPTH_REDUCTION, 4.0, 0.8),
            OptimizationTarget(OptimizationType.FIDELITY_IMPROVEMENT, 0.95, 0.6)
        ]
        
        # Test optimization with different algorithms
        algorithms = [
            OptimizationAlgorithm.GENETIC,
            OptimizationAlgorithm.SIMULATED_ANNEALING,
            OptimizationAlgorithm.PARTICLE_SWARM
        ]
        
        for algorithm in algorithms:
            result = await self.quantum_optimizer.optimize_circuit(
                self.circuit_data, targets, algorithm
            )
            
            self.assertIsNotNone(result)
            self.assertTrue(result.success)
            self.assertIsInstance(result.optimized_circuit, dict)
            self.assertIsInstance(result.optimization_metrics, dict)
            self.assertGreaterEqual(result.performance_improvement, 0.0)
            self.assertGreater(result.optimization_time, 0.0)
            
            print(f"  ✅ {algorithm.value} optimization successful: {result.performance_improvement:.2%} improvement")
    
    async def _async_test_optimization_task_management(self):
        """Test optimization task management asynchronously."""
        print("\n🔄 Testing Async Optimization Task Management...")
        
        # Submit multiple optimization tasks
        task_ids = []
        for i in range(5):
            task_id = await self.optimization_manager.submit_optimization_task(
                self.circuit_data, f"async_optimization_{i}", OptimizationPriority.HIGH
            )
            task_ids.append(task_id)
            self.assertIsNotNone(task_id)
            self.assertIsInstance(task_id, str)
        
        print(f"  ✅ Submitted {len(task_ids)} async optimization tasks")
        
        # Monitor task completion
        completed_tasks = 0
        max_wait = 15
        start_time = time.time()
        
        while completed_tasks < len(task_ids) and (time.time() - start_time) < max_wait:
            for task_id in task_ids:
                status = await self.optimization_manager.get_optimization_status(task_id)
                if status and status['status'] in ['completed', 'failed', 'cancelled']:
                    completed_tasks += 1
            
            if completed_tasks < len(task_ids):
                await asyncio.sleep(0.1)
        
        print(f"  ✅ {completed_tasks}/{len(task_ids)} async tasks completed")
        
        # Get results for completed tasks
        for task_id in task_ids:
            result = await self.optimization_manager.get_optimization_results(task_id)
            if result:
                self.assertIsInstance(result, dict)
                self.assertIn('task_id', result)
                self.assertIn('optimization_type', result)
                self.assertIn('success', result)
                self.assertIn('performance_improvement', result)
        
        print("  ✅ Async optimization results retrieval successful")
    
    async def _async_test_optimization_integration(self):
        """Test optimization integration asynchronously."""
        print("\n🔄 Testing Async Optimization Integration...")
        
        # Test integrated optimization workflow
        # 1. Submit optimization task
        task_id = await self.optimization_manager.submit_optimization_task(
            self.circuit_data, "integration_test", OptimizationPriority.HIGH
        )
        
        # 2. Wait for task completion
        max_wait = 10
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            status = await self.optimization_manager.get_optimization_status(task_id)
            if status and status['status'] in ['completed', 'failed', 'cancelled']:
                break
            await asyncio.sleep(0.1)
        
        # 3. Get optimization results
        result = await self.optimization_manager.get_optimization_results(task_id)
        if result:
            self.assertIsInstance(result, dict)
            self.assertIn('task_id', result)
            self.assertIn('optimization_type', result)
            self.assertIn('success', result)
            self.assertIn('performance_improvement', result)
            self.assertIn('optimization_time', result)
            self.assertIn('recommendations', result)
        
        print("  ✅ Async optimization integration successful")
    
    def test_async_suite(self):
        """Run async test suite."""
        print("\n🔄 Running Asynchronous Tests...")
        asyncio.run(self._async_test_execution_graph_optimization())
        asyncio.run(self._async_test_quantum_circuit_optimization())
        asyncio.run(self._async_test_optimization_task_management())
        asyncio.run(self._async_test_optimization_integration())
        print("✅ All asynchronous tests completed successfully!")

if __name__ == "__main__":
    unittest.main()
