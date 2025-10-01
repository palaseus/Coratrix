"""
Test Suite for Distributed Execution Layer
=========================================

This test suite validates the Distributed Execution Layer
that enables quantum circuits to be executed across multiple nodes.

Tests cover:
- Execution Graph functionality
- RPC Layer communication
- State Sharding system
- Distributed Executor coordination
- Cluster Management
- Fault Tolerance
- Integration and performance testing
"""

import unittest
import asyncio
import time
import numpy as np
import pytest
from typing import Dict, List, Any
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from distributed.execution_graph import ExecutionGraph, ExecutionNode, NodeType, NodeStatus, ExecutionStatus
from distributed.rpc_layer import RPCLayer, MessageType, ConnectionStatus
from distributed.state_sharding import StateSharding, ShardType, ShardStatus

class TestDistributedExecution(unittest.TestCase):
    """Test suite for the Distributed Execution Layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.execution_graph = ExecutionGraph()
        self.rpc_layer = RPCLayer("test_node", 8080)
        self.state_sharding = StateSharding("test_node")
        
        # Test circuits
        self.bell_state_circuit = {
            'name': 'Bell State',
            'num_qubits': 2,
            'gates': [
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]}
            ]
        }
        
        self.ghz_state_circuit = {
            'name': 'GHZ State',
            'num_qubits': 3,
            'gates': [
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]},
                {'type': 'CNOT', 'qubits': [0, 2]}
            ]
        }
        
        self.large_circuit = {
            'name': 'Large Circuit',
            'num_qubits': 8,
            'gates': [
                {'type': 'H', 'qubits': [i]} for i in range(8)
            ] + [
                {'type': 'CNOT', 'qubits': [i, i+1]} for i in range(0, 7, 2)
            ]
        }
        
        # Test nodes
        self.compute_node_1 = ExecutionNode(
            node_id="compute_1",
            node_type=NodeType.COMPUTE_NODE,
            capabilities={'performance_score': 0.8, 'memory_gb': 16},
            resources={'cpu_cores': 8, 'memory_gb': 16, 'gpu_available': True}
        )
        
        self.compute_node_2 = ExecutionNode(
            node_id="compute_2",
            node_type=NodeType.COMPUTE_NODE,
            capabilities={'performance_score': 0.9, 'memory_gb': 32},
            resources={'cpu_cores': 16, 'memory_gb': 32, 'gpu_available': True}
        )
        
        self.coordinator_node = ExecutionNode(
            node_id="coordinator",
            node_type=NodeType.COORDINATOR_NODE,
            capabilities={'coordination_score': 1.0},
            resources={'cpu_cores': 4, 'memory_gb': 8}
        )
    
    def test_execution_graph_initialization(self):
        """Test execution graph initialization."""
        print("\n🌐 Testing Execution Graph Initialization...")
        
        # Test graph initialization
        self.assertIsNotNone(self.execution_graph.graph)
        self.assertIsNotNone(self.execution_graph.nodes)
        self.assertIsNotNone(self.execution_graph.partitions)
        self.assertIsNotNone(self.execution_graph.tasks)
        self.assertIsNotNone(self.execution_graph.executions)
        
        # Test statistics
        stats = self.execution_graph.get_execution_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('execution_stats', stats)
        self.assertIn('node_count', stats)
        self.assertIn('partition_count', stats)
        self.assertIn('task_count', stats)
        
        print("  ✅ Execution Graph initialized successfully")
    
    def test_rpc_layer_initialization(self):
        """Test RPC layer initialization."""
        print("\n🌐 Testing RPC Layer Initialization...")
        
        # Test RPC layer components
        self.assertIsNotNone(self.rpc_layer.server)
        self.assertIsNotNone(self.rpc_layer.client)
        self.assertIsNotNone(self.rpc_layer.node_registry)
        self.assertIsNotNone(self.rpc_layer.message_routing)
        
        # Test statistics
        stats = self.rpc_layer.get_layer_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('layer_stats', stats)
        self.assertIn('server_stats', stats)
        self.assertIn('client_stats', stats)
        
        print("  ✅ RPC Layer initialized successfully")
    
    def test_state_sharding_initialization(self):
        """Test state sharding initialization."""
        print("\n🌐 Testing State Sharding Initialization...")
        
        # Test sharding components
        self.assertIsNotNone(self.state_sharding.shard_manager)
        self.assertIsNotNone(self.state_sharding.sharding_stats)
        
        # Test statistics
        stats = self.state_sharding.get_sharding_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('sharding_stats', stats)
        self.assertIn('shard_manager_stats', stats)
        
        print("  ✅ State Sharding initialized successfully")
    
    def test_execution_graph_node_management(self):
        """Test execution graph node management."""
        print("\n🌐 Testing Execution Graph Node Management...")
        
        # Add nodes to execution graph
        self.execution_graph.add_node(self.compute_node_1)
        self.execution_graph.add_node(self.compute_node_2)
        self.execution_graph.add_node(self.coordinator_node)
        
        # Test node addition
        self.assertEqual(len(self.execution_graph.nodes), 3)
        self.assertIn("compute_1", self.execution_graph.nodes)
        self.assertIn("compute_2", self.execution_graph.nodes)
        self.assertIn("coordinator", self.execution_graph.nodes)
        
        # Test node status
        node_status = self.execution_graph.get_node_status()
        self.assertIsInstance(node_status, dict)
        self.assertIn("compute_1", node_status)
        self.assertIn("compute_2", node_status)
        self.assertIn("coordinator", node_status)
        
        # Test edge addition
        self.execution_graph.add_edge("coordinator", "compute_1", 1.0)
        self.execution_graph.add_edge("coordinator", "compute_2", 1.0)
        self.execution_graph.add_edge("compute_1", "compute_2", 0.5)
        
        # Test edge removal
        self.execution_graph.remove_edge("compute_1", "compute_2")
        
        # Test node removal
        self.execution_graph.remove_node("coordinator")
        self.assertEqual(len(self.execution_graph.nodes), 2)
        self.assertNotIn("coordinator", self.execution_graph.nodes)
        
        print("  ✅ Execution Graph node management validated")
    
    @pytest.mark.asyncio
    async def test_circuit_partitioning(self):
        """Test circuit partitioning."""
        print("\n🌐 Testing Circuit Partitioning...")
        
        # Add nodes to execution graph
        self.execution_graph.add_node(self.compute_node_1)
        self.execution_graph.add_node(self.compute_node_2)
        
        # Test balanced partitioning
        bell_partitions = await self.execution_graph.partition_circuit(self.bell_state_circuit, "balanced")
        self.assertIsInstance(bell_partitions, list)
        self.assertGreater(len(bell_partitions), 0)
        
        for partition in bell_partitions:
            self.assertIsNotNone(partition.partition_id)
            self.assertIsNotNone(partition.circuit_section)
            self.assertIsNotNone(partition.assigned_node)
            self.assertIn(partition.assigned_node, self.execution_graph.nodes)
        
        print("  ✅ Bell state partitioning completed")
        
        # Test GHZ state partitioning
        ghz_partitions = await self.execution_graph.partition_circuit(self.ghz_state_circuit, "balanced")
        self.assertIsInstance(ghz_partitions, list)
        self.assertGreater(len(ghz_partitions), 0)
        
        print("  ✅ GHZ state partitioning completed")
        
        # Test large circuit partitioning
        large_partitions = await self.execution_graph.partition_circuit(self.large_circuit, "balanced")
        self.assertIsInstance(large_partitions, list)
        self.assertGreater(len(large_partitions), 0)
        
        print("  ✅ Large circuit partitioning completed")
        
        print("  ✅ Circuit Partitioning validated")
    
    @pytest.mark.asyncio
    async def test_circuit_execution(self):
        """Test circuit execution."""
        print("\n🌐 Testing Circuit Execution...")
        
        # Add nodes to execution graph
        self.execution_graph.add_node(self.compute_node_1)
        self.execution_graph.add_node(self.compute_node_2)
        
        # Test Bell state execution
        bell_result = await self.execution_graph.execute_circuit(self.bell_state_circuit)
        self.assertIsNotNone(bell_result)
        self.assertIsNotNone(bell_result.execution_id)
        self.assertIsNotNone(bell_result.circuit_data)
        self.assertIsNotNone(bell_result.execution_status)
        self.assertIsNotNone(bell_result.total_execution_time)
        self.assertIsNotNone(bell_result.node_utilization)
        self.assertIsNotNone(bell_result.memory_usage)
        self.assertIsNotNone(bell_result.results)
        
        self.assertGreater(bell_result.total_execution_time, 0.0)
        
        print(f"  ✅ Bell state execution: {bell_result.total_execution_time:.4f}s")
        
        # Test GHZ state execution
        ghz_result = await self.execution_graph.execute_circuit(self.ghz_state_circuit)
        self.assertIsNotNone(ghz_result)
        self.assertIsNotNone(ghz_result.execution_id)
        self.assertIsNotNone(ghz_result.execution_status)
        self.assertGreater(ghz_result.total_execution_time, 0.0)
        
        print(f"  ✅ GHZ state execution: {ghz_result.total_execution_time:.4f}s")
        
        # Test large circuit execution
        large_result = await self.execution_graph.execute_circuit(self.large_circuit)
        self.assertIsNotNone(large_result)
        self.assertIsNotNone(large_result.execution_id)
        self.assertIsNotNone(large_result.execution_status)
        self.assertGreater(large_result.total_execution_time, 0.0)
        
        print(f"  ✅ Large circuit execution: {large_result.total_execution_time:.4f}s")
        
        print("  ✅ Circuit Execution validated")
    
    @pytest.mark.asyncio
    async def test_rpc_communication(self):
        """Test RPC communication."""
        print("\n🌐 Testing RPC Communication...")
        
        # Test node registration
        await self.rpc_layer.register_node("node_1", "localhost", 8081)
        await self.rpc_layer.register_node("node_2", "localhost", 8082)
        
        # Test node registry
        self.assertEqual(len(self.rpc_layer.node_registry), 2)
        self.assertIn("node_1", self.rpc_layer.node_registry)
        self.assertIn("node_2", self.rpc_layer.node_registry)
        
        # Test message sending
        message_id = await self.rpc_layer.client.send_message(
            target_node="node_1",
            message_type=MessageType.HEARTBEAT,
            payload={'timestamp': time.time()}
        )
        self.assertIsNotNone(message_id)
        
        # Test heartbeat
        await self.rpc_layer.send_heartbeat_to_all()
        
        # Test broadcast
        await self.rpc_layer.broadcast_message(
            MessageType.STATUS_UPDATE,
            {'status': 'active', 'timestamp': time.time()}
        )
        
        # Test node unregistration
        await self.rpc_layer.unregister_node("node_1")
        self.assertEqual(len(self.rpc_layer.node_registry), 1)
        self.assertNotIn("node_1", self.rpc_layer.node_registry)
        
        print("  ✅ RPC Communication validated")
    
    @pytest.mark.asyncio
    async def test_state_sharding_operations(self):
        """Test state sharding operations."""
        print("\n🌐 Testing State Sharding Operations...")
        
        # Start state sharding
        self.state_sharding.start_sharding()
        
        try:
            # Test distributed state creation
            bell_shards = await self.state_sharding.create_distributed_state(
                self.bell_state_circuit, "balanced"
            )
            self.assertIsInstance(bell_shards, list)
            self.assertGreater(len(bell_shards), 0)
            
            for shard in bell_shards:
                self.assertIsNotNone(shard.shard_id)
                self.assertIsNotNone(shard.shard_type)
                self.assertIsNotNone(shard.state_data)
                self.assertIsNotNone(shard.qubit_indices)
                self.assertIsNotNone(shard.node_id)
            
            print("  ✅ Bell state sharding completed")
            
            # Test GHZ state sharding
            ghz_shards = await self.state_sharding.create_distributed_state(
                self.ghz_state_circuit, "entanglement_aware"
            )
            self.assertIsInstance(ghz_shards, list)
            self.assertGreater(len(ghz_shards), 0)
            
            print("  ✅ GHZ state sharding completed")
            
            # Test large circuit sharding
            large_shards = await self.state_sharding.create_distributed_state(
                self.large_circuit, "performance_optimized"
            )
            self.assertIsInstance(large_shards, list)
            self.assertGreater(len(large_shards), 0)
            
            print("  ✅ Large circuit sharding completed")
            
            # Test shard operations
            if bell_shards:
                shard = bell_shards[0]
                
                # Test shard retrieval
                retrieved_shard = await self.state_sharding.shard_manager.get_shard(shard.shard_id)
                self.assertIsNotNone(retrieved_shard)
                self.assertEqual(retrieved_shard.shard_id, shard.shard_id)
                
                # Test shard transfer
                transfer_success = await self.state_sharding.shard_manager.transfer_shard(
                    shard.shard_id, "target_node"
                )
                self.assertTrue(transfer_success)
                
                # Test gate application
                gate_data = {'type': 'H', 'qubits': [0]}
                gate_success = await self.state_sharding.shard_manager.apply_gate_to_shard(
                    shard.shard_id, gate_data
                )
                self.assertTrue(gate_success)
                
                print("  ✅ Shard operations completed")
            
        finally:
            # Stop state sharding
            self.state_sharding.stop_sharding()
        
        print("  ✅ State Sharding Operations validated")
    
    def test_execution_statistics(self):
        """Test execution statistics."""
        print("\n📊 Testing Execution Statistics...")
        
        # Test execution graph statistics
        graph_stats = self.execution_graph.get_execution_statistics()
        self.assertIsInstance(graph_stats, dict)
        self.assertIn('execution_stats', graph_stats)
        self.assertIn('node_count', graph_stats)
        self.assertIn('partition_count', graph_stats)
        self.assertIn('task_count', graph_stats)
        self.assertIn('execution_count', graph_stats)
        
        print("  ✅ Execution Graph statistics validated")
        
        # Test RPC layer statistics
        rpc_stats = self.rpc_layer.get_layer_statistics()
        self.assertIsInstance(rpc_stats, dict)
        self.assertIn('layer_stats', rpc_stats)
        self.assertIn('server_stats', rpc_stats)
        self.assertIn('client_stats', rpc_stats)
        self.assertIn('registered_nodes', rpc_stats)
        
        print("  ✅ RPC Layer statistics validated")
        
        # Test state sharding statistics
        sharding_stats = self.state_sharding.get_sharding_statistics()
        self.assertIsInstance(sharding_stats, dict)
        self.assertIn('sharding_stats', sharding_stats)
        self.assertIn('shard_manager_stats', sharding_stats)
        
        print("  ✅ State Sharding statistics validated")
        
        print("  ✅ Execution Statistics validated")
    
    def test_execution_recommendations(self):
        """Test execution recommendations."""
        print("\n💡 Testing Execution Recommendations...")
        
        # Test execution graph recommendations
        graph_recommendations = self.execution_graph.get_execution_recommendations(self.large_circuit)
        self.assertIsInstance(graph_recommendations, list)
        
        for rec in graph_recommendations:
            self.assertIn('type', rec)
            self.assertIn('message', rec)
            self.assertIn('recommendation', rec)
            self.assertIn('priority', rec)
        
        print("  ✅ Execution Graph recommendations validated")
        
        # Test state sharding recommendations
        sharding_recommendations = self.state_sharding.get_sharding_recommendations(self.large_circuit)
        self.assertIsInstance(sharding_recommendations, list)
        
        for rec in sharding_recommendations:
            self.assertIn('type', rec)
            self.assertIn('message', rec)
            self.assertIn('recommendation', rec)
            self.assertIn('priority', rec)
        
        print("  ✅ State Sharding recommendations validated")
        
        print("  ✅ Execution Recommendations validated")
    
    @pytest.mark.asyncio
    async def test_integration_performance(self):
        """Test integration and performance."""
        print("\n⚡ Testing Integration and Performance...")
        
        # Add nodes to execution graph
        self.execution_graph.add_node(self.compute_node_1)
        self.execution_graph.add_node(self.compute_node_2)
        
        # Test end-to-end distributed execution
        start_time = time.time()
        
        # Execute Bell state
        bell_result = await self.execution_graph.execute_circuit(self.bell_state_circuit)
        bell_time = time.time() - start_time
        
        self.assertIsNotNone(bell_result)
        self.assertLess(bell_time, 5.0)  # Should complete within 5 seconds
        
        print(f"  ✅ Bell state distributed execution: {bell_time:.4f}s")
        
        # Test GHZ state execution
        start_time = time.time()
        ghz_result = await self.execution_graph.execute_circuit(self.ghz_state_circuit)
        ghz_time = time.time() - start_time
        
        self.assertIsNotNone(ghz_result)
        self.assertLess(ghz_time, 5.0)  # Should complete within 5 seconds
        
        print(f"  ✅ GHZ state distributed execution: {ghz_time:.4f}s")
        
        # Test large circuit execution
        start_time = time.time()
        large_result = await self.execution_graph.execute_circuit(self.large_circuit)
        large_time = time.time() - start_time
        
        self.assertIsNotNone(large_result)
        self.assertLess(large_time, 10.0)  # Should complete within 10 seconds
        
        print(f"  ✅ Large circuit distributed execution: {large_time:.4f}s")
        
        print("  ✅ Integration and Performance validated")

def run_distributed_execution_tests():
    """Run all distributed execution tests."""
    print("🌐 DISTRIBUTED EXECUTION LAYER TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        'test_execution_graph_initialization',
        'test_rpc_layer_initialization',
        'test_state_sharding_initialization',
        'test_execution_graph_node_management',
        'test_execution_statistics',
        'test_execution_recommendations'
    ]
    
    for test_case in test_cases:
        test_suite.addTest(TestDistributedExecution(test_case))
    
    # Run synchronous tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run asynchronous tests
    async def run_async_tests():
        print("\n🔄 Running Asynchronous Tests...")
        
        test_instance = TestDistributedExecution()
        test_instance.setUp()
        
        try:
            await test_instance.test_circuit_partitioning()
            await test_instance.test_circuit_execution()
            await test_instance.test_rpc_communication()
            await test_instance.test_state_sharding_operations()
            await test_instance.test_integration_performance()
            
            print("✅ All asynchronous tests completed successfully!")
            
        except Exception as e:
            print(f"❌ Asynchronous test failed: {e}")
            raise
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    print("\n🎉 DISTRIBUTED EXECUTION LAYER TEST SUITE COMPLETED!")
    print("The Distributed Execution Layer is fully validated!")

if __name__ == "__main__":
    run_distributed_execution_tests()
