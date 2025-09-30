"""
Coratrix 4.0 Quantum OS - Distributed Execution Layer
====================================================

The Distributed Execution Layer enables Coratrix 4.0 to execute quantum
circuits across multiple nodes with state sharding and RPC communication.

This is the GOD-TIER distributed quantum computing system that transforms
Coratrix into a truly scalable quantum OS.

Key Features:
- Node-based execution graph
- RPC layer for state transfer
- State sharding across nodes
- Cluster expansion capabilities
- Fault tolerance and recovery
- Performance monitoring and load balancing
"""

from .execution_graph import ExecutionGraph, ExecutionNode, NodeType
from .rpc_layer import RPCLayer, RPCServer, RPCClient
from .state_sharding import StateSharder, ShardManager
from .distributed_executor import DistributedExecutor, DistributedConfig
from .cluster_manager import ClusterManager, NodeManager
from .fault_tolerance import FaultTolerance, RecoveryManager

__all__ = [
    'ExecutionGraph',
    'ExecutionNode',
    'NodeType',
    'RPCLayer',
    'RPCServer',
    'RPCClient',
    'StateSharder',
    'ShardManager',
    'DistributedExecutor',
    'DistributedConfig',
    'ClusterManager',
    'NodeManager',
    'FaultTolerance',
    'RecoveryManager'
]
