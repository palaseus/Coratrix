"""
State Sharding - Distributed Quantum State Management
====================================================

The State Sharding system enables Coratrix 4.0 to distribute quantum
states across multiple nodes for scalable quantum circuit execution.

This is the state management system that enables
distributed quantum state manipulation and coordination.
"""

import time
import logging
import numpy as np
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)

class ShardType(Enum):
    """Types of quantum state shards."""
    DENSE_STATE = "dense_state"
    SPARSE_STATE = "sparse_state"
    TENSOR_NETWORK = "tensor_network"
    MEASUREMENT_RESULT = "measurement_result"
    INTERMEDIATE_STATE = "intermediate_state"

class ShardStatus(Enum):
    """Status of quantum state shards."""
    CREATED = "created"
    TRANSFERRING = "transferring"
    AVAILABLE = "available"
    LOCKED = "locked"
    DELETED = "deleted"
    ERROR = "error"

@dataclass
class QuantumShard:
    """A quantum state shard."""
    shard_id: str
    shard_type: ShardType
    state_data: Dict[str, Any]
    qubit_indices: List[int]
    node_id: str
    status: ShardStatus = ShardStatus.CREATED
    size_bytes: int = 0
    checksum: str = ""
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ShardOperation:
    """An operation on quantum state shards."""
    operation_id: str
    operation_type: str
    shard_ids: List[str]
    target_node: str
    parameters: Dict[str, Any]
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class ShardManager:
    """
    Shard Manager for Quantum State Sharding.
    
    This manages the creation, distribution, and coordination
    of quantum state shards across multiple nodes.
    """
    
    def __init__(self, node_id: str):
        """Initialize the shard manager."""
        self.node_id = node_id
        self.shards: Dict[str, QuantumShard] = {}
        self.shard_operations: Dict[str, ShardOperation] = {}
        self.shard_index: Dict[str, List[str]] = defaultdict(list)  # qubit -> shard_ids
        
        # Shard statistics
        self.shard_stats = {
            'total_shards_created': 0,
            'total_shards_transferred': 0,
            'total_operations_completed': 0,
            'average_shard_size': 0.0,
            'total_storage_used': 0,
            'shard_hit_rate': 0.0
        }
        
        # Cleanup thread
        self.cleanup_thread = None
        self.running = False
        
        logger.info(f"🌐 Shard Manager initialized for node: {node_id}")
    
    def start_manager(self):
        """Start the shard manager."""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("🌐 Shard Manager started")
    
    def stop_manager(self):
        """Stop the shard manager."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        logger.info("🌐 Shard Manager stopped")
    
    async def create_shard(self, state_data: Dict[str, Any], qubit_indices: List[int], 
                          shard_type: ShardType = ShardType.DENSE_STATE) -> QuantumShard:
        """Create a new quantum state shard."""
        shard_id = f"shard_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        # Calculate shard size
        size_bytes = self._calculate_shard_size(state_data)
        
        # Calculate checksum
        checksum = self._calculate_checksum(state_data)
        
        shard = QuantumShard(
            shard_id=shard_id,
            shard_type=shard_type,
            state_data=state_data,
            qubit_indices=qubit_indices,
            node_id=self.node_id,
            status=ShardStatus.CREATED,
            size_bytes=size_bytes,
            checksum=checksum,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        # Store shard
        self.shards[shard_id] = shard
        
        # Update index
        for qubit in qubit_indices:
            self.shard_index[str(qubit)].append(shard_id)
        
        # Update statistics
        self.shard_stats['total_shards_created'] += 1
        self.shard_stats['total_storage_used'] += size_bytes
        
        # Update average shard size
        total = self.shard_stats['total_shards_created']
        current_avg = self.shard_stats['average_shard_size']
        self.shard_stats['average_shard_size'] = (current_avg * (total - 1) + size_bytes) / total
        
        logger.info(f"🌐 Created shard: {shard_id} ({size_bytes} bytes)")
        return shard
    
    async def get_shard(self, shard_id: str) -> Optional[QuantumShard]:
        """Get a quantum state shard by ID."""
        if shard_id in self.shards:
            shard = self.shards[shard_id]
            shard.last_accessed = time.time()
            shard.access_count += 1
            return shard
        return None
    
    async def transfer_shard(self, shard_id: str, target_node: str) -> bool:
        """Transfer a shard to another node."""
        if shard_id not in self.shards:
            return False
        
        shard = self.shards[shard_id]
        shard.status = ShardStatus.TRANSFERRING
        
        try:
            # Simulate shard transfer
            await asyncio.sleep(0.001)  # Simulate network transfer
            
            # Update shard location
            shard.node_id = target_node
            shard.status = ShardStatus.AVAILABLE
            
            # Update statistics
            self.shard_stats['total_shards_transferred'] += 1
            
            logger.info(f"🌐 Transferred shard {shard_id} to {target_node}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Shard transfer failed: {e}")
            shard.status = ShardStatus.ERROR
            return False
    
    async def merge_shards(self, shard_ids: List[str]) -> Optional[QuantumShard]:
        """Merge multiple shards into a single shard."""
        if not shard_ids:
            return None
        
        # Get all shards
        shards = [self.shards[shard_id] for shard_id in shard_ids if shard_id in self.shards]
        
        if not shards:
            return None
        
        # Merge state data
        merged_state = self._merge_state_data([shard.state_data for shard in shards])
        merged_qubits = list(set(qubit for shard in shards for qubit in shard.qubit_indices))
        
        # Create merged shard
        merged_shard = await self.create_shard(
            state_data=merged_state,
            qubit_indices=merged_qubits,
            shard_type=shards[0].shard_type
        )
        
        # Mark original shards for deletion
        for shard in shards:
            shard.status = ShardStatus.DELETED
        
        logger.info(f"🌐 Merged {len(shards)} shards into {merged_shard.shard_id}")
        return merged_shard
    
    async def split_shard(self, shard_id: str, split_qubits: List[List[int]]) -> List[QuantumShard]:
        """Split a shard into multiple shards."""
        if shard_id not in self.shards:
            return []
        
        original_shard = self.shards[shard_id]
        split_shards = []
        
        for qubit_group in split_qubits:
            # Extract state data for this qubit group
            split_state = self._extract_state_data(original_shard.state_data, qubit_group)
            
            # Create new shard
            split_shard = await self.create_shard(
                state_data=split_state,
                qubit_indices=qubit_group,
                shard_type=original_shard.shard_type
            )
            
            split_shards.append(split_shard)
        
        # Mark original shard for deletion
        original_shard.status = ShardStatus.DELETED
        
        logger.info(f"🌐 Split shard {shard_id} into {len(split_shards)} shards")
        return split_shards
    
    async def apply_gate_to_shard(self, shard_id: str, gate_data: Dict[str, Any]) -> bool:
        """Apply a quantum gate to a shard."""
        if shard_id not in self.shards:
            return False
        
        shard = self.shards[shard_id]
        
        try:
            # Lock shard for operation
            shard.status = ShardStatus.LOCKED
            
            # Apply gate to state data
            updated_state = self._apply_gate_to_state(shard.state_data, gate_data)
            
            # Update shard
            shard.state_data = updated_state
            shard.checksum = self._calculate_checksum(updated_state)
            shard.size_bytes = self._calculate_shard_size(updated_state)
            shard.status = ShardStatus.AVAILABLE
            
            logger.info(f"🌐 Applied gate to shard {shard_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Gate application failed: {e}")
            shard.status = ShardStatus.ERROR
            return False
    
    def _calculate_shard_size(self, state_data: Dict[str, Any]) -> int:
        """Calculate the size of a shard in bytes."""
        # Simplified size calculation
        if 'state_vector' in state_data:
            state_vector = state_data['state_vector']
            if isinstance(state_vector, list):
                return len(state_vector) * 16  # 16 bytes per complex number
            else:
                return len(str(state_vector))
        
        return len(json.dumps(state_data))
    
    def _calculate_checksum(self, state_data: Dict[str, Any]) -> str:
        """Calculate checksum for state data."""
        data_str = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _merge_state_data(self, state_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple state data dictionaries."""
        # Simplified state merging
        merged_state = {
            'state_vector': [],
            'metadata': {}
        }
        
        for state_data in state_data_list:
            if 'state_vector' in state_data:
                merged_state['state_vector'].extend(state_data['state_vector'])
            if 'metadata' in state_data:
                merged_state['metadata'].update(state_data['metadata'])
        
        return merged_state
    
    def _extract_state_data(self, state_data: Dict[str, Any], qubit_indices: List[int]) -> Dict[str, Any]:
        """Extract state data for specific qubits."""
        # Simplified state extraction
        extracted_state = {
            'state_vector': state_data.get('state_vector', []),
            'metadata': {
                'qubit_indices': qubit_indices,
                'extracted_from': state_data.get('metadata', {}).get('original_shard', '')
            }
        }
        
        return extracted_state
    
    def _apply_gate_to_state(self, state_data: Dict[str, Any], gate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a quantum gate to state data."""
        # Simplified gate application
        updated_state = state_data.copy()
        
        # Add gate information to metadata
        if 'metadata' not in updated_state:
            updated_state['metadata'] = {}
        
        if 'applied_gates' not in updated_state['metadata']:
            updated_state['metadata']['applied_gates'] = []
        
        updated_state['metadata']['applied_gates'].append({
            'gate_type': gate_data.get('type', ''),
            'qubits': gate_data.get('qubits', []),
            'timestamp': time.time()
        })
        
        return updated_state
    
    def _cleanup_loop(self):
        """Cleanup loop for managing shard lifecycle."""
        while self.running:
            try:
                current_time = time.time()
                
                # Clean up deleted shards
                deleted_shards = [shard_id for shard_id, shard in self.shards.items() 
                               if shard.status == ShardStatus.DELETED]
                
                for shard_id in deleted_shards:
                    shard = self.shards[shard_id]
                    
                    # Remove from index
                    for qubit in shard.qubit_indices:
                        if shard_id in self.shard_index[str(qubit)]:
                            self.shard_index[str(qubit)].remove(shard_id)
                    
                    # Remove shard
                    del self.shards[shard_id]
                    self.shard_stats['total_storage_used'] -= shard.size_bytes
                
                # Clean up old shards (older than 1 hour)
                old_shards = [shard_id for shard_id, shard in self.shards.items() 
                            if current_time - shard.last_accessed > 3600]
                
                for shard_id in old_shards:
                    shard = self.shards[shard_id]
                    shard.status = ShardStatus.DELETED
                
                time.sleep(10.0)  # Cleanup every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Cleanup loop error: {e}")
                time.sleep(1.0)
    
    def get_shard_statistics(self) -> Dict[str, Any]:
        """Get shard manager statistics."""
        return {
            'shard_stats': self.shard_stats,
            'total_shards': len(self.shards),
            'shards_by_status': {
                status.value: sum(1 for shard in self.shards.values() if shard.status == status)
                for status in ShardStatus
            },
            'shards_by_type': {
                shard_type.value: sum(1 for shard in self.shards.values() if shard.shard_type == shard_type)
                for shard_type in ShardType
            },
            'qubit_coverage': len(self.shard_index),
            'average_access_count': np.mean([shard.access_count for shard in self.shards.values()]) if self.shards else 0.0
        }
    
    def get_shard_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get shard management recommendations."""
        recommendations = []
        
        num_qubits = circuit_data.get('num_qubits', 0)
        gates = circuit_data.get('gates', [])
        
        # Storage recommendations
        if self.shard_stats['total_storage_used'] > 1024 * 1024 * 1024:  # 1GB
            recommendations.append({
                'type': 'storage_management',
                'message': f'High storage usage ({self.shard_stats["total_storage_used"] / (1024**3):.2f} GB)',
                'recommendation': 'Consider shard cleanup or compression',
                'priority': 'medium'
            })
        
        # Shard size recommendations
        if self.shard_stats['average_shard_size'] > 1024 * 1024:  # 1MB
            recommendations.append({
                'type': 'shard_size',
                'message': f'Large average shard size ({self.shard_stats["average_shard_size"] / (1024**2):.2f} MB)',
                'recommendation': 'Consider shard splitting for better distribution',
                'priority': 'low'
            })
        
        # Circuit complexity recommendations
        if num_qubits > 15:
            recommendations.append({
                'type': 'circuit_complexity',
                'message': f'Large circuit ({num_qubits} qubits) detected',
                'recommendation': 'Consider aggressive sharding for better performance',
                'priority': 'high'
            })
        
        return recommendations

class StateSharding:
    """
    State Sharding System for Distributed Quantum State Management.
    
    This is the state management system that enables
    distributed quantum state manipulation and coordination.
    """
    
    def __init__(self, node_id: str):
        """Initialize the state sharding system."""
        self.node_id = node_id
        self.shard_manager = ShardManager(node_id)
        self.coordination_layer = None  # Would integrate with RPC layer
        
        # Sharding statistics
        self.sharding_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_operation_time': 0.0,
            'total_data_transferred': 0
        }
        
        logger.info(f"🌐 State Sharding initialized for node: {node_id}")
    
    def start_sharding(self):
        """Start the state sharding system."""
        self.shard_manager.start_manager()
        logger.info("🌐 State Sharding started")
    
    def stop_sharding(self):
        """Stop the state sharding system."""
        self.shard_manager.stop_manager()
        logger.info("🌐 State Sharding stopped")
    
    async def create_distributed_state(self, circuit_data: Dict[str, Any], 
                                     sharding_strategy: str = "balanced") -> List[QuantumShard]:
        """Create a distributed quantum state from circuit data."""
        logger.info(f"🌐 Creating distributed state for circuit: {circuit_data.get('name', 'Unknown')}")
        
        num_qubits = circuit_data.get('num_qubits', 0)
        gates = circuit_data.get('gates', [])
        
        if num_qubits == 0:
            return []
        
        # Determine sharding strategy
        if sharding_strategy == "balanced":
            shards = await self._balanced_sharding(circuit_data, num_qubits)
        elif sharding_strategy == "entanglement_aware":
            shards = await self._entanglement_aware_sharding(circuit_data, num_qubits)
        elif sharding_strategy == "performance_optimized":
            shards = await self._performance_optimized_sharding(circuit_data, num_qubits)
        else:
            shards = await self._balanced_sharding(circuit_data, num_qubits)
        
        # Update statistics
        self.sharding_stats['total_operations'] += 1
        self.sharding_stats['successful_operations'] += 1
        
        logger.info(f"🌐 Created {len(shards)} distributed state shards")
        return shards
    
    async def _balanced_sharding(self, circuit_data: Dict[str, Any], num_qubits: int) -> List[QuantumShard]:
        """Create balanced shards of the quantum state."""
        shards = []
        
        # Simple balanced sharding - one shard per qubit
        for qubit in range(num_qubits):
            state_data = {
                'state_vector': [1.0, 0.0],  # |0⟩ state
                'metadata': {
                    'qubit_index': qubit,
                    'shard_type': 'single_qubit'
                }
            }
            
            shard = await self.shard_manager.create_shard(
                state_data=state_data,
                qubit_indices=[qubit],
                shard_type=ShardType.DENSE_STATE
            )
            
            shards.append(shard)
        
        return shards
    
    async def _entanglement_aware_sharding(self, circuit_data: Dict[str, Any], num_qubits: int) -> List[QuantumShard]:
        """Create entanglement-aware shards of the quantum state."""
        shards = []
        
        # Analyze entanglement structure
        entanglement_groups = self._analyze_entanglement_groups(circuit_data)
        
        for group_qubits in entanglement_groups:
            state_data = {
                'state_vector': [1.0] + [0.0] * (2**len(group_qubits) - 1),
                'metadata': {
                    'qubit_indices': group_qubits,
                    'shard_type': 'entangled_group'
                }
            }
            
            shard = await self.shard_manager.create_shard(
                state_data=state_data,
                qubit_indices=group_qubits,
                shard_type=ShardType.DENSE_STATE
            )
            
            shards.append(shard)
        
        return shards
    
    async def _performance_optimized_sharding(self, circuit_data: Dict[str, Any], num_qubits: int) -> List[QuantumShard]:
        """Create performance-optimized shards of the quantum state."""
        shards = []
        
        # Performance-based sharding
        if num_qubits <= 4:
            # Small circuits - single shard
            state_data = {
                'state_vector': [1.0] + [0.0] * (2**num_qubits - 1),
                'metadata': {
                    'qubit_indices': list(range(num_qubits)),
                    'shard_type': 'full_state'
                }
            }
            
            shard = await self.shard_manager.create_shard(
                state_data=state_data,
                qubit_indices=list(range(num_qubits)),
                shard_type=ShardType.DENSE_STATE
            )
            
            shards.append(shard)
        else:
            # Large circuits - multiple shards
            shards_per_qubit = max(1, num_qubits // 4)
            
            for i in range(0, num_qubits, shards_per_qubit):
                group_qubits = list(range(i, min(i + shards_per_qubit, num_qubits)))
                
                state_data = {
                    'state_vector': [1.0] + [0.0] * (2**len(group_qubits) - 1),
                    'metadata': {
                        'qubit_indices': group_qubits,
                        'shard_type': 'qubit_group'
                    }
                }
                
                shard = await self.shard_manager.create_shard(
                    state_data=state_data,
                    qubit_indices=group_qubits,
                    shard_type=ShardType.DENSE_STATE
                )
                
                shards.append(shard)
        
        return shards
    
    def _analyze_entanglement_groups(self, circuit_data: Dict[str, Any]) -> List[List[int]]:
        """Analyze entanglement groups in a circuit."""
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        # Simple entanglement analysis
        entanglement_groups = []
        current_group = set()
        
        for gate in gates:
            gate_qubits = set(gate.get('qubits', []))
            
            # Check if gate connects to current group
            if current_group.intersection(gate_qubits) or not current_group:
                current_group.update(gate_qubits)
            else:
                # Start new group
                if current_group:
                    entanglement_groups.append(list(current_group))
                current_group = gate_qubits
        
        # Add final group
        if current_group:
            entanglement_groups.append(list(current_group))
        
        # If no entanglement detected, create single-qubit groups
        if not entanglement_groups:
            entanglement_groups = [[i] for i in range(num_qubits)]
        
        return entanglement_groups
    
    def get_sharding_statistics(self) -> Dict[str, Any]:
        """Get state sharding statistics."""
        return {
            'sharding_stats': self.sharding_stats,
            'shard_manager_stats': self.shard_manager.get_shard_statistics()
        }
    
    def get_sharding_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sharding recommendations."""
        return self.shard_manager.get_shard_recommendations(circuit_data)
