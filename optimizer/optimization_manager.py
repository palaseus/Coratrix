"""
Optimization Manager - Centralized Optimization Management
========================================================

The Optimization Manager provides centralized management
for all optimization operations in the system.
"""

import time
import logging
import numpy as np
import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import hashlib
import networkx as nx

logger = logging.getLogger(__name__)

class OptimizationPriority(Enum):
    """Optimization priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OptimizationStatus(Enum):
    """Optimization status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OptimizationTask:
    """An optimization task."""
    task_id: str
    circuit_data: Dict[str, Any]
    optimization_type: str
    priority: OptimizationPriority
    status: OptimizationStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class OptimizationConfig:
    """Configuration for optimization operations."""
    max_concurrent_optimizations: int = 5
    optimization_timeout: float = 300.0
    retry_attempts: int = 3
    enable_parallel_optimization: bool = True
    optimization_queue_size: int = 1000

class OptimizationManager:
    """
    Optimization Manager for Centralized Optimization Management.
    
    This provides centralized management for all optimization operations
    in the system.
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize the optimization manager."""
        self.config = config
        self.optimization_queue: deque = deque()
        self.active_optimizations: Dict[str, OptimizationTask] = {}
        self.completed_optimizations: deque = deque(maxlen=1000)
        self.optimization_workers: List[threading.Thread] = []
        self.optimization_lock = threading.Lock()
        self.optimization_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'average_optimization_time': 0.0,
            'queue_size': 0
        }
        
        # Start optimization workers
        self._start_optimization_workers()
        
        logger.info("🎨 Optimization Manager initialized - Centralized optimization management active")
    
    def _start_optimization_workers(self):
        """Start optimization worker threads."""
        for i in range(self.config.max_concurrent_optimizations):
            worker = threading.Thread(target=self._optimization_worker, daemon=True)
            worker.start()
            self.optimization_workers.append(worker)
    
    def _optimization_worker(self):
        """Optimization worker thread."""
        while True:
            try:
                # Get next optimization task
                task = None
                with self.optimization_lock:
                    if self.optimization_queue:
                        task = self.optimization_queue.popleft()
                        task.status = OptimizationStatus.RUNNING
                        task.started_at = time.time()
                        self.active_optimizations[task.task_id] = task
                
                if task:
                    # Execute optimization
                    asyncio.run(self._execute_optimization(task))
                else:
                    # No tasks available, sleep briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Optimization worker error: {e}")
                time.sleep(1.0)
    
    async def _execute_optimization(self, task: OptimizationTask):
        """Execute an optimization task."""
        try:
            logger.info(f"🎨 Executing optimization task: {task.task_id}")
            
            # Simulate optimization execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Create optimization result
            result = {
                'task_id': task.task_id,
                'optimization_type': task.optimization_type,
                'success': True,
                'performance_improvement': np.random.random() * 0.5,
                'optimization_time': time.time() - task.started_at,
                'recommendations': [f"Optimization completed for {task.optimization_type}"]
            }
            
            # Update task status
            task.status = OptimizationStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            
            # Move to completed optimizations
            with self.optimization_lock:
                self.completed_optimizations.append(task)
                if task.task_id in self.active_optimizations:
                    del self.active_optimizations[task.task_id]
            
            # Update statistics
            self._update_optimization_stats(task)
            
            logger.info(f"✅ Optimization task completed: {task.task_id}")
            
        except Exception as e:
            logger.error(f"❌ Optimization task failed: {task.task_id} - {e}")
            task.status = OptimizationStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            with self.optimization_lock:
                if task.task_id in self.active_optimizations:
                    del self.active_optimizations[task.task_id]
    
    async def submit_optimization_task(self, circuit_data: Dict[str, Any], 
                                     optimization_type: str,
                                     priority: OptimizationPriority = OptimizationPriority.MEDIUM) -> str:
        """Submit an optimization task."""
        task_id = f"opt_{int(time.time() * 1000)}_{hashlib.md5(str(circuit_data).encode()).hexdigest()[:8]}"
        
        task = OptimizationTask(
            task_id=task_id,
            circuit_data=circuit_data,
            optimization_type=optimization_type,
            priority=priority,
            status=OptimizationStatus.PENDING,
            created_at=time.time()
        )
        
        with self.optimization_lock:
            self.optimization_queue.append(task)
            self.optimization_stats['total_tasks'] += 1
            self.optimization_stats['queue_size'] = len(self.optimization_queue)
        
        logger.info(f"🎨 Optimization task submitted: {task_id} ({optimization_type}, {priority.value})")
        return task_id
    
    async def get_optimization_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get optimization task status."""
        with self.optimization_lock:
            # Check active optimizations
            if task_id in self.active_optimizations:
                task = self.active_optimizations[task_id]
                return {
                    'task_id': task_id,
                    'status': task.status.value,
                    'optimization_type': task.optimization_type,
                    'priority': task.priority.value,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'result': task.result,
                    'error': task.error
                }
            
            # Check completed optimizations
            for task in self.completed_optimizations:
                if task.task_id == task_id:
                    return {
                        'task_id': task_id,
                        'status': task.status.value,
                        'optimization_type': task.optimization_type,
                        'priority': task.priority.value,
                        'created_at': task.created_at,
                        'started_at': task.started_at,
                        'completed_at': task.completed_at,
                        'result': task.result,
                        'error': task.error
                    }
        
        return None
    
    async def cancel_optimization_task(self, task_id: str) -> bool:
        """Cancel an optimization task."""
        with self.optimization_lock:
            # Check if task is in queue
            for i, task in enumerate(self.optimization_queue):
                if task.task_id == task_id:
                    self.optimization_queue.remove(task)
                    task.status = OptimizationStatus.CANCELLED
                    task.completed_at = time.time()
                    self.completed_optimizations.append(task)
                    self.optimization_stats['cancelled_tasks'] += 1
                    logger.info(f"✅ Optimization task cancelled: {task_id}")
                    return True
            
            # Check if task is active
            if task_id in self.active_optimizations:
                task = self.active_optimizations[task_id]
                task.status = OptimizationStatus.CANCELLED
                task.completed_at = time.time()
                del self.active_optimizations[task_id]
                self.completed_optimizations.append(task)
                self.optimization_stats['cancelled_tasks'] += 1
                logger.info(f"✅ Optimization task cancelled: {task_id}")
                return True
        
        return False
    
    async def get_optimization_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get optimization results for a task."""
        with self.optimization_lock:
            # Check completed optimizations
            for task in self.completed_optimizations:
                if task.task_id == task_id and task.status == OptimizationStatus.COMPLETED:
                    return task.result
        
        return None
    
    async def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        with self.optimization_lock:
            return {
                'optimization_stats': self.optimization_stats,
                'active_optimizations': len(self.active_optimizations),
                'queue_size': len(self.optimization_queue),
                'completed_optimizations': len(self.completed_optimizations),
                'worker_threads': len(self.optimization_workers)
            }
    
    async def get_optimization_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimization recommendations."""
        recommendations = []
        
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        # Circuit complexity recommendations
        if len(gates) > 100:
            recommendations.append({
                'type': 'complexity',
                'message': f'Large circuit ({len(gates)} gates) detected',
                'recommendation': 'Consider using high-priority optimization',
                'priority': 'high'
            })
        
        # Qubit count recommendations
        if num_qubits > 20:
            recommendations.append({
                'type': 'scalability',
                'message': f'Large qubit count ({num_qubits}) detected',
                'recommendation': 'Consider using parallel optimization',
                'priority': 'medium'
            })
        
        # Performance recommendations
        if self.optimization_stats['average_optimization_time'] > 10.0:
            recommendations.append({
                'type': 'performance',
                'message': f'High average optimization time ({self.optimization_stats["average_optimization_time"]:.2f}s)',
                'recommendation': 'Consider using faster optimization algorithms',
                'priority': 'low'
            })
        
        return recommendations
    
    def _update_optimization_stats(self, task: OptimizationTask):
        """Update optimization statistics."""
        if task.status == OptimizationStatus.COMPLETED:
            self.optimization_stats['completed_tasks'] += 1
        elif task.status == OptimizationStatus.FAILED:
            self.optimization_stats['failed_tasks'] += 1
        elif task.status == OptimizationStatus.CANCELLED:
            self.optimization_stats['cancelled_tasks'] += 1
        
        # Update average optimization time
        if task.completed_at and task.started_at:
            optimization_time = task.completed_at - task.started_at
            total = self.optimization_stats['completed_tasks']
            current_avg = self.optimization_stats['average_optimization_time']
            self.optimization_stats['average_optimization_time'] = (
                (current_avg * (total - 1) + optimization_time) / total
            )
    
    async def cleanup_old_optimizations(self, max_age_hours: float = 24.0):
        """Clean up old optimization results."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self.optimization_lock:
            # Remove old completed optimizations
            old_optimizations = []
            for task in self.completed_optimizations:
                if current_time - task.completed_at > max_age_seconds:
                    old_optimizations.append(task)
            
            for task in old_optimizations:
                self.completed_optimizations.remove(task)
            
            logger.info(f"🧹 Cleaned up {len(old_optimizations)} old optimization results")
    
    async def shutdown(self):
        """Shutdown the optimization manager."""
        logger.info("🛑 Shutting down optimization manager...")
        
        # Wait for active optimizations to complete
        while self.active_optimizations:
            await asyncio.sleep(0.1)
        
        # Clear queues
        with self.optimization_lock:
            self.optimization_queue.clear()
            self.completed_optimizations.clear()
        
        logger.info("✅ Optimization manager shutdown complete")
