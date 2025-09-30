"""
Cost Analyzer - Intelligent Resource Cost Optimization
=====================================================

The Cost Analyzer is the financial brain of Coratrix 4.0's Quantum OS.
It provides intelligent cost analysis and optimization for quantum circuit
execution across different backends and resources.

This enables:
- Real-time cost analysis for execution decisions
- Resource optimization based on cost-benefit analysis
- Budget-aware execution planning
- Cost prediction and optimization recommendations
- Multi-backend cost comparison and optimization

This makes Coratrix 4.0 not just powerful, but economically intelligent.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class CostType(Enum):
    """Types of costs in quantum execution."""
    COMPUTATION = "computation"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    LATENCY = "latency"
    RELIABILITY = "reliability"

class OptimizationStrategy(Enum):
    """Resource optimization strategies."""
    COST_MINIMIZATION = "cost_minimization"
    PERFORMANCE_MAXIMIZATION = "performance_maximization"
    BALANCED = "balanced"
    BUDGET_CONSTRAINED = "budget_constrained"
    TIME_CONSTRAINED = "time_constrained"

@dataclass
class CostProfile:
    """Cost profile for a quantum execution."""
    computation_cost: float
    memory_cost: float
    network_cost: float
    storage_cost: float
    latency_cost: float
    reliability_cost: float
    total_cost: float
    cost_breakdown: Dict[str, float]

@dataclass
class ResourceRequirement:
    """Resource requirements for quantum execution."""
    cpu_cores: int
    memory_gb: float
    gpu_memory_gb: float
    network_bandwidth_mbps: float
    storage_gb: float
    execution_time_seconds: float
    reliability_requirement: float

@dataclass
class BackendCostModel:
    """Cost model for a backend."""
    backend_id: str
    base_computation_cost: float  # per operation
    memory_cost_per_gb: float
    network_cost_per_mb: float
    storage_cost_per_gb: float
    latency_penalty: float
    reliability_bonus: float
    scaling_factors: Dict[str, float]

class CostAnalyzer:
    """
    Intelligent Cost Analyzer for Quantum OS.
    
    This analyzer provides comprehensive cost analysis and optimization
    for quantum circuit execution, enabling economically intelligent
    resource allocation and execution planning.
    """
    
    def __init__(self, backends: Dict[str, Any] = None, telemetry_collector: Any = None):
        """Initialize the cost analyzer."""
        self.cost_models: Dict[str, BackendCostModel] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.cost_optimization_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cost analysis algorithms
        self.optimization_strategies = {
            OptimizationStrategy.COST_MINIMIZATION: self._minimize_cost,
            OptimizationStrategy.PERFORMANCE_MAXIMIZATION: self._maximize_performance,
            OptimizationStrategy.BALANCED: self._balanced_optimization,
            OptimizationStrategy.BUDGET_CONSTRAINED: self._budget_constrained_optimization,
            OptimizationStrategy.TIME_CONSTRAINED: self._time_constrained_optimization
        }
        
        logger.info("💰 Cost Analyzer initialized - Economic intelligence active")
    
    def register_backend_cost_model(self, backend_id: str, cost_model: BackendCostModel):
        """Register a cost model for a backend."""
        self.cost_models[backend_id] = cost_model
        logger.info(f"💰 Cost model registered for backend: {backend_id}")
    
    def analyze_execution_cost(self, circuit_data: Dict[str, Any], 
                             backend_id: str, 
                             resource_requirements: ResourceRequirement) -> CostProfile:
        """
        Analyze the cost of executing a circuit on a specific backend.
        
        This is the GOD-TIER cost analysis method that provides comprehensive
        cost breakdown for quantum circuit execution.
        """
        if backend_id not in self.cost_models:
            raise ValueError(f"Backend {backend_id} not registered")
        
        cost_model = self.cost_models[backend_id]
        
        # Calculate individual cost components
        computation_cost = self._calculate_computation_cost(circuit_data, cost_model, resource_requirements)
        memory_cost = self._calculate_memory_cost(resource_requirements, cost_model)
        network_cost = self._calculate_network_cost(resource_requirements, cost_model)
        storage_cost = self._calculate_storage_cost(resource_requirements, cost_model)
        latency_cost = self._calculate_latency_cost(resource_requirements, cost_model)
        reliability_cost = self._calculate_reliability_cost(resource_requirements, cost_model)
        
        # Calculate total cost
        total_cost = (
            computation_cost + memory_cost + network_cost + 
            storage_cost + latency_cost + reliability_cost
        )
        
        # Create cost breakdown
        cost_breakdown = {
            'computation': computation_cost,
            'memory': memory_cost,
            'network': network_cost,
            'storage': storage_cost,
            'latency': latency_cost,
            'reliability': reliability_cost
        }
        
        cost_profile = CostProfile(
            computation_cost=computation_cost,
            memory_cost=memory_cost,
            network_cost=network_cost,
            storage_cost=storage_cost,
            latency_cost=latency_cost,
            reliability_cost=reliability_cost,
            total_cost=total_cost,
            cost_breakdown=cost_breakdown
        )
        
        # Store analysis for learning
        self._store_cost_analysis(circuit_data, backend_id, cost_profile, resource_requirements)
        
        logger.info(f"💰 Cost analysis completed: {total_cost:.4f} total cost for {backend_id}")
        return cost_profile
    
    def _calculate_computation_cost(self, circuit_data: Dict[str, Any], 
                                  cost_model: BackendCostModel, 
                                  resource_requirements: ResourceRequirement) -> float:
        """Calculate computation cost for circuit execution."""
        num_qubits = circuit_data.get('num_qubits', 0)
        num_gates = len(circuit_data.get('gates', []))
        
        # Base computation cost
        base_cost = cost_model.base_computation_cost * num_gates
        
        # Scaling based on qubit count
        qubit_scaling = cost_model.scaling_factors.get('qubits', 1.0)
        qubit_cost = base_cost * (qubit_scaling ** num_qubits)
        
        # Scaling based on execution time
        time_scaling = cost_model.scaling_factors.get('time', 1.0)
        time_cost = qubit_cost * (time_scaling ** resource_requirements.execution_time_seconds)
        
        return time_cost
    
    def _calculate_memory_cost(self, resource_requirements: ResourceRequirement, 
                             cost_model: BackendCostModel) -> float:
        """Calculate memory cost for execution."""
        memory_gb = resource_requirements.memory_gb
        return cost_model.memory_cost_per_gb * memory_gb
    
    def _calculate_network_cost(self, resource_requirements: ResourceRequirement, 
                              cost_model: BackendCostModel) -> float:
        """Calculate network cost for execution."""
        bandwidth_mbps = resource_requirements.network_bandwidth_mbps
        # Convert to MB and apply cost
        bandwidth_mb = bandwidth_mbps / 8  # Convert Mbps to MB/s
        return cost_model.network_cost_per_mb * bandwidth_mb
    
    def _calculate_storage_cost(self, resource_requirements: ResourceRequirement, 
                              cost_model: BackendCostModel) -> float:
        """Calculate storage cost for execution."""
        storage_gb = resource_requirements.storage_gb
        return cost_model.storage_cost_per_gb * storage_gb
    
    def _calculate_latency_cost(self, resource_requirements: ResourceRequirement, 
                              cost_model: BackendCostModel) -> float:
        """Calculate latency cost for execution."""
        execution_time = resource_requirements.execution_time_seconds
        return cost_model.latency_penalty * execution_time
    
    def _calculate_reliability_cost(self, resource_requirements: ResourceRequirement, 
                                  cost_model: BackendCostModel) -> float:
        """Calculate reliability cost for execution."""
        reliability_req = resource_requirements.reliability_requirement
        # Higher reliability requirements increase cost
        reliability_multiplier = 1.0 + (1.0 - reliability_req) * 2.0
        return cost_model.reliability_bonus * reliability_multiplier
    
    def _store_cost_analysis(self, circuit_data: Dict[str, Any], backend_id: str, 
                          cost_profile: CostProfile, resource_requirements: ResourceRequirement):
        """Store cost analysis for learning and optimization."""
        analysis = {
            'timestamp': time.time(),
            'circuit_data': circuit_data,
            'backend_id': backend_id,
            'cost_profile': cost_profile.__dict__,
            'resource_requirements': resource_requirements.__dict__
        }
        
        self.execution_history.append(analysis)
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]
    
    def optimize_execution_cost(self, circuit_data: Dict[str, Any], 
                              available_backends: List[str],
                              optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                              constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize execution cost across available backends.
        
        This is the GOD-TIER optimization method that finds the most
        cost-effective execution strategy.
        """
        if not available_backends:
            raise ValueError("No backends available for optimization")
        
        # Analyze costs for all backends
        backend_costs = {}
        for backend_id in available_backends:
            if backend_id in self.cost_models:
                # Estimate resource requirements
                resource_requirements = self._estimate_resource_requirements(circuit_data, backend_id)
                
                # Analyze cost
                cost_profile = self.analyze_execution_cost(circuit_data, backend_id, resource_requirements)
                backend_costs[backend_id] = {
                    'cost_profile': cost_profile,
                    'resource_requirements': resource_requirements
                }
        
        # Apply optimization strategy
        optimization_algorithm = self.optimization_strategies.get(optimization_strategy)
        if optimization_algorithm:
            optimal_backend = optimization_algorithm(backend_costs, constraints or {})
        else:
            optimal_backend = self._balanced_optimization(backend_costs, constraints or {})
        
        # Generate optimization report
        optimization_report = {
            'optimal_backend': optimal_backend,
            'backend_costs': {k: v['cost_profile'].__dict__ for k, v in backend_costs.items()},
            'optimization_strategy': optimization_strategy.value,
            'constraints': constraints,
            'recommendations': self._generate_cost_recommendations(backend_costs, optimal_backend)
        }
        
        logger.info(f"💰 Cost optimization completed: {optimal_backend} selected")
        return optimization_report
    
    def _estimate_resource_requirements(self, circuit_data: Dict[str, Any], 
                                      backend_id: str) -> ResourceRequirement:
        """Estimate resource requirements for circuit execution."""
        num_qubits = circuit_data.get('num_qubits', 0)
        num_gates = len(circuit_data.get('gates', []))
        
        # Estimate based on circuit characteristics
        cpu_cores = min(4, max(1, num_qubits // 4))
        memory_gb = (2 ** num_qubits) * 16 / (1024 ** 3)  # Theoretical memory
        gpu_memory_gb = memory_gb * 0.5 if num_qubits > 10 else 0
        network_bandwidth_mbps = 100  # Default network requirement
        storage_gb = memory_gb * 0.1  # Storage for intermediate results
        execution_time_seconds = num_gates * 0.001 * (2 ** min(num_qubits, 15))
        reliability_requirement = 0.9  # Default reliability requirement
        
        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            network_bandwidth_mbps=network_bandwidth_mbps,
            storage_gb=storage_gb,
            execution_time_seconds=execution_time_seconds,
            reliability_requirement=reliability_requirement
        )
    
    def _minimize_cost(self, backend_costs: Dict[str, Dict[str, Any]], 
                      constraints: Dict[str, Any]) -> str:
        """Minimize total execution cost."""
        min_cost = float('inf')
        optimal_backend = None
        
        for backend_id, cost_data in backend_costs.items():
            total_cost = cost_data['cost_profile'].total_cost
            
            # Check constraints
            if self._satisfies_constraints(cost_data, constraints):
                if total_cost < min_cost:
                    min_cost = total_cost
                    optimal_backend = backend_id
        
        return optimal_backend or list(backend_costs.keys())[0]
    
    def _maximize_performance(self, backend_costs: Dict[str, Dict[str, Any]], 
                            constraints: Dict[str, Any]) -> str:
        """Maximize execution performance (minimize execution time)."""
        min_time = float('inf')
        optimal_backend = None
        
        for backend_id, cost_data in backend_costs.items():
            execution_time = cost_data['resource_requirements'].execution_time_seconds
            
            # Check constraints
            if self._satisfies_constraints(cost_data, constraints):
                if execution_time < min_time:
                    min_time = execution_time
                    optimal_backend = backend_id
        
        return optimal_backend or list(backend_costs.keys())[0]
    
    def _balanced_optimization(self, backend_costs: Dict[str, Dict[str, Any]], 
                             constraints: Dict[str, Any]) -> str:
        """Balanced optimization considering both cost and performance."""
        best_score = -1.0
        optimal_backend = None
        
        for backend_id, cost_data in backend_costs.items():
            if not self._satisfies_constraints(cost_data, constraints):
                continue
            
            cost_profile = cost_data['cost_profile']
            resource_requirements = cost_data['resource_requirements']
            
            # Calculate balanced score
            cost_score = 1.0 / (1.0 + cost_profile.total_cost)
            performance_score = 1.0 / (1.0 + resource_requirements.execution_time_seconds)
            
            # Weighted combination
            balanced_score = 0.6 * cost_score + 0.4 * performance_score
            
            if balanced_score > best_score:
                best_score = balanced_score
                optimal_backend = backend_id
        
        return optimal_backend or list(backend_costs.keys())[0]
    
    def _budget_constrained_optimization(self, backend_costs: Dict[str, Dict[str, Any]], 
                                       constraints: Dict[str, Any]) -> str:
        """Optimize within budget constraints."""
        budget = constraints.get('budget', float('inf'))
        
        # Filter backends within budget
        affordable_backends = {}
        for backend_id, cost_data in backend_costs.items():
            if cost_data['cost_profile'].total_cost <= budget:
                affordable_backends[backend_id] = cost_data
        
        if not affordable_backends:
            # If no backends within budget, return cheapest
            return self._minimize_cost(backend_costs, constraints)
        
        # Optimize among affordable backends
        return self._balanced_optimization(affordable_backends, constraints)
    
    def _time_constrained_optimization(self, backend_costs: Dict[str, Dict[str, Any]], 
                                     constraints: Dict[str, Any]) -> str:
        """Optimize within time constraints."""
        max_time = constraints.get('max_execution_time', float('inf'))
        
        # Filter backends within time constraint
        timely_backends = {}
        for backend_id, cost_data in backend_costs.items():
            if cost_data['resource_requirements'].execution_time_seconds <= max_time:
                timely_backends[backend_id] = cost_data
        
        if not timely_backends:
            # If no backends within time constraint, return fastest
            return self._maximize_performance(backend_costs, constraints)
        
        # Optimize among timely backends
        return self._balanced_optimization(timely_backends, constraints)
    
    def _satisfies_constraints(self, cost_data: Dict[str, Any], 
                             constraints: Dict[str, Any]) -> bool:
        """Check if cost data satisfies constraints."""
        cost_profile = cost_data['cost_profile']
        resource_requirements = cost_data['resource_requirements']
        
        # Check budget constraint
        if 'budget' in constraints:
            if cost_profile.total_cost > constraints['budget']:
                return False
        
        # Check time constraint
        if 'max_execution_time' in constraints:
            if resource_requirements.execution_time_seconds > constraints['max_execution_time']:
                return False
        
        # Check memory constraint
        if 'max_memory' in constraints:
            if resource_requirements.memory_gb > constraints['max_memory']:
                return False
        
        # Check reliability constraint
        if 'min_reliability' in constraints:
            if resource_requirements.reliability_requirement < constraints['min_reliability']:
                return False
        
        return True
    
    def _generate_cost_recommendations(self, backend_costs: Dict[str, Dict[str, Any]], 
                                     optimal_backend: str) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if optimal_backend not in backend_costs:
            return recommendations
        
        optimal_cost = backend_costs[optimal_backend]['cost_profile']
        
        # Analyze cost breakdown
        cost_breakdown = optimal_cost.cost_breakdown
        max_cost_component = max(cost_breakdown.items(), key=lambda x: x[1])
        
        if max_cost_component[0] == 'computation':
            recommendations.append({
                'type': 'computation_optimization',
                'message': 'High computation cost detected',
                'recommendation': 'Consider circuit optimization or backend with better computation efficiency'
            })
        
        if max_cost_component[0] == 'memory':
            recommendations.append({
                'type': 'memory_optimization',
                'message': 'High memory cost detected',
                'recommendation': 'Consider sparse operations or memory-efficient backends'
            })
        
        if max_cost_component[0] == 'network':
            recommendations.append({
                'type': 'network_optimization',
                'message': 'High network cost detected',
                'recommendation': 'Consider local execution or network optimization'
            })
        
        # Compare with other backends
        for backend_id, cost_data in backend_costs.items():
            if backend_id != optimal_backend:
                cost_diff = cost_data['cost_profile'].total_cost - optimal_cost.total_cost
                if cost_diff > 0:
                    recommendations.append({
                        'type': 'backend_comparison',
                        'message': f'{backend_id} is {cost_diff:.2f} more expensive',
                        'recommendation': f'Stick with {optimal_backend} for cost efficiency'
                    })
        
        return recommendations
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """Get cost analysis statistics."""
        if not self.execution_history:
            return {'message': 'No cost analysis history available'}
        
        # Calculate statistics
        total_executions = len(self.execution_history)
        total_cost = sum(h['cost_profile']['total_cost'] for h in self.execution_history)
        avg_cost = total_cost / total_executions
        
        # Backend usage statistics
        backend_usage = defaultdict(int)
        for h in self.execution_history:
            backend_usage[h['backend_id']] += 1
        
        # Cost trend analysis
        recent_costs = [h['cost_profile']['total_cost'] for h in self.execution_history[-10:]]
        cost_trend = 'stable'
        if len(recent_costs) >= 5:
            if np.mean(recent_costs[-5:]) > np.mean(recent_costs[:5]) * 1.1:
                cost_trend = 'increasing'
            elif np.mean(recent_costs[-5:]) < np.mean(recent_costs[:5]) * 0.9:
                cost_trend = 'decreasing'
        
        return {
            'total_executions': total_executions,
            'total_cost': total_cost,
            'average_cost': avg_cost,
            'backend_usage': dict(backend_usage),
            'cost_trend': cost_trend,
            'recent_costs': recent_costs[-10:]
        }

class ResourceOptimizer:
    """
    Resource Optimizer for Quantum OS.
    
    This optimizer provides intelligent resource allocation and optimization
    recommendations for quantum circuit execution.
    """
    
    def __init__(self, cost_analyzer: CostAnalyzer = None):
        """Initialize the resource optimizer."""
        self.cost_analyzer = cost_analyzer or CostAnalyzer()
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("🔧 Resource Optimizer initialized - Intelligent optimization active")
    
    def optimize_resource_allocation(self, circuit_data: Dict[str, Any], 
                                   available_backends: List[str],
                                   optimization_goals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize resource allocation for quantum circuit execution.
        
        This is the GOD-TIER optimization method that provides intelligent
        resource allocation recommendations.
        """
        optimization_goals = optimization_goals or {
            'minimize_cost': True,
            'maximize_performance': False,
            'ensure_reliability': True
        }
        
        # Analyze costs for all backends
        backend_analysis = {}
        for backend_id in available_backends:
            if backend_id in self.cost_analyzer.cost_models:
                resource_requirements = self.cost_analyzer._estimate_resource_requirements(circuit_data, backend_id)
                cost_profile = self.cost_analyzer.analyze_execution_cost(circuit_data, backend_id, resource_requirements)
                
                backend_analysis[backend_id] = {
                    'cost_profile': cost_profile,
                    'resource_requirements': resource_requirements,
                    'optimization_score': self._calculate_optimization_score(cost_profile, resource_requirements, optimization_goals)
                }
        
        # Find optimal backend
        optimal_backend = max(backend_analysis.keys(), 
                            key=lambda k: backend_analysis[k]['optimization_score'])
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            circuit_data, backend_analysis, optimal_backend
        )
        
        optimization_result = {
            'optimal_backend': optimal_backend,
            'optimization_score': backend_analysis[optimal_backend]['optimization_score'],
            'backend_analysis': {k: {
                'cost_profile': v['cost_profile'].__dict__,
                'resource_requirements': v['resource_requirements'].__dict__,
                'optimization_score': v['optimization_score']
            } for k, v in backend_analysis.items()},
            'recommendations': recommendations,
            'optimization_goals': optimization_goals
        }
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'circuit_data': circuit_data,
            'optimization_result': optimization_result
        })
        
        logger.info(f"🔧 Resource optimization completed: {optimal_backend} selected")
        return optimization_result
    
    def _calculate_optimization_score(self, cost_profile: CostProfile, 
                                    resource_requirements: ResourceRequirement,
                                    optimization_goals: Dict[str, Any]) -> float:
        """Calculate optimization score for a backend."""
        score = 0.5  # Base score
        
        # Cost optimization
        if optimization_goals.get('minimize_cost', False):
            cost_score = 1.0 / (1.0 + cost_profile.total_cost)
            score += 0.3 * cost_score
        
        # Performance optimization
        if optimization_goals.get('maximize_performance', False):
            performance_score = 1.0 / (1.0 + resource_requirements.execution_time_seconds)
            score += 0.3 * performance_score
        
        # Reliability optimization
        if optimization_goals.get('ensure_reliability', False):
            reliability_score = resource_requirements.reliability_requirement
            score += 0.2 * reliability_score
        
        # Memory efficiency
        memory_efficiency = 1.0 / (1.0 + resource_requirements.memory_gb)
        score += 0.2 * memory_efficiency
        
        return min(score, 1.0)
    
    def _generate_optimization_recommendations(self, circuit_data: Dict[str, Any], 
                                             backend_analysis: Dict[str, Any],
                                             optimal_backend: str) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Circuit-specific recommendations
        num_qubits = circuit_data.get('num_qubits', 0)
        if num_qubits > 15:
            recommendations.append({
                'type': 'large_circuit_optimization',
                'message': f'Large circuit detected ({num_qubits} qubits)',
                'recommendation': 'Consider distributed execution or circuit decomposition',
                'priority': 'high'
            })
        
        # Resource optimization recommendations
        optimal_analysis = backend_analysis[optimal_backend]
        resource_requirements = optimal_analysis['resource_requirements']
        
        if resource_requirements.memory_gb > 1.0:
            recommendations.append({
                'type': 'memory_optimization',
                'message': f'High memory requirement ({resource_requirements.memory_gb:.2f} GB)',
                'recommendation': 'Consider sparse operations or memory-efficient backends',
                'priority': 'medium'
            })
        
        if resource_requirements.execution_time_seconds > 10.0:
            recommendations.append({
                'type': 'performance_optimization',
                'message': f'Long execution time ({resource_requirements.execution_time_seconds:.2f}s)',
                'recommendation': 'Consider parallel execution or faster backends',
                'priority': 'medium'
            })
        
        # Cost optimization recommendations
        cost_profile = optimal_analysis['cost_profile']
        if cost_profile.total_cost > 1.0:
            recommendations.append({
                'type': 'cost_optimization',
                'message': f'High execution cost ({cost_profile.total_cost:.4f})',
                'recommendation': 'Consider cost optimization or alternative backends',
                'priority': 'low'
            })
        
        return recommendations
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        # Calculate statistics
        total_optimizations = len(self.optimization_history)
        
        # Backend selection statistics
        backend_selections = defaultdict(int)
        for h in self.optimization_history:
            backend_selections[h['optimization_result']['optimal_backend']] += 1
        
        # Average optimization scores
        avg_scores = []
        for h in self.optimization_history:
            avg_scores.append(h['optimization_result']['optimization_score'])
        
        return {
            'total_optimizations': total_optimizations,
            'backend_selections': dict(backend_selections),
            'average_optimization_score': np.mean(avg_scores) if avg_scores else 0,
            'recent_optimizations': self.optimization_history[-10:]
        }
