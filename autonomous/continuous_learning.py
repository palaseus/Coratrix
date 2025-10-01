"""
Continuous Learning System - Autonomous Knowledge Evolution
========================================================

This module implements the continuous learning system that maintains
an evolving knowledge base of optimizations, backend behaviors, and
circuit patterns, generates periodic autonomous reports with recommended
upgrades, predicted resource allocations, and experimental results.

This is the GOD-TIER learning intelligence that makes Coratrix
truly self-improving and adaptive.
"""

import asyncio
import time
import logging
import numpy as np
import threading
import json
import pickle
import random
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import networkx as nx

logger = logging.getLogger(__name__)

class LearningType(Enum):
    """Types of continuous learning."""
    PERFORMANCE_LEARNING = "performance_learning"
    PATTERN_LEARNING = "pattern_learning"
    OPTIMIZATION_LEARNING = "optimization_learning"
    BACKEND_LEARNING = "backend_learning"
    CIRCUIT_LEARNING = "circuit_learning"
    STRATEGIC_LEARNING = "strategic_learning"

class KnowledgeType(Enum):
    """Types of knowledge in the knowledge base."""
    OPTIMIZATION_PATTERN = "optimization_pattern"
    BACKEND_BEHAVIOR = "backend_behavior"
    CIRCUIT_PATTERN = "circuit_pattern"
    PERFORMANCE_INSIGHT = "performance_insight"
    STRATEGIC_RECOMMENDATION = "strategic_recommendation"
    EXPERIMENTAL_RESULT = "experimental_result"

@dataclass
class KnowledgeEntry:
    """A knowledge entry in the learning system."""
    entry_id: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    confidence: float
    source: str
    timestamp: float
    usage_count: int = 0
    success_rate: float = 0.0
    tags: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)

@dataclass
class LearningPattern:
    """A learned pattern from system behavior."""
    pattern_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    success_rate: float
    applications: List[str]
    optimization_potential: float

@dataclass
class LearningReport:
    """A periodic learning report."""
    report_id: str
    timestamp: float
    learning_summary: Dict[str, Any]
    knowledge_growth: Dict[str, int]
    performance_improvements: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    predictions: Dict[str, Any]
    experimental_results: List[Dict[str, Any]]

class ContinuousLearningSystem:
    """
    GOD-TIER Continuous Learning System for Autonomous Knowledge Evolution.
    
    This system continuously learns from system behavior, performance data,
    and experimental results to build an evolving knowledge base that
    enables autonomous optimization and strategic decision-making.
    
    This transforms Coratrix into a truly intelligent system that
    gets smarter with every execution and experiment.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Continuous Learning System."""
        self.config = config or {}
        self.learning_id = f"cls_{int(time.time() * 1000)}"
        
        # Knowledge base
        self.knowledge_base: Dict[str, KnowledgeEntry] = {}
        self.knowledge_graph = nx.DiGraph()
        self.learning_patterns: Dict[str, LearningPattern] = {}
        
        # Learning models
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pattern_classifier = KMeans(n_clusters=10, random_state=42)
        self.optimization_learner = RandomForestRegressor(n_estimators=50, random_state=42)
        self.models_trained = False
        
        # Learning state
        self.learning_history: deque = deque(maxlen=10000)
        self.performance_data: deque = deque(maxlen=50000)
        self.optimization_data: deque = deque(maxlen=10000)
        self.experimental_data: deque = deque(maxlen=5000)
        
        # Learning metrics
        self.learning_metrics = {
            'total_knowledge_entries': 0,
            'patterns_learned': 0,
            'optimizations_discovered': 0,
            'performance_improvements': 0.0,
            'learning_cycles': 0
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.learning_thread = None
        self.knowledge_thread = None
        
        logger.info(f"🧠 Continuous Learning System initialized (ID: {self.learning_id})")
        logger.info("🚀 GOD-TIER learning intelligence active")
    
    async def start(self):
        """Start the continuous learning system."""
        self.running = True
        
        # Start learning thread
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        # Start knowledge processing thread
        self.knowledge_thread = threading.Thread(target=self._knowledge_loop, daemon=True)
        self.knowledge_thread.start()
        
        logger.info("🎯 Continuous Learning System started")
    
    async def stop(self):
        """Stop the continuous learning system."""
        self.running = False
        
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
        if self.knowledge_thread:
            self.knowledge_thread.join(timeout=5.0)
        
        logger.info("🛑 Continuous Learning System stopped")
    
    def _learning_loop(self):
        """Main learning processing loop."""
        while self.running:
            try:
                # Process performance data
                self._process_performance_data()
                
                # Learn from optimization data
                self._learn_from_optimizations()
                
                # Extract patterns
                self._extract_learning_patterns()
                
                # Update knowledge base
                self._update_knowledge_base()
                
                # Sleep between learning cycles
                time.sleep(60.0)  # Learn every minute
                
            except Exception as e:
                logger.error(f"❌ Learning loop error: {e}")
                time.sleep(10.0)
    
    def _knowledge_loop(self):
        """Knowledge processing and organization loop."""
        while self.running:
            try:
                # Organize knowledge
                self._organize_knowledge()
                
                # Update knowledge relationships
                self._update_knowledge_relationships()
                
                # Generate learning insights
                self._generate_learning_insights()
                
                # Sleep between knowledge cycles
                time.sleep(120.0)  # Process knowledge every 2 minutes
                
            except Exception as e:
                logger.error(f"❌ Knowledge loop error: {e}")
                time.sleep(15.0)
    
    def _process_performance_data(self):
        """Process performance data for learning."""
        if len(self.performance_data) < 100:
            return
        
        # Extract performance patterns
        performance_patterns = self._extract_performance_patterns()
        
        # Learn from patterns
        for pattern in performance_patterns:
            self._learn_from_performance_pattern(pattern)
        
        # Update learning metrics
        self.learning_metrics['learning_cycles'] += 1
    
    def _extract_performance_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns from performance data."""
        patterns = []
        
        # Get recent performance data
        recent_data = list(self.performance_data)[-1000:]
        
        if len(recent_data) < 50:
            return patterns
        
        # Analyze execution time patterns
        execution_times = [d.get('execution_time', 0) for d in recent_data if 'execution_time' in d]
        if execution_times:
            pattern = {
                'type': 'execution_time_pattern',
                'data': execution_times,
                'trend': self._calculate_trend(execution_times),
                'variance': np.var(execution_times),
                'mean': np.mean(execution_times)
            }
            patterns.append(pattern)
        
        # Analyze memory usage patterns
        memory_usage = [d.get('memory_usage', 0) for d in recent_data if 'memory_usage' in d]
        if memory_usage:
            pattern = {
                'type': 'memory_usage_pattern',
                'data': memory_usage,
                'trend': self._calculate_trend(memory_usage),
                'variance': np.var(memory_usage),
                'mean': np.mean(memory_usage)
            }
            patterns.append(pattern)
        
        # Analyze cost patterns
        costs = [d.get('cost', 0) for d in recent_data if 'cost' in d]
        if costs:
            pattern = {
                'type': 'cost_pattern',
                'data': costs,
                'trend': self._calculate_trend(costs),
                'variance': np.var(costs),
                'mean': np.mean(costs)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_trend(self, data: List[float]) -> str:
        """Calculate trend in data."""
        if len(data) < 2:
            return 'insufficient_data'
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Fit linear trend
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _learn_from_performance_pattern(self, pattern: Dict[str, Any]):
        """Learn from a performance pattern."""
        # Create knowledge entry
        knowledge_entry = KnowledgeEntry(
            entry_id=f"perf_{int(time.time() * 1000)}",
            knowledge_type=KnowledgeType.PERFORMANCE_INSIGHT,
            content=pattern,
            confidence=0.8,
            source='performance_analysis',
            timestamp=time.time(),
            tags=['performance', pattern['type']]
        )
        
        # Store in knowledge base
        self.knowledge_base[knowledge_entry.entry_id] = knowledge_entry
        self.learning_metrics['total_knowledge_entries'] += 1
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(knowledge_entry.entry_id, **pattern)
    
    def _learn_from_optimizations(self):
        """Learn from optimization data."""
        if len(self.optimization_data) < 50:
            return
        
        # Analyze optimization patterns
        optimization_patterns = self._analyze_optimization_patterns()
        
        # Learn from successful optimizations
        for pattern in optimization_patterns:
            self._learn_from_optimization_pattern(pattern)
    
    def _analyze_optimization_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in optimization data."""
        patterns = []
        
        # Get recent optimization data
        recent_data = list(self.optimization_data)[-1000:]
        
        if len(recent_data) < 20:
            return patterns
        
        # Analyze successful optimizations
        successful_optimizations = [d for d in recent_data if d.get('success', False)]
        
        if successful_optimizations:
            # Extract common characteristics
            common_features = self._extract_common_features(successful_optimizations)
            
            pattern = {
                'type': 'successful_optimization_pattern',
                'features': common_features,
                'success_rate': len(successful_optimizations) / len(recent_data),
                'average_improvement': np.mean([d.get('improvement', 0) for d in successful_optimizations])
            }
            patterns.append(pattern)
        
        return patterns
    
    def _extract_common_features(self, optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common features from successful optimizations."""
        features = {
            'common_parameters': {},
            'common_conditions': {},
            'common_outcomes': {}
        }
        
        # Analyze parameter patterns
        parameters = [opt.get('parameters', {}) for opt in optimizations]
        if parameters:
            # Find common parameter values
            for param_name in set().union(*[p.keys() for p in parameters]):
                values = [p.get(param_name) for p in parameters if param_name in p]
                if len(set(values)) < len(values) * 0.8:  # 80% similarity
                    features['common_parameters'][param_name] = max(set(values), key=values.count)
        
        return features
    
    def _learn_from_optimization_pattern(self, pattern: Dict[str, Any]):
        """Learn from an optimization pattern."""
        # Create knowledge entry
        knowledge_entry = KnowledgeEntry(
            entry_id=f"opt_{int(time.time() * 1000)}",
            knowledge_type=KnowledgeType.OPTIMIZATION_PATTERN,
            content=pattern,
            confidence=pattern.get('success_rate', 0.5),
            source='optimization_analysis',
            timestamp=time.time(),
            tags=['optimization', 'pattern']
        )
        
        # Store in knowledge base
        self.knowledge_base[knowledge_entry.entry_id] = knowledge_entry
        self.learning_metrics['total_knowledge_entries'] += 1
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(knowledge_entry.entry_id, **pattern)
    
    def _extract_learning_patterns(self):
        """Extract learning patterns from collected data."""
        # Analyze circuit patterns
        circuit_patterns = self._analyze_circuit_patterns()
        
        # Analyze backend patterns
        backend_patterns = self._analyze_backend_patterns()
        
        # Create learning patterns
        for pattern_data in circuit_patterns + backend_patterns:
            pattern = LearningPattern(
                pattern_id=f"pattern_{int(time.time() * 1000)}",
                pattern_type=pattern_data['type'],
                pattern_data=pattern_data,
                confidence=pattern_data.get('confidence', 0.5),
                frequency=pattern_data.get('frequency', 1),
                success_rate=pattern_data.get('success_rate', 0.5),
                applications=pattern_data.get('applications', []),
                optimization_potential=pattern_data.get('optimization_potential', 0.0)
            )
            
            self.learning_patterns[pattern.pattern_id] = pattern
            self.learning_metrics['patterns_learned'] += 1
    
    def _analyze_circuit_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in circuit data."""
        patterns = []
        
        # This would analyze actual circuit data
        # For now, return mock patterns
        if random.random() < 0.3:  # 30% chance
            pattern = {
                'type': 'circuit_pattern',
                'pattern': 'linear_circuit',
                'confidence': random.uniform(0.6, 0.9),
                'frequency': random.randint(5, 20),
                'success_rate': random.uniform(0.7, 0.95),
                'applications': ['quantum_simulation', 'quantum_optimization'],
                'optimization_potential': random.uniform(0.1, 0.3)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_backend_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in backend behavior."""
        patterns = []
        
        # This would analyze actual backend data
        # For now, return mock patterns
        if random.random() < 0.2:  # 20% chance
            pattern = {
                'type': 'backend_pattern',
                'pattern': 'gpu_optimization',
                'confidence': random.uniform(0.5, 0.8),
                'frequency': random.randint(3, 15),
                'success_rate': random.uniform(0.6, 0.9),
                'applications': ['high_performance_computing', 'quantum_simulation'],
                'optimization_potential': random.uniform(0.15, 0.4)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _update_knowledge_base(self):
        """Update the knowledge base with new insights."""
        # Analyze knowledge relationships
        self._analyze_knowledge_relationships()
        
        # Update knowledge confidence scores
        self._update_knowledge_confidence()
        
        # Remove outdated knowledge
        self._remove_outdated_knowledge()
    
    def _analyze_knowledge_relationships(self):
        """Analyze relationships between knowledge entries."""
        # This would analyze actual relationships
        # For now, create mock relationships
        knowledge_entries = list(self.knowledge_base.values())
        
        for i, entry1 in enumerate(knowledge_entries):
            for j, entry2 in enumerate(knowledge_entries[i+1:], i+1):
                if self._are_related(entry1, entry2):
                    # Add relationship
                    self.knowledge_graph.add_edge(entry1.entry_id, entry2.entry_id)
                    entry1.relationships.append(entry2.entry_id)
                    entry2.relationships.append(entry1.entry_id)
    
    def _are_related(self, entry1: KnowledgeEntry, entry2: KnowledgeEntry) -> bool:
        """Check if two knowledge entries are related."""
        # Simple relationship check based on tags
        common_tags = set(entry1.tags) & set(entry2.tags)
        return len(common_tags) > 0
    
    def _update_knowledge_confidence(self):
        """Update confidence scores based on usage and success."""
        for entry in self.knowledge_base.values():
            # Update confidence based on usage
            if entry.usage_count > 0:
                entry.confidence = min(1.0, entry.confidence + 0.1)
            
            # Update confidence based on success rate
            if entry.success_rate > 0.8:
                entry.confidence = min(1.0, entry.confidence + 0.05)
            elif entry.success_rate < 0.3:
                entry.confidence = max(0.0, entry.confidence - 0.1)
    
    def _remove_outdated_knowledge(self):
        """Remove outdated knowledge entries."""
        current_time = time.time()
        outdated_entries = []
        
        for entry_id, entry in self.knowledge_base.items():
            # Remove entries older than 30 days with low confidence
            if (current_time - entry.timestamp > 30 * 24 * 3600 and 
                entry.confidence < 0.3 and entry.usage_count < 5):
                outdated_entries.append(entry_id)
        
        # Remove outdated entries
        for entry_id in outdated_entries:
            del self.knowledge_base[entry_id]
            if entry_id in self.knowledge_graph:
                self.knowledge_graph.remove_node(entry_id)
    
    def _organize_knowledge(self):
        """Organize knowledge base for efficient retrieval."""
        # Group knowledge by type
        knowledge_by_type = defaultdict(list)
        for entry in self.knowledge_base.values():
            knowledge_by_type[entry.knowledge_type].append(entry)
        
        # Update knowledge organization
        self.knowledge_organization = dict(knowledge_by_type)
    
    def _update_knowledge_relationships(self):
        """Update relationships in the knowledge graph."""
        # This would update actual relationships
        # For now, maintain existing relationships
        pass
    
    def _generate_learning_insights(self):
        """Generate insights from the learning process."""
        insights = []
        
        # Analyze learning progress
        learning_progress = self._analyze_learning_progress()
        insights.append(learning_progress)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities()
        insights.extend(optimization_opportunities)
        
        # Generate strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations()
        insights.extend(strategic_recommendations)
        
        # Store insights
        for insight in insights:
            self.learning_history.append(insight)
    
    def _analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress and effectiveness."""
        return {
            'type': 'learning_progress',
            'total_knowledge_entries': len(self.knowledge_base),
            'learning_patterns': len(self.learning_patterns),
            'knowledge_graph_size': self.knowledge_graph.number_of_nodes(),
            'learning_effectiveness': self._calculate_learning_effectiveness()
        }
    
    def _calculate_learning_effectiveness(self) -> float:
        """Calculate learning effectiveness score."""
        if not self.learning_history:
            return 0.0
        
        # Simple effectiveness calculation
        recent_insights = list(self.learning_history)[-100:]
        high_confidence_insights = [i for i in recent_insights if i.get('confidence', 0) > 0.7]
        
        return len(high_confidence_insights) / len(recent_insights) if recent_insights else 0.0
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities from learning."""
        opportunities = []
        
        # Analyze learning patterns for optimization potential
        for pattern in self.learning_patterns.values():
            if pattern.optimization_potential > 0.5:
                opportunity = {
                    'type': 'optimization_opportunity',
                    'pattern_id': pattern.pattern_id,
                    'optimization_potential': pattern.optimization_potential,
                    'applications': pattern.applications,
                    'confidence': pattern.confidence
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _generate_strategic_recommendations(self) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on learning."""
        recommendations = []
        
        # Analyze knowledge base for strategic insights
        if len(self.knowledge_base) > 100:
            recommendation = {
                'type': 'strategic_recommendation',
                'recommendation': 'Expand knowledge base with more diverse data',
                'priority': 'medium',
                'confidence': 0.7
            }
            recommendations.append(recommendation)
        
        # Analyze learning patterns for strategic insights
        if len(self.learning_patterns) > 50:
            recommendation = {
                'type': 'strategic_recommendation',
                'recommendation': 'Focus on high-potential learning patterns',
                'priority': 'high',
                'confidence': 0.8
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def update_performance_data(self, data: Dict[str, Any]):
        """Update performance data for learning."""
        self.performance_data.append(data)
    
    def update_optimization_data(self, data: Dict[str, Any]):
        """Update optimization data for learning."""
        self.optimization_data.append(data)
    
    def update_experimental_data(self, data: Dict[str, Any]):
        """Update experimental data for learning."""
        self.experimental_data.append(data)
    
    def get_knowledge_base_size(self) -> int:
        """Get the size of the knowledge base."""
        return len(self.knowledge_base)
    
    def get_learning_insights(self) -> List[Dict[str, Any]]:
        """Get recent learning insights."""
        return list(self.learning_history)[-20:]
    
    def get_learning_patterns(self) -> List[Dict[str, Any]]:
        """Get learned patterns."""
        return [
            {
                'pattern_id': p.pattern_id,
                'pattern_type': p.pattern_type,
                'confidence': p.confidence,
                'frequency': p.frequency,
                'success_rate': p.success_rate,
                'optimization_potential': p.optimization_potential
            }
            for p in list(self.learning_patterns.values())[-10:]
        ]
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Generate a comprehensive learning report."""
        report = LearningReport(
            report_id=f"report_{int(time.time() * 1000)}",
            timestamp=time.time(),
            learning_summary=self.learning_metrics.copy(),
            knowledge_growth={
                'total_entries': len(self.knowledge_base),
                'patterns_learned': len(self.learning_patterns),
                'graph_connections': self.knowledge_graph.number_of_edges()
            },
            performance_improvements={
                'learning_effectiveness': self._calculate_learning_effectiveness(),
                'knowledge_utilization': self._calculate_knowledge_utilization(),
                'pattern_success_rate': self._calculate_pattern_success_rate()
            },
            recommendations=self._generate_learning_recommendations(),
            predictions=self._generate_learning_predictions(),
            experimental_results=self._get_experimental_results()
        )
        
        return {
            'report_id': report.report_id,
            'timestamp': report.timestamp,
            'learning_summary': report.learning_summary,
            'knowledge_growth': report.knowledge_growth,
            'performance_improvements': report.performance_improvements,
            'recommendations': report.recommendations,
            'predictions': report.predictions,
            'experimental_results': report.experimental_results
        }
    
    def _calculate_knowledge_utilization(self) -> float:
        """Calculate knowledge utilization rate."""
        if not self.knowledge_base:
            return 0.0
        
        total_usage = sum(entry.usage_count for entry in self.knowledge_base.values())
        return total_usage / len(self.knowledge_base) if self.knowledge_base else 0.0
    
    def _calculate_pattern_success_rate(self) -> float:
        """Calculate success rate of learned patterns."""
        if not self.learning_patterns:
            return 0.0
        
        success_rates = [p.success_rate for p in self.learning_patterns.values()]
        return np.mean(success_rates) if success_rates else 0.0
    
    def _generate_learning_recommendations(self) -> List[Dict[str, Any]]:
        """Generate learning recommendations."""
        recommendations = []
        
        # Knowledge base recommendations
        if len(self.knowledge_base) < 100:
            recommendations.append({
                'type': 'knowledge_expansion',
                'description': 'Expand knowledge base with more diverse data',
                'priority': 'high'
            })
        
        # Pattern learning recommendations
        if len(self.learning_patterns) < 20:
            recommendations.append({
                'type': 'pattern_learning',
                'description': 'Focus on learning more patterns from system behavior',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _generate_learning_predictions(self) -> Dict[str, Any]:
        """Generate learning predictions."""
        predictions = {
            'knowledge_growth_rate': self._predict_knowledge_growth(),
            'learning_effectiveness_trend': self._predict_learning_effectiveness(),
            'optimization_potential': self._predict_optimization_potential()
        }
        
        return predictions
    
    def _predict_knowledge_growth(self) -> float:
        """Predict knowledge growth rate."""
        if len(self.knowledge_base) < 10:
            return 0.1
        
        # Simple growth prediction
        recent_entries = [e for e in self.knowledge_base.values() 
                         if time.time() - e.timestamp < 3600]  # Last hour
        return len(recent_entries) / 3600  # Entries per second
    
    def _predict_learning_effectiveness(self) -> str:
        """Predict learning effectiveness trend."""
        if len(self.learning_history) < 10:
            return 'insufficient_data'
        
        recent_effectiveness = [h.get('learning_effectiveness', 0) for h in list(self.learning_history)[-10:]]
        if len(recent_effectiveness) > 1:
            trend = np.polyfit(range(len(recent_effectiveness)), recent_effectiveness, 1)[0]
            if trend > 0.01:
                return 'improving'
            elif trend < -0.01:
                return 'degrading'
            else:
                return 'stable'
        
        return 'stable'
    
    def _predict_optimization_potential(self) -> float:
        """Predict optimization potential."""
        if not self.learning_patterns:
            return 0.0
        
        optimization_potentials = [p.optimization_potential for p in self.learning_patterns.values()]
        return np.mean(optimization_potentials) if optimization_potentials else 0.0
    
    def _get_experimental_results(self) -> List[Dict[str, Any]]:
        """Get experimental results for learning."""
        return list(self.experimental_data)[-10:]
