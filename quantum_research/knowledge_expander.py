"""
Knowledge Expander

This module provides knowledge expansion capabilities for documenting
algorithmic discoveries and optimization strategies autonomously. The
expander maintains an evolving repository of innovations and insights
and suggests speculative research directions.

Author: Quantum Research Engine - Coratrix 4.0
"""

import asyncio
import time
import logging
import numpy as np
import random
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge."""
    ALGORITHM_DISCOVERY = "algorithm_discovery"
    OPTIMIZATION_STRATEGY = "optimization_strategy"
    PERFORMANCE_INSIGHT = "performance_insight"
    THEORETICAL_BREAKTHROUGH = "theoretical_breakthrough"
    EXPERIMENTAL_RESULT = "experimental_result"
    RESEARCH_DIRECTION = "research_direction"

class InsightLevel(Enum):
    """Levels of insight."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    BREAKTHROUGH = "breakthrough"

@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base."""
    entry_id: str
    knowledge_type: KnowledgeType
    title: str
    content: str
    insights: List[str]
    tags: List[str]
    confidence_score: float
    novelty_score: float
    practical_value: float
    theoretical_significance: float
    related_entries: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

@dataclass
class ResearchDirection:
    """Research direction suggestion."""
    direction_id: str
    title: str
    description: str
    potential_impact: float
    feasibility: float
    novelty: float
    related_knowledge: List[str]
    suggested_experiments: List[str]
    expected_outcomes: List[str]
    created_at: float = field(default_factory=time.time)

class KnowledgeExpander:
    """
    Knowledge expander for documenting discoveries and insights.
    
    This class can document all algorithmic discoveries and optimization
    strategies autonomously, maintain an evolving repository of innovations
    and insights, and suggest speculative research directions.
    """
    
    def __init__(self):
        """Initialize the knowledge expander."""
        self.expander_id = f"ke_{int(time.time() * 1000)}"
        self.running = False
        self.knowledge_base = {}
        self.research_directions = []
        self.insight_patterns = defaultdict(list)
        self.knowledge_graph = nx.Graph()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clustering_model = KMeans(n_clusters=10, random_state=42)
        
        # Knowledge statistics
        self.knowledge_statistics = defaultdict(int)
        self.insight_statistics = defaultdict(list)
        self.research_statistics = defaultdict(list)
        
        logger.info(f"Knowledge Expander initialized: {self.expander_id}")
    
    async def start(self):
        """Start the knowledge expander."""
        if self.running:
            logger.warning("Knowledge expander is already running")
            return
        
        self.running = True
        logger.info("Knowledge Expander started")
        
        # Start background tasks
        asyncio.create_task(self._knowledge_processing())
        asyncio.create_task(self._insight_analysis())
        asyncio.create_task(self._research_direction_generation())
        asyncio.create_task(self._knowledge_graph_updating())
    
    async def stop(self):
        """Stop the knowledge expander."""
        if not self.running:
            logger.warning("Knowledge expander is not running")
            return
        
        self.running = False
        logger.info("Knowledge Expander stopped")
    
    async def _knowledge_processing(self):
        """Process new knowledge entries."""
        while self.running:
            try:
                # Process any pending knowledge entries
                await self._process_pending_knowledge()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in knowledge processing: {e}")
                await asyncio.sleep(1.0)
    
    async def _insight_analysis(self):
        """Analyze insights and patterns."""
        while self.running:
            try:
                if len(self.knowledge_base) > 10:
                    await self._analyze_insight_patterns()
                    await self._identify_knowledge_gaps()
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error in insight analysis: {e}")
                await asyncio.sleep(1.0)
    
    async def _research_direction_generation(self):
        """Generate research directions."""
        while self.running:
            try:
                if len(self.knowledge_base) > 20:
                    await self._generate_research_directions()
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Error in research direction generation: {e}")
                await asyncio.sleep(1.0)
    
    async def _knowledge_graph_updating(self):
        """Update knowledge graph."""
        while self.running:
            try:
                if len(self.knowledge_base) > 5:
                    await self._update_knowledge_graph()
                
                await asyncio.sleep(15.0)
                
            except Exception as e:
                logger.error(f"Error in knowledge graph updating: {e}")
                await asyncio.sleep(1.0)
    
    async def document_discovery(self, discovery: Dict[str, Any]) -> str:
        """Document a new discovery."""
        try:
            entry_id = f"entry_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            # Determine knowledge type
            knowledge_type = await self._determine_knowledge_type(discovery)
            
            # Extract insights
            insights = await self._extract_insights(discovery)
            
            # Generate tags
            tags = await self._generate_tags(discovery, insights)
            
            # Calculate scores
            confidence_score = await self._calculate_confidence_score(discovery)
            novelty_score = await self._calculate_novelty_score(discovery)
            practical_value = await self._calculate_practical_value(discovery)
            theoretical_significance = await self._calculate_theoretical_significance(discovery)
            
            # Create knowledge entry
            entry = KnowledgeEntry(
                entry_id=entry_id,
                knowledge_type=knowledge_type,
                title=discovery.get('title', f"Discovery {entry_id}"),
                content=discovery.get('content', ''),
                insights=insights,
                tags=tags,
                confidence_score=confidence_score,
                novelty_score=novelty_score,
                practical_value=practical_value,
                theoretical_significance=theoretical_significance
            )
            
            # Store in knowledge base
            self.knowledge_base[entry_id] = entry
            
            # Update statistics
            self.knowledge_statistics[knowledge_type.value] += 1
            
            logger.info(f"Documented discovery {entry_id}: {knowledge_type.value}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error documenting discovery: {e}")
            return None
    
    async def _determine_knowledge_type(self, discovery: Dict[str, Any]) -> KnowledgeType:
        """Determine the knowledge type of a discovery."""
        content = discovery.get('content', '').lower()
        title = discovery.get('title', '').lower()
        
        # Check for algorithm-related keywords
        if any(keyword in content or keyword in title for keyword in ['algorithm', 'quantum algorithm', 'optimization']):
            return KnowledgeType.ALGORITHM_DISCOVERY
        
        # Check for optimization-related keywords
        if any(keyword in content or keyword in title for keyword in ['optimization', 'improvement', 'enhancement']):
            return KnowledgeType.OPTIMIZATION_STRATEGY
        
        # Check for performance-related keywords
        if any(keyword in content or keyword in title for keyword in ['performance', 'benchmark', 'speed', 'efficiency']):
            return KnowledgeType.PERFORMANCE_INSIGHT
        
        # Check for theoretical keywords
        if any(keyword in content or keyword in title for keyword in ['theoretical', 'proof', 'mathematical', 'formal']):
            return KnowledgeType.THEORETICAL_BREAKTHROUGH
        
        # Check for experimental keywords
        if any(keyword in content or keyword in title for keyword in ['experiment', 'test', 'validation', 'result']):
            return KnowledgeType.EXPERIMENTAL_RESULT
        
        # Default to research direction
        return KnowledgeType.RESEARCH_DIRECTION
    
    async def _extract_insights(self, discovery: Dict[str, Any]) -> List[str]:
        """Extract insights from a discovery."""
        insights = []
        
        # Extract insights from content
        content = discovery.get('content', '')
        if content:
            # Simple insight extraction based on keywords
            if 'novel' in content.lower():
                insights.append("Novel approach identified")
            if 'breakthrough' in content.lower():
                insights.append("Breakthrough potential detected")
            if 'optimization' in content.lower():
                insights.append("Optimization opportunity identified")
            if 'entanglement' in content.lower():
                insights.append("Entanglement-based insight")
            if 'hybrid' in content.lower():
                insights.append("Hybrid classical-quantum approach")
        
        # Extract insights from performance metrics
        performance_metrics = discovery.get('performance_metrics', {})
        if performance_metrics:
            if performance_metrics.get('execution_time', 0) < 1.0:
                insights.append("Fast execution time achieved")
            if performance_metrics.get('accuracy', 0) > 0.9:
                insights.append("High accuracy achieved")
            if performance_metrics.get('scalability', 0) > 0.8:
                insights.append("Good scalability demonstrated")
        
        # Extract insights from algorithm characteristics
        algorithm_type = discovery.get('algorithm_type', '')
        if algorithm_type:
            insights.append(f"Algorithm type: {algorithm_type}")
        
        innovation_level = discovery.get('innovation_level', '')
        if innovation_level:
            insights.append(f"Innovation level: {innovation_level}")
        
        return insights
    
    async def _generate_tags(self, discovery: Dict[str, Any], insights: List[str]) -> List[str]:
        """Generate tags for a discovery."""
        tags = []
        
        # Add tags based on algorithm type
        algorithm_type = discovery.get('algorithm_type', '')
        if algorithm_type:
            tags.append(algorithm_type.lower().replace('_', ' '))
        
        # Add tags based on insights
        for insight in insights:
            if 'novel' in insight.lower():
                tags.append('novel')
            if 'breakthrough' in insight.lower():
                tags.append('breakthrough')
            if 'optimization' in insight.lower():
                tags.append('optimization')
            if 'entanglement' in insight.lower():
                tags.append('entanglement')
            if 'hybrid' in insight.lower():
                tags.append('hybrid')
        
        # Add tags based on performance metrics
        performance_metrics = discovery.get('performance_metrics', {})
        if performance_metrics.get('execution_time', 0) < 1.0:
            tags.append('fast')
        if performance_metrics.get('accuracy', 0) > 0.9:
            tags.append('accurate')
        if performance_metrics.get('scalability', 0) > 0.8:
            tags.append('scalable')
        
        # Add tags based on complexity
        complexity = discovery.get('complexity', '')
        if complexity:
            tags.append(complexity.lower())
        
        return list(set(tags))  # Remove duplicates
    
    async def _calculate_confidence_score(self, discovery: Dict[str, Any]) -> float:
        """Calculate confidence score for a discovery."""
        score = 0.0
        
        # Base confidence from discovery characteristics
        if 'confidence_score' in discovery:
            score += discovery['confidence_score'] * 0.4
        
        # Confidence from performance metrics
        performance_metrics = discovery.get('performance_metrics', {})
        if performance_metrics:
            avg_performance = np.mean(list(performance_metrics.values()))
            score += avg_performance * 0.3
        
        # Confidence from novelty
        novelty_score = discovery.get('novelty_score', 0.5)
        score += novelty_score * 0.2
        
        # Confidence from practical applicability
        practical_applicability = discovery.get('practical_applicability', 0.5)
        score += practical_applicability * 0.1
        
        return min(1.0, score)
    
    async def _calculate_novelty_score(self, discovery: Dict[str, Any]) -> float:
        """Calculate novelty score for a discovery."""
        score = 0.0
        
        # Base novelty from innovation level
        innovation_level = discovery.get('innovation_level', 'moderate')
        innovation_scores = {
            'incremental': 0.2,
            'moderate': 0.4,
            'breakthrough': 0.7,
            'revolutionary': 1.0
        }
        score += innovation_scores.get(innovation_level.lower(), 0.4)
        
        # Novelty from algorithm type
        algorithm_type = discovery.get('algorithm_type', '')
        if 'novel' in algorithm_type.lower() or 'new' in algorithm_type.lower():
            score += 0.2
        
        # Novelty from entanglement patterns
        entanglement_patterns = discovery.get('entanglement_patterns', [])
        if entanglement_patterns:
            score += 0.1 * len(entanglement_patterns)
        
        # Novelty from state encodings
        state_encodings = discovery.get('state_encodings', [])
        if state_encodings:
            score += 0.1 * len(state_encodings)
        
        return min(1.0, score)
    
    async def _calculate_practical_value(self, discovery: Dict[str, Any]) -> float:
        """Calculate practical value for a discovery."""
        score = 0.0
        
        # Practical value from practical applicability
        practical_applicability = discovery.get('practical_applicability', 0.5)
        score += practical_applicability * 0.4
        
        # Practical value from performance metrics
        performance_metrics = discovery.get('performance_metrics', {})
        if performance_metrics:
            # Weight different metrics
            if 'execution_time' in performance_metrics:
                time_score = max(0, 1 - performance_metrics['execution_time'] / 10.0)
                score += time_score * 0.2
            
            if 'accuracy' in performance_metrics:
                score += performance_metrics['accuracy'] * 0.2
            
            if 'scalability' in performance_metrics:
                score += performance_metrics['scalability'] * 0.2
        
        return min(1.0, score)
    
    async def _calculate_theoretical_significance(self, discovery: Dict[str, Any]) -> float:
        """Calculate theoretical significance for a discovery."""
        score = 0.0
        
        # Theoretical significance from innovation level
        innovation_level = discovery.get('innovation_level', 'moderate')
        innovation_scores = {
            'incremental': 0.1,
            'moderate': 0.3,
            'breakthrough': 0.7,
            'revolutionary': 1.0
        }
        score += innovation_scores.get(innovation_level.lower(), 0.3)
        
        # Theoretical significance from algorithm type
        algorithm_type = discovery.get('algorithm_type', '')
        if 'theoretical' in algorithm_type.lower() or 'mathematical' in algorithm_type.lower():
            score += 0.3
        
        # Theoretical significance from complexity
        complexity = discovery.get('complexity', 'moderate')
        complexity_scores = {
            'simple': 0.1,
            'moderate': 0.3,
            'complex': 0.6,
            'extreme': 0.9
        }
        score += complexity_scores.get(complexity.lower(), 0.3)
        
        return min(1.0, score)
    
    async def _process_pending_knowledge(self):
        """Process any pending knowledge entries."""
        # This would process any pending knowledge entries
        # For now, it's a placeholder
        pass
    
    async def _analyze_insight_patterns(self):
        """Analyze insight patterns in the knowledge base."""
        try:
            if len(self.knowledge_base) < 5:
                return
            
            # Extract all insights
            all_insights = []
            for entry in self.knowledge_base.values():
                all_insights.extend(entry.insights)
            
            if not all_insights:
                return
            
            # Vectorize insights
            insight_vectors = self.vectorizer.fit_transform(all_insights)
            
            # Cluster insights
            if len(all_insights) >= 5:
                clusters = self.clustering_model.fit_predict(insight_vectors)
                
                # Analyze clusters
                for i, cluster in enumerate(clusters):
                    self.insight_patterns[f"cluster_{cluster}"].append(all_insights[i])
            
            logger.info(f"Analyzed insight patterns: {len(self.insight_patterns)} clusters")
            
        except Exception as e:
            logger.error(f"Error analyzing insight patterns: {e}")
    
    async def _identify_knowledge_gaps(self):
        """Identify knowledge gaps in the knowledge base."""
        try:
            # Analyze knowledge distribution
            knowledge_types = [entry.knowledge_type.value for entry in self.knowledge_base.values()]
            type_counts = defaultdict(int)
            for kt in knowledge_types:
                type_counts[kt] += 1
            
            # Identify underrepresented areas
            total_entries = len(self.knowledge_base)
            if total_entries > 0:
                underrepresented = []
                for kt, count in type_counts.items():
                    if count / total_entries < 0.1:  # Less than 10% representation
                        underrepresented.append(kt)
                
                if underrepresented:
                    logger.info(f"Identified underrepresented knowledge areas: {underrepresented}")
            
        except Exception as e:
            logger.error(f"Error identifying knowledge gaps: {e}")
    
    async def _generate_research_directions(self):
        """Generate research directions based on knowledge base."""
        try:
            if len(self.knowledge_base) < 10:
                return
            
            # Analyze knowledge base for research opportunities
            research_opportunities = await self._identify_research_opportunities()
            
            for opportunity in research_opportunities:
                direction = await self._create_research_direction(opportunity)
                if direction:
                    self.research_directions.append(direction)
            
            logger.info(f"Generated {len(research_opportunities)} research directions")
            
        except Exception as e:
            logger.error(f"Error generating research directions: {e}")
    
    async def _identify_research_opportunities(self) -> List[Dict[str, Any]]:
        """Identify research opportunities from knowledge base."""
        opportunities = []
        
        # Analyze knowledge types for gaps
        knowledge_types = [entry.knowledge_type for entry in self.knowledge_base.values()]
        type_counts = defaultdict(int)
        for kt in knowledge_types:
            type_counts[kt] += 1
        
        # Identify underrepresented areas
        total_entries = len(self.knowledge_base)
        for kt, count in type_counts.items():
            if count / total_entries < 0.15:  # Less than 15% representation
                opportunities.append({
                    'type': 'knowledge_gap',
                    'area': kt.value,
                    'description': f"Underrepresented area: {kt.value}",
                    'potential_impact': 0.7,
                    'feasibility': 0.8
                })
        
        # Analyze insight patterns for opportunities
        for pattern_name, insights in self.insight_patterns.items():
            if len(insights) >= 3:  # Significant pattern
                opportunities.append({
                    'type': 'insight_pattern',
                    'area': pattern_name,
                    'description': f"Research opportunity in {pattern_name}",
                    'potential_impact': 0.8,
                    'feasibility': 0.6
                })
        
        return opportunities
    
    async def _create_research_direction(self, opportunity: Dict[str, Any]) -> Optional[ResearchDirection]:
        """Create a research direction from an opportunity."""
        try:
            direction_id = f"rd_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            # Generate research direction content
            title = f"Research Direction: {opportunity['area'].title()}"
            description = opportunity['description']
            
            # Generate related knowledge
            related_knowledge = []
            for entry in self.knowledge_base.values():
                if opportunity['area'] in entry.tags or opportunity['area'] in entry.title.lower():
                    related_knowledge.append(entry.entry_id)
            
            # Generate suggested experiments
            suggested_experiments = await self._generate_suggested_experiments(opportunity)
            
            # Generate expected outcomes
            expected_outcomes = await self._generate_expected_outcomes(opportunity)
            
            direction = ResearchDirection(
                direction_id=direction_id,
                title=title,
                description=description,
                potential_impact=opportunity['potential_impact'],
                feasibility=opportunity['feasibility'],
                novelty=random.uniform(0.6, 1.0),
                related_knowledge=related_knowledge,
                suggested_experiments=suggested_experiments,
                expected_outcomes=expected_outcomes
            )
            
            return direction
            
        except Exception as e:
            logger.error(f"Error creating research direction: {e}")
            return None
    
    async def _generate_suggested_experiments(self, opportunity: Dict[str, Any]) -> List[str]:
        """Generate suggested experiments for a research direction."""
        experiments = []
        
        if opportunity['type'] == 'knowledge_gap':
            experiments.append(f"Conduct comprehensive study in {opportunity['area']}")
            experiments.append(f"Develop novel approaches for {opportunity['area']}")
            experiments.append(f"Validate existing theories in {opportunity['area']}")
        
        elif opportunity['type'] == 'insight_pattern':
            experiments.append(f"Investigate {opportunity['area']} pattern in detail")
            experiments.append(f"Explore applications of {opportunity['area']} pattern")
            experiments.append(f"Optimize {opportunity['area']} pattern performance")
        
        return experiments
    
    async def _generate_expected_outcomes(self, opportunity: Dict[str, Any]) -> List[str]:
        """Generate expected outcomes for a research direction."""
        outcomes = []
        
        outcomes.append(f"Improved understanding of {opportunity['area']}")
        outcomes.append(f"Novel algorithms or methods in {opportunity['area']}")
        outcomes.append(f"Performance improvements in {opportunity['area']}")
        outcomes.append(f"New theoretical insights in {opportunity['area']}")
        
        return outcomes
    
    async def _update_knowledge_graph(self):
        """Update the knowledge graph with new entries."""
        try:
            # Add new nodes
            for entry_id, entry in self.knowledge_base.items():
                if not self.knowledge_graph.has_node(entry_id):
                    self.knowledge_graph.add_node(entry_id, **{
                        'title': entry.title,
                        'knowledge_type': entry.knowledge_type.value,
                        'confidence_score': entry.confidence_score,
                        'novelty_score': entry.novelty_score
                    })
            
            # Add edges based on similarity
            for entry_id1, entry1 in self.knowledge_base.items():
                for entry_id2, entry2 in self.knowledge_base.items():
                    if entry_id1 != entry_id2:
                        similarity = await self._calculate_entry_similarity(entry1, entry2)
                        if similarity > 0.5:  # High similarity threshold
                            if not self.knowledge_graph.has_edge(entry_id1, entry_id2):
                                self.knowledge_graph.add_edge(entry_id1, entry_id2, weight=similarity)
            
            logger.info(f"Updated knowledge graph: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}")
    
    async def _calculate_entry_similarity(self, entry1: KnowledgeEntry, entry2: KnowledgeEntry) -> float:
        """Calculate similarity between two knowledge entries."""
        similarity = 0.0
        
        # Type similarity
        if entry1.knowledge_type == entry2.knowledge_type:
            similarity += 0.3
        
        # Tag similarity
        common_tags = set(entry1.tags) & set(entry2.tags)
        if entry1.tags or entry2.tags:
            tag_similarity = len(common_tags) / max(len(entry1.tags), len(entry2.tags))
            similarity += tag_similarity * 0.3
        
        # Content similarity (simplified)
        if entry1.content and entry2.content:
            # Simple word overlap
            words1 = set(entry1.content.lower().split())
            words2 = set(entry2.content.lower().split())
            if words1 or words2:
                content_similarity = len(words1 & words2) / max(len(words1), len(words2))
                similarity += content_similarity * 0.4
        
        return similarity
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        total_entries = len(self.knowledge_base)
        
        # Knowledge type distribution
        type_distribution = defaultdict(int)
        for entry in self.knowledge_base.values():
            type_distribution[entry.knowledge_type.value] += 1
        
        # Confidence distribution
        confidence_scores = [entry.confidence_score for entry in self.knowledge_base.values()]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Novelty distribution
        novelty_scores = [entry.novelty_score for entry in self.knowledge_base.values()]
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0.0
        
        # Practical value distribution
        practical_values = [entry.practical_value for entry in self.knowledge_base.values()]
        avg_practical_value = np.mean(practical_values) if practical_values else 0.0
        
        return {
            'expander_id': self.expander_id,
            'running': self.running,
            'total_entries': total_entries,
            'type_distribution': dict(type_distribution),
            'research_directions': len(self.research_directions),
            'insight_patterns': len(self.insight_patterns),
            'knowledge_graph_nodes': self.knowledge_graph.number_of_nodes(),
            'knowledge_graph_edges': self.knowledge_graph.number_of_edges(),
            'average_confidence': avg_confidence,
            'average_novelty': avg_novelty,
            'average_practical_value': avg_practical_value
        }
    
    def get_knowledge_entries(self, knowledge_type: Optional[KnowledgeType] = None) -> List[KnowledgeEntry]:
        """Get knowledge entries, optionally filtered by type."""
        entries = list(self.knowledge_base.values())
        
        if knowledge_type:
            entries = [entry for entry in entries if entry.knowledge_type == knowledge_type]
        
        # Sort by confidence score
        entries.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return entries
    
    def get_research_directions(self) -> List[ResearchDirection]:
        """Get research directions."""
        # Sort by potential impact
        directions = sorted(self.research_directions, key=lambda x: x.potential_impact, reverse=True)
        return directions
    
    def get_insight_patterns(self) -> Dict[str, List[str]]:
        """Get insight patterns."""
        return dict(self.insight_patterns)
    
    def get_knowledge_graph(self) -> nx.Graph:
        """Get the knowledge graph."""
        return self.knowledge_graph
