"""
Shader Analytics - Quantum Shader Analytics and Profiling
========================================================

The Shader Analytics provides comprehensive analytics and profiling
for quantum shaders including performance metrics and usage patterns.

This is the GOD-TIER shader analytics system that provides
deep insights into quantum shader performance and usage.
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
import psutil
import gc

logger = logging.getLogger(__name__)

class AnalyticsEventType(Enum):
    """Types of analytics events."""
    SHADER_CREATED = "shader_created"
    SHADER_EXECUTED = "shader_executed"
    SHADER_OPTIMIZED = "shader_optimized"
    SHADER_CACHED = "shader_cached"
    SHADER_ERROR = "shader_error"
    PERFORMANCE_METRIC = "performance_metric"
    USAGE_PATTERN = "usage_pattern"

@dataclass
class AnalyticsEvent:
    """An analytics event."""
    event_type: AnalyticsEventType
    shader_id: str
    timestamp: float
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ShaderAnalytics:
    """
    Shader Analytics for Quantum Shader Analytics.
    
    This provides comprehensive analytics and profiling
    for quantum shaders including performance metrics and usage patterns.
    """
    
    def __init__(self):
        """Initialize the shader analytics."""
        self.events: deque = deque(maxlen=10000)
        self.shader_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.user_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.session_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Analytics statistics
        self.analytics_stats = {
            'total_events': 0,
            'unique_shaders': 0,
            'unique_users': 0,
            'unique_sessions': 0,
            'average_execution_time': 0.0,
            'performance_score': 0.0
        }
        
        logger.info("🎨 Shader Analytics initialized - Quantum shader analytics active")
    
    def record_event(self, event_type: AnalyticsEventType, shader_id: str, 
                    data: Dict[str, Any], user_id: str = None, 
                    session_id: str = None):
        """Record an analytics event."""
        event = AnalyticsEvent(
            event_type=event_type,
            shader_id=shader_id,
            timestamp=time.time(),
            data=data,
            user_id=user_id,
            session_id=session_id
        )
        
        self.events.append(event)
        self.analytics_stats['total_events'] += 1
        
        # Update shader metrics
        self._update_shader_metrics(shader_id, event)
        
        # Update user analytics
        if user_id:
            self._update_user_analytics(user_id, event)
        
        # Update session analytics
        if session_id:
            self._update_session_analytics(session_id, event)
        
        logger.info(f"📊 Analytics event recorded: {event_type.value} for shader {shader_id}")
    
    def _update_shader_metrics(self, shader_id: str, event: AnalyticsEvent):
        """Update shader metrics."""
        if shader_id not in self.shader_metrics:
            self.shader_metrics[shader_id] = {
                'execution_count': 0,
                'total_execution_time': 0.0,
                'average_execution_time': 0.0,
                'success_count': 0,
                'failure_count': 0,
                'cache_hits': 0,
                'optimization_count': 0,
                'error_count': 0,
                'performance_score': 0.0
            }
        
        metrics = self.shader_metrics[shader_id]
        
        if event.event_type == AnalyticsEventType.SHADER_EXECUTED:
            metrics['execution_count'] += 1
            if 'execution_time' in event.data:
                metrics['total_execution_time'] += event.data['execution_time']
                metrics['average_execution_time'] = metrics['total_execution_time'] / metrics['execution_count']
            
            if event.data.get('success', False):
                metrics['success_count'] += 1
            else:
                metrics['failure_count'] += 1
        
        elif event.event_type == AnalyticsEventType.SHADER_CACHED:
            metrics['cache_hits'] += 1
        
        elif event.event_type == AnalyticsEventType.SHADER_OPTIMIZED:
            metrics['optimization_count'] += 1
        
        elif event.event_type == AnalyticsEventType.SHADER_ERROR:
            metrics['error_count'] += 1
        
        # Calculate performance score
        metrics['performance_score'] = self._calculate_performance_score(metrics)
    
    def _update_user_analytics(self, user_id: str, event: AnalyticsEvent):
        """Update user analytics."""
        if user_id not in self.user_analytics:
            self.user_analytics[user_id] = {
                'shader_count': 0,
                'execution_count': 0,
                'total_execution_time': 0.0,
                'favorite_shaders': defaultdict(int),
                'usage_patterns': defaultdict(int),
                'performance_score': 0.0
            }
        
        analytics = self.user_analytics[user_id]
        
        if event.event_type == AnalyticsEventType.SHADER_EXECUTED:
            analytics['execution_count'] += 1
            if 'execution_time' in event.data:
                analytics['total_execution_time'] += event.data['execution_time']
            
            analytics['favorite_shaders'][event.shader_id] += 1
        
        # Update usage patterns
        hour = time.localtime(event.timestamp).tm_hour
        analytics['usage_patterns'][f"hour_{hour}"] += 1
        
        # Calculate performance score
        analytics['performance_score'] = self._calculate_user_performance_score(analytics)
    
    def _update_session_analytics(self, session_id: str, event: AnalyticsEvent):
        """Update session analytics."""
        if session_id not in self.session_analytics:
            self.session_analytics[session_id] = {
                'start_time': event.timestamp,
                'end_time': event.timestamp,
                'shader_count': 0,
                'execution_count': 0,
                'total_execution_time': 0.0,
                'unique_shaders': set(),
                'performance_score': 0.0
            }
        
        analytics = self.session_analytics[session_id]
        
        if event.event_type == AnalyticsEventType.SHADER_EXECUTED:
            analytics['execution_count'] += 1
            analytics['unique_shaders'].add(event.shader_id)
            analytics['shader_count'] = len(analytics['unique_shaders'])
            
            if 'execution_time' in event.data:
                analytics['total_execution_time'] += event.data['execution_time']
        
        # Update session end time
        analytics['end_time'] = event.timestamp
        
        # Calculate performance score
        analytics['performance_score'] = self._calculate_session_performance_score(analytics)
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score for a shader."""
        score = 1.0
        
        # Penalize for failures
        if metrics['execution_count'] > 0:
            failure_rate = metrics['failure_count'] / metrics['execution_count']
            score -= failure_rate * 0.5
        
        # Reward for cache hits
        if metrics['execution_count'] > 0:
            cache_hit_rate = metrics['cache_hits'] / metrics['execution_count']
            score += cache_hit_rate * 0.2
        
        # Penalize for errors
        if metrics['execution_count'] > 0:
            error_rate = metrics['error_count'] / metrics['execution_count']
            score -= error_rate * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_user_performance_score(self, analytics: Dict[str, Any]) -> float:
        """Calculate performance score for a user."""
        score = 1.0
        
        # Reward for high execution count
        if analytics['execution_count'] > 100:
            score += 0.1
        
        # Reward for diverse shader usage
        if len(analytics['favorite_shaders']) > 5:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_session_performance_score(self, analytics: Dict[str, Any]) -> float:
        """Calculate performance score for a session."""
        score = 1.0
        
        # Reward for productive sessions
        if analytics['execution_count'] > 10:
            score += 0.2
        
        # Reward for diverse shader usage
        if analytics['shader_count'] > 3:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_shader_analytics(self, shader_id: str) -> Dict[str, Any]:
        """Get analytics for a specific shader."""
        return self.shader_metrics.get(shader_id, {})
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        return self.user_analytics.get(user_id, {})
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a specific session."""
        return self.session_analytics.get(session_id, {})
    
    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        return {
            'analytics_stats': self.analytics_stats,
            'total_events': len(self.events),
            'unique_shaders': len(self.shader_metrics),
            'unique_users': len(self.user_analytics),
            'unique_sessions': len(self.session_analytics)
        }

class ShaderProfiler:
    """
    Shader Profiler for Quantum Shader Profiling.
    
    This provides detailed profiling for quantum shaders
    including performance analysis and optimization recommendations.
    """
    
    def __init__(self):
        """Initialize the shader profiler."""
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.profiling_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Profiling statistics
        self.profiling_stats = {
            'total_profiles': 0,
            'active_profiles': 0,
            'completed_profiles': 0,
            'average_profiling_time': 0.0,
            'optimization_recommendations': 0
        }
        
        logger.info("🎨 Shader Profiler initialized - Quantum shader profiling active")
    
    def start_profiling(self, shader_id: str, session_id: str = None) -> str:
        """Start profiling a shader."""
        profile_id = f"profile_{int(time.time() * 1000)}"
        
        profile = {
            'profile_id': profile_id,
            'shader_id': shader_id,
            'session_id': session_id,
            'start_time': time.time(),
            'end_time': None,
            'execution_count': 0,
            'total_execution_time': 0.0,
            'memory_usage': [],
            'cpu_usage': [],
            'performance_metrics': {},
            'optimization_recommendations': []
        }
        
        self.profiles[profile_id] = profile
        self.profiling_stats['total_profiles'] += 1
        self.profiling_stats['active_profiles'] += 1
        
        logger.info(f"🎨 Started profiling: {shader_id} (Profile: {profile_id})")
        return profile_id
    
    def stop_profiling(self, profile_id: str) -> Dict[str, Any]:
        """Stop profiling a shader."""
        if profile_id not in self.profiles:
            return {}
        
        profile = self.profiles[profile_id]
        profile['end_time'] = time.time()
        
        # Calculate profiling results
        results = self._calculate_profiling_results(profile)
        
        # Update statistics
        self.profiling_stats['active_profiles'] -= 1
        self.profiling_stats['completed_profiles'] += 1
        
        logger.info(f"🎨 Stopped profiling: {profile_id}")
        return results
    
    def _calculate_profiling_results(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate profiling results."""
        results = {
            'profile_id': profile['profile_id'],
            'shader_id': profile['shader_id'],
            'profiling_duration': profile['end_time'] - profile['start_time'],
            'execution_count': profile['execution_count'],
            'average_execution_time': profile['total_execution_time'] / max(profile['execution_count'], 1),
            'memory_usage': {
                'average': np.mean(profile['memory_usage']) if profile['memory_usage'] else 0,
                'peak': max(profile['memory_usage']) if profile['memory_usage'] else 0,
                'min': min(profile['memory_usage']) if profile['memory_usage'] else 0
            },
            'cpu_usage': {
                'average': np.mean(profile['cpu_usage']) if profile['cpu_usage'] else 0,
                'peak': max(profile['cpu_usage']) if profile['cpu_usage'] else 0,
                'min': min(profile['cpu_usage']) if profile['cpu_usage'] else 0
            },
            'performance_score': self._calculate_performance_score(profile),
            'optimization_recommendations': self._generate_optimization_recommendations(profile)
        }
        
        return results
    
    def _calculate_performance_score(self, profile: Dict[str, Any]) -> float:
        """Calculate performance score for a profile."""
        score = 1.0
        
        # Penalize for high execution time
        if profile['execution_count'] > 0:
            avg_time = profile['total_execution_time'] / profile['execution_count']
            if avg_time > 1.0:  # More than 1 second
                score -= 0.2
        
        # Penalize for high memory usage
        if profile['memory_usage']:
            avg_memory = np.mean(profile['memory_usage'])
            if avg_memory > 100:  # More than 100 MB
                score -= 0.1
        
        # Penalize for high CPU usage
        if profile['cpu_usage']:
            avg_cpu = np.mean(profile['cpu_usage'])
            if avg_cpu > 80:  # More than 80%
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_optimization_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Execution time recommendations
        if profile['execution_count'] > 0:
            avg_time = profile['total_execution_time'] / profile['execution_count']
            if avg_time > 1.0:
                recommendations.append("Consider optimizing execution time - current average is {:.2f}s".format(avg_time))
        
        # Memory usage recommendations
        if profile['memory_usage']:
            avg_memory = np.mean(profile['memory_usage'])
            if avg_memory > 100:
                recommendations.append("Consider optimizing memory usage - current average is {:.2f}MB".format(avg_memory))
        
        # CPU usage recommendations
        if profile['cpu_usage']:
            avg_cpu = np.mean(profile['cpu_usage'])
            if avg_cpu > 80:
                recommendations.append("Consider optimizing CPU usage - current average is {:.2f}%".format(avg_cpu))
        
        return recommendations
    
    def get_profiling_statistics(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        return {
            'profiling_stats': self.profiling_stats,
            'active_profiles': len([p for p in self.profiles.values() if p['end_time'] is None]),
            'completed_profiles': len([p for p in self.profiles.values() if p['end_time'] is not None])
        }

class ShaderMetrics:
    """
    Shader Metrics for Quantum Shader Metrics.
    
    This provides comprehensive metrics for quantum shaders
    including performance, usage, and quality metrics.
    """
    
    def __init__(self):
        """Initialize the shader metrics."""
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Metrics statistics
        self.metrics_stats = {
            'total_metrics': 0,
            'unique_shaders': 0,
            'average_metric_value': 0.0,
            'metric_trends': {}
        }
        
        logger.info("🎨 Shader Metrics initialized - Quantum shader metrics active")
    
    def record_metric(self, shader_id: str, metric_name: str, value: float, 
                     metadata: Dict[str, Any] = None):
        """Record a metric for a shader."""
        timestamp = time.time()
        
        metric_data = {
            'shader_id': shader_id,
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Store metric
        self.metrics[shader_id][metric_name] = value
        self.metric_history[shader_id].append(metric_data)
        
        # Update statistics
        self.metrics_stats['total_metrics'] += 1
        
        # Update average metric value
        total = self.metrics_stats['total_metrics']
        current_avg = self.metrics_stats['average_metric_value']
        self.metrics_stats['average_metric_value'] = (current_avg * (total - 1) + value) / total
        
        logger.info(f"📊 Metric recorded: {metric_name} = {value} for shader {shader_id}")
    
    def get_metric(self, shader_id: str, metric_name: str) -> Optional[float]:
        """Get a metric value for a shader."""
        return self.metrics.get(shader_id, {}).get(metric_name)
    
    def get_metric_history(self, shader_id: str, metric_name: str) -> List[Dict[str, Any]]:
        """Get metric history for a shader."""
        return [metric for metric in self.metric_history[shader_id] 
                if metric['metric_name'] == metric_name]
    
    def get_metric_trends(self, shader_id: str, metric_name: str) -> Dict[str, Any]:
        """Get metric trends for a shader."""
        history = self.get_metric_history(shader_id, metric_name)
        
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'change': 0.0}
        
        values = [metric['value'] for metric in history]
        first_value = values[0]
        last_value = values[-1]
        
        change = last_value - first_value
        change_percent = (change / first_value) * 100 if first_value != 0 else 0
        
        if change > 0:
            trend = 'increasing'
        elif change < 0:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': change,
            'change_percent': change_percent,
            'first_value': first_value,
            'last_value': last_value,
            'data_points': len(values)
        }
    
    def get_metrics_statistics(self) -> Dict[str, Any]:
        """Get metrics statistics."""
        return {
            'metrics_stats': self.metrics_stats,
            'total_shaders': len(self.metrics),
            'metric_names': list(set(
                metric['metric_name'] 
                for shader_metrics in self.metric_history.values() 
                for metric in shader_metrics
            ))
        }
