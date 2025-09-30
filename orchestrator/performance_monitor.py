"""
Performance Monitor - Real-Time Quantum OS Telemetry
==================================================

The Performance Monitor is the nervous system of Coratrix 4.0's Quantum OS.
It collects, analyzes, and provides real-time telemetry about:

- Execution performance across all backends
- Resource utilization and bottlenecks
- Circuit execution patterns and optimization opportunities
- System health and reliability metrics
- Auto-tuning feedback loops

This enables the Quantum OS to self-optimize and adapt in real-time,
creating a truly intelligent and adaptive quantum computing platform.
"""

import time
import logging
import numpy as np
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a quantum execution."""
    execution_id: str
    backend_type: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    circuit_characteristics: Dict[str, Any] = field(default_factory=dict)
    optimization_opportunities: List[str] = field(default_factory=list)

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    active_executions: int
    queue_length: int

@dataclass
class BackendPerformanceProfile:
    """Performance profile for a specific backend."""
    backend_type: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    average_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    reliability_score: float = 1.0
    performance_trend: str = "stable"  # improving, stable, degrading
    last_updated: float = 0.0

class TelemetryCollector:
    """
    Real-Time Telemetry Collector for Quantum OS.
    
    This collector gathers comprehensive telemetry data from all components
    of the Quantum OS, enabling real-time monitoring, analysis, and optimization.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize the telemetry collector."""
        self.buffer_size = buffer_size
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.system_metrics_buffer: deque = deque(maxlen=buffer_size)
        self.backend_profiles: Dict[str, BackendPerformanceProfile] = {}
        
        # Threading
        self.collection_thread = None
        self.running = False
        self.collection_interval = 1.0  # seconds
        
        # Callbacks
        self.metrics_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        logger.info("📊 Telemetry Collector initialized - Real-time monitoring active")
    
    def start_collection(self):
        """Start telemetry collection."""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("🎯 Telemetry collection started")
    
    def stop_collection(self):
        """Stop telemetry collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("🛑 Telemetry collection stopped")
    
    def collect(self, metrics: Dict[str, Any]):
        """Collect telemetry metrics."""
        timestamp = time.time()
        metrics['timestamp'] = timestamp
        
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        # Update backend profiles
        if 'backend_type' in metrics:
            self._update_backend_profile(metrics)
        
        # Trigger callbacks
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"❌ Metrics callback error: {e}")
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            system_metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage_percent=psutil.disk_usage('/').percent,
                network_io=psutil.net_io_counters()._asdict(),
                active_executions=0,  # Will be updated by orchestrator
                queue_length=0  # Will be updated by orchestrator
            )
            
            self.system_metrics_buffer.append(system_metrics)
            
        except Exception as e:
            logger.error(f"❌ System metrics collection error: {e}")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                self.collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"❌ Collection loop error: {e}")
                time.sleep(1.0)
    
    def _update_backend_profile(self, metrics: Dict[str, Any]):
        """Update backend performance profile."""
        backend_type = metrics.get('backend_type', 'unknown')
        
        if backend_type not in self.backend_profiles:
            self.backend_profiles[backend_type] = BackendPerformanceProfile(backend_type=backend_type)
        
        profile = self.backend_profiles[backend_type]
        
        # Update execution counts
        profile.total_executions += 1
        
        if metrics.get('success', True):
            profile.successful_executions += 1
        else:
            profile.failed_executions += 1
        
        # Update performance metrics
        execution_time = metrics.get('execution_time', 0.0)
        memory_usage = metrics.get('memory_usage_mb', 0.0)
        
        # Rolling average
        if profile.total_executions == 1:
            profile.average_execution_time = execution_time
            profile.average_memory_usage = memory_usage
        else:
            alpha = 0.1  # Learning rate
            profile.average_execution_time = (1 - alpha) * profile.average_execution_time + alpha * execution_time
            profile.average_memory_usage = (1 - alpha) * profile.average_memory_usage + alpha * memory_usage
        
        # Update peak memory
        profile.peak_memory_usage = max(profile.peak_memory_usage, memory_usage)
        
        # Update reliability score
        profile.reliability_score = profile.successful_executions / profile.total_executions
        
        # Update performance trend
        self._update_performance_trend(profile, execution_time)
        
        profile.last_updated = time.time()
    
    def _update_performance_trend(self, profile: BackendPerformanceProfile, execution_time: float):
        """Update performance trend for a backend."""
        # Simple trend analysis
        if profile.total_executions < 10:
            profile.performance_trend = "stable"
            return
        
        # Get recent execution times
        recent_metrics = [m for m in self.metrics_buffer 
                         if m.get('backend_type') == profile.backend_type and 
                         m.get('timestamp', 0) > time.time() - 300]  # Last 5 minutes
        
        if len(recent_metrics) < 5:
            return
        
        recent_times = [m.get('execution_time', 0) for m in recent_metrics[-10:]]
        
        # Calculate trend
        if len(recent_times) >= 5:
            first_half = np.mean(recent_times[:len(recent_times)//2])
            second_half = np.mean(recent_times[len(recent_times)//2:])
            
            if second_half < first_half * 0.9:
                profile.performance_trend = "improving"
            elif second_half > first_half * 1.1:
                profile.performance_trend = "degrading"
            else:
                profile.performance_trend = "stable"
    
    def add_metrics_callback(self, callback: Callable):
        """Add a metrics callback."""
        self.metrics_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'total_metrics': len(self.metrics_buffer),
            'backend_profiles': {k: v.__dict__ for k, v in self.backend_profiles.items()},
            'system_metrics_count': len(self.system_metrics_buffer),
            'collection_running': self.running,
            'buffer_utilization': len(self.metrics_buffer) / self.buffer_size
        }
    
    def get_backend_performance(self, backend_type: str) -> Optional[Dict[str, Any]]:
        """Get performance data for a specific backend."""
        if backend_type in self.backend_profiles:
            return self.backend_profiles[backend_type].__dict__
        return None
    
    def get_system_metrics_history(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get system metrics history."""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [m.__dict__ for m in self.system_metrics_buffer 
                if m.timestamp > cutoff_time]

class PerformanceMonitor:
    """
    Real-Time Performance Monitor for Quantum OS.
    
    This monitor provides comprehensive performance analysis and optimization
    recommendations for the Quantum OS, enabling self-tuning and adaptive behavior.
    """
    
    def __init__(self, telemetry_collector: TelemetryCollector = None):
        """Initialize the performance monitor."""
        self.telemetry_collector = telemetry_collector or TelemetryCollector()
        self.analysis_thread = None
        self.running = False
        self.analysis_interval = 5.0  # seconds
        
        # Performance analysis
        self.performance_insights: List[Dict[str, Any]] = []
        self.optimization_recommendations: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'execution_time_ms': 1000.0,
            'memory_usage_mb': 1000.0,
            'cpu_usage_percent': 90.0,
            'reliability_threshold': 0.8
        }
        
        logger.info("📈 Performance Monitor initialized - Real-time analysis active")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        logger.info("🎯 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5.0)
        logger.info("🛑 Performance monitoring stopped")
    
    def _analysis_loop(self):
        """Main analysis loop."""
        while self.running:
            try:
                self._analyze_performance()
                self._generate_insights()
                self._check_alerts()
                time.sleep(self.analysis_interval)
            except Exception as e:
                logger.error(f"❌ Analysis loop error: {e}")
                time.sleep(1.0)
    
    def _analyze_performance(self):
        """Analyze current performance metrics."""
        if not self.telemetry_collector.metrics_buffer:
            return
        
        # Analyze recent metrics
        recent_metrics = list(self.telemetry_collector.metrics_buffer)[-100:]  # Last 100 metrics
        
        # Performance analysis
        execution_times = [m.get('execution_time', 0) for m in recent_metrics if 'execution_time' in m]
        memory_usage = [m.get('memory_usage_mb', 0) for m in recent_metrics if 'memory_usage_mb' in m]
        success_rates = [m.get('success', True) for m in recent_metrics]
        
        analysis = {
            'timestamp': time.time(),
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'max_execution_time': np.max(execution_times) if execution_times else 0,
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'max_memory_usage': np.max(memory_usage) if memory_usage else 0,
            'success_rate': sum(success_rates) / len(success_rates) if success_rates else 1.0,
            'total_metrics': len(recent_metrics)
        }
        
        self.performance_insights.append(analysis)
        
        # Keep only recent insights
        if len(self.performance_insights) > 1000:
            self.performance_insights = self.performance_insights[-500:]
    
    def _generate_insights(self):
        """Generate performance insights and recommendations."""
        if len(self.performance_insights) < 2:
            return
        
        recent_insights = self.performance_insights[-10:]
        
        # Analyze trends
        execution_times = [i['avg_execution_time'] for i in recent_insights]
        memory_usage = [i['avg_memory_usage'] for i in recent_insights]
        success_rates = [i['success_rate'] for i in recent_insights]
        
        # Generate recommendations
        recommendations = []
        
        # Execution time recommendations
        if len(execution_times) >= 5:
            trend = np.polyfit(range(len(execution_times)), execution_times, 1)[0]
            if trend > 0.1:  # Increasing execution time
                recommendations.append({
                    'type': 'performance_degradation',
                    'message': 'Execution times are increasing',
                    'severity': 'warning',
                    'recommendation': 'Consider optimizing circuit or switching backends'
                })
        
        # Memory usage recommendations
        if len(memory_usage) >= 5:
            avg_memory = np.mean(memory_usage)
            if avg_memory > self.alert_thresholds['memory_usage_mb']:
                recommendations.append({
                    'type': 'high_memory_usage',
                    'message': f'High memory usage: {avg_memory:.1f} MB',
                    'severity': 'warning',
                    'recommendation': 'Consider using sparse operations or reducing circuit size'
                })
        
        # Success rate recommendations
        if len(success_rates) >= 5:
            avg_success_rate = np.mean(success_rates)
            if avg_success_rate < self.alert_thresholds['reliability_threshold']:
                recommendations.append({
                    'type': 'low_reliability',
                    'message': f'Low success rate: {avg_success_rate:.2%}',
                    'severity': 'error',
                    'recommendation': 'Check backend health and circuit validity'
                })
        
        # Store recommendations
        for rec in recommendations:
            rec['timestamp'] = time.time()
            self.optimization_recommendations.append(rec)
        
        # Keep only recent recommendations
        if len(self.optimization_recommendations) > 100:
            self.optimization_recommendations = self.optimization_recommendations[-50:]
    
    def _check_alerts(self):
        """Check for performance alerts."""
        if not self.performance_insights:
            return
        
        latest_insights = self.performance_insights[-1]
        
        # Check alert thresholds
        alerts = []
        
        if latest_insights['avg_execution_time'] > self.alert_thresholds['execution_time_ms']:
            alerts.append({
                'type': 'high_execution_time',
                'message': f'High execution time: {latest_insights["avg_execution_time"]:.1f}ms',
                'severity': 'warning'
            })
        
        if latest_insights['avg_memory_usage'] > self.alert_thresholds['memory_usage_mb']:
            alerts.append({
                'type': 'high_memory_usage',
                'message': f'High memory usage: {latest_insights["avg_memory_usage"]:.1f}MB',
                'severity': 'warning'
            })
        
        if latest_insights['success_rate'] < self.alert_thresholds['reliability_threshold']:
            alerts.append({
                'type': 'low_success_rate',
                'message': f'Low success rate: {latest_insights["success_rate"]:.2%}',
                'severity': 'error'
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.telemetry_collector.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"❌ Alert callback error: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'monitoring_active': self.running,
            'total_insights': len(self.performance_insights),
            'total_recommendations': len(self.optimization_recommendations),
            'recent_insights': self.performance_insights[-10:] if self.performance_insights else [],
            'recent_recommendations': self.optimization_recommendations[-10:] if self.optimization_recommendations else [],
            'alert_thresholds': self.alert_thresholds,
            'backend_profiles': self.telemetry_collector.backend_profiles
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations."""
        return self.optimization_recommendations[-20:]  # Last 20 recommendations
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]):
        """Update alert thresholds."""
        self.alert_thresholds.update(thresholds)
        logger.info(f"📊 Alert thresholds updated: {thresholds}")
    
    def get_backend_recommendations(self, circuit_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get backend recommendations for a specific circuit."""
        recommendations = []
        
        # Analyze circuit characteristics
        num_qubits = circuit_characteristics.get('num_qubits', 0)
        sparsity_ratio = circuit_characteristics.get('sparsity_ratio', 0.0)
        entanglement_complexity = circuit_characteristics.get('entanglement_complexity', 0.0)
        
        # Generate recommendations based on characteristics
        if num_qubits > 15:
            recommendations.append({
                'type': 'large_circuit',
                'message': f'Large circuit ({num_qubits} qubits) detected',
                'recommendation': 'Consider using sparse-tensor hybrid engine or distributed execution',
                'priority': 'high'
            })
        
        if sparsity_ratio > 0.5:
            recommendations.append({
                'type': 'sparse_circuit',
                'message': f'Sparse circuit detected (sparsity: {sparsity_ratio:.2%})',
                'recommendation': 'Use sparse-tensor engine for optimal performance',
                'priority': 'medium'
            })
        
        if entanglement_complexity > 0.7:
            recommendations.append({
                'type': 'high_entanglement',
                'message': f'High entanglement complexity ({entanglement_complexity:.2f})',
                'recommendation': 'Consider tensor network simulation or distributed execution',
                'priority': 'high'
            })
        
        return recommendations
