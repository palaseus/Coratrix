"""
Performance Monitor - Real-Time Performance Visualization
========================================================

The Performance Monitor provides real-time visualization of quantum
circuit execution performance metrics and system statistics.

This is the GOD-TIER performance monitoring system that provides
comprehensive insights into quantum circuit execution.
"""

import time
import logging
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    NETWORK_LATENCY = "network_latency"
    ENTANGLEMENT_ENTROPY = "entanglement_entropy"
    GATE_COUNT = "gate_count"
    CIRCUIT_DEPTH = "circuit_depth"

@dataclass
class PerformanceMetric:
    """A performance metric."""
    metric_type: MetricType
    value: float
    timestamp: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricsRenderer:
    """Renderer for performance metrics visualization."""
    
    def __init__(self):
        """Initialize the metrics renderer."""
        self.render_stats = {
            'total_metrics_rendered': 0,
            'average_render_time': 0.0,
            'charts_generated': 0,
            'real_time_updates': 0
        }
        
        logger.info("🎨 Metrics Renderer initialized - Performance visualization active")
    
    async def render_performance_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Render performance metrics visualization."""
        logger.info("🎨 Rendering performance metrics")
        
        start_time = time.time()
        
        try:
            # Generate time series charts
            time_series_charts = await self._generate_time_series_charts(metrics)
            
            # Generate real-time gauges
            real_time_gauges = await self._generate_real_time_gauges(metrics)
            
            # Generate performance heatmaps
            performance_heatmaps = await self._generate_performance_heatmaps(metrics)
            
            # Generate system overview
            system_overview = await self._generate_system_overview(metrics)
            
            # Create render output
            render_output = {
                'time_series_charts': time_series_charts,
                'real_time_gauges': real_time_gauges,
                'performance_heatmaps': performance_heatmaps,
                'system_overview': system_overview,
                'metadata': {
                    'render_time': time.time() - start_time,
                    'metrics_count': len(metrics),
                    'time_range': self._calculate_time_range(metrics)
                }
            }
            
            # Update statistics
            self._update_render_stats(time.time() - start_time)
            
            logger.info(f"✅ Performance metrics rendered in {time.time() - start_time:.4f}s")
            return render_output
            
        except Exception as e:
            logger.error(f"❌ Performance metrics rendering failed: {e}")
            raise
    
    async def _generate_time_series_charts(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Generate time series charts for metrics."""
        charts = {}
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type.value].append(metric)
        
        # Generate chart for each metric type
        for metric_type, metric_list in metrics_by_type.items():
            chart_data = {
                'type': 'line',
                'data': [
                    {
                        'timestamp': metric.timestamp,
                        'value': metric.value,
                        'unit': metric.unit
                    }
                    for metric in metric_list
                ],
                'title': f'{metric_type.replace("_", " ").title()} Over Time',
                'x_axis': 'Time',
                'y_axis': metric_list[0].unit if metric_list else 'Value'
            }
            charts[metric_type] = chart_data
        
        return charts
    
    async def _generate_real_time_gauges(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Generate real-time gauges for current metrics."""
        gauges = {}
        
        # Get latest metrics
        latest_metrics = self._get_latest_metrics(metrics)
        
        for metric_type, metric in latest_metrics.items():
            gauge_data = {
                'type': 'gauge',
                'value': metric.value,
                'unit': metric.unit,
                'min_value': 0.0,
                'max_value': self._get_max_value_for_metric_type(metric_type),
                'color': self._get_color_for_metric_value(metric_type, metric.value),
                'title': f'{metric_type.replace("_", " ").title()}',
                'status': self._get_status_for_metric_value(metric_type, metric.value)
            }
            gauges[metric_type] = gauge_data
        
        return gauges
    
    async def _generate_performance_heatmaps(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Generate performance heatmaps."""
        heatmaps = {}
        
        # Group metrics by time windows
        time_windows = self._create_time_windows(metrics)
        
        for window_name, window_metrics in time_windows.items():
            heatmap_data = {
                'type': 'heatmap',
                'data': self._create_heatmap_data(window_metrics),
                'title': f'Performance Heatmap - {window_name}',
                'color_scale': 'viridis'
            }
            heatmaps[window_name] = heatmap_data
        
        return heatmaps
    
    async def _generate_system_overview(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Generate system overview dashboard."""
        latest_metrics = self._get_latest_metrics(metrics)
        
        overview = {
            'system_status': self._calculate_system_status(latest_metrics),
            'performance_score': self._calculate_performance_score(latest_metrics),
            'resource_utilization': self._calculate_resource_utilization(latest_metrics),
            'alerts': self._generate_alerts(latest_metrics),
            'recommendations': self._generate_recommendations(latest_metrics)
        }
        
        return overview
    
    def _get_latest_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, PerformanceMetric]:
        """Get the latest metric for each type."""
        latest_metrics = {}
        
        for metric in metrics:
            metric_type = metric.metric_type.value
            if metric_type not in latest_metrics or metric.timestamp > latest_metrics[metric_type].timestamp:
                latest_metrics[metric_type] = metric
        
        return latest_metrics
    
    def _get_max_value_for_metric_type(self, metric_type: str) -> float:
        """Get maximum value for a metric type."""
        max_values = {
            'execution_time': 10.0,  # seconds
            'memory_usage': 1000.0,  # MB
            'cpu_utilization': 100.0,  # percentage
            'gpu_utilization': 100.0,  # percentage
            'network_latency': 1000.0,  # ms
            'entanglement_entropy': 1.0,  # normalized
            'gate_count': 1000.0,  # count
            'circuit_depth': 100.0  # depth
        }
        
        return max_values.get(metric_type, 100.0)
    
    def _get_color_for_metric_value(self, metric_type: str, value: float) -> str:
        """Get color for a metric value."""
        max_value = self._get_max_value_for_metric_type(metric_type)
        normalized_value = value / max_value
        
        if normalized_value < 0.3:
            return 'green'
        elif normalized_value < 0.7:
            return 'yellow'
        else:
            return 'red'
    
    def _get_status_for_metric_value(self, metric_type: str, value: float) -> str:
        """Get status for a metric value."""
        max_value = self._get_max_value_for_metric_type(metric_type)
        normalized_value = value / max_value
        
        if normalized_value < 0.3:
            return 'good'
        elif normalized_value < 0.7:
            return 'warning'
        else:
            return 'critical'
    
    def _create_time_windows(self, metrics: List[PerformanceMetric]) -> Dict[str, List[PerformanceMetric]]:
        """Create time windows for heatmap generation."""
        if not metrics:
            return {}
        
        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Create time windows
        current_time = time.time()
        windows = {
            'last_minute': [],
            'last_5_minutes': [],
            'last_hour': []
        }
        
        for metric in sorted_metrics:
            time_diff = current_time - metric.timestamp
            
            if time_diff <= 60:  # Last minute
                windows['last_minute'].append(metric)
            if time_diff <= 300:  # Last 5 minutes
                windows['last_5_minutes'].append(metric)
            if time_diff <= 3600:  # Last hour
                windows['last_hour'].append(metric)
        
        return windows
    
    def _create_heatmap_data(self, metrics: List[PerformanceMetric]) -> List[List[float]]:
        """Create heatmap data from metrics."""
        if not metrics:
            return []
        
        # Group metrics by type and time
        heatmap_data = []
        
        # Simplified heatmap data generation
        for metric in metrics:
            heatmap_data.append([
                metric.timestamp,
                hash(metric.metric_type.value) % 100,  # Simplified metric type hash
                metric.value
            ])
        
        return heatmap_data
    
    def _calculate_system_status(self, latest_metrics: Dict[str, PerformanceMetric]) -> str:
        """Calculate overall system status."""
        if not latest_metrics:
            return 'unknown'
        
        critical_count = 0
        warning_count = 0
        
        for metric in latest_metrics.values():
            status = self._get_status_for_metric_value(metric.metric_type.value, metric.value)
            if status == 'critical':
                critical_count += 1
            elif status == 'warning':
                warning_count += 1
        
        if critical_count > 0:
            return 'critical'
        elif warning_count > 0:
            return 'warning'
        else:
            return 'good'
    
    def _calculate_performance_score(self, latest_metrics: Dict[str, PerformanceMetric]) -> float:
        """Calculate overall performance score."""
        if not latest_metrics:
            return 0.0
        
        scores = []
        for metric in latest_metrics.values():
            max_value = self._get_max_value_for_metric_type(metric.metric_type.value)
            normalized_value = metric.value / max_value
            score = 1.0 - min(normalized_value, 1.0)  # Invert so lower is better
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_resource_utilization(self, latest_metrics: Dict[str, PerformanceMetric]) -> Dict[str, float]:
        """Calculate resource utilization."""
        utilization = {}
        
        for metric_type, metric in latest_metrics.items():
            if 'utilization' in metric_type or 'usage' in metric_type:
                utilization[metric_type] = metric.value
        
        return utilization
    
    def _generate_alerts(self, latest_metrics: Dict[str, PerformanceMetric]) -> List[Dict[str, Any]]:
        """Generate performance alerts."""
        alerts = []
        
        for metric_type, metric in latest_metrics.items():
            status = self._get_status_for_metric_value(metric_type, metric.value)
            
            if status == 'critical':
                alerts.append({
                    'type': 'critical',
                    'message': f'{metric_type.replace("_", " ").title()} is critical: {metric.value} {metric.unit}',
                    'timestamp': metric.timestamp
                })
            elif status == 'warning':
                alerts.append({
                    'type': 'warning',
                    'message': f'{metric_type.replace("_", " ").title()} is high: {metric.value} {metric.unit}',
                    'timestamp': metric.timestamp
                })
        
        return alerts
    
    def _generate_recommendations(self, latest_metrics: Dict[str, PerformanceMetric]) -> List[Dict[str, Any]]:
        """Generate performance recommendations."""
        recommendations = []
        
        for metric_type, metric in latest_metrics.items():
            status = self._get_status_for_metric_value(metric_type, metric.value)
            
            if status == 'critical':
                recommendations.append({
                    'type': 'critical',
                    'message': f'Optimize {metric_type.replace("_", " ").lower()}',
                    'recommendation': f'Consider reducing {metric_type.replace("_", " ").lower()} for better performance'
                })
            elif status == 'warning':
                recommendations.append({
                    'type': 'warning',
                    'message': f'Monitor {metric_type.replace("_", " ").lower()}',
                    'recommendation': f'Keep an eye on {metric_type.replace("_", " ").lower()} trends'
                })
        
        return recommendations
    
    def _calculate_time_range(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """Calculate time range for metrics."""
        if not metrics:
            return {'min': 0, 'max': 0, 'duration': 0}
        
        timestamps = [metric.timestamp for metric in metrics]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        return {
            'min': min_time,
            'max': max_time,
            'duration': max_time - min_time
        }
    
    def _update_render_stats(self, render_time: float):
        """Update rendering statistics."""
        self.render_stats['total_metrics_rendered'] += 1
        
        # Update average render time
        total = self.render_stats['total_metrics_rendered']
        current_avg = self.render_stats['average_render_time']
        self.render_stats['average_render_time'] = (current_avg * (total - 1) + render_time) / total
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """Get metrics renderer statistics."""
        return {
            'render_stats': self.render_stats
        }

class VizPerformanceMonitor:
    """
    Performance Monitor for Real-Time Performance Visualization.
    
    This is the GOD-TIER performance monitoring system that provides
    comprehensive insights into quantum circuit execution.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics_renderer = MetricsRenderer()
        self.performance_metrics: deque = deque(maxlen=10000)
        
        # Performance statistics
        self.perf_stats = {
            'total_metrics_collected': 0,
            'average_metric_value': 0.0,
            'performance_trends': {},
            'system_health_score': 1.0
        }
        
        logger.info("🎨 Performance Monitor initialized - Real-time monitoring active")
    
    async def collect_metric(self, metric_type: MetricType, value: float, 
                           unit: str, metadata: Dict[str, Any] = None):
        """Collect a performance metric."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            unit=unit,
            metadata=metadata or {}
        )
        
        self.performance_metrics.append(metric)
        self.perf_stats['total_metrics_collected'] += 1
        
        # Update average metric value
        self._update_average_metric_value(value)
        
        logger.info(f"📊 Collected metric: {metric_type.value} = {value} {unit}")
    
    async def get_visualization_metrics(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get visualization metrics for a circuit."""
        # Generate simulated metrics for the circuit
        metrics = await self._generate_circuit_metrics(circuit_data)
        
        # Render performance visualization
        visualization = await self.metrics_renderer.render_performance_metrics(metrics)
        
        return visualization
    
    async def _generate_circuit_metrics(self, circuit_data: Dict[str, Any]) -> List[PerformanceMetric]:
        """Generate performance metrics for a circuit."""
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        metrics = []
        current_time = time.time()
        
        # Generate execution time metric
        execution_time = len(gates) * 0.001  # Simulated execution time
        metrics.append(PerformanceMetric(
            metric_type=MetricType.EXECUTION_TIME,
            value=execution_time,
            timestamp=current_time,
            unit='seconds'
        ))
        
        # Generate memory usage metric
        memory_usage = (2 ** num_qubits) * 16 / (1024 * 1024)  # MB
        metrics.append(PerformanceMetric(
            metric_type=MetricType.MEMORY_USAGE,
            value=memory_usage,
            timestamp=current_time,
            unit='MB'
        ))
        
        # Generate CPU utilization metric
        cpu_utilization = min(100.0, num_qubits * 5.0)  # Simulated CPU usage
        metrics.append(PerformanceMetric(
            metric_type=MetricType.CPU_UTILIZATION,
            value=cpu_utilization,
            timestamp=current_time,
            unit='%'
        ))
        
        # Generate GPU utilization metric
        gpu_utilization = min(100.0, num_qubits * 3.0)  # Simulated GPU usage
        metrics.append(PerformanceMetric(
            metric_type=MetricType.GPU_UTILIZATION,
            value=gpu_utilization,
            timestamp=current_time,
            unit='%'
        ))
        
        # Generate entanglement entropy metric
        entanglement_entropy = min(1.0, len(gates) * 0.1)  # Simulated entanglement
        metrics.append(PerformanceMetric(
            metric_type=MetricType.ENTANGLEMENT_ENTROPY,
            value=entanglement_entropy,
            timestamp=current_time,
            unit='normalized'
        ))
        
        # Generate gate count metric
        metrics.append(PerformanceMetric(
            metric_type=MetricType.GATE_COUNT,
            value=len(gates),
            timestamp=current_time,
            unit='count'
        ))
        
        # Generate circuit depth metric
        circuit_depth = len(gates)  # Simplified depth calculation
        metrics.append(PerformanceMetric(
            metric_type=MetricType.CIRCUIT_DEPTH,
            value=circuit_depth,
            timestamp=current_time,
            unit='depth'
        ))
        
        return metrics
    
    def _update_average_metric_value(self, value: float):
        """Update average metric value."""
        total = self.perf_stats['total_metrics_collected']
        current_avg = self.perf_stats['average_metric_value']
        self.perf_stats['average_metric_value'] = (current_avg * (total - 1) + value) / total
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance monitor statistics."""
        return {
            'perf_stats': self.perf_stats,
            'metrics_renderer_stats': self.metrics_renderer.get_render_statistics(),
            'metrics_history_size': len(self.performance_metrics)
        }
    
    def get_performance_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get performance recommendations."""
        recommendations = []
        
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        # Performance recommendations
        if len(gates) > 100:
            recommendations.append({
                'type': 'performance',
                'message': f'Large circuit ({len(gates)} gates) detected',
                'recommendation': 'Consider circuit optimization for better performance',
                'priority': 'medium'
            })
        
        # Memory recommendations
        if num_qubits > 15:
            recommendations.append({
                'type': 'memory',
                'message': f'Large qubit count ({num_qubits}) detected',
                'recommendation': 'Consider using sparse representation for memory efficiency',
                'priority': 'high'
            })
        
        # System health recommendations
        if self.perf_stats['system_health_score'] < 0.8:
            recommendations.append({
                'type': 'system_health',
                'message': f'Low system health score ({self.perf_stats["system_health_score"]:.2f})',
                'recommendation': 'Monitor system resources and optimize circuit execution',
                'priority': 'high'
            })
        
        return recommendations
