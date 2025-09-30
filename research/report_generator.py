"""
Comprehensive report generator for quantum exploration.

This module provides detailed reporting capabilities including
JSON output, interactive CLI summaries, and structured analysis.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from research.visualization_engine import VisualizationEngine


class ReportGenerator:
    """
    Comprehensive report generator for quantum exploration.
    
    Generates detailed reports including JSON output,
    interactive CLI summaries, and structured analysis.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.visualization_engine = VisualizationEngine()
        self.report_data = {}
        
    def generate_comprehensive_report(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report from exploration data.
        
        Args:
            exploration_data: Complete exploration data
        
        Returns:
            Comprehensive report
        """
        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)
        
        # Generate report sections
        report = {
            'report_metadata': self._generate_report_metadata(),
            'executive_summary': self._generate_executive_summary(exploration_data),
            'algorithm_analysis': self._generate_algorithm_analysis(exploration_data),
            'entanglement_analysis': self._generate_entanglement_analysis(exploration_data),
            'visualization_summary': self._generate_visualization_summary(exploration_data),
            'performance_metrics': self._generate_performance_metrics(exploration_data),
            'recommendations': self._generate_recommendations(exploration_data),
            'technical_details': self._generate_technical_details(exploration_data)
        }
        
        # Generate visualizations
        visualization_summary = self.visualization_engine.generate_comprehensive_visualization(exploration_data)
        report['visualizations'] = visualization_summary
        
        # Generate ASCII summary
        ascii_summary = self.visualization_engine.generate_ascii_summary(exploration_data)
        report['ascii_summary'] = ascii_summary
        
        # Store report data
        self.report_data = report
        
        print("✅ Comprehensive report generated")
        
        return report
    
    def _generate_report_metadata(self) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'generation_time': datetime.now().isoformat(),
            'report_version': '2.0.0',
            'generator': 'Coratrix Research-Grade Quantum Explorer',
            'timestamp': time.time()
        }
    
    def _generate_executive_summary(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        # Get metadata from exploration_data directly (not nested under 'exploration_metadata')
        num_qubits = exploration_data.get('num_qubits', 0)
        use_gpu = exploration_data.get('use_gpu', False)
        use_sparse = exploration_data.get('use_sparse', False)
        algorithms = exploration_data.get('algorithms', {})
        
        # Calculate summary statistics
        total_algorithms = len(algorithms)
        successful_algorithms = 0
        entanglement_entropies = []
        entangled_count = 0
        
        for algo_name, algo_data in algorithms.items():
            # Check if algorithm was successful based on various criteria
            is_successful = False
            
            # Check for execution time (basic success indicator)
            if 'execution_time' in algo_data and algo_data['execution_time'] > 0:
                is_successful = True
            
            # Check for success probability (Grover's algorithm)
            if 'success_probability' in algo_data:
                success_prob = algo_data['success_probability']
                if success_prob > 0.9:  # Threshold for high success
                    is_successful = True
            
            # Check for fidelity (teleportation)
            if 'fidelity' in algo_data:
                fidelity = algo_data['fidelity']
                if fidelity > 0.9:  # Threshold for high fidelity
                    is_successful = True
            
            # Check for proper state preparation (GHZ, W states)
            if 'entanglement_analysis' in algo_data:
                ent_analysis = algo_data['entanglement_analysis']
                if ent_analysis.get('is_entangled', False):
                    is_successful = True
            
            if is_successful:
                successful_algorithms += 1
            
            # Collect entanglement data
            if 'entanglement_analysis' in algo_data:
                ent_analysis = algo_data['entanglement_analysis']
                entanglement_entropies.append(ent_analysis.get('entanglement_entropy', 0.0))
                if ent_analysis.get('is_entangled', False):
                    entangled_count += 1
        
        success_rate = successful_algorithms / total_algorithms if total_algorithms > 0 else 0.0
        avg_entanglement_entropy = sum(entanglement_entropies) / len(entanglement_entropies) if entanglement_entropies else 0.0
        
        return {
            'total_algorithms': total_algorithms,
            'successful_algorithms': successful_algorithms,
            'success_rate': success_rate,
            'num_qubits': num_qubits,
            'use_gpu': use_gpu,
            'use_sparse': use_sparse,
            'entangled_states': entangled_count,
            'average_entanglement_entropy': avg_entanglement_entropy,
            'max_entanglement_entropy': max(entanglement_entropies) if entanglement_entropies else 0.0
        }
    
    def _generate_algorithm_analysis(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed algorithm analysis."""
        algorithms = exploration_data.get('algorithms', {})
        analysis = {}
        
        for algo_name, algo_data in algorithms.items():
            algo_analysis = {
                'algorithm_name': algo_data.get('algorithm', algo_name),
                'execution_time': algo_data.get('execution_time', 0.0),
                'final_state': algo_data.get('final_state', 'N/A'),
                'probabilities': algo_data.get('probabilities', []),
                'entanglement_analysis': algo_data.get('entanglement_analysis', {}),
                'success_metrics': self._extract_algorithm_success_metrics(algo_data)
            }
            
            analysis[algo_name] = algo_analysis
        
        return analysis
    
    def _extract_algorithm_success_metrics(self, algo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract success metrics from algorithm data."""
        metrics = {}
        
        # Extract various success metrics
        if 'success_probability' in algo_data:
            metrics['success_probability'] = algo_data['success_probability']
        
        if 'fidelity' in algo_data:
            metrics['fidelity'] = algo_data['fidelity']
        
        if 'measurement_results' in algo_data:
            metrics['measurement_results'] = algo_data['measurement_results']
        
        if 'result' in algo_data:
            metrics['result'] = algo_data['result']
        
        return metrics
    
    def _generate_entanglement_analysis(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive entanglement analysis."""
        algorithms = exploration_data.get('algorithms', {})
        
        # Collect entanglement data
        entanglement_data = []
        bell_states = []
        entangled_states = []
        
        for algo_name, algo_data in algorithms.items():
            if 'entanglement_analysis' in algo_data:
                ent_analysis = algo_data['entanglement_analysis']
                entanglement_data.append({
                    'algorithm': algo_name,
                    'entanglement_entropy': ent_analysis.get('entanglement_entropy', 0.0),
                    'concurrence': ent_analysis.get('concurrence', 0.0),
                    'negativity': ent_analysis.get('negativity', 0.0),
                    'is_entangled': ent_analysis.get('is_entangled', False),
                    'bell_state_type': ent_analysis.get('bell_state_type'),
                    'entanglement_rank': ent_analysis.get('entanglement_rank', 0)
                })
                
                if ent_analysis.get('is_bell_state', False):
                    bell_states.append({
                        'algorithm': algo_name,
                        'bell_state_type': ent_analysis.get('bell_state_type')
                    })
                
                if ent_analysis.get('is_entangled', False):
                    entangled_states.append(algo_name)
        
        # Calculate statistics
        entropies = [data['entanglement_entropy'] for data in entanglement_data]
        concurrences = [data['concurrence'] for data in entanglement_data]
        negativities = [data['negativity'] for data in entanglement_data]
        
        return {
            'entanglement_data': entanglement_data,
            'bell_states': bell_states,
            'entangled_states': entangled_states,
            'statistics': {
                'total_measurements': len(entanglement_data),
                'entangled_count': len(entangled_states),
                'bell_state_count': len(bell_states),
                'average_entropy': sum(entropies) / len(entropies) if entropies else 0.0,
                'max_entropy': max(entropies) if entropies else 0.0,
                'average_concurrence': sum(concurrences) / len(concurrences) if concurrences else 0.0,
                'max_concurrence': max(concurrences) if concurrences else 0.0,
                'average_negativity': sum(negativities) / len(negativities) if negativities else 0.0,
                'max_negativity': max(negativities) if negativities else 0.0
            }
        }
    
    def _generate_visualization_summary(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization summary."""
        visualizations = exploration_data.get('visualizations', {})
        
        return {
            'total_visualizations': len(visualizations),
            'visualization_types': list(visualizations.keys()),
            'visualization_data': visualizations
        }
    
    def _generate_performance_metrics(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance metrics."""
        algorithms = exploration_data.get('algorithms', {})
        
        # Calculate performance metrics
        execution_times = [algo.get('execution_time', 0.0) for algo in algorithms.values() 
                          if 'execution_time' in algo]
        
        total_execution_time = sum(execution_times)
        average_execution_time = total_execution_time / len(execution_times) if execution_times else 0.0
        
        # Get metadata from exploration_data directly
        num_qubits = exploration_data.get('num_qubits', 0)
        use_gpu = exploration_data.get('use_gpu', False)
        use_sparse = exploration_data.get('use_sparse', False)
        state_dimension = 2 ** num_qubits if num_qubits > 0 else 0
        
        return {
            'total_execution_time': total_execution_time,
            'average_execution_time': average_execution_time,
            'num_qubits': num_qubits,
            'state_dimension': state_dimension,
            'memory_estimate': state_dimension * 16,  # 16 bytes per complex number
            'use_gpu': use_gpu,
            'use_sparse': use_sparse
        }
    
    def _generate_recommendations(self, exploration_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on exploration results."""
        recommendations = []
        
        # Get metadata from exploration_data directly
        num_qubits = exploration_data.get('num_qubits', 0)
        use_gpu = exploration_data.get('use_gpu', False)
        use_sparse = exploration_data.get('use_sparse', False)
        algorithms = exploration_data.get('algorithms', {})
        
        # GPU recommendations
        if num_qubits >= 8 and not use_gpu:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'recommendation': 'Enable GPU acceleration for large quantum systems',
                'reason': f'System has {num_qubits} qubits, GPU acceleration would improve performance'
            })
        
        # Sparse matrix recommendations
        if num_qubits >= 10 and not use_sparse:
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'recommendation': 'Use sparse matrix representation for very large systems',
                'reason': f'System has {num_qubits} qubits, sparse matrices would reduce memory usage'
            })
        
        # Entanglement recommendations
        entanglement_data = []
        for algo_data in algorithms.values():
            if 'entanglement_analysis' in algo_data:
                ent_analysis = algo_data['entanglement_analysis']
                entanglement_data.append(ent_analysis.get('entanglement_entropy', 0.0))
        
        if entanglement_data:
            avg_entanglement = sum(entanglement_data) / len(entanglement_data)
            if avg_entanglement > 0.8:
                recommendations.append({
                    'type': 'algorithm',
                    'priority': 'medium',
                    'recommendation': 'Consider entanglement-based quantum algorithms',
                    'reason': f'High average entanglement entropy ({avg_entanglement:.4f}) detected'
                })
        
        # Success rate recommendations - use the same logic as executive summary
        successful_algorithms = 0
        for algo_name, algo_data in algorithms.items():
            is_successful = False
            
            # Check for execution time (basic success indicator)
            if 'execution_time' in algo_data and algo_data['execution_time'] > 0:
                is_successful = True
            
            # Check for success probability (Grover's algorithm)
            if 'success_probability' in algo_data:
                success_prob = algo_data['success_probability']
                if success_prob > 0.9:  # Threshold for high success
                    is_successful = True
            
            # Check for fidelity (teleportation)
            if 'fidelity' in algo_data:
                fidelity = algo_data['fidelity']
                if fidelity > 0.9:  # Threshold for high fidelity
                    is_successful = True
            
            # Check for proper state preparation (GHZ, W states)
            if 'entanglement_analysis' in algo_data:
                ent_analysis = algo_data['entanglement_analysis']
                if ent_analysis.get('is_entangled', False):
                    is_successful = True
            
            if is_successful:
                successful_algorithms += 1
        
        total_algorithms = len(algorithms)
        success_rate = successful_algorithms / total_algorithms if total_algorithms > 0 else 0.0
        
        if success_rate < 0.8:
            recommendations.append({
                'type': 'reliability',
                'priority': 'high',
                'recommendation': 'Investigate algorithm execution failures',
                'reason': f'Success rate is {success_rate:.2%}, below expected threshold'
            })
        
        return recommendations
    
    def _generate_technical_details(self, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical details section."""
        # Get metadata from exploration_data directly
        num_qubits = exploration_data.get('num_qubits', 0)
        use_gpu = exploration_data.get('use_gpu', False)
        use_sparse = exploration_data.get('use_sparse', False)
        algorithms = exploration_data.get('algorithms', {})
        
        # Calculate successful algorithms using the same logic as executive summary
        successful_algorithms = 0
        for algo_name, algo_data in algorithms.items():
            is_successful = False
            
            # Check for execution time (basic success indicator)
            if 'execution_time' in algo_data and algo_data['execution_time'] > 0:
                is_successful = True
            
            # Check for success probability (Grover's algorithm)
            if 'success_probability' in algo_data:
                success_prob = algo_data['success_probability']
                if success_prob > 0.9:  # Threshold for high success
                    is_successful = True
            
            # Check for fidelity (teleportation)
            if 'fidelity' in algo_data:
                fidelity = algo_data['fidelity']
                if fidelity > 0.9:  # Threshold for high fidelity
                    is_successful = True
            
            # Check for proper state preparation (GHZ, W states)
            if 'entanglement_analysis' in algo_data:
                ent_analysis = algo_data['entanglement_analysis']
                if ent_analysis.get('is_entangled', False):
                    is_successful = True
            
            if is_successful:
                successful_algorithms += 1
        
        return {
            'system_configuration': {
                'num_qubits': num_qubits,
                'use_gpu': use_gpu,
                'use_sparse': use_sparse,
                'state_dimension': 2 ** num_qubits if num_qubits > 0 else 0
            },
            'algorithm_details': {
                algo_name: {
                    'execution_time': algo_data.get('execution_time', 0.0),
                    'final_state': algo_data.get('final_state', 'N/A'),
                    'probabilities_count': len(algo_data.get('probabilities', [])),
                    'entanglement_entropy': algo_data.get('entanglement_analysis', {}).get('entanglement_entropy', 0.0)
                }
                for algo_name, algo_data in algorithms.items()
            },
            'exploration_statistics': {
                'total_algorithms': len(algorithms),
                'successful_algorithms': successful_algorithms,
                'total_execution_time': sum(algo.get('execution_time', 0.0) 
                                          for algo in algorithms.values()),
                'average_entanglement_entropy': sum(
                    algo.get('entanglement_analysis', {}).get('entanglement_entropy', 0.0)
                    for algo in algorithms.values()
                ) / len(algorithms) if algorithms else 0.0
            }
        }
    
    def save_report_to_file(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save report to JSON file."""
        import os
        
        # Create reports/exploration directory if it doesn't exist
        reports_dir = "reports/exploration"
        os.makedirs(reports_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"quantum_exploration_report_{timestamp}.json"
        
        # Ensure filename is in the reports/exploration directory
        if not filename.startswith(reports_dir):
            filename = os.path.join(reports_dir, filename)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {filename}")
        return filename
    
    def print_interactive_summary(self, report: Dict[str, Any]):
        """Print interactive CLI summary."""
        print("\n" + "="*80)
        print("🔬 QUANTUM EXPLORATION INTERACTIVE SUMMARY")
        print("="*80)
        
        # Executive summary
        exec_summary = report.get('executive_summary', {})
        print(f"\n📊 EXECUTIVE SUMMARY:")
        print(f"   Total Algorithms: {exec_summary.get('total_algorithms', 0)}")
        print(f"   Successful: {exec_summary.get('successful_algorithms', 0)}")
        print(f"   Success Rate: {exec_summary.get('success_rate', 0.0):.2%}")
        print(f"   Qubits: {exec_summary.get('num_qubits', 0)}")
        print(f"   Entangled States: {exec_summary.get('entangled_states', 0)}")
        print(f"   Avg Entanglement Entropy: {exec_summary.get('average_entanglement_entropy', 0.0):.4f}")
        
        # Algorithm results
        print(f"\n🔬 ALGORITHM RESULTS:")
        algo_analysis = report.get('algorithm_analysis', {})
        for algo_name, algo_data in algo_analysis.items():
            print(f"   {algo_name}:")
            print(f"     Execution Time: {algo_data.get('execution_time', 0.0):.4f}s")
            print(f"     Final State: {algo_data.get('final_state', 'N/A')}")
            ent_analysis = algo_data.get('entanglement_analysis', {})
            print(f"     Entangled: {ent_analysis.get('is_entangled', False)}")
            print(f"     Entropy: {ent_analysis.get('entanglement_entropy', 0.0):.4f}")
        
        # Entanglement analysis
        print(f"\n🔗 ENTANGLEMENT ANALYSIS:")
        ent_analysis = report.get('entanglement_analysis', {})
        stats = ent_analysis.get('statistics', {})
        print(f"   Total Measurements: {stats.get('total_measurements', 0)}")
        print(f"   Entangled States: {stats.get('entangled_count', 0)}")
        print(f"   Bell States: {stats.get('bell_state_count', 0)}")
        print(f"   Average Entropy: {stats.get('average_entropy', 0.0):.4f}")
        print(f"   Max Entropy: {stats.get('max_entropy', 0.0):.4f}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        perf_metrics = report.get('performance_metrics', {})
        print(f"   Total Execution Time: {perf_metrics.get('total_execution_time', 0.0):.4f}s")
        print(f"   Average Execution Time: {perf_metrics.get('average_execution_time', 0.0):.4f}s")
        print(f"   State Dimension: {perf_metrics.get('state_dimension', 0)}")
        print(f"   Memory Estimate: {perf_metrics.get('memory_estimate', 0)} bytes")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = report.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec.get('priority', 'medium').upper()}] {rec.get('recommendation', 'N/A')}")
            print(f"      Reason: {rec.get('reason', 'N/A')}")
        
        print("\n" + "="*80)
        print("✅ QUANTUM EXPLORATION COMPLETE")
        print("="*80)
