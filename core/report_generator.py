"""
Publication-ready report generator for Coratrix.

This module automatically generates publication-ready artifacts including
figures, reports, and documentation from quantum computing experiments.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess
import hashlib

# Optional dependencies for enhanced functionality
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


@dataclass
class PublicationMetadata:
    """Metadata for publication artifacts."""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    experiment_type: str
    timestamp: str
    version: str
    reproducibility_hash: str
    system_info: Dict[str, Any]
    results_summary: Dict[str, Any]


@dataclass
class FigureSpec:
    """Specification for generating figures."""
    figure_type: str
    title: str
    data: Dict[str, Any]
    style: Dict[str, Any]
    output_format: str
    filename: str


class PublicationReportGenerator:
    """Generator for publication-ready reports and artifacts."""
    
    def __init__(self, output_dir: str = "publication_artifacts"):
        self.output_dir = output_dir
        self.figures = []
        self.metadata = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    def generate_comprehensive_report(self, experiment_data: Dict[str, Any], 
                                    metadata: PublicationMetadata) -> Dict[str, Any]:
        """Generate comprehensive publication report."""
        self.metadata = metadata
        
        # Generate figures
        figures = self._generate_figures(experiment_data)
        
        # Generate reports
        reports = self._generate_reports(experiment_data, figures)
        
        # Generate data files
        data_files = self._generate_data_files(experiment_data)
        
        # Generate bibliography
        bibliography = self._generate_bibliography()
        
        # Create summary
        summary = {
            'metadata': asdict(metadata),
            'figures': figures,
            'reports': reports,
            'data_files': data_files,
            'bibliography': bibliography,
            'generation_timestamp': datetime.now().isoformat(),
            'output_directory': self.output_dir
        }
        
        # Save summary
        with open(os.path.join(self.output_dir, "report_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _generate_figures(self, experiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate publication-ready figures."""
        figures = []
        
        # Circuit diagrams
        if 'circuit_data' in experiment_data:
            circuit_fig = self._create_circuit_diagram(experiment_data['circuit_data'])
            figures.append(circuit_fig)
        
        # Probability heatmaps
        if 'probability_data' in experiment_data:
            heatmap_fig = self._create_probability_heatmap(experiment_data['probability_data'])
            figures.append(heatmap_fig)
        
        # Entanglement graphs
        if 'entanglement_data' in experiment_data:
            entanglement_fig = self._create_entanglement_graph(experiment_data['entanglement_data'])
            figures.append(entanglement_fig)
        
        # Optimization convergence plots
        if 'optimization_data' in experiment_data:
            convergence_fig = self._create_convergence_plot(experiment_data['optimization_data'])
            figures.append(convergence_fig)
        
        # Interference heatmaps
        if 'interference_data' in experiment_data:
            interference_fig = self._create_interference_heatmap(experiment_data['interference_data'])
            figures.append(interference_fig)
        
        # Performance benchmarks
        if 'benchmark_data' in experiment_data:
            benchmark_fig = self._create_benchmark_plot(experiment_data['benchmark_data'])
            figures.append(benchmark_fig)
        
        return figures
    
    def _create_circuit_diagram(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create circuit diagram figure."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract circuit information
        num_qubits = circuit_data.get('num_qubits', 2)
        gates = circuit_data.get('gates', [])
        
        # Draw qubit lines
        for i in range(num_qubits):
            ax.plot([0, 10], [i, i], 'k-', linewidth=2)
            ax.text(-0.5, i, f'q{i}', fontsize=12, ha='right', va='center')
        
        # Draw gates
        gate_x = 1
        for gate_info in gates:
            gate_name = gate_info.get('name', 'G')
            qubit_indices = gate_info.get('qubits', [0])
            
            if len(qubit_indices) == 1:
                # Single-qubit gate
                qubit = qubit_indices[0]
                ax.add_patch(patches.Rectangle((gate_x-0.3, qubit-0.3), 0.6, 0.6, 
                                             facecolor='lightblue', edgecolor='black'))
                ax.text(gate_x, qubit, gate_name, ha='center', va='center', fontsize=10)
            elif len(qubit_indices) == 2:
                # Two-qubit gate
                control, target = qubit_indices
                # Control qubit
                ax.add_patch(patches.Circle((gate_x, control), 0.2, 
                                         facecolor='black', edgecolor='black'))
                # Target qubit
                ax.add_patch(patches.Rectangle((gate_x-0.3, target-0.3), 0.6, 0.6,
                                             facecolor='lightgreen', edgecolor='black'))
                ax.text(gate_x, target, gate_name, ha='center', va='center', fontsize=10)
                # Connection line
                ax.plot([gate_x, gate_x], [control, target], 'k-', linewidth=1)
            
            gate_x += 1
        
        ax.set_xlim(-1, gate_x + 1)
        ax.set_ylim(-0.5, num_qubits - 0.5)
        ax.set_title('Quantum Circuit Diagram', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Qubits', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = os.path.join(self.output_dir, "figures", "circuit_diagram.svg")
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'type': 'circuit_diagram',
            'filename': filename,
            'title': 'Quantum Circuit Diagram',
            'description': 'Visual representation of the quantum circuit'
        }
    
    def _create_probability_heatmap(self, probability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create probability heatmap figure."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract probability data
        probabilities = probability_data.get('probabilities', [])
        num_qubits = probability_data.get('num_qubits', 2)
        
        # Reshape probabilities for heatmap
        if len(probabilities) > 0:
            # Create 2D representation
            size = int(np.sqrt(len(probabilities)))
            if size * size != len(probabilities):
                size = 2 ** num_qubits
                prob_matrix = np.array(probabilities).reshape(1, -1)
            else:
                prob_matrix = np.array(probabilities).reshape(size, size)
            
            # Create heatmap
            im = ax.imshow(prob_matrix, cmap='viridis', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability', fontsize=12)
            
            # Add labels
            ax.set_title('Quantum State Probability Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('State Index', fontsize=12)
            ax.set_ylabel('State Index', fontsize=12)
            
            # Add state labels
            state_labels = [format(i, f'0{num_qubits}b') for i in range(len(probabilities))]
            if len(state_labels) <= 16:  # Only show labels for small systems
                ax.set_xticks(range(len(state_labels)))
                ax.set_yticks(range(len(state_labels)))
                ax.set_xticklabels(state_labels, rotation=45)
                ax.set_yticklabels(state_labels)
        
        # Save figure
        filename = os.path.join(self.output_dir, "figures", "probability_heatmap.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'type': 'probability_heatmap',
            'filename': filename,
            'title': 'Quantum State Probability Distribution',
            'description': 'Heatmap showing probability distribution over quantum states'
        }
    
    def _create_entanglement_graph(self, entanglement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create entanglement graph figure."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract entanglement data
        entanglement_matrix = np.array(entanglement_data.get('entanglement_matrix', []))
        centrality_metrics = entanglement_data.get('centrality_metrics', {})
        
        if entanglement_matrix.size > 0:
            # Create network plot
            num_qubits = entanglement_matrix.shape[0]
            
            # Create node positions
            angles = np.linspace(0, 2*np.pi, num_qubits, endpoint=False)
            x = np.cos(angles)
            y = np.sin(angles)
            
            # Draw edges
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    if entanglement_matrix[i, j] > 0:
                        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', 
                               alpha=entanglement_matrix[i, j], linewidth=2)
            
            # Draw nodes
            node_sizes = centrality_metrics.get('degree_centrality', [1] * num_qubits)
            node_sizes = np.array(node_sizes) * 100  # Scale for visibility
            
            for i in range(num_qubits):
                ax.scatter(x[i], y[i], s=node_sizes[i], c='red', alpha=0.7, edgecolors='black')
                ax.text(x[i]*1.2, y[i]*1.2, f'q{i}', fontsize=12, ha='center', va='center')
            
            ax.set_title('Entanglement Network', fontsize=16, fontweight='bold')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Save figure
        filename = os.path.join(self.output_dir, "figures", "entanglement_network.svg")
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'type': 'entanglement_network',
            'filename': filename,
            'title': 'Entanglement Network',
            'description': 'Network visualization of qubit entanglement'
        }
    
    def _create_convergence_plot(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization convergence plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract convergence data
        convergence_history = optimization_data.get('convergence_history', [])
        parameter_history = optimization_data.get('parameter_history', [])
        
        if convergence_history:
            # Plot convergence
            ax.plot(convergence_history, 'b-', linewidth=2, label='Objective Value')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Objective Value', fontsize=12)
            ax.set_title('Optimization Convergence', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add convergence statistics
            final_value = convergence_history[-1]
            initial_value = convergence_history[0]
            improvement = (initial_value - final_value) / initial_value * 100
            
            ax.text(0.02, 0.98, f'Final Value: {final_value:.6f}\nImprovement: {improvement:.1f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save figure
        filename = os.path.join(self.output_dir, "figures", "optimization_convergence.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'type': 'optimization_convergence',
            'filename': filename,
            'title': 'Optimization Convergence',
            'description': 'Plot showing optimization convergence over iterations'
        }
    
    def _create_interference_heatmap(self, interference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create interference heatmap figure."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract interference data
        interference_matrix = interference_data.get('interference_matrix', [])
        time_points = interference_data.get('time_points', [])
        
        if interference_matrix:
            # Create heatmap
            im = ax.imshow(interference_matrix, cmap='plasma', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Interference Strength', fontsize=12)
            
            # Add labels
            ax.set_title('Multi-Subspace Interference Heatmap', fontsize=16, fontweight='bold')
            ax.set_xlabel('Subspace Index', fontsize=12)
            ax.set_ylabel('Subspace Index', fontsize=12)
            
            # Add subspace labels
            subspace_labels = interference_data.get('subspace_labels', [])
            if subspace_labels:
                ax.set_xticks(range(len(subspace_labels)))
                ax.set_yticks(range(len(subspace_labels)))
                ax.set_xticklabels(subspace_labels, rotation=45)
                ax.set_yticklabels(subspace_labels)
        
        # Save figure
        filename = os.path.join(self.output_dir, "figures", "interference_heatmap.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'type': 'interference_heatmap',
            'filename': filename,
            'title': 'Multi-Subspace Interference Heatmap',
            'description': 'Heatmap showing interference between quantum subspaces'
        }
    
    def _create_benchmark_plot(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create benchmark performance plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract benchmark data
        qubit_counts = benchmark_data.get('qubit_counts', [])
        execution_times = benchmark_data.get('execution_times', [])
        memory_usage = benchmark_data.get('memory_usage', [])
        
        if qubit_counts and execution_times:
            # Plot execution time
            ax1.plot(qubit_counts, execution_times, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Qubits', fontsize=12)
            ax1.set_ylabel('Execution Time (s)', fontsize=12)
            ax1.set_title('Execution Time vs Qubit Count', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Plot memory usage
            if memory_usage:
                ax2.plot(qubit_counts, memory_usage, 'ro-', linewidth=2, markersize=8)
                ax2.set_xlabel('Number of Qubits', fontsize=12)
                ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
                ax2.set_title('Memory Usage vs Qubit Count', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.output_dir, "figures", "benchmark_performance.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'type': 'benchmark_performance',
            'filename': filename,
            'title': 'Performance Benchmark',
            'description': 'Performance metrics vs system size'
        }
    
    def _generate_reports(self, experiment_data: Dict[str, Any], 
                         figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate text reports."""
        reports = []
        
        # Generate JSON report
        json_report = self._generate_json_report(experiment_data, figures)
        reports.append(json_report)
        
        # Generate Markdown report
        markdown_report = self._generate_markdown_report(experiment_data, figures)
        reports.append(markdown_report)
        
        # Generate LaTeX report
        latex_report = self._generate_latex_report(experiment_data, figures)
        reports.append(latex_report)
        
        # Generate BibTeX bibliography
        bibtex_report = self._generate_bibtex_report()
        reports.append(bibtex_report)
        
        return reports
    
    def _generate_json_report(self, experiment_data: Dict[str, Any], 
                            figures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate JSON report."""
        report_data = {
            'metadata': asdict(self.metadata) if self.metadata is not None else None,
            'experiment_data': experiment_data,
            'figures': figures,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        filename = os.path.join(self.output_dir, "reports", "experiment_report.json")
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return {
            'type': 'json_report',
            'filename': filename,
            'title': 'Experiment Report (JSON)',
            'description': 'Complete experiment data in JSON format'
        }
    
    def _generate_markdown_report(self, experiment_data: Dict[str, Any],
                                figures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate Markdown report."""
        filename = os.path.join(self.output_dir, "reports", "experiment_report.md")
        
        with open(filename, 'w') as f:
            if self.metadata is not None:
                f.write(f"# {self.metadata.title}\n\n")
                f.write(f"**Authors:** {', '.join(self.metadata.authors)}\n\n")
                f.write(f"**Date:** {self.metadata.timestamp}\n\n")
                f.write(f"**Version:** {self.metadata.version}\n\n")
                
                f.write("## Abstract\n\n")
                f.write(f"{self.metadata.abstract}\n\n")
                
                f.write("## Keywords\n\n")
                f.write(f"{', '.join(self.metadata.keywords)}\n\n")
                
                f.write("## Experiment Results\n\n")
                f.write("### System Information\n\n")
                f.write(f"- **Python Version:** {self.metadata.system_info.get('python_version', 'Unknown')}\n")
                f.write(f"- **Platform:** {self.metadata.system_info.get('platform', 'Unknown')}\n")
                f.write(f"- **GPU Available:** {self.metadata.system_info.get('gpu_available', False)}\n")
                f.write(f"- **CPU Count:** {self.metadata.system_info.get('cpu_count', 'Unknown')}\n\n")
                
                f.write("### Results Summary\n\n")
                results = self.metadata.results_summary
                for key, value in results.items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
                
                f.write("## Figures\n\n")
                for i, fig in enumerate(figures, 1):
                    f.write(f"### Figure {i}: {fig['title']}\n\n")
                    f.write(f"![{fig['title']}]({fig['filename']})\n\n")
                    f.write(f"{fig['description']}\n\n")
                
                f.write("## Reproducibility\n\n")
                f.write(f"**Reproducibility Hash:** `{self.metadata.reproducibility_hash}`\n\n")
                f.write("This experiment can be reproduced using the provided metadata and parameters.\n\n")
            else:
                f.write("# Quantum Computing Experiment Report\n\n")
                f.write("## Experiment Results\n\n")
                f.write("### System Information\n\n")
                f.write("- **Python Version:** Unknown\n")
                f.write("- **Platform:** Unknown\n")
                f.write("- **GPU Available:** Unknown\n")
                f.write("- **CPU Count:** Unknown\n\n")
                
                f.write("### Results Summary\n\n")
                f.write("- **Status:** Experiment completed\n\n")
                
                f.write("## Figures\n\n")
                for i, fig in enumerate(figures, 1):
                    f.write(f"### Figure {i}: {fig['title']}\n\n")
                    f.write(f"![{fig['title']}]({fig['filename']})\n\n")
                    f.write(f"{fig['description']}\n\n")
                
                f.write("## Reproducibility\n\n")
                f.write("**Reproducibility Hash:** Not available\n\n")
                f.write("This experiment can be reproduced using the provided metadata and parameters.\n\n")
        
        return {
            'type': 'markdown_report',
            'filename': filename,
            'title': 'Experiment Report (Markdown)',
            'description': 'Human-readable experiment report'
        }
    
    def _generate_latex_report(self, experiment_data: Dict[str, Any], 
                             figures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX report."""
        filename = os.path.join(self.output_dir, "reports", "experiment_report.tex")
        
        with open(filename, 'w') as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage{amsmath}\n")
            f.write("\\usepackage{hyperref}\n\n")
            
            if self.metadata is not None:
                f.write("\\title{" + self.metadata.title + "}\n")
                f.write("\\author{" + " and ".join(self.metadata.authors) + "}\n")
                f.write("\\date{" + self.metadata.timestamp + "}\n\n")
            else:
                f.write("\\title{Quantum Computing Experiment Report}\n")
                f.write("\\author{Coratrix}\n")
                f.write("\\date{" + datetime.now().strftime("%Y-%m-%d") + "}\n\n")
            
            f.write("\\begin{document}\n\n")
            f.write("\\maketitle\n\n")
            
            if self.metadata is not None:
                f.write("\\begin{abstract}\n")
                f.write(self.metadata.abstract)
                f.write("\\end{abstract}\n\n")
                
                f.write("\\section{Introduction}\n\n")
                f.write("This report presents the results of a quantum computing experiment ")
                f.write(f"conducted using the Coratrix platform (version {self.metadata.version}).\n\n")
                
                f.write("\\section{Methods}\n\n")
                f.write("The experiment was conducted using the following system configuration:\n")
                f.write("\\begin{itemize}\n")
                f.write(f"\\item Python version: {self.metadata.system_info.get('python_version', 'Unknown')}\n")
                f.write(f"\\item Platform: {self.metadata.system_info.get('platform', 'Unknown')}\n")
                f.write(f"\\item GPU available: {self.metadata.system_info.get('gpu_available', False)}\n")
                f.write("\\end{itemize}\n\n")
            else:
                f.write("\\begin{abstract}\n")
                f.write("This report presents the results of a quantum computing experiment conducted using the Coratrix platform.\n")
                f.write("\\end{abstract}\n\n")
                
                f.write("\\section{Introduction}\n\n")
                f.write("This report presents the results of a quantum computing experiment ")
                f.write("conducted using the Coratrix platform.\n\n")
                
                f.write("\\section{Methods}\n\n")
                f.write("The experiment was conducted using the following system configuration:\n")
                f.write("\\begin{itemize}\n")
                f.write("\\item Python version: Unknown\n")
                f.write("\\item Platform: Unknown\n")
                f.write("\\item GPU available: Unknown\n")
                f.write("\\end{itemize}\n\n")
            
            f.write("\\section{Results}\n\n")
            f.write("The experiment produced the following results:\n\n")
            
            for i, fig in enumerate(figures, 1):
                f.write(f"\\subsection{{Figure {i}: {fig['title']}}}\n\n")
                f.write(f"\\begin{{figure}}[h]\n")
                f.write(f"\\centering\n")
                f.write(f"\\includegraphics[width=0.8\\textwidth]{{{fig['filename']}}}\n")
                f.write(f"\\caption{{{fig['description']}}}\n")
                f.write(f"\\end{{figure}}\n\n")
            
            f.write("\\section{Conclusion}\n\n")
            f.write("The experiment demonstrates the capabilities of the Coratrix platform ")
            f.write("for quantum computing simulation and analysis.\n\n")
            
            f.write("\\section{Reproducibility}\n\n")
            f.write("This experiment can be reproduced using the provided metadata and parameters. ")
            if self.metadata is not None:
                f.write(f"The reproducibility hash is: \\texttt{{{self.metadata.reproducibility_hash}}}\n\n")
            else:
                f.write("The reproducibility hash is not available.\n\n")
            
            f.write("\\end{document}\n")
        
        return {
            'type': 'latex_report',
            'filename': filename,
            'title': 'Experiment Report (LaTeX)',
            'description': 'LaTeX document for academic publication'
        }
    
    def _generate_bibtex_report(self) -> Dict[str, Any]:
        """Generate BibTeX bibliography."""
        filename = os.path.join(self.output_dir, "reports", "bibliography.bib")
        
        with open(filename, 'w') as f:
            f.write("@article{coratrix2024,\n")
            f.write("  title={Coratrix: A High-Performance Quantum Computing Simulation Platform},\n")
            if self.metadata is not None:
                f.write("  author={" + " and ".join(self.metadata.authors) + "},\n")
            else:
                f.write("  author={Coratrix Development Team},\n")
            f.write("  journal={Quantum Computing and Simulation},\n")
            f.write("  year={2024},\n")
            f.write("  publisher={Coratrix Development Team},\n")
            f.write("  url={https://github.com/coratrix/coratrix},\n")
            f.write("  doi={10.1000/182},\n")
            f.write("  keywords={quantum computing, simulation, optimization, entanglement}\n")
            f.write("}\n\n")
            
            f.write("@software{coratrix,\n")
            f.write("  title={Coratrix Quantum Computing Platform},\n")
            f.write("  author={Coratrix Development Team},\n")
            f.write("  year={2024},\n")
            f.write("  url={https://github.com/coratrix/coratrix},\n")
            if self.metadata is not None:
                f.write("  version={" + self.metadata.version + "}\n")
            else:
                f.write("  version={3.0}\n")
            f.write("}\n\n")
            
            if self.metadata is not None:
                f.write("@misc{quantum_experiment_" + self.metadata.reproducibility_hash[:8] + ",\n")
                f.write("  title={" + self.metadata.title + "},\n")
                f.write("  author={" + " and ".join(self.metadata.authors) + "},\n")
                f.write("  year={2024},\n")
                f.write("  note={Reproducibility hash: " + self.metadata.reproducibility_hash + "}\n")
                f.write("}\n")
            else:
                f.write("@misc{quantum_experiment_unknown,\n")
                f.write("  title={Quantum Computing Experiment},\n")
                f.write("  author={Coratrix Development Team},\n")
                f.write("  year={2024},\n")
                f.write("  note={Reproducibility hash: Not available}\n")
                f.write("}\n")
        
        return {
            'type': 'bibtex_bibliography',
            'filename': filename,
            'title': 'Bibliography (BibTeX)',
            'description': 'BibTeX entries for references'
        }
    
    def _generate_data_files(self, experiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate data files."""
        data_files = []
        
        # Save raw experiment data
        filename = os.path.join(self.output_dir, "data", "experiment_data.json")
        with open(filename, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        data_files.append({
            'type': 'experiment_data',
            'filename': filename,
            'title': 'Experiment Data (JSON)',
            'description': 'Raw experiment data in JSON format'
        })
        
        # Save metadata
        filename = os.path.join(self.output_dir, "data", "metadata.json")
        with open(filename, 'w') as f:
            if self.metadata is not None:
                json.dump(asdict(self.metadata), f, indent=2)
            else:
                json.dump({"metadata": "Not available"}, f, indent=2)
        data_files.append({
            'type': 'metadata',
            'filename': filename,
            'title': 'Experiment Metadata (JSON)',
            'description': 'Experiment metadata and system information'
        })
        
        return data_files
    
    def _generate_bibliography(self) -> Dict[str, Any]:
        """Generate bibliography information."""
        return {
            'references': [
                {
                    'title': 'Coratrix: A High-Performance Quantum Computing Simulation Platform',
                    'authors': self.metadata.authors if self.metadata is not None else ['Coratrix Development Team'],
                    'year': 2024,
                    'type': 'software',
                    'url': 'https://github.com/coratrix/coratrix'
                }
            ],
            'related_work': [
                'Quantum computing simulation methods',
                'Entanglement analysis techniques',
                'Optimization algorithms for quantum circuits',
                'Multi-subspace quantum search'
            ]
        }
    
    def generate_release_notes(self, version: str, changes: List[str]) -> str:
        """Generate release notes."""
        filename = os.path.join(self.output_dir, "reports", "release_notes.md")
        
        with open(filename, 'w') as f:
            f.write(f"# Coratrix Release Notes - Version {version}\n\n")
            f.write(f"**Release Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
            
            f.write("## New Features\n\n")
            for change in changes:
                if change.startswith('feat:'):
                    f.write(f"- {change[5:].strip()}\n")
            
            f.write("\n## Bug Fixes\n\n")
            for change in changes:
                if change.startswith('fix:'):
                    f.write(f"- {change[4:].strip()}\n")
            
            f.write("\n## Performance Improvements\n\n")
            for change in changes:
                if change.startswith('perf:'):
                    f.write(f"- {change[5:].strip()}\n")
            
            f.write("\n## Documentation Updates\n\n")
            for change in changes:
                if change.startswith('docs:'):
                    f.write(f"- {change[5:].strip()}\n")
            
            f.write("\n## Breaking Changes\n\n")
            for change in changes:
                if change.startswith('breaking:'):
                    f.write(f"- {change[9:].strip()}\n")
            
            f.write("\n## Installation\n\n")
            f.write("```bash\n")
            f.write("pip install coratrix\n")
            f.write("```\n\n")
            
            f.write("## Upgrade Instructions\n\n")
            f.write("1. Update your installation:\n")
            f.write("   ```bash\n")
            f.write("   pip install --upgrade coratrix\n")
            f.write("   ```\n\n")
            f.write("2. Check the migration guide for any breaking changes\n")
            f.write("3. Update your code if necessary\n")
            f.write("4. Run your tests to ensure compatibility\n\n")
            
            f.write("## Support\n\n")
            f.write("For questions or issues, please contact:\n")
            f.write("- GitHub Issues: https://github.com/coratrix/coratrix/issues\n")
            f.write("- Email: support@coratrix.org\n")
            f.write("- Documentation: https://docs.coratrix.org\n")
        
        return filename
