"""
Test suite for the publication report generator.
"""

import unittest
import os
import json
import tempfile
import shutil
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.report_generator import PublicationReportGenerator, PublicationMetadata, FigureSpec


class TestPublicationReportGenerator(unittest.TestCase):
    """Test cases for the publication report generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = PublicationReportGenerator(output_dir=self.temp_dir)
        
        # Create sample metadata
        self.metadata = PublicationMetadata(
            title="Test Quantum Experiment",
            authors=["Test Author"],
            abstract="This is a test experiment for quantum computing.",
            keywords=["quantum", "computing", "test"],
            experiment_type="quantum_simulation",
            timestamp="2024-01-01T00:00:00Z",
            version="1.0.0",
            reproducibility_hash="test_hash_12345",
            system_info={
                "python_version": "3.9.0",
                "platform": "Linux",
                "gpu_available": True,
                "cpu_count": 8
            },
            results_summary={
                "fidelity": 0.99,
                "entanglement_entropy": 1.0,
                "execution_time": 2.5
            }
        )
        
        # Set metadata on generator
        self.generator.metadata = self.metadata
        
        # Create sample experiment data
        self.experiment_data = {
            "circuit_data": {
                "num_qubits": 2,
                "gates": [
                    {"name": "H", "qubits": [0]},
                    {"name": "CNOT", "qubits": [0, 1]}
                ]
            },
            "probability_data": {
                "probabilities": [0.5, 0.0, 0.0, 0.5],
                "num_qubits": 2
            },
            "entanglement_data": {
                "entanglement_matrix": [[0, 1], [1, 0]],
                "centrality_metrics": {
                    "degree_centrality": [1, 1]
                }
            },
            "optimization_data": {
                "convergence_history": [1.0, 0.8, 0.6, 0.4, 0.2],
                "parameter_history": [[1, 2], [1.5, 2.5], [2, 3], [2.5, 3.5], [3, 4]]
            },
            "interference_data": {
                "interference_matrix": [[1, 0.5], [0.5, 1]],
                "time_points": [0, 1],
                "subspace_labels": ["A", "B"]
            },
            "benchmark_data": {
                "qubit_counts": [2, 4, 6, 8],
                "execution_times": [0.1, 0.5, 2.0, 8.0],
                "memory_usage": [1, 4, 16, 64]
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.output_dir, self.temp_dir)
        self.assertIsNotNone(self.generator.metadata)  # Metadata is set in setUp
        self.assertEqual(self.generator.figures, [])
        
        # Check directory creation
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "figures")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "reports")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "data")))
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        report = self.generator.generate_comprehensive_report(
            self.experiment_data, self.metadata
        )
        
        # Check report structure
        self.assertIn('metadata', report)
        self.assertIn('figures', report)
        self.assertIn('reports', report)
        self.assertIn('data_files', report)
        self.assertIn('bibliography', report)
        
        # Check metadata
        self.assertEqual(report['metadata']['title'], self.metadata.title)
        self.assertEqual(report['metadata']['authors'], self.metadata.authors)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "report_summary.json")))
        self.assertTrue(len(report['figures']) > 0)
        self.assertTrue(len(report['reports']) > 0)
        self.assertTrue(len(report['data_files']) > 0)
    
    def test_figure_generation(self):
        """Test figure generation."""
        figures = self.generator._generate_figures(self.experiment_data)
        
        # Check that figures were generated
        self.assertGreater(len(figures), 0)
        
        # Check figure types
        figure_types = [fig['type'] for fig in figures]
        expected_types = [
            'circuit_diagram',
            'probability_heatmap',
            'entanglement_network',
            'optimization_convergence',
            'interference_heatmap',
            'benchmark_performance'
        ]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, figure_types)
        
        # Check that figure files exist
        for fig in figures:
            self.assertTrue(os.path.exists(fig['filename']))
            self.assertIn('title', fig)
            self.assertIn('description', fig)
    
    def test_circuit_diagram_generation(self):
        """Test circuit diagram generation."""
        circuit_data = {
            "num_qubits": 3,
            "gates": [
                {"name": "H", "qubits": [0]},
                {"name": "CNOT", "qubits": [0, 1]},
                {"name": "X", "qubits": [2]}
            ]
        }
        
        fig = self.generator._create_circuit_diagram(circuit_data)
        
        self.assertEqual(fig['type'], 'circuit_diagram')
        self.assertTrue(os.path.exists(fig['filename']))
        self.assertIn('circuit_diagram.svg', fig['filename'])
    
    def test_probability_heatmap_generation(self):
        """Test probability heatmap generation."""
        probability_data = {
            "probabilities": [0.25, 0.25, 0.25, 0.25],
            "num_qubits": 2
        }
        
        fig = self.generator._create_probability_heatmap(probability_data)
        
        self.assertEqual(fig['type'], 'probability_heatmap')
        self.assertTrue(os.path.exists(fig['filename']))
        self.assertIn('probability_heatmap.png', fig['filename'])
    
    def test_entanglement_graph_generation(self):
        """Test entanglement graph generation."""
        entanglement_data = {
            "entanglement_matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            "centrality_metrics": {
                "degree_centrality": [1, 2, 1]
            }
        }
        
        fig = self.generator._create_entanglement_graph(entanglement_data)
        
        self.assertEqual(fig['type'], 'entanglement_network')
        self.assertTrue(os.path.exists(fig['filename']))
        self.assertIn('entanglement_network.svg', fig['filename'])
    
    def test_convergence_plot_generation(self):
        """Test convergence plot generation."""
        optimization_data = {
            "convergence_history": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
            "parameter_history": [[1, 2], [1.5, 2.5], [2, 3], [2.5, 3.5], [3, 4], [3.5, 4.5]]
        }
        
        fig = self.generator._create_convergence_plot(optimization_data)
        
        self.assertEqual(fig['type'], 'optimization_convergence')
        self.assertTrue(os.path.exists(fig['filename']))
        self.assertIn('optimization_convergence.png', fig['filename'])
    
    def test_interference_heatmap_generation(self):
        """Test interference heatmap generation."""
        interference_data = {
            "interference_matrix": [[1, 0.5], [0.5, 1]],
            "time_points": [0, 1],
            "subspace_labels": ["A", "B"]
        }
        
        fig = self.generator._create_interference_heatmap(interference_data)
        
        self.assertEqual(fig['type'], 'interference_heatmap')
        self.assertTrue(os.path.exists(fig['filename']))
        self.assertIn('interference_heatmap.png', fig['filename'])
    
    def test_benchmark_plot_generation(self):
        """Test benchmark plot generation."""
        benchmark_data = {
            "qubit_counts": [2, 4, 6, 8, 10],
            "execution_times": [0.1, 0.5, 2.0, 8.0, 32.0],
            "memory_usage": [1, 4, 16, 64, 256]
        }
        
        fig = self.generator._create_benchmark_plot(benchmark_data)
        
        self.assertEqual(fig['type'], 'benchmark_performance')
        self.assertTrue(os.path.exists(fig['filename']))
        self.assertIn('benchmark_performance.png', fig['filename'])
    
    def test_report_generation(self):
        """Test report generation."""
        figures = self.generator._generate_figures(self.experiment_data)
        reports = self.generator._generate_reports(self.experiment_data, figures)
        
        # Check that reports were generated
        self.assertGreater(len(reports), 0)
        
        # Check report types
        report_types = [report['type'] for report in reports]
        expected_types = [
            'json_report',
            'markdown_report',
            'latex_report',
            'bibtex_bibliography'
        ]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, report_types)
        
        # Check that report files exist
        for report in reports:
            self.assertTrue(os.path.exists(report['filename']))
            self.assertIn('title', report)
            self.assertIn('description', report)
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        figures = self.generator._generate_figures(self.experiment_data)
        report = self.generator._generate_json_report(self.experiment_data, figures)
        
        self.assertEqual(report['type'], 'json_report')
        self.assertTrue(os.path.exists(report['filename']))
        
        # Check JSON content
        with open(report['filename'], 'r') as f:
            data = json.load(f)
        
        self.assertIn('metadata', data)
        self.assertIn('experiment_data', data)
        self.assertIn('figures', data)
    
    def test_markdown_report_generation(self):
        """Test Markdown report generation."""
        figures = self.generator._generate_figures(self.experiment_data)
        report = self.generator._generate_markdown_report(self.experiment_data, figures)
        
        self.assertEqual(report['type'], 'markdown_report')
        self.assertTrue(os.path.exists(report['filename']))
        
        # Check Markdown content
        with open(report['filename'], 'r') as f:
            content = f.read()
        
        self.assertIn(self.metadata.title, content)
        self.assertIn(self.metadata.abstract, content)
        self.assertIn('## Figures', content)
    
    def test_latex_report_generation(self):
        """Test LaTeX report generation."""
        figures = self.generator._generate_figures(self.experiment_data)
        report = self.generator._generate_latex_report(self.experiment_data, figures)
        
        self.assertEqual(report['type'], 'latex_report')
        self.assertTrue(os.path.exists(report['filename']))
        
        # Check LaTeX content
        with open(report['filename'], 'r') as f:
            content = f.read()
        
        self.assertIn('\\documentclass{article}', content)
        self.assertIn(self.metadata.title, content)
        self.assertIn('\\begin{abstract}', content)
        self.assertIn('\\end{abstract}', content)
    
    def test_bibtex_report_generation(self):
        """Test BibTeX report generation."""
        report = self.generator._generate_bibtex_report()
        
        self.assertEqual(report['type'], 'bibtex_bibliography')
        self.assertTrue(os.path.exists(report['filename']))
        
        # Check BibTeX content
        with open(report['filename'], 'r') as f:
            content = f.read()
        
        self.assertIn('@article{coratrix2024,', content)
        self.assertIn('@software{coratrix,', content)
        self.assertIn('@misc{quantum_experiment_', content)
    
    def test_data_file_generation(self):
        """Test data file generation."""
        data_files = self.generator._generate_data_files(self.experiment_data)
        
        # Check that data files were generated
        self.assertGreater(len(data_files), 0)
        
        # Check file types
        file_types = [file['type'] for file in data_files]
        expected_types = ['experiment_data', 'metadata']
        
        for expected_type in expected_types:
            self.assertIn(expected_type, file_types)
        
        # Check that files exist
        for data_file in data_files:
            self.assertTrue(os.path.exists(data_file['filename']))
    
    def test_bibliography_generation(self):
        """Test bibliography generation."""
        bibliography = self.generator._generate_bibliography()
        
        self.assertIn('references', bibliography)
        self.assertIn('related_work', bibliography)
        
        # Check references
        references = bibliography['references']
        self.assertGreater(len(references), 0)
        
        # Check first reference
        first_ref = references[0]
        self.assertIn('title', first_ref)
        self.assertIn('authors', first_ref)
        self.assertIn('year', first_ref)
        self.assertIn('type', first_ref)
        self.assertIn('url', first_ref)
    
    def test_release_notes_generation(self):
        """Test release notes generation."""
        version = "1.0.0"
        changes = [
            "feat: Add GPU acceleration support",
            "fix: Resolve memory leak in sparse matrices",
            "perf: Optimize gate application for large circuits",
            "docs: Update API documentation",
            "breaking: Change default noise model parameters"
        ]
        
        filename = self.generator.generate_release_notes(version, changes)
        
        self.assertTrue(os.path.exists(filename))
        
        # Check content
        with open(filename, 'r') as f:
            content = f.read()
        
        self.assertIn(f"# Coratrix Release Notes - Version {version}", content)
        self.assertIn("## New Features", content)
        self.assertIn("## Bug Fixes", content)
        self.assertIn("## Performance Improvements", content)
        self.assertIn("## Documentation Updates", content)
        self.assertIn("## Breaking Changes", content)
    
    def test_empty_data_handling(self):
        """Test handling of empty experiment data."""
        empty_data = {}
        figures = self.generator._generate_figures(empty_data)
        
        # Should handle empty data gracefully
        self.assertIsInstance(figures, list)
        self.assertEqual(len(figures), 0)
    
    def test_missing_data_handling(self):
        """Test handling of missing data fields."""
        incomplete_data = {
            "circuit_data": {"num_qubits": 2}  # Missing gates
        }
        
        # Should not crash with missing data
        figures = self.generator._generate_figures(incomplete_data)
        self.assertIsInstance(figures, list)
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Test with minimal metadata
        minimal_metadata = PublicationMetadata(
            title="Test",
            authors=["Author"],
            abstract="Test abstract",
            keywords=["test"],
            experiment_type="test",
            timestamp="2024-01-01T00:00:00Z",
            version="1.0.0",
            reproducibility_hash="test",
            system_info={},
            results_summary={}
        )
        
        # Should work with minimal metadata
        report = self.generator.generate_comprehensive_report(
            self.experiment_data, minimal_metadata
        )
        
        self.assertIn('metadata', report)
        self.assertEqual(report['metadata']['title'], "Test")


if __name__ == '__main__':
    unittest.main()
