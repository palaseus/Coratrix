"""
Tests for reproducibility and security utilities.

This module tests deterministic seeds, metadata tracking, and security features.
"""

import unittest
import sys
import os
import tempfile
import json
import numpy as np
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reproducibility import (
    ReproducibilityManager, SecurityManager, DeterministicRandom,
    ReproducibilityChecker, SystemMetadata, ExperimentMetadata
)


class TestReproducibilityManager(unittest.TestCase):
    """Test cases for reproducibility manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ReproducibilityManager(session_id="test_session", random_seed=42)
    
    def test_manager_initialization(self):
        """Test reproducibility manager initialization."""
        self.assertEqual(self.manager.session_id, "test_session")
        self.assertEqual(self.manager.random_seed, 42)
        self.assertIsInstance(self.manager.system_metadata, SystemMetadata)
        self.assertEqual(len(self.manager.experiments), 0)
    
    def test_system_metadata_collection(self):
        """Test system metadata collection."""
        metadata = self.manager.system_metadata
        
        # Check required fields
        self.assertIsInstance(metadata.timestamp, str)
        self.assertIsInstance(metadata.python_version, str)
        self.assertIsInstance(metadata.platform, str)
        self.assertIsInstance(metadata.architecture, str)
        self.assertIsInstance(metadata.cpu_count, int)
        self.assertIsInstance(metadata.memory_total_gb, float)
        self.assertIsInstance(metadata.gpu_available, bool)
        self.assertIsInstance(metadata.numpy_version, str)
        self.assertIsInstance(metadata.scipy_version, str)
        self.assertIsInstance(metadata.random_seed, int)
        self.assertIsInstance(metadata.session_id, str)
        
        # Check value ranges
        self.assertGreater(metadata.cpu_count, 0)
        self.assertGreater(metadata.memory_total_gb, 0)
        self.assertGreaterEqual(metadata.gpu_memory_gb, 0)
        self.assertEqual(metadata.random_seed, 42)
        self.assertEqual(metadata.session_id, "test_session")
    
    def test_experiment_creation(self):
        """Test experiment creation."""
        experiment_type = "quantum_simulation"
        parameters = {"num_qubits": 4, "shots": 1000}
        
        experiment_id = self.manager.create_experiment(experiment_type, parameters)
        
        # Check experiment ID format
        self.assertIsInstance(experiment_id, str)
        self.assertGreater(len(experiment_id), 0)
        
        # Check experiment was added
        self.assertEqual(len(self.manager.experiments), 1)
        
        # Check experiment metadata
        experiment = self.manager.experiments[0]
        self.assertEqual(experiment.experiment_id, experiment_id)
        self.assertEqual(experiment.session_id, "test_session")
        self.assertEqual(experiment.experiment_type, experiment_type)
        self.assertEqual(experiment.parameters, parameters)
        self.assertIsInstance(experiment.reproducibility_hash, str)
        self.assertGreater(len(experiment.reproducibility_hash), 0)
    
    def test_experiment_update(self):
        """Test experiment update."""
        experiment_id = self.manager.create_experiment("test", {"param": "value"})
        
        # Update experiment
        self.manager.update_experiment(
            experiment_id, 
            execution_time=1.5, 
            success=True, 
            output_files=["output1.txt", "output2.txt"]
        )
        
        # Check update
        experiment = self.manager.get_experiment_metadata(experiment_id)
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment.execution_time, 1.5)
        self.assertTrue(experiment.success)
        self.assertEqual(experiment.output_files, ["output1.txt", "output2.txt"])
    
    def test_experiment_metadata_retrieval(self):
        """Test experiment metadata retrieval."""
        experiment_id = self.manager.create_experiment("test", {"param": "value"})
        
        # Retrieve existing experiment
        experiment = self.manager.get_experiment_metadata(experiment_id)
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment.experiment_id, experiment_id)
        
        # Retrieve non-existent experiment
        non_existent = self.manager.get_experiment_metadata("non_existent")
        self.assertIsNone(non_existent)
    
    def test_session_export(self):
        """Test session metadata export."""
        # Create some experiments
        self.manager.create_experiment("test1", {"param1": "value1"})
        self.manager.create_experiment("test2", {"param2": "value2"})
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.manager.export_session_metadata(temp_file)
            
            # Check file was created
            self.assertTrue(os.path.exists(temp_file))
            
            # Check file content
            with open(temp_file, 'r') as f:
                data = json.load(f)
                
                # Check required fields
                self.assertIn('session_id', data)
                self.assertIn('random_seed', data)
                self.assertIn('system_metadata', data)
                self.assertIn('experiments', data)
                
                # Check experiments
                self.assertEqual(len(data['experiments']), 2)
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_reproducibility_verification(self):
        """Test reproducibility verification."""
        experiment_id = self.manager.create_experiment("test", {"param": "value"})
        
        # Verify reproducibility (should pass for same session)
        is_reproducible, message = self.manager.verify_reproducibility(experiment_id)
        self.assertTrue(is_reproducible)
        self.assertEqual(message, "Reproducibility verified")
    
    def test_session_summary(self):
        """Test session summary generation."""
        # Create some experiments
        exp1 = self.manager.create_experiment("test1", {"param1": "value1"})
        exp2 = self.manager.create_experiment("test2", {"param2": "value2"})
        
        # Update experiments
        self.manager.update_experiment(exp1, 1.0, True, ["output1.txt"])
        self.manager.update_experiment(exp2, 2.0, False, [], "Error occurred")
        
        # Get session summary
        summary = self.manager.get_session_summary()
        
        # Check summary fields
        self.assertIn('session_id', summary)
        self.assertIn('random_seed', summary)
        self.assertIn('total_experiments', summary)
        self.assertIn('successful_experiments', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('total_execution_time', summary)
        self.assertIn('system_metadata', summary)
        
        # Check values
        self.assertEqual(summary['total_experiments'], 2)
        self.assertEqual(summary['successful_experiments'], 1)
        self.assertEqual(summary['success_rate'], 0.5)
        self.assertEqual(summary['total_execution_time'], 3.0)


class TestSecurityManager(unittest.TestCase):
    """Test cases for security manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.security_manager = SecurityManager(privacy_mode=True)
    
    def test_security_manager_initialization(self):
        """Test security manager initialization."""
        self.assertTrue(self.security_manager.privacy_mode)
        self.assertIsInstance(self.security_manager.sensitive_fields, list)
        self.assertGreater(len(self.security_manager.sensitive_fields), 0)
    
    def test_sensitive_data_redaction(self):
        """Test sensitive data redaction."""
        data = {
            'git_commit_hash': 'abc123',
            'working_directory': '/home/user/project',
            'session_id': 'session123',
            'experiment_id': 'exp456',
            'normal_field': 'normal_value',
            'system_metadata': {
                'git_commit_hash': 'def456',
                'working_directory': '/home/user/project',
                'normal_field': 'normal_value'
            }
        }
        
        redacted_data = self.security_manager.redact_sensitive_data(data)
        
        # Check sensitive fields are redacted
        self.assertEqual(redacted_data['git_commit_hash'], '[REDACTED]')
        self.assertEqual(redacted_data['working_directory'], '[REDACTED]')
        self.assertEqual(redacted_data['session_id'], '[REDACTED]')
        self.assertEqual(redacted_data['experiment_id'], '[REDACTED]')
        
        # Check normal fields are preserved
        self.assertEqual(redacted_data['normal_field'], 'normal_value')
        
        # Check nested sensitive fields are redacted
        system_metadata = redacted_data['system_metadata']
        self.assertEqual(system_metadata['git_commit_hash'], '[REDACTED]')
        self.assertEqual(system_metadata['working_directory'], '[REDACTED]')
        self.assertEqual(system_metadata['normal_field'], 'normal_value')
    
    def test_privacy_report_creation(self):
        """Test privacy report creation."""
        metadata = {
            'git_commit_hash': 'abc123',
            'working_directory': '/home/user/project',
            'normal_field': 'normal_value'
        }
        
        privacy_report = self.security_manager.create_privacy_report(metadata)
        
        # Check privacy report structure
        self.assertIn('privacy_notice', privacy_report)
        self.assertIn('metadata', privacy_report)
        
        # Check privacy notice
        privacy_notice = privacy_report['privacy_notice']
        self.assertTrue(privacy_notice['privacy_mode'])
        self.assertIn('redacted_fields', privacy_notice)
        self.assertIn('redaction_timestamp', privacy_notice)
        
        # Check metadata is redacted
        redacted_metadata = privacy_report['metadata']
        self.assertEqual(redacted_metadata['git_commit_hash'], '[REDACTED]')
        self.assertEqual(redacted_metadata['working_directory'], '[REDACTED]')
        self.assertEqual(redacted_metadata['normal_field'], 'normal_value')


class TestDeterministicRandom(unittest.TestCase):
    """Test cases for deterministic random number generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.deterministic_random = DeterministicRandom(seed=42)
    
    def test_deterministic_random_initialization(self):
        """Test deterministic random initialization."""
        self.assertEqual(self.deterministic_random.seed, 42)
        self.assertIsNotNone(self.deterministic_random.np_random)
        self.assertIsNotNone(self.deterministic_random.python_random)
    
    def test_random_state_management(self):
        """Test random state management."""
        # Get initial state
        initial_state = self.deterministic_random.get_random_state()
        
        # Generate some random numbers
        rand1 = self.deterministic_random.random()
        rand2 = self.deterministic_random.random()
        
        # Set state back
        self.deterministic_random.set_random_state(initial_state)
        
        # Generate random numbers again
        rand3 = self.deterministic_random.random()
        rand4 = self.deterministic_random.random()
        
        # Should be the same
        self.assertAlmostEqual(rand1, rand3, places=10)
        self.assertAlmostEqual(rand2, rand4, places=10)
    
    def test_random_number_generation(self):
        """Test random number generation."""
        # Test random float
        rand_float = self.deterministic_random.random()
        self.assertIsInstance(rand_float, float)
        self.assertGreaterEqual(rand_float, 0.0)
        self.assertLess(rand_float, 1.0)
        
        # Test random integer
        rand_int = self.deterministic_random.randint(0, 10)
        self.assertIsInstance(rand_int, int)
        self.assertGreaterEqual(rand_int, 0)
        self.assertLess(rand_int, 10)
        
        # Test random choice
        choices = ['a', 'b', 'c', 'd']
        choice = self.deterministic_random.choice(choices)
        self.assertIn(choice, choices)
        
        # Test shuffle
        original = [1, 2, 3, 4, 5]
        shuffled = self.deterministic_random.shuffle(original)
        self.assertEqual(len(shuffled), len(original))
        self.assertEqual(set(shuffled), set(original))
        self.assertNotEqual(shuffled, original)  # Should be shuffled


class TestReproducibilityChecker(unittest.TestCase):
    """Test cases for reproducibility checker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = ReproducibilityChecker(tolerance=1e-9)
    
    def test_numerical_reproducibility_check(self):
        """Test numerical reproducibility checking."""
        values1 = [1.0, 2.0, 3.0, 4.0]
        values2 = [1.0, 2.0, 3.0, 4.0]
        
        is_reproducible, max_diff = self.checker.check_numerical_reproducibility(values1, values2)
        self.assertTrue(is_reproducible)
        self.assertAlmostEqual(max_diff, 0.0, places=10)
        
        # Test with small differences
        values3 = [1.0, 2.0, 3.0, 4.0000000001]
        is_reproducible, max_diff = self.checker.check_numerical_reproducibility(values1, values3)
        self.assertTrue(is_reproducible)
        self.assertLess(max_diff, 1e-9)
        
        # Test with large differences
        values4 = [1.0, 2.0, 3.0, 5.0]
        is_reproducible, max_diff = self.checker.check_numerical_reproducibility(values1, values4)
        self.assertFalse(is_reproducible)
        self.assertGreater(max_diff, 1e-9)
    
    def test_quantum_state_reproducibility_check(self):
        """Test quantum state reproducibility checking."""
        from core.qubit import QuantumState
        
        # Create identical states
        state1 = QuantumState(2)
        state1.set_amplitude(0, 1.0/np.sqrt(2))
        state1.set_amplitude(3, 1.0/np.sqrt(2))
        state1.normalize()
        
        state2 = QuantumState(2)
        state2.set_amplitude(0, 1.0/np.sqrt(2))
        state2.set_amplitude(3, 1.0/np.sqrt(2))
        state2.normalize()
        
        is_reproducible, max_diff = self.checker.check_quantum_state_reproducibility(state1, state2)
        self.assertTrue(is_reproducible)
        self.assertLess(max_diff, 1e-10)
        
        # Test with different states
        state3 = QuantumState(2)
        state3.set_amplitude(0, 1.0)
        
        is_reproducible, max_diff = self.checker.check_quantum_state_reproducibility(state1, state3)
        self.assertFalse(is_reproducible)
        self.assertGreater(max_diff, 1e-10)
    
    def test_circuit_reproducibility_check(self):
        """Test circuit reproducibility checking."""
        from core.circuit import QuantumCircuit
        from core.gates import HGate, CNOTGate
        
        # Create identical circuits
        circuit1 = QuantumCircuit(2)
        circuit1.add_gate(HGate(), [0])
        circuit1.add_gate(CNOTGate(), [0, 1])
        
        circuit2 = QuantumCircuit(2)
        circuit2.add_gate(HGate(), [0])
        circuit2.add_gate(CNOTGate(), [0, 1])
        
        is_reproducible, message = self.checker.check_circuit_reproducibility(circuit1, circuit2)
        self.assertTrue(is_reproducible)
        self.assertEqual(message, "Circuits are reproducible")
        
        # Test with different circuits
        circuit3 = QuantumCircuit(2)
        circuit3.add_gate(HGate(), [0])
        circuit3.add_gate(HGate(), [1])  # Different gate
        
        is_reproducible, message = self.checker.check_circuit_reproducibility(circuit1, circuit3)
        self.assertFalse(is_reproducible)
        self.assertIn("Different gate", message)
        
        # Test with different number of qubits
        circuit4 = QuantumCircuit(3)
        circuit4.add_gate(HGate(), [0])
        circuit4.add_gate(CNOTGate(), [0, 1])
        
        is_reproducible, message = self.checker.check_circuit_reproducibility(circuit1, circuit4)
        self.assertFalse(is_reproducible)
        self.assertEqual(message, "Different number of qubits")


class TestIntegration(unittest.TestCase):
    """Integration tests for reproducibility and security."""
    
    def test_full_reproducibility_workflow(self):
        """Test full reproducibility workflow."""
        # Create reproducibility manager
        manager = ReproducibilityManager(session_id="integration_test", random_seed=123)
        
        # Create experiment
        experiment_id = manager.create_experiment(
            "integration_test", 
            {"num_qubits": 3, "shots": 1000}
        )
        
        # Update experiment
        manager.update_experiment(experiment_id, 2.5, True, ["output.txt"])
        
        # Verify reproducibility
        is_reproducible, message = manager.verify_reproducibility(experiment_id)
        self.assertTrue(is_reproducible)
        
        # Get session summary
        summary = manager.get_session_summary()
        self.assertEqual(summary['total_experiments'], 1)
        self.assertEqual(summary['successful_experiments'], 1)
        self.assertEqual(summary['success_rate'], 1.0)
    
    def test_security_and_privacy_integration(self):
        """Test security and privacy integration."""
        # Create security manager
        security_manager = SecurityManager(privacy_mode=True)
        
        # Create metadata with sensitive data
        metadata = {
            'git_commit_hash': 'sensitive_hash',
            'working_directory': '/sensitive/path',
            'normal_field': 'normal_value'
        }
        
        # Create privacy report
        privacy_report = security_manager.create_privacy_report(metadata)
        
        # Check that sensitive data is redacted
        redacted_metadata = privacy_report['metadata']
        self.assertEqual(redacted_metadata['git_commit_hash'], '[REDACTED]')
        self.assertEqual(redacted_metadata['working_directory'], '[REDACTED]')
        self.assertEqual(redacted_metadata['normal_field'], 'normal_value')
    
    def test_deterministic_execution(self):
        """Test deterministic execution."""
        # Create deterministic random generator
        random_gen = DeterministicRandom(seed=456)
        
        # Generate random numbers
        rand1 = random_gen.random()
        rand2 = random_gen.random()
        
        # Create new generator with same seed
        random_gen2 = DeterministicRandom(seed=456)
        
        # Generate random numbers
        rand3 = random_gen2.random()
        rand4 = random_gen2.random()
        
        # Should be identical
        self.assertAlmostEqual(rand1, rand3, places=10)
        self.assertAlmostEqual(rand2, rand4, places=10)


if __name__ == '__main__':
    unittest.main()
