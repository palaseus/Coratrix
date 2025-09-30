"""
Tests for hardware interface functionality.

This module tests OpenQASM import/export and backend interface functionality.
"""

import unittest
import sys
import os
import tempfile
import json
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.openqasm_interface import OpenQASMInterface, OpenQASMParser, OpenQASMExporter
from hardware.backend_interface import (
    BackendManager, CoratrixSimulatorBackend, NoisySimulatorBackend,
    BackendResult, BackendCapabilities, IBMQStubBackend
)
from core.circuit import QuantumCircuit
from core.gates import HGate, XGate, CNOTGate


class TestOpenQASMInterface(unittest.TestCase):
    """Test cases for OpenQASM interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = OpenQASMInterface()
        self.parser = OpenQASMParser()
        self.exporter = OpenQASMExporter()
    
    def test_parse_simple_circuit(self):
        """Test parsing a simple OpenQASM circuit."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        """
        
        circuit = self.parser.parse_string(qasm_string)
        
        self.assertEqual(circuit.num_qubits, 2)
        self.assertEqual(len(circuit.gates), 2)
        
        # Check first gate (H)
        gate1, qubits1 = circuit.gates[0]
        self.assertEqual(gate1.name, "H")
        self.assertEqual(qubits1, [0])
        
        # Check second gate (CNOT)
        gate2, qubits2 = circuit.gates[1]
        self.assertEqual(gate2.name, "CNOT")
        self.assertEqual(qubits2, [0, 1])
    
    def test_parse_parameterized_gates(self):
        """Test parsing parameterized gates."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(1.57) q[0];
        ry(0.5) q[0];
        rz(2.0) q[0];
        """
        
        circuit = self.parser.parse_string(qasm_string)
        
        self.assertEqual(circuit.num_qubits, 1)
        self.assertEqual(len(circuit.gates), 3)
        
        # Check parameterized gates
        for i, (gate, qubits) in enumerate(circuit.gates):
            self.assertEqual(qubits, [0])
            if i == 0:
                self.assertEqual(gate.name, "Rx")
                self.assertAlmostEqual(gate.parameters['theta'], 1.57, places=2)
            elif i == 1:
                self.assertEqual(gate.name, "Ry")
                self.assertAlmostEqual(gate.parameters['theta'], 0.5, places=2)
            elif i == 2:
                self.assertEqual(gate.name, "Rz")
                self.assertAlmostEqual(gate.parameters['theta'], 2.0, places=2)
    
    def test_export_circuit(self):
        """Test exporting a circuit to OpenQASM."""
        circuit = QuantumCircuit(2)
        circuit.add_gate(HGate(), [0])
        circuit.add_gate(CNOTGate(), [0, 1])
        
        qasm_string = self.exporter.circuit_to_qasm(circuit)
        
        self.assertIn("OPENQASM 2.0", qasm_string)
        self.assertIn("qreg q[2]", qasm_string)
        self.assertIn("h q[0]", qasm_string)
        self.assertIn("cx q[0],q[1]", qasm_string)
    
    def test_export_parameterized_circuit(self):
        """Test exporting parameterized circuit."""
        from core.advanced_gates import RxGate, RyGate, RzGate
        
        circuit = QuantumCircuit(1)
        circuit.add_gate(RxGate(1.57), [0])
        circuit.add_gate(RyGate(0.5), [0])
        circuit.add_gate(RzGate(2.0), [0])
        
        qasm_string = self.exporter.circuit_to_qasm(circuit)
        
        self.assertIn("rx(1.57) q[0]", qasm_string)
        self.assertIn("ry(0.5) q[0]", qasm_string)
        self.assertIn("rz(2.0) q[0]", qasm_string)
    
    def test_validate_qasm(self):
        """Test OpenQASM validation."""
        valid_qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        """
        
        is_valid, errors = self.interface.validate_qasm(valid_qasm)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        invalid_qasm = """
        OPENQASM 2.0;
        qreg q[2];
        unknown_gate q[0];
        """
        
        is_valid, errors = self.interface.validate_qasm(invalid_qasm)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_file_operations(self):
        """Test file import/export operations."""
        # Create a test circuit
        circuit = QuantumCircuit(2)
        circuit.add_gate(HGate(), [0])
        circuit.add_gate(CNOTGate(), [0, 1])
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False) as f:
            temp_file = f.name
            self.interface.export_circuit(circuit, temp_file)
        
        try:
            # Import from file
            imported_circuit = self.interface.import_circuit(temp_file)
            
            self.assertEqual(imported_circuit.num_qubits, circuit.num_qubits)
            self.assertEqual(len(imported_circuit.gates), len(circuit.gates))
            
        finally:
            # Clean up
            os.unlink(temp_file)


class TestBackendInterface(unittest.TestCase):
    """Test cases for backend interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend_manager = BackendManager()
        self.simulator_backend = CoratrixSimulatorBackend()
        self.noisy_backend = NoisySimulatorBackend()
    
    def test_backend_registration(self):
        """Test backend registration and retrieval."""
        # Check that default backends are registered
        backends = self.backend_manager.list_backends()
        self.assertGreater(len(backends), 0)
        
        # Check specific backends
        self.assertIn("coratrix_simulator", backends)
        self.assertIn("noisy_simulator", backends)
    
    def test_backend_capabilities(self):
        """Test backend capabilities."""
        capabilities = self.backend_manager.get_backend_capabilities("coratrix_simulator")
        
        self.assertIsNotNone(capabilities)
        self.assertGreater(capabilities.max_qubits, 0)
        self.assertGreater(len(capabilities.supported_gates), 0)
        self.assertTrue(capabilities.supports_measurement)
    
    def test_circuit_execution(self):
        """Test circuit execution on backends."""
        # Create a simple circuit
        circuit = QuantumCircuit(2)
        circuit.add_gate(HGate(), [0])
        circuit.add_gate(CNOTGate(), [0, 1])
        
        # Execute on simulator backend
        result = self.backend_manager.execute_circuit(circuit, "coratrix_simulator", shots=100)
        
        self.assertTrue(result.success)
        self.assertGreater(result.execution_time, 0)
        self.assertGreater(len(result.counts), 0)
        self.assertIn("backend_name", result.backend_info)
    
    def test_circuit_validation(self):
        """Test circuit validation."""
        # Create a circuit that exceeds backend limits
        circuit = QuantumCircuit(20)  # Exceeds most backend limits
        
        result = self.backend_manager.execute_circuit(circuit, "coratrix_simulator", shots=100)
        
        # Should fail validation
        self.assertFalse(result.success)
        self.assertIn("validation", result.error_message.lower())
    
    def test_noisy_simulator(self):
        """Test noisy simulator backend."""
        # Create a simple circuit
        circuit = QuantumCircuit(2)
        circuit.add_gate(HGate(), [0])
        circuit.add_gate(CNOTGate(), [0, 1])
        
        # Execute on noisy simulator
        result = self.backend_manager.execute_circuit(circuit, "noisy_simulator", shots=100)
        
        self.assertTrue(result.success)
        self.assertGreater(result.execution_time, 0)
        self.assertIn("noise_model", result.backend_info)
    
    def test_backend_availability(self):
        """Test backend availability checking."""
        # Check that backends report availability correctly
        for backend_name in self.backend_manager.list_backends():
            backend = self.backend_manager.get_backend(backend_name)
            self.assertIsNotNone(backend)
            self.assertTrue(backend.is_available())
    
    def test_execution_timeout(self):
        """Test execution timeout handling."""
        # Create a circuit that might take a long time
        circuit = QuantumCircuit(8)
        for _ in range(100):  # Many gates
            circuit.add_gate(HGate(), [0])
            circuit.add_gate(XGate(), [0])
        
        result = self.backend_manager.execute_circuit(circuit, "coratrix_simulator", shots=10)
        
        # Should either succeed or fail gracefully
        self.assertIsInstance(result.success, bool)
        self.assertGreaterEqual(result.execution_time, 0)
    
    def test_shot_count_handling(self):
        """Test different shot counts."""
        circuit = QuantumCircuit(2)
        circuit.add_gate(HGate(), [0])
        circuit.add_gate(CNOTGate(), [0, 1])
        
        # Test different shot counts
        for shots in [1, 10, 100, 1000]:
            result = self.backend_manager.execute_circuit(circuit, "coratrix_simulator", shots=shots)
            
            self.assertTrue(result.success)
            self.assertGreater(result.execution_time, 0)
            
            # Check that results make sense
            total_counts = sum(result.counts.values())
            self.assertLessEqual(total_counts, shots)  # Should not exceed requested shots


class TestHardwareIntegration(unittest.TestCase):
    """Integration tests for hardware interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = OpenQASMInterface()
        self.backend_manager = BackendManager()
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from OpenQASM to execution."""
        # Create OpenQASM string
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        """
        
        # Import circuit
        circuit = self.interface.import_circuit_string(qasm_string)
        
        # Execute on backend
        result = self.backend_manager.execute_circuit(circuit, "coratrix_simulator", shots=100)
        
        # Verify results
        self.assertTrue(result.success)
        self.assertGreater(len(result.counts), 0)
        
        # Check that we get expected Bell state results
        total_counts = sum(result.counts.values())
        self.assertLessEqual(total_counts, 100)
    
    def test_export_import_roundtrip(self):
        """Test export-import roundtrip."""
        # Create original circuit
        original_circuit = QuantumCircuit(2)
        original_circuit.add_gate(HGate(), [0])
        original_circuit.add_gate(CNOTGate(), [0, 1])
        
        # Export to OpenQASM
        qasm_string = self.interface.circuit_to_qasm(original_circuit)
        
        # Import back
        imported_circuit = self.interface.import_circuit_string(qasm_string)
        
        # Compare circuits
        self.assertEqual(imported_circuit.num_qubits, original_circuit.num_qubits)
        self.assertEqual(len(imported_circuit.gates), len(original_circuit.gates))
        
        # Execute both circuits and compare results
        result1 = self.backend_manager.execute_circuit(original_circuit, "coratrix_simulator", shots=100)
        result2 = self.backend_manager.execute_circuit(imported_circuit, "coratrix_simulator", shots=100)
        
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        
        # Results should be similar (allowing for measurement randomness)
        self.assertGreater(len(result1.counts), 0)
        self.assertGreater(len(result2.counts), 0)


if __name__ == '__main__':
    unittest.main()
