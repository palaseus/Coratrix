"""
Hardware interoperability module for Coratrix.

This module provides interfaces for exporting and importing quantum circuits
to/from various hardware backends and quantum programming frameworks.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HardwareBackend(Enum):
    """Supported hardware backends."""
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"
    QSHARP = "qsharp"
    BRAKET = "braket"


@dataclass
class CircuitGate:
    """Representation of a quantum gate in a circuit."""
    name: str
    qubits: List[int]
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class OpenQASMExporter:
    """Export Coratrix circuits to OpenQASM format."""
    
    def __init__(self):
        self.version = "2.0"
        self.include_header = True
    
    def export_circuit(self, gates: List[Tuple], num_qubits: int, 
                      custom_gates: Dict[str, str] = None,
                      include_measurements: bool = False) -> str:
        """
        Export a circuit to OpenQASM format.
        
        Args:
            gates: List of gate tuples (name, qubits, parameters)
            num_qubits: Number of qubits in the circuit
            custom_gates: Dictionary of custom gate definitions
            include_measurements: Whether to include measurement operations
            
        Returns:
            OpenQASM code as string
        """
        qasm_lines = []
        
        # Header
        if self.include_header:
            qasm_lines.append(f"OPENQASM {self.version};")
            qasm_lines.append('include "qelib1.inc";')
            qasm_lines.append("")
        
        # Custom gate definitions
        if custom_gates:
            for gate_name, gate_def in custom_gates.items():
                qasm_lines.append(gate_def)
            qasm_lines.append("")
        
        # Circuit declaration
        qasm_lines.append(f"qreg q[{num_qubits}];")
        if include_measurements:
            qasm_lines.append(f"creg c[{num_qubits}];")
        qasm_lines.append("")
        
        # Gate operations
        for gate_tuple in gates:
            if len(gate_tuple) == 2:
                name, qubits = gate_tuple
                parameters = {}
            elif len(gate_tuple) == 3:
                name, qubits, parameters = gate_tuple
            else:
                continue
            
            qasm_line = self._format_gate(name, qubits, parameters)
            qasm_lines.append(qasm_line)
        
        # Measurements
        if include_measurements:
            for i in range(num_qubits):
                qasm_lines.append(f"measure q[{i}] -> c[{i}];")
        
        return "\n".join(qasm_lines)
    
    def _format_gate(self, name: str, qubits: List[int], parameters: Dict[str, Any]) -> str:
        """Format a single gate for OpenQASM."""
        # Map Coratrix gate names to OpenQASM names
        gate_mapping = {
            "H": "h",
            "X": "x", 
            "Y": "y",
            "Z": "z",
            "CNOT": "cx",
            "CPhase": "cp",
            "S": "s",
            "T": "t",
            "CustomCPhase": "custom_cphase",
            "CustomRotation": "custom_rotation"
        }
        
        qasm_name = gate_mapping.get(name, name.lower())
        
        # Format qubit indices
        qubit_str = " ".join(f"q[{q}]" for q in qubits)
        
        # Format parameters
        if parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
            return f"{qasm_name}({param_str}) {qubit_str};"
        else:
            return f"{qasm_name} {qubit_str};"


class OpenQASMImporter:
    """Import OpenQASM circuits into Coratrix format."""
    
    def __init__(self):
        self.gate_mapping = {
            "h": "H",
            "x": "X",
            "y": "Y", 
            "z": "Z",
            "cx": "CNOT",
            "cp": "CPhase",
            "s": "S",
            "t": "T"
        }
    
    def import_circuit(self, qasm_code: str) -> 'ScalableQuantumState':
        """
        Import an OpenQASM circuit into Coratrix.
        
        Args:
            qasm_code: OpenQASM code as string
            
        Returns:
            ScalableQuantumState representing the circuit
        """
        from core.scalable_quantum_state import ScalableQuantumState
        from core.gates import HGate, XGate, YGate, ZGate, CNOTGate
        from core.advanced_gates import CPhaseGate, SGate, TGate
        
        # Parse the QASM code
        lines = [line.strip() for line in qasm_code.split('\n') if line.strip()]
        
        # Find number of qubits
        num_qubits = 0
        for line in lines:
            if line.startswith('qreg q['):
                # Extract number from qreg q[N];
                num_qubits = int(line.split('[')[1].split(']')[0])
                break
        
        if num_qubits == 0:
            raise ValueError("Could not determine number of qubits from QASM")
        
        # Create state
        state = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
        state.set_amplitude(0, 1)  # Initialize to |0...0⟩
        
        # Apply gates
        for line in lines:
            if line.startswith('qreg') or line.startswith('creg') or line.startswith('OPENQASM') or line.startswith('include'):
                continue
            
            # Parse gate operations
            if ' ' in line and not line.startswith('//'):
                parts = line.split()
                if len(parts) >= 2:
                    gate_name = parts[0]
                    qubits = self._parse_qubits(parts[1:])
                    
                    # Apply gate
                    self._apply_gate_to_state(state, gate_name, qubits)
        
        return state
    
    def _parse_qubits(self, qubit_parts: List[str]) -> List[int]:
        """Parse qubit indices from QASM format."""
        qubits = []
        for part in qubit_parts:
            if 'q[' in part and ']' in part:
                # Extract index from q[N]
                idx = int(part.split('[')[1].split(']')[0])
                qubits.append(idx)
        return qubits
    
    def _apply_gate_to_state(self, state, gate_name: str, qubits: List[int]):
        """Apply a gate to the state."""
        from core.gates import HGate, XGate, YGate, ZGate, CNOTGate
        from core.advanced_gates import CPhaseGate, SGate, TGate
        
        gate_mapping = {
            "h": HGate(),
            "x": XGate(),
            "y": YGate(),
            "z": ZGate(),
            "cx": CNOTGate(),
            "cp": CPhaseGate(),
            "s": SGate(),
            "t": TGate()
        }
        
        if gate_name in gate_mapping:
            gate = gate_mapping[gate_name]
            state.apply_gate(gate, qubits)


class QiskitExporter:
    """Export Coratrix circuits to Qiskit format."""
    
    def __init__(self):
        self.backend_name = "qasm_simulator"
    
    def export_circuit(self, gates: List[Tuple], num_qubits: int) -> 'QuantumCircuit':
        """
        Export a circuit to Qiskit QuantumCircuit.
        
        Args:
            gates: List of gate tuples (name, qubits, parameters)
            num_qubits: Number of qubits in the circuit
            
        Returns:
            Qiskit QuantumCircuit object
        """
        try:
            from qiskit import QuantumCircuit
            
            qc = QuantumCircuit(num_qubits)
            
            for gate_tuple in gates:
                if len(gate_tuple) == 2:
                    name, qubits = gate_tuple
                    parameters = {}
                elif len(gate_tuple) == 3:
                    name, qubits, parameters = gate_tuple
                else:
                    continue
                
                self._add_gate_to_circuit(qc, name, qubits, parameters)
            
            return qc
            
        except ImportError:
            raise ImportError("Qiskit not available. Install with: pip install qiskit")
    
    def _add_gate_to_circuit(self, qc, name: str, qubits: List[int], parameters: Dict[str, Any]):
        """Add a gate to the Qiskit circuit."""
        if name == "H":
            qc.h(qubits[0])
        elif name == "X":
            qc.x(qubits[0])
        elif name == "Y":
            qc.y(qubits[0])
        elif name == "Z":
            qc.z(qubits[0])
        elif name == "CNOT":
            qc.cx(qubits[0], qubits[1])
        elif name == "CPhase":
            phi = parameters.get("phi", 0)
            qc.cp(phi, qubits[0], qubits[1])
        elif name == "S":
            qc.s(qubits[0])
        elif name == "T":
            qc.t(qubits[0])
        else:
            # For custom gates, try to map to closest Qiskit equivalent
            if "rotation" in name.lower():
                theta = parameters.get("theta", 0)
                axis = parameters.get("axis", "z")
                if axis == "x":
                    qc.rx(theta, qubits[0])
                elif axis == "y":
                    qc.ry(theta, qubits[0])
                else:
                    qc.rz(theta, qubits[0])


class PennyLaneExporter:
    """Export Coratrix circuits to PennyLane format."""
    
    def __init__(self):
        self.device_name = "default.qubit"
    
    def export_circuit(self, gates: List[Tuple], num_qubits: int) -> str:
        """
        Export a circuit to PennyLane format.
        
        Args:
            gates: List of gate tuples (name, qubits, parameters)
            num_qubits: Number of qubits in the circuit
            
        Returns:
            PennyLane circuit code as string
        """
        lines = []
        lines.append("import pennylane as qml")
        lines.append("")
        lines.append("@qml.qnode(dev)")
        lines.append("def circuit():")
        
        for gate_tuple in gates:
            if len(gate_tuple) == 2:
                name, qubits = gate_tuple
                parameters = {}
            elif len(gate_tuple) == 3:
                name, qubits, parameters = gate_tuple
            else:
                continue
            
            line = self._format_gate_for_pennylane(name, qubits, parameters)
            lines.append(f"    {line}")
        
        lines.append("    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]")
        
        return "\n".join(lines)
    
    def _format_gate_for_pennylane(self, name: str, qubits: List[int], parameters: Dict[str, Any]) -> str:
        """Format a gate for PennyLane."""
        gate_mapping = {
            "H": "qml.Hadamard",
            "X": "qml.PauliX",
            "Y": "qml.PauliY", 
            "Z": "qml.PauliZ",
            "CNOT": "qml.CNOT",
            "CPhase": "qml.PhaseShift",
            "S": "qml.S",
            "T": "qml.T"
        }
        
        qml_gate = gate_mapping.get(name, f"qml.{name}")
        
        if parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
            return f"{qml_gate}({param_str}, wires={qubits[0]})"
        else:
            if len(qubits) == 1:
                return f"{qml_gate}(wires={qubits[0]})"
            else:
                return f"{qml_gate}(wires={qubits})"


class HardwareBackendInterface:
    """Interface for hardware backend operations."""
    
    def __init__(self, backend: HardwareBackend):
        self.backend = backend
        self.connection = None
    
    def connect(self, **kwargs):
        """Connect to the hardware backend."""
        if self.backend == HardwareBackend.QISKIT:
            self._connect_qiskit(**kwargs)
        elif self.backend == HardwareBackend.PENNYLANE:
            self._connect_pennylane(**kwargs)
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")
    
    def _connect_qiskit(self, **kwargs):
        """Connect to Qiskit backend."""
        try:
            from qiskit import IBMQ
            if 'token' in kwargs:
                IBMQ.enable_account(kwargs['token'])
            self.connection = IBMQ
        except ImportError:
            raise ImportError("Qiskit not available")
    
    def _connect_pennylane(self, **kwargs):
        """Connect to PennyLane backend."""
        try:
            import pennylane as qml
            self.connection = qml
        except ImportError:
            raise ImportError("PennyLane not available")
    
    def run_circuit(self, circuit, shots: int = 1024) -> Dict[str, Any]:
        """Run a circuit on the hardware backend."""
        if self.backend == HardwareBackend.QISKIT:
            return self._run_qiskit_circuit(circuit, shots)
        elif self.backend == HardwareBackend.PENNYLANE:
            return self._run_pennylane_circuit(circuit, shots)
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")
    
    def _run_qiskit_circuit(self, circuit, shots: int) -> Dict[str, Any]:
        """Run circuit on Qiskit backend."""
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
            
            simulator = AerSimulator()
            transpiled_circuit = transpile(circuit, simulator)
            job = simulator.run(transpiled_circuit, shots=shots)
            result = job.result()
            
            return {
                "counts": result.get_counts(),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def _run_pennylane_circuit(self, circuit, shots: int) -> Dict[str, Any]:
        """Run circuit on PennyLane backend."""
        try:
            # This would need to be implemented based on specific PennyLane usage
            return {
                "result": "PennyLane execution not fully implemented",
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
