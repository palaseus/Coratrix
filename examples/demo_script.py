#!/usr/bin/env python3
"""
Demonstration script for Coratrix quantum computer.

This script shows how to use the Coratrix framework to create
and manipulate quantum states, demonstrating key quantum
computing concepts.
"""

import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qubit import QuantumState
from core.gates import HGate, CNOTGate, XGate, YGate, ZGate
from core.circuit import QuantumCircuit
from core.measurement import Measurement
from vm.parser import QuantumParser
from vm.executor import QuantumExecutor


def demonstrate_basic_quantum_operations():
    """Demonstrate basic quantum operations."""
    print("=" * 60)
    print("CORATRIX QUANTUM COMPUTER DEMONSTRATION")
    print("=" * 60)
    
    # Create a 2-qubit quantum circuit
    circuit = QuantumCircuit(2)
    print(f"Initial state: {circuit.get_state()}")
    print(f"Initial probabilities: {circuit.get_probabilities()}")
    print()
    
    # Apply Hadamard gate to create superposition
    h_gate = HGate()
    circuit.apply_gate(h_gate, [0])
    print(f"After H on q0: {circuit.get_state()}")
    print(f"Probabilities: {circuit.get_probabilities()}")
    print()
    
    # Apply CNOT to create entanglement
    cnot_gate = CNOTGate()
    circuit.apply_gate(cnot_gate, [0, 1])
    print(f"After CNOT: {circuit.get_state()}")
    print(f"Probabilities: {circuit.get_probabilities()}")
    print()
    
    # This is now a Bell state (|00⟩ + |11⟩)/√2
    print("This is a Bell state - maximally entangled!")
    print()


def demonstrate_measurement():
    """Demonstrate quantum measurement and state collapse."""
    print("=" * 60)
    print("QUANTUM MEASUREMENT DEMONSTRATION")
    print("=" * 60)
    
    # Create Bell state
    circuit = QuantumCircuit(2)
    h_gate = HGate()
    cnot_gate = CNOTGate()
    
    circuit.apply_gate(h_gate, [0])
    circuit.apply_gate(cnot_gate, [0, 1])
    
    print(f"Before measurement: {circuit.get_state()}")
    print(f"Probabilities: {circuit.get_probabilities()}")
    print()
    
    # Measure the state multiple times
    print("Measuring the Bell state 10 times:")
    for i in range(10):
        # Reset to Bell state
        circuit.reset()
        circuit.apply_gate(h_gate, [0])
        circuit.apply_gate(cnot_gate, [0, 1])
        
        # Measure
        measurement = Measurement(circuit.get_state())
        results = measurement.measure_all()
        print(f"Measurement {i+1}: {results}")
    
    print()
    print("Notice: Entangled qubits always give correlated results!")
    print()


def demonstrate_vm_execution():
    """Demonstrate the virtual machine execution."""
    print("=" * 60)
    print("VIRTUAL MACHINE EXECUTION DEMONSTRATION")
    print("=" * 60)
    
    # Create a quantum script
    script = """
# Bell state preparation
H q0
CNOT q0,q1
MEASURE
"""
    
    print("Quantum script:")
    print(script)
    print()
    
    # Parse and execute
    parser = QuantumParser()
    instructions = parser.parse_script(script)
    
    executor = QuantumExecutor(2)
    results = executor.execute_instructions(instructions)
    
    print(f"Execution results: {results}")
    print(f"Final state: {executor.get_state_string()}")
    print()


def demonstrate_entanglement_detection():
    """Demonstrate entanglement detection."""
    print("=" * 60)
    print("ENTANGLEMENT DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create Bell state
    circuit = QuantumCircuit(2)
    h_gate = HGate()
    cnot_gate = CNOTGate()
    
    circuit.apply_gate(h_gate, [0])
    circuit.apply_gate(cnot_gate, [0, 1])
    
    # Check entanglement
    executor = QuantumExecutor(2)
    executor.circuit = circuit
    entanglement_info = executor.get_entanglement_info()
    
    print(f"Quantum state: {circuit.get_state()}")
    print(f"Entanglement info: {entanglement_info}")
    print()
    
    # Create separable state for comparison
    circuit.reset()
    circuit.apply_gate(h_gate, [0])  # Only q0 in superposition
    
    executor.circuit = circuit
    entanglement_info = executor.get_entanglement_info()
    
    print(f"Separable state: {circuit.get_state()}")
    print(f"Entanglement info: {entanglement_info}")
    print()


def demonstrate_gate_operations():
    """Demonstrate various gate operations."""
    print("=" * 60)
    print("GATE OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    gates = {
        'X': XGate(),
        'Y': YGate(),
        'Z': ZGate(),
        'H': HGate(),
        'CNOT': CNOTGate()
    }
    
    for gate_name, gate in gates.items():
        print(f"\n{gate_name} Gate:")
        if gate_name != 'CNOT':  # CNOT requires 2 qubits
            print(f"Matrix for 1-qubit system:")
            matrix = gate.get_matrix(1, [0])
            print(matrix)
        else:
            print("Matrix for 2-qubit system:")
            matrix = gate.get_matrix(2, [0, 1])
            print(matrix)
        print()
        
        # Test on |0⟩ state
        if gate_name != 'CNOT':  # CNOT requires 2 qubits
            circuit = QuantumCircuit(1)
            gate.apply(circuit.get_state(), [0])
            print(f"X|0⟩ = {circuit.get_state()}")
        else:
            circuit = QuantumCircuit(2)
            gate.apply(circuit.get_state(), [0, 1])
            print(f"CNOT|00⟩ = {circuit.get_state()}")
        print()


def main():
    """Run all demonstrations."""
    try:
        demonstrate_basic_quantum_operations()
        demonstrate_measurement()
        demonstrate_vm_execution()
        demonstrate_entanglement_detection()
        demonstrate_gate_operations()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("Coratrix is ready for quantum computing!")
        print("Try running: python main.py --interactive")
        print("Or: python main.py --script examples/bell_state.qasm")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
