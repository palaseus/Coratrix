# Bell State Preparation and Measurement
# This script demonstrates the creation of a Bell state (|00⟩ + |11⟩)/√2
# and its measurement properties.

# Start with |00⟩ state
# Apply Hadamard gate to qubit 0 to create superposition
H q0

# Apply CNOT gate to create entanglement
# Control: q0, Target: q1
CNOT q0,q1

# The state is now (|00⟩ + |11⟩)/√2 - a Bell state
# Measure both qubits to see the correlation
MEASURE
