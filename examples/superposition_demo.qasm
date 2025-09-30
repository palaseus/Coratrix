# Superposition Demonstration
# This script shows how the Hadamard gate creates superposition states

# Start with |00⟩ state
# Apply Hadamard to qubit 0 to create (|0⟩ + |1⟩)/√2
H q0

# Apply Hadamard to qubit 1 to create (|0⟩ + |1⟩)/√2
H q1

# The state is now (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
# This is a uniform superposition of all 2-qubit basis states
MEASURE
