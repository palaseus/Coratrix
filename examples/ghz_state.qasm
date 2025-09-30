# GHZ State Preparation (3-qubit)
# Creates the maximally entangled GHZ state:
# |GHZ⟩ = (|000⟩ + |111⟩)/√2

# Start with |000⟩ state
# Apply Hadamard to first qubit to create superposition
H q0

# Apply CNOT gates to create entanglement
CNOT q0,q1
CNOT q1,q2

# The state is now (|000⟩ + |111⟩)/√2 - a GHZ state
# Measure to see the correlation
MEASURE
