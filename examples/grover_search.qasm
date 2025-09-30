# Grover's Search Algorithm (2-qubit version)
# This script demonstrates Grover's algorithm for searching
# in a 2-qubit database (4 items)

# Step 1: Initialize qubits in uniform superposition
H q0
H q1

# Step 2: Oracle function (marks the target state |11⟩)
# For |11⟩, apply a phase flip
CNOT q0,q1
Z q1
CNOT q0,q1

# Step 3: Diffusion operator (inversion about the mean)
H q0
H q1
X q0
X q1
CNOT q0,q1
Z q1
CNOT q0,q1
X q0
X q1
H q0
H q1

# Step 4: Measure to get the target state
MEASURE
