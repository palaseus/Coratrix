# Advanced Grover's Search Algorithm (3-qubit)
# Searches for the target state |111⟩ in a 3-qubit database

# Step 1: Initialize qubits in uniform superposition
H q0
H q1
H q2

# Step 2: Oracle function (marks the target state |111⟩)
# Apply phase flip to |111⟩ state
CNOT q0,q1
CNOT q1,q2
Z q2
CNOT q1,q2
CNOT q0,q1

# Step 3: Diffusion operator (inversion about the mean)
H q0
H q1
H q2
X q0
X q1
X q2
CNOT q0,q1
CNOT q1,q2
Z q2
CNOT q1,q2
CNOT q0,q1
X q0
X q1
X q2
H q0
H q1
H q2

# Step 4: Measure to get the target state
MEASURE
