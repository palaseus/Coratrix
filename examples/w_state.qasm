# W State Preparation (3-qubit)
# Creates the W state:
# |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3

# Start with |100⟩ state
X q0

# Apply controlled operations to create W state
H q1
CNOT q0,q1
H q2
CNOT q0,q2

# The state is now (|100⟩ + |010⟩ + |001⟩)/√3 - a W state
# Measure to see the distribution
MEASURE
