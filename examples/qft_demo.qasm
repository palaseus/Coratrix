# Quantum Fourier Transform Demo (3-qubit)
# Applies QFT to create superposition of all computational basis states

# Start with |000⟩ state
# Apply QFT to create superposition
H q0
# Controlled phase gates would go here in a full implementation
# For simplicity, we'll use H gates

H q1
H q2

# The state is now a superposition of all 3-qubit basis states
# Measure to see the distribution
MEASURE
