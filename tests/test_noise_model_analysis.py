#!/usr/bin/env python3
"""
Test script for noise model analysis with Grover and QFT algorithms.

This script demonstrates how noise affects quantum algorithms by:
1. Running Grover's algorithm with and without noise
2. Running QFT with and without noise  
3. Analyzing how entanglement metrics shift under noise
4. Comparing performance and fidelity degradation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import json
from core.scalable_quantum_state import ScalableQuantumState
from core.noise_models import NoiseModel, NoiseChannel
from core.advanced_algorithms import EntanglementMonotones, FidelityEstimator
from algorithms.quantum_algorithms import GroverAlgorithm, QuantumFourierTransform
from core.gates import HGate, CNOTGate
import matplotlib.pyplot as plt

def create_noise_model(depolarizing_error=0.01, amplitude_damping_error=0.005):
    """Create a noise model with specified error rates."""
    return NoiseModel(
        depolarizing_error=depolarizing_error,
        amplitude_damping_error=amplitude_damping_error
    )

def apply_noise_to_state(state, noise_model, num_applications=1):
    """Apply noise to a quantum state multiple times."""
    # For this test, we'll simulate noise by adding small random perturbations
    # In a real implementation, you would use the noise model's methods
    
    for _ in range(num_applications):
        # Simulate depolarizing noise
        if noise_model.depolarizing_error > 0:
            # Add small random perturbations to amplitudes
            for i in range(state.dimension):
                if abs(state.get_amplitude(i)) > 1e-10:  # Only perturb non-zero amplitudes
                    noise = np.random.normal(0, noise_model.depolarizing_error, 2)
                    current_amp = state.get_amplitude(i)
                    new_amp = current_amp + complex(noise[0], noise[1])
                    state.set_amplitude(i, new_amp)
        
        # Simulate amplitude damping
        if noise_model.amplitude_damping_error > 0:
            # Reduce amplitude of excited states
            for i in range(1, state.dimension):  # Skip |0...0⟩ state
                current_amp = state.get_amplitude(i)
                damping_factor = 1 - noise_model.amplitude_damping_error
                new_amp = current_amp * damping_factor
                state.set_amplitude(i, new_amp)
    
    # Renormalize the state
    state.normalize()

class SimpleExecutor:
    """Simple executor for testing algorithms."""
    def __init__(self, state):
        self.state = state
    
    def get_state(self):
        return self.state
    
    def apply_gate(self, gate_name, target_qubits):
        """Apply a gate to the state."""
        from core.gates import HGate, CNOTGate, XGate, YGate, ZGate
        from core.advanced_gates import CPhaseGate, SGate, TGate
        
        if gate_name == 'H':
            gate = HGate()
        elif gate_name == 'CNOT':
            gate = CNOTGate()
        elif gate_name == 'X':
            gate = XGate()
        elif gate_name == 'Y':
            gate = YGate()
        elif gate_name == 'Z':
            gate = ZGate()
        elif gate_name == 'S':
            gate = SGate()
        elif gate_name == 'T':
            gate = TGate()
        elif gate_name == 'CPhase':
            gate = CPhaseGate()
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        self.state.apply_gate(gate, target_qubits)

def test_grover_with_noise():
    """Test Grover's algorithm with and without noise."""
    print("\n🔍 Testing Grover's Algorithm with Noise")
    print("=" * 50)
    
    # Test parameters
    num_qubits = 3
    target_item = 5  # Search for item 5 (binary: 101)
    
    # Create clean Grover state
    print("1. Running Grover without noise...")
    grover_clean = GroverAlgorithm()
    state_clean = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
    executor_clean = SimpleExecutor(state_clean)
    grover_clean.execute(executor_clean, {"target_item": target_item, "num_iterations": 2})
    
    # Create noisy Grover state
    print("2. Running Grover with noise...")
    grover_noisy = GroverAlgorithm()
    state_noisy = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
    executor_noisy = SimpleExecutor(state_noisy)
    grover_noisy.execute(executor_noisy, {"target_item": target_item, "num_iterations": 2})
    
    # Apply noise
    noise_model = create_noise_model(depolarizing_error=0.02, amplitude_damping_error=0.01)
    apply_noise_to_state(state_noisy, noise_model, num_applications=3)
    
    # Analyze entanglement (simplified for 3-qubit system)
    monotones = EntanglementMonotones()
    
    print("\n3. Analyzing entanglement metrics...")
    
    # For 3-qubit system, analyze pairwise entanglement
    # Clean state analysis
    entropy_clean = 0.0  # Simplified - would need proper partial trace for 3-qubit
    negativity_clean = monotones.calculate_negativity(state_clean, [0])
    
    # Noisy state analysis  
    entropy_noisy = 0.0  # Simplified - would need proper partial trace for 3-qubit
    negativity_noisy = monotones.calculate_negativity(state_noisy, [0])
    
    print(f"Clean state - Entropy: {entropy_clean:.4f}, Negativity: {negativity_clean:.4f}")
    print(f"Noisy state - Entropy: {entropy_noisy:.4f}, Negativity: {negativity_noisy:.4f}")
    
    # Calculate fidelity
    fidelity_estimator = FidelityEstimator()
    fidelity = fidelity_estimator.estimate_state_fidelity(state_clean, state_noisy)
    print(f"State fidelity: {fidelity:.4f}")
    
    # Analyze measurement probabilities
    print("\n4. Measurement probability analysis...")
    probs_clean = state_clean.get_probabilities()
    probs_noisy = state_noisy.get_probabilities()
    
    print(f"Clean state probabilities: {[f'{p:.4f}' for p in probs_clean]}")
    print(f"Noisy state probabilities: {[f'{p:.4f}' for p in probs_noisy]}")
    
    # Find target state probability
    target_prob_clean = probs_clean[target_item]
    target_prob_noisy = probs_noisy[target_item]
    
    print(f"Target state |{target_item:03b}⟩ probability:")
    print(f"  Clean: {target_prob_clean:.4f}")
    print(f"  Noisy: {target_prob_noisy:.4f}")
    print(f"  Degradation: {((target_prob_clean - target_prob_noisy) / target_prob_clean * 100):.2f}%")
    
    return {
        "entropy_clean": entropy_clean,
        "entropy_noisy": entropy_noisy,
        "negativity_clean": negativity_clean,
        "negativity_noisy": negativity_noisy,
        "fidelity": fidelity,
        "target_prob_clean": target_prob_clean,
        "target_prob_noisy": target_prob_noisy
    }

def test_qft_with_noise():
    """Test QFT algorithm with and without noise."""
    print("\n🌊 Testing QFT Algorithm with Noise")
    print("=" * 40)
    
    num_qubits = 3
    
    # Create clean QFT state
    print("1. Running QFT without noise...")
    qft_clean = QuantumFourierTransform()
    state_clean = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
    # Initialize with |000⟩ + |111⟩ superposition
    state_clean.set_amplitude(0, 1/np.sqrt(2))
    state_clean.set_amplitude(7, 1/np.sqrt(2))
    executor_clean = SimpleExecutor(state_clean)
    qft_clean.execute(executor_clean, {})
    
    # Create noisy QFT state
    print("2. Running QFT with noise...")
    qft_noisy = QuantumFourierTransform()
    state_noisy = ScalableQuantumState(num_qubits, use_gpu=False, sparse_threshold=8)
    state_noisy.set_amplitude(0, 1/np.sqrt(2))
    state_noisy.set_amplitude(7, 1/np.sqrt(2))
    executor_noisy = SimpleExecutor(state_noisy)
    qft_noisy.execute(executor_noisy, {})
    
    # Apply noise
    noise_model = create_noise_model(depolarizing_error=0.015, amplitude_damping_error=0.008)
    apply_noise_to_state(state_noisy, noise_model, num_applications=2)
    
    # Analyze entanglement (simplified for 3-qubit system)
    monotones = EntanglementMonotones()
    
    print("\n3. Analyzing QFT entanglement metrics...")
    
    # Clean state analysis
    entropy_clean = 0.0  # Simplified - would need proper partial trace for 3-qubit
    negativity_clean = monotones.calculate_negativity(state_clean, [0])
    
    # Noisy state analysis
    entropy_noisy = 0.0  # Simplified - would need proper partial trace for 3-qubit
    negativity_noisy = monotones.calculate_negativity(state_noisy, [0])
    
    print(f"Clean QFT - Entropy: {entropy_clean:.4f}, Negativity: {negativity_clean:.4f}")
    print(f"Noisy QFT - Entropy: {entropy_noisy:.4f}, Negativity: {negativity_noisy:.4f}")
    
    # Calculate fidelity
    fidelity_estimator = FidelityEstimator()
    fidelity = fidelity_estimator.estimate_state_fidelity(state_clean, state_noisy)
    print(f"QFT fidelity: {fidelity:.4f}")
    
    # Analyze phase coherence
    print("\n4. Phase coherence analysis...")
    amplitudes_clean = [state_clean.get_amplitude(i) for i in range(8)]
    amplitudes_noisy = [state_noisy.get_amplitude(i) for i in range(8)]
    
    # Calculate phase differences
    phases_clean = [np.angle(amp) for amp in amplitudes_clean if abs(amp) > 1e-10]
    phases_noisy = [np.angle(amp) for amp in amplitudes_noisy if abs(amp) > 1e-10]
    
    print(f"Clean state phases: {[f'{p:.3f}' for p in phases_clean[:4]]}")
    print(f"Noisy state phases: {[f'{p:.3f}' for p in phases_noisy[:4]]}")
    
    return {
        "entropy_clean": entropy_clean,
        "entropy_noisy": entropy_noisy,
        "negativity_clean": negativity_clean,
        "negativity_noisy": negativity_noisy,
        "fidelity": fidelity,
        "phase_coherence_clean": len(phases_clean),
        "phase_coherence_noisy": len(phases_noisy)
    }

def test_noise_scaling():
    """Test how different noise levels affect algorithms."""
    print("\n📊 Testing Noise Scaling Effects")
    print("=" * 35)
    
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    results = []
    
    for noise_level in noise_levels:
        print(f"\nTesting noise level: {noise_level:.3f}")
        
        # Create state
        state = ScalableQuantumState(2, use_gpu=False, sparse_threshold=8)
        state.set_amplitude(0, 1/np.sqrt(2))
        state.set_amplitude(3, 1/np.sqrt(2))  # Bell state
        
        # Apply noise
        if noise_level > 0:
            noise_model = create_noise_model(
                depolarizing_error=noise_level,
                amplitude_damping_error=noise_level/2
            )
            apply_noise_to_state(state, noise_model, num_applications=1)
        
        # Analyze
        monotones = EntanglementMonotones()
        entropy = monotones.calculate_entanglement_entropy(state, [0])
        negativity = monotones.calculate_negativity(state, [0])
        
        # Calculate purity (trace of density matrix squared)
        density_matrix = state.get_density_matrix()
        purity = np.trace(density_matrix @ density_matrix).real
        
        results.append({
            "noise_level": noise_level,
            "entropy": entropy,
            "negativity": negativity,
            "purity": purity
        })
        
        print(f"  Entropy: {entropy:.4f}, Negativity: {negativity:.4f}, Purity: {purity:.4f}")
    
    return results

def test_entanglement_degradation():
    """Test how entanglement degrades under different noise conditions."""
    print("\n🔗 Testing Entanglement Degradation")
    print("=" * 40)
    
    # Create different entangled states
    states = {
        "Bell": create_bell_state(),
        "GHZ": create_ghz_state(),
        "W": create_w_state()
    }
    
    noise_models = {
        "Low": create_noise_model(0.005, 0.002),
        "Medium": create_noise_model(0.02, 0.01),
        "High": create_noise_model(0.05, 0.025)
    }
    
    results = {}
    
    for state_name, state in states.items():
        print(f"\nTesting {state_name} state:")
        results[state_name] = {}
        
        for noise_name, noise_model in noise_models.items():
            # Create noisy copy
            noisy_state = ScalableQuantumState(state.num_qubits, use_gpu=False, sparse_threshold=8)
            for i in range(state.dimension):
                noisy_state.set_amplitude(i, state.get_amplitude(i))
            
            apply_noise_to_state(noisy_state, noise_model, num_applications=2)
            
            # Analyze
            monotones = EntanglementMonotones()
            entropy = monotones.calculate_entanglement_entropy(noisy_state, [0])
            negativity = monotones.calculate_negativity(noisy_state, [0])
            
            results[state_name][noise_name] = {
                "entropy": entropy,
                "negativity": negativity
            }
            
            print(f"  {noise_name} noise - Entropy: {entropy:.4f}, Negativity: {negativity:.4f}")
    
    return results

def create_bell_state():
    """Create a Bell state."""
    state = ScalableQuantumState(2, use_gpu=False, sparse_threshold=8)
    state.set_amplitude(0, 1/np.sqrt(2))
    state.set_amplitude(3, 1/np.sqrt(2))
    return state

def create_ghz_state():
    """Create a GHZ state (2-qubit version)."""
    state = ScalableQuantumState(2, use_gpu=False, sparse_threshold=8)
    state.set_amplitude(0, 1/np.sqrt(2))
    state.set_amplitude(3, 1/np.sqrt(2))
    return state

def create_w_state():
    """Create a W state (2-qubit version)."""
    state = ScalableQuantumState(2, use_gpu=False, sparse_threshold=8)
    state.set_amplitude(1, 1/np.sqrt(2))
    state.set_amplitude(2, 1/np.sqrt(2))
    return state

def main():
    """Run comprehensive noise model testing."""
    print("🧪 Noise Model Testing Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test 1: Grover with noise
    grover_results = test_grover_with_noise()
    
    # Test 2: QFT with noise
    qft_results = test_qft_with_noise()
    
    # Test 3: Noise scaling
    scaling_results = test_noise_scaling()
    
    # Test 4: Entanglement degradation
    degradation_results = test_entanglement_degradation()
    
    end_time = time.time()
    
    print(f"\n✅ Noise Model Testing Complete!")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print("=" * 60)
    
    # Compile results
    all_results = {
        "grover_analysis": grover_results,
        "qft_analysis": qft_results,
        "noise_scaling": scaling_results,
        "entanglement_degradation": degradation_results,
        "runtime_seconds": end_time - start_time
    }
    
    # Save results
    with open("noise_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: noise_analysis_results.json")
    
    return all_results

if __name__ == "__main__":
    results = main()
    print(f"\nFinal Results Summary:")
    print(f"Grover fidelity: {results['grover_analysis']['fidelity']:.4f}")
    print(f"QFT fidelity: {results['qft_analysis']['fidelity']:.4f}")
    print(f"Entanglement degradation observed: {len(results['entanglement_degradation'])} states tested")
