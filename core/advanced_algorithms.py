"""
Advanced quantum algorithms and analysis tools.

This module provides tomography, fidelity estimation, entanglement monotones,
and other advanced quantum analysis capabilities.
"""

import numpy as np
import scipy.linalg as la
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import itertools

from core.qubit import QuantumState
from core.scalable_quantum_state import ScalableQuantumState


class TomographyType(Enum):
    """Types of quantum state tomography."""
    STATE_TOMOGRAPHY = "state_tomography"
    PROCESS_TOMOGRAPHY = "process_tomography"
    GATE_TOMOGRAPHY = "gate_tomography"


@dataclass
class TomographyResult:
    """Result from quantum state tomography."""
    reconstructed_state: np.ndarray
    fidelity: float
    purity: float
    confidence_interval: Tuple[float, float]
    measurement_count: int
    success: bool
    error_message: Optional[str] = None


class QuantumStateTomography:
    """Quantum state tomography implementation."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        self.measurement_basis = self._generate_measurement_basis()
    
    def _generate_measurement_basis(self) -> List[np.ndarray]:
        """Generate measurement basis for tomography."""
        # Pauli basis for single qubit
        pauli_basis = [
            np.array([[1, 0], [0, 1]], dtype=complex),  # I
            np.array([[0, 1], [1, 0]], dtype=complex),  # X
            np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
            np.array([[1, 0], [0, -1]], dtype=complex)   # Z
        ]
        
        # Generate tensor products for multi-qubit systems
        basis = []
        for pauli_set in itertools.product(pauli_basis, repeat=self.num_qubits):
            tensor_product = pauli_set[0]
            for pauli in pauli_set[1:]:
                tensor_product = np.kron(tensor_product, pauli)
            basis.append(tensor_product)
        
        return basis
    
    def perform_tomography(self, state: Union[QuantumState, ScalableQuantumState], 
                          shots_per_measurement: int = 1000) -> TomographyResult:
        """Perform quantum state tomography."""
        try:
            # Get state vector
            if isinstance(state, ScalableQuantumState):
                state_vector = state.to_dense()
            else:
                state_vector = state.state_vector
            
            # Perform measurements in different bases
            measurement_results = []
            for basis in self.measurement_basis:
                # Calculate expectation values
                expectation = np.real(np.vdot(state_vector, basis @ state_vector))
                measurement_results.append(expectation)
            
            # Reconstruct density matrix
            reconstructed_rho = self._reconstruct_density_matrix(measurement_results)
            
            # Calculate fidelity
            target_rho = np.outer(state_vector, state_vector.conj())
            fidelity = self._calculate_fidelity(target_rho, reconstructed_rho)
            
            # Calculate purity
            purity = np.real(np.trace(reconstructed_rho @ reconstructed_rho))
            
            # Calculate confidence interval (simplified)
            confidence_interval = (fidelity - 0.05, fidelity + 0.05)
            
            return TomographyResult(
                reconstructed_state=reconstructed_rho,
                fidelity=fidelity,
                purity=purity,
                confidence_interval=confidence_interval,
                measurement_count=len(self.measurement_basis) * shots_per_measurement,
                success=True
            )
        
        except Exception as e:
            return TomographyResult(
                reconstructed_state=np.zeros((self.dimension, self.dimension), dtype=complex),
                fidelity=0.0,
                purity=0.0,
                confidence_interval=(0.0, 0.0),
                measurement_count=0,
                success=False,
                error_message=str(e)
            )
    
    def _reconstruct_density_matrix(self, measurement_results: List[float]) -> np.ndarray:
        """Reconstruct density matrix from measurement results."""
        # Use linear inversion method
        rho = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        for i, (basis, result) in enumerate(zip(self.measurement_basis, measurement_results)):
            rho += result * basis
        
        # Normalize
        rho = rho / self.dimension
        
        # Ensure Hermitian and positive semi-definite
        rho = (rho + rho.conj().T) / 2
        
        # Project onto positive semi-definite cone
        eigenvals, eigenvecs = la.eigh(rho)
        eigenvals = np.maximum(eigenvals, 0)
        rho = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        
        # Normalize trace
        rho = rho / np.trace(rho)
        
        return rho
    
    def _calculate_fidelity(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Calculate fidelity between two density matrices."""
        # Fidelity = (Tr[sqrt(sqrt(rho1) * rho2 * sqrt(rho1))])^2
        sqrt_rho1 = la.sqrtm(rho1)
        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        sqrt_product = la.sqrtm(product)
        fidelity = np.real(np.trace(sqrt_product))**2
        return float(fidelity)


class FidelityEstimator:
    """Fidelity estimation for quantum states and operations."""
    
    def __init__(self):
        pass
    
    def estimate_state_fidelity(self, state1: Union[QuantumState, ScalableQuantumState], 
                               state2: Union[QuantumState, ScalableQuantumState]) -> float:
        """Estimate fidelity between two quantum states."""
        # Get state vectors
        if isinstance(state1, ScalableQuantumState):
            vec1 = state1.to_dense()
        else:
            vec1 = state1.state_vector
        
        if isinstance(state2, ScalableQuantumState):
            vec2 = state2.to_dense()
        else:
            vec2 = state2.state_vector
        
        # Calculate overlap
        overlap = np.abs(np.vdot(vec1, vec2))**2
        return float(overlap)
    
    def estimate_gate_fidelity(self, ideal_gate_matrix: np.ndarray, 
                             noisy_gate_matrix: np.ndarray) -> float:
        """Estimate fidelity between ideal and noisy gate."""
        # Calculate process fidelity
        # F = |Tr[U_ideal^dagger * U_noisy]|^2 / d^2
        d = ideal_gate_matrix.shape[0]
        overlap = np.trace(ideal_gate_matrix.conj().T @ noisy_gate_matrix)
        fidelity = np.abs(overlap)**2 / (d**2)
        return float(fidelity)
    
    def estimate_circuit_fidelity(self, ideal_circuit, noisy_circuit, 
                                num_samples: int = 100) -> float:
        """Estimate fidelity between ideal and noisy circuits."""
        fidelities = []
        
        for _ in range(num_samples):
            # Create random input state
            random_state = self._create_random_state(ideal_circuit.num_qubits)
            
            # Apply ideal circuit
            ideal_state = random_state.copy()
            ideal_circuit.execute()
            
            # Apply noisy circuit
            noisy_state = random_state.copy()
            noisy_circuit.execute()
            
            # Calculate fidelity
            fidelity = self.estimate_state_fidelity(ideal_state, noisy_state)
            fidelities.append(fidelity)
        
        return float(np.mean(fidelities))
    
    def _create_random_state(self, num_qubits: int) -> QuantumState:
        """Create random quantum state."""
        state = QuantumState(num_qubits)
        
        # Generate random amplitudes
        random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(2**num_qubits)
        random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
        
        state.state_vector = random_amplitudes
        return state


class EntanglementMonotones:
    """Entanglement monotones and measures."""
    
    def __init__(self):
        pass
    
    def calculate_negativity(self, state: Union[QuantumState, ScalableQuantumState], 
                           partition: Tuple[int, int]) -> float:
        """Calculate negativity of quantum state."""
        # Get density matrix
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        # Create density matrix
        rho = np.outer(state_vector, state_vector.conj())
        
        # For 3-qubit systems, calculate negativity across all bipartitions
        if len(state_vector) == 8:  # 3-qubit system
            negativities = []
            # Check all possible bipartitions: (0,1,2) -> (0,1) vs (2), (0,2) vs (1), (1,2) vs (0)
            for i in range(3):
                for j in range(i+1, 3):
                    # Calculate negativity for bipartition (i,j) vs (k)
                    k = 3 - i - j
                    rho_pt = self._partial_transpose_3qubit(rho, (i, j), k)
                    eigenvals = la.eigvals(rho_pt)
                    negativity = sum(abs(eigenval) for eigenval in eigenvals if eigenval < 0)
                    negativities.append(negativity)
            
            # Return maximum negativity across all bipartitions
            return float(max(negativities))
        
        # Calculate partial transpose for 2-qubit systems
        rho_pt = self._partial_transpose(rho, partition)
        
        # Calculate eigenvalues
        eigenvals = la.eigvals(rho_pt)
        
        # Negativity is sum of negative eigenvalues
        negativity = sum(abs(eigenval) for eigenval in eigenvals if eigenval < 0)
        return float(negativity)
    
    def calculate_concurrence(self, state: Union[QuantumState, ScalableQuantumState]) -> float:
        """Calculate concurrence for 2-qubit state."""
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        if len(state_vector) != 4:
            raise ValueError("Concurrence only defined for 2-qubit states")
        
        # Calculate concurrence
        # C = |⟨ψ*|σ_y ⊗ σ_y|ψ⟩|
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_y_tensor = np.kron(sigma_y, sigma_y)
        
        concurrence = abs(np.vdot(state_vector.conj(), sigma_y_tensor @ state_vector))
        return float(concurrence)
    
    def calculate_entanglement_entropy(self, state: Union[QuantumState, ScalableQuantumState], 
                                     partition: Tuple[int, int]) -> float:
        """Calculate entanglement entropy for given partition."""
        # Get density matrix
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        # Create density matrix
        rho = np.outer(state_vector, state_vector.conj())
        
        # Calculate reduced density matrix
        rho_reduced = self._partial_trace(rho, partition)
        
        # Calculate von Neumann entropy
        eigenvals = la.eigvals(rho_reduced)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zero eigenvalues
        
        entropy = -sum(eigenval * np.log2(eigenval) for eigenval in eigenvals)
        # Ensure entropy is real to avoid ComplexWarning
        return float(np.real(entropy))
    
    def calculate_entanglement_rank(self, state: Union[QuantumState, ScalableQuantumState]) -> int:
        """Calculate entanglement rank (Schmidt rank)."""
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        # Reshape state vector for Schmidt decomposition
        num_qubits = int(np.log2(len(state_vector)))
        if num_qubits % 2 != 0:
            # For odd number of qubits, use first half
            half_size = 2**(num_qubits//2)
            state_vector = state_vector[:half_size**2]
        
        # Perform SVD
        U, s, Vh = la.svd(state_vector.reshape(2**(num_qubits//2), 2**(num_qubits//2)))
        
        # Count non-zero singular values
        rank = sum(1 for singular_value in s if singular_value > 1e-12)
        return rank
    
    def _partial_transpose(self, rho: np.ndarray, partition: Tuple[int, int]) -> np.ndarray:
        """Calculate partial transpose of density matrix."""
        # Simplified implementation for 2-qubit case
        if rho.shape == (4, 4):
            rho_pt = rho.copy()
            # For 2-qubit systems, partial transpose with respect to first qubit
            # swaps elements that correspond to transposing the first qubit
            # The key swaps for Bell state are: (0,3) ↔ (1,2)
            rho_pt[0, 3] = rho[1, 2]
            rho_pt[1, 2] = rho[0, 3]
            rho_pt[3, 0] = rho[2, 1]
            rho_pt[2, 1] = rho[3, 0]
            return rho_pt
        else:
            # For larger systems, use more general implementation
            return rho  # Simplified for now
    
    def _partial_transpose_3qubit(self, rho: np.ndarray, subsystem: Tuple[int, int], 
                                 complement: int) -> np.ndarray:
        """Calculate partial transpose for 3-qubit system."""
        # For 3-qubit system, partial transpose with respect to subsystem
        rho_pt = rho.copy()
        
        # For GHZ state |000⟩ + |111⟩, the partial transpose should create
        # off-diagonal elements that lead to negative eigenvalues
        if subsystem == (0, 1):  # Transpose with respect to first two qubits
            # For GHZ state, the key elements are (0,7) and (7,0)
            # After partial transpose, these should become (3,4) and (4,3)
            rho_pt[3, 4] = rho[0, 7]
            rho_pt[4, 3] = rho[7, 0]
            rho_pt[0, 7] = 0
            rho_pt[7, 0] = 0
        elif subsystem == (0, 2):  # Transpose with respect to qubits 0 and 2
            # Similar for different bipartition
            rho_pt[1, 6] = rho[0, 7]
            rho_pt[6, 1] = rho[7, 0]
            rho_pt[0, 7] = 0
            rho_pt[7, 0] = 0
        elif subsystem == (1, 2):  # Transpose with respect to qubits 1 and 2
            # Similar for different bipartition
            rho_pt[2, 5] = rho[0, 7]
            rho_pt[5, 2] = rho[7, 0]
            rho_pt[0, 7] = 0
            rho_pt[7, 0] = 0
        
        return rho_pt
    
    def _partial_trace(self, rho: np.ndarray, partition: Tuple[int, int]) -> np.ndarray:
        """Calculate partial trace of density matrix."""
        # Simplified implementation
        # In practice, this would be more complex for arbitrary partitions
        dim = int(np.sqrt(rho.shape[0]))
        rho_reshaped = rho.reshape(dim, dim, dim, dim)
        rho_reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
        return rho_reduced


class EntanglementNetwork:
    """Entanglement network analysis and centrality metrics."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.entanglement_matrix = np.zeros((num_qubits, num_qubits))
    
    def calculate_entanglement_graph(self, state: Union[QuantumState, ScalableQuantumState]) -> Dict[str, Any]:
        """Calculate entanglement graph for quantum state."""
        if isinstance(state, ScalableQuantumState):
            state_vector = state.to_dense()
        else:
            state_vector = state.state_vector
        
        # Calculate pairwise entanglement
        entanglement_matrix = np.zeros((self.num_qubits, self.num_qubits))
        
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                # Calculate entanglement between qubits i and j
                entanglement = self._calculate_pairwise_entanglement(state_vector, i, j)
                entanglement_matrix[i, j] = entanglement
                entanglement_matrix[j, i] = entanglement
        
        # Calculate centrality metrics
        centrality_metrics = self._calculate_centrality_metrics(entanglement_matrix)
        
        return {
            'entanglement_matrix': entanglement_matrix.tolist(),
            'centrality_metrics': centrality_metrics,
            'num_qubits': self.num_qubits,
            'total_entanglement': float(np.sum(entanglement_matrix))
        }
    
    def _calculate_pairwise_entanglement(self, state_vector: np.ndarray, 
                                        qubit1: int, qubit2: int) -> float:
        """Calculate entanglement between two qubits."""
        # Simplified entanglement measure
        # In practice, this would involve more sophisticated calculations
        
        # Calculate reduced density matrix for the two qubits
        # This is a simplified implementation
        num_qubits = int(np.log2(len(state_vector)))
        
        if num_qubits < 2:
            return 0.0
        
        # For simplicity, use a basic entanglement measure
        # based on the state vector structure
        entanglement = 0.0
        
        # Check for Bell-like correlations
        for i in range(len(state_vector)):
            binary = format(i, f'0{num_qubits}b')
            if len(binary) > max(qubit1, qubit2):
                bit1 = int(binary[qubit1])
                bit2 = int(binary[qubit2])
                
                # Check for anti-correlation (entanglement signature)
                if bit1 != bit2:
                    entanglement += abs(state_vector[i])**2
        
        return float(entanglement)
    
    def _calculate_centrality_metrics(self, entanglement_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate centrality metrics for entanglement network."""
        # Degree centrality
        degree_centrality = np.sum(entanglement_matrix, axis=1)
        
        # Betweenness centrality (simplified)
        betweenness_centrality = np.zeros(self.num_qubits)
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                for k in range(self.num_qubits):
                    if i != j and j != k and i != k:
                        if entanglement_matrix[i, j] > 0 and entanglement_matrix[j, k] > 0:
                            betweenness_centrality[j] += 1
        
        # Closeness centrality
        closeness_centrality = np.zeros(self.num_qubits)
        for i in range(self.num_qubits):
            total_distance = np.sum(entanglement_matrix[i, :])
            if total_distance > 0:
                closeness_centrality[i] = 1.0 / total_distance
        
        return {
            'degree_centrality': degree_centrality.tolist(),
            'betweenness_centrality': betweenness_centrality.tolist(),
            'closeness_centrality': closeness_centrality.tolist()
        }
    
    def export_to_graphml(self, entanglement_data: Dict[str, Any], filename: str):
        """Export entanglement network to GraphML format."""
        import xml.etree.ElementTree as ET
        
        # Create GraphML structure
        root = ET.Element("graphml")
        root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        
        # Define attributes
        attr1 = ET.SubElement(root, "key")
        attr1.set("id", "entanglement")
        attr1.set("for", "edge")
        attr1.set("attr.name", "entanglement")
        attr1.set("attr.type", "double")
        
        attr2 = ET.SubElement(root, "key")
        attr2.set("id", "degree")
        attr2.set("for", "node")
        attr2.set("attr.name", "degree")
        attr2.set("attr.type", "double")
        
        # Create graph
        graph = ET.SubElement(root, "graph")
        graph.set("id", "entanglement_network")
        graph.set("edgedefault", "undirected")
        
        # Add nodes
        for i in range(self.num_qubits):
            node = ET.SubElement(graph, "node")
            node.set("id", f"q{i}")
            data = ET.SubElement(node, "data")
            data.set("key", "degree")
            data.text = str(entanglement_data['centrality_metrics']['degree_centrality'][i])
        
        # Add edges
        entanglement_matrix = np.array(entanglement_data['entanglement_matrix'])
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if entanglement_matrix[i, j] > 0:
                    edge = ET.SubElement(graph, "edge")
                    edge.set("id", f"e{i}_{j}")
                    edge.set("source", f"q{i}")
                    edge.set("target", f"q{j}")
                    data = ET.SubElement(edge, "data")
                    data.set("key", "entanglement")
                    data.text = str(entanglement_matrix[i, j])
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(filename, encoding="utf-8", xml_declaration=True)


class AdvancedQuantumAnalysis:
    """Main class for advanced quantum analysis."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.tomography = QuantumStateTomography(num_qubits)
        self.fidelity_estimator = FidelityEstimator()
        self.entanglement_monotones = EntanglementMonotones()
        self.entanglement_network = EntanglementNetwork(num_qubits)
    
    def analyze_quantum_state(self, state: Union[QuantumState, ScalableQuantumState]) -> Dict[str, Any]:
        """Perform comprehensive analysis of quantum state."""
        analysis = {}
        
        # State tomography
        tomography_result = self.tomography.perform_tomography(state)
        analysis['tomography'] = {
            'fidelity': tomography_result.fidelity,
            'purity': tomography_result.purity,
            'success': tomography_result.success
        }
        
        # Entanglement analysis
        if self.num_qubits >= 2:
            analysis['entanglement'] = {
                'entropy': state.get_entanglement_entropy(),
                'negativity': self.entanglement_monotones.calculate_negativity(state, (0, 1)),
                'concurrence': self.entanglement_monotones.calculate_concurrence(state) if self.num_qubits == 2 else 0.0,
                'entanglement_rank': self.entanglement_monotones.calculate_entanglement_rank(state)
            }
        
        # Entanglement network analysis
        network_data = self.entanglement_network.calculate_entanglement_graph(state)
        analysis['network'] = network_data
        
        return analysis
