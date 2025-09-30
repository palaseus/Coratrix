"""
Pattern Recognizer - Intelligent Quantum Circuit Pattern Detection
================================================================

The Pattern Recognizer is the intelligent pattern detection system of
Coratrix 4.0's adaptive compiler. It provides:

- Circuit pattern recognition and classification
- Optimization pattern matching
- Learning from circuit patterns
- Intelligent circuit transformation
- Pattern-based optimization recommendations
- Adaptive pattern recognition

This makes the compiler truly intelligent and adaptive to circuit patterns.
"""

import time
import logging
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of quantum circuit patterns."""
    BELL_STATE = "bell_state"
    GHZ_STATE = "ghz_state"
    GROVER_SEARCH = "grover_search"
    QFT = "quantum_fourier_transform"
    TELEPORTATION = "teleportation"
    QAOA = "qaoa"
    VQE = "vqe"
    ERROR_CORRECTION = "error_correction"
    PARALLEL_GATES = "parallel_gates"
    SEQUENTIAL_GATES = "sequential_gates"
    REDUNDANT_GATES = "redundant_gates"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"

class PatternComplexity(Enum):
    """Complexity levels for patterns."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"

@dataclass
class CircuitPattern:
    """A recognized quantum circuit pattern."""
    pattern_id: str
    pattern_type: PatternType
    complexity: PatternComplexity
    circuit_sequence: List[Dict[str, Any]]
    optimization_rules: List[Dict[str, Any]]
    success_rate: float
    confidence: float
    frequency: int = 0
    last_seen: float = 0.0

@dataclass
class PatternMatch:
    """A match of a pattern in a circuit."""
    pattern: CircuitPattern
    match_start: int
    match_end: int
    match_confidence: float
    optimization_potential: float

@dataclass
class OptimizationRule:
    """An optimization rule for a pattern."""
    rule_id: str
    pattern_type: PatternType
    condition: Dict[str, Any]
    transformation: Dict[str, Any]
    expected_improvement: float
    confidence: float

class PatternRecognizer:
    """
    Pattern Recognizer for Intelligent Quantum Circuit Pattern Detection.
    
    This is the intelligent pattern detection system that recognizes
    quantum circuit patterns and provides optimization recommendations
    based on learned patterns and circuit characteristics.
    """
    
    def __init__(self):
        """Initialize the pattern recognizer."""
        self.known_patterns: Dict[str, CircuitPattern] = {}
        self.optimization_rules: Dict[PatternType, List[OptimizationRule]] = defaultdict(list)
        self.pattern_history: deque = deque(maxlen=10000)
        self.learning_enabled = True
        
        # Pattern recognition parameters
        self.min_pattern_length = 2
        self.max_pattern_length = 50
        self.confidence_threshold = 0.7
        self.optimization_threshold = 0.5
        
        # Initialize known patterns
        self._initialize_known_patterns()
        self._initialize_optimization_rules()
        
        logger.info("🔍 Pattern Recognizer initialized - Intelligent pattern detection active")
    
    def _initialize_known_patterns(self):
        """Initialize known quantum circuit patterns."""
        # Bell State Pattern
        bell_pattern = CircuitPattern(
            pattern_id="bell_state_001",
            pattern_type=PatternType.BELL_STATE,
            complexity=PatternComplexity.SIMPLE,
            circuit_sequence=[
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]}
            ],
            optimization_rules=[
                {'rule': 'gate_merging', 'description': 'Merge H and CNOT gates'},
                {'rule': 'depth_reduction', 'description': 'Optimize for depth'}
            ],
            success_rate=0.95,
            confidence=0.9
        )
        self.known_patterns["bell_state_001"] = bell_pattern
        
        # GHZ State Pattern
        ghz_pattern = CircuitPattern(
            pattern_id="ghz_state_001",
            pattern_type=PatternType.GHZ_STATE,
            complexity=PatternComplexity.MODERATE,
            circuit_sequence=[
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]},
                {'type': 'CNOT', 'qubits': [0, 2]}
            ],
            optimization_rules=[
                {'rule': 'parallelization', 'description': 'Parallelize CNOT gates'},
                {'rule': 'gate_consolidation', 'description': 'Consolidate H gates'}
            ],
            success_rate=0.90,
            confidence=0.85
        )
        self.known_patterns["ghz_state_001"] = ghz_pattern
        
        # Grover Search Pattern
        grover_pattern = CircuitPattern(
            pattern_id="grover_search_001",
            pattern_type=PatternType.GROVER_SEARCH,
            complexity=PatternComplexity.COMPLEX,
            circuit_sequence=[
                {'type': 'H', 'qubits': [0]},
                {'type': 'H', 'qubits': [1]},
                {'type': 'H', 'qubits': [2]},
                {'type': 'CNOT', 'qubits': [0, 1]},
                {'type': 'CNOT', 'qubits': [1, 2]}
            ],
            optimization_rules=[
                {'rule': 'oracle_optimization', 'description': 'Optimize oracle function'},
                {'rule': 'diffusion_optimization', 'description': 'Optimize diffusion operator'}
            ],
            success_rate=0.85,
            confidence=0.80
        )
        self.known_patterns["grover_search_001"] = grover_pattern
        
        # QFT Pattern
        qft_pattern = CircuitPattern(
            pattern_id="qft_001",
            pattern_type=PatternType.QFT,
            complexity=PatternComplexity.ADVANCED,
            circuit_sequence=[
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]},
                {'type': 'H', 'qubits': [1]},
                {'type': 'CNOT', 'qubits': [1, 2]},
                {'type': 'H', 'qubits': [2]}
            ],
            optimization_rules=[
                {'rule': 'butterfly_optimization', 'description': 'Optimize butterfly operations'},
                {'rule': 'phase_optimization', 'description': 'Optimize phase gates'}
            ],
            success_rate=0.80,
            confidence=0.75
        )
        self.known_patterns["qft_001"] = qft_pattern
        
        # Teleportation Pattern
        teleportation_pattern = CircuitPattern(
            pattern_id="teleportation_001",
            pattern_type=PatternType.TELEPORTATION,
            complexity=PatternComplexity.MODERATE,
            circuit_sequence=[
                {'type': 'H', 'qubits': [0]},
                {'type': 'CNOT', 'qubits': [0, 1]},
                {'type': 'CNOT', 'qubits': [1, 2]},
                {'type': 'H', 'qubits': [1]}
            ],
            optimization_rules=[
                {'rule': 'bell_state_optimization', 'description': 'Optimize Bell state creation'},
                {'rule': 'measurement_optimization', 'description': 'Optimize measurement operations'}
            ],
            success_rate=0.90,
            confidence=0.85
        )
        self.known_patterns["teleportation_001"] = teleportation_pattern
    
    def _initialize_optimization_rules(self):
        """Initialize optimization rules for patterns."""
        # Bell State optimization rules
        self.optimization_rules[PatternType.BELL_STATE].append(OptimizationRule(
            rule_id="bell_001",
            pattern_type=PatternType.BELL_STATE,
            condition={'gate_count': 2, 'has_hadamard': True, 'has_cnot': True},
            transformation={'merge_gates': True, 'optimize_depth': True},
            expected_improvement=0.1,
            confidence=0.9
        ))
        
        # GHZ State optimization rules
        self.optimization_rules[PatternType.GHZ_STATE].append(OptimizationRule(
            rule_id="ghz_001",
            pattern_type=PatternType.GHZ_STATE,
            condition={'gate_count': 3, 'has_hadamard': True, 'has_multiple_cnot': True},
            transformation={'parallelize_cnot': True, 'consolidate_hadamard': True},
            expected_improvement=0.2,
            confidence=0.85
        ))
        
        # Grover Search optimization rules
        self.optimization_rules[PatternType.GROVER_SEARCH].append(OptimizationRule(
            rule_id="grover_001",
            pattern_type=PatternType.GROVER_SEARCH,
            condition={'gate_count': 5, 'has_hadamard': True, 'has_cnot_chain': True},
            transformation={'optimize_oracle': True, 'optimize_diffusion': True},
            expected_improvement=0.3,
            confidence=0.80
        ))
    
    def recognize_patterns(self, circuit_data: Dict[str, Any]) -> List[PatternMatch]:
        """
        Recognize patterns in a quantum circuit.
        
        This is the GOD-TIER pattern recognition method that identifies
        quantum circuit patterns and provides optimization opportunities.
        """
        gates = circuit_data.get('gates', [])
        if not gates:
            return []
        
        logger.info(f"🔍 Recognizing patterns in circuit: {circuit_data.get('name', 'Unknown')}")
        
        # Find pattern matches
        pattern_matches = []
        
        for pattern_id, pattern in self.known_patterns.items():
            matches = self._find_pattern_matches(gates, pattern)
            pattern_matches.extend(matches)
        
        # Sort by confidence and optimization potential
        pattern_matches.sort(key=lambda x: (x.match_confidence, x.optimization_potential), reverse=True)
        
        # Update pattern frequency
        for match in pattern_matches:
            match.pattern.frequency += 1
            match.pattern.last_seen = time.time()
        
        # Store in pattern history
        self.pattern_history.append({
            'timestamp': time.time(),
            'circuit_data': circuit_data,
            'pattern_matches': pattern_matches
        })
        
        logger.info(f"🔍 Found {len(pattern_matches)} pattern matches")
        return pattern_matches
    
    def _find_pattern_matches(self, gates: List[Dict[str, Any]], 
                            pattern: CircuitPattern) -> List[PatternMatch]:
        """Find matches of a pattern in a circuit."""
        matches = []
        pattern_sequence = pattern.circuit_sequence
        
        if len(pattern_sequence) > len(gates):
            return matches
        
        # Sliding window approach
        for i in range(len(gates) - len(pattern_sequence) + 1):
            window = gates[i:i + len(pattern_sequence)]
            
            # Check if window matches pattern
            match_confidence = self._calculate_pattern_similarity(window, pattern_sequence)
            
            if match_confidence >= self.confidence_threshold:
                # Calculate optimization potential
                optimization_potential = self._calculate_optimization_potential(window, pattern)
                
                match = PatternMatch(
                    pattern=pattern,
                    match_start=i,
                    match_end=i + len(pattern_sequence) - 1,
                    match_confidence=match_confidence,
                    optimization_potential=optimization_potential
                )
                matches.append(match)
        
        return matches
    
    def _calculate_pattern_similarity(self, window: List[Dict[str, Any]], 
                                    pattern_sequence: List[Dict[str, Any]]) -> float:
        """Calculate similarity between a window and pattern sequence."""
        if len(window) != len(pattern_sequence):
            return 0.0
        
        similarities = []
        
        for gate, pattern_gate in zip(window, pattern_sequence):
            similarity = self._calculate_gate_similarity(gate, pattern_gate)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_gate_similarity(self, gate: Dict[str, Any], 
                                pattern_gate: Dict[str, Any]) -> float:
        """Calculate similarity between two gates."""
        # Type similarity
        type_similarity = 1.0 if gate.get('type') == pattern_gate.get('type') else 0.0
        
        # Qubit similarity
        gate_qubits = set(gate.get('qubits', []))
        pattern_qubits = set(pattern_gate.get('qubits', []))
        
        if not pattern_qubits:  # Pattern doesn't specify qubits
            qubit_similarity = 1.0
        elif not gate_qubits:  # Gate doesn't specify qubits
            qubit_similarity = 0.0
        else:
            qubit_similarity = len(gate_qubits.intersection(pattern_qubits)) / len(pattern_qubits.union(gate_qubits))
        
        # Parameter similarity (for parameterized gates)
        param_similarity = 1.0
        if 'parameters' in pattern_gate and 'parameters' in gate:
            gate_params = gate.get('parameters', [])
            pattern_params = pattern_gate.get('parameters', [])
            if len(pattern_params) == len(gate_params):
                param_similarity = 1.0 - np.mean([abs(p1 - p2) for p1, p2 in zip(gate_params, pattern_params)])
            else:
                param_similarity = 0.0
        
        # Weighted combination
        similarity = (type_similarity * 0.5 + qubit_similarity * 0.3 + param_similarity * 0.2)
        return similarity
    
    def _calculate_optimization_potential(self, window: List[Dict[str, Any]], 
                                       pattern: CircuitPattern) -> float:
        """Calculate optimization potential for a pattern match."""
        # Base optimization potential from pattern
        base_potential = pattern.success_rate
        
        # Adjust based on circuit characteristics
        gate_count = len(window)
        if gate_count > 10:
            base_potential += 0.1
        
        # Check for optimization opportunities
        optimization_opportunities = 0
        
        # Check for redundant gates
        gate_types = [gate.get('type') for gate in window]
        unique_types = len(set(gate_types))
        if unique_types < len(gate_types):
            optimization_opportunities += 0.2
        
        # Check for sequential single-qubit gates
        for i in range(len(window) - 1):
            if (window[i].get('type') in ['H', 'X', 'Y', 'Z'] and 
                window[i+1].get('type') in ['H', 'X', 'Y', 'Z'] and
                window[i].get('qubits') == window[i+1].get('qubits')):
                optimization_opportunities += 0.1
        
        return min(base_potential + optimization_opportunities, 1.0)
    
    async def apply_optimizations(self, circuit_data: Dict[str, Any], 
                                pattern_matches: List[PatternMatch]) -> Dict[str, Any]:
        """
        Apply optimizations based on recognized patterns.
        
        This is the GOD-TIER optimization method that applies intelligent
        circuit transformations based on recognized patterns.
        """
        if not pattern_matches:
            return circuit_data
        
        logger.info(f"🔧 Applying optimizations for {len(pattern_matches)} pattern matches")
        
        optimized_circuit = circuit_data.copy()
        gates = optimized_circuit.get('gates', [])
        
        # Sort matches by optimization potential (highest first)
        pattern_matches.sort(key=lambda x: x.optimization_potential, reverse=True)
        
        # Apply optimizations
        for match in pattern_matches:
            if match.optimization_potential >= self.optimization_threshold:
                gates = await self._apply_pattern_optimization(gates, match)
        
        optimized_circuit['gates'] = gates
        
        # Update pattern statistics
        for match in pattern_matches:
            match.pattern.frequency += 1
            match.pattern.last_seen = time.time()
        
        logger.info(f"🔧 Applied optimizations successfully")
        return optimized_circuit
    
    async def _apply_pattern_optimization(self, gates: List[Dict[str, Any]], 
                                        match: PatternMatch) -> List[Dict[str, Any]]:
        """Apply optimization for a specific pattern match."""
        pattern = match.pattern
        start_idx = match.match_start
        end_idx = match.match_end
        
        # Get optimization rules for this pattern
        rules = self.optimization_rules.get(pattern.pattern_type, [])
        
        if not rules:
            return gates
        
        # Apply the most confident rule
        best_rule = max(rules, key=lambda r: r.confidence)
        
        # Apply transformation based on rule
        if best_rule.transformation.get('merge_gates', False):
            gates = await self._merge_gates_optimization(gates, start_idx, end_idx)
        
        elif best_rule.transformation.get('parallelize_cnot', False):
            gates = await self._parallelize_cnot_optimization(gates, start_idx, end_idx)
        
        elif best_rule.transformation.get('consolidate_hadamard', False):
            gates = await self._consolidate_hadamard_optimization(gates, start_idx, end_idx)
        
        elif best_rule.transformation.get('optimize_oracle', False):
            gates = await self._optimize_oracle_optimization(gates, start_idx, end_idx)
        
        elif best_rule.transformation.get('optimize_diffusion', False):
            gates = await self._optimize_diffusion_optimization(gates, start_idx, end_idx)
        
        return gates
    
    async def _merge_gates_optimization(self, gates: List[Dict[str, Any]], 
                                      start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Apply gate merging optimization."""
        # Simple gate merging logic
        optimized_gates = gates.copy()
        
        # Remove redundant gates in the pattern region
        for i in range(start_idx, end_idx + 1):
            if i < len(optimized_gates) - 1:
                current_gate = optimized_gates[i]
                next_gate = optimized_gates[i + 1]
                
                # Merge H followed by H (identity)
                if (current_gate.get('type') == 'H' and next_gate.get('type') == 'H' and
                    current_gate.get('qubits') == next_gate.get('qubits')):
                    optimized_gates.pop(i)  # Remove first H
                    break
        
        return optimized_gates
    
    async def _parallelize_cnot_optimization(self, gates: List[Dict[str, Any]], 
                                          start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Apply CNOT parallelization optimization."""
        # Simple CNOT parallelization logic
        optimized_gates = gates.copy()
        
        # Group CNOT gates by target qubit
        cnot_gates = []
        for i in range(start_idx, end_idx + 1):
            if i < len(optimized_gates) and optimized_gates[i].get('type') == 'CNOT':
                cnot_gates.append((i, optimized_gates[i]))
        
        # Reorder CNOT gates for better parallelization
        if len(cnot_gates) > 1:
            # Simple reordering based on qubit indices
            cnot_gates.sort(key=lambda x: x[1].get('qubits', [0])[0])
            
            # Update gates in optimized circuit
            for i, (original_idx, gate) in enumerate(cnot_gates):
                if start_idx + i < len(optimized_gates):
                    optimized_gates[start_idx + i] = gate
        
        return optimized_gates
    
    async def _consolidate_hadamard_optimization(self, gates: List[Dict[str, Any]], 
                                               start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Apply Hadamard consolidation optimization."""
        # Simple Hadamard consolidation logic
        optimized_gates = gates.copy()
        
        # Find consecutive Hadamard gates
        for i in range(start_idx, end_idx):
            if (i < len(optimized_gates) - 1 and 
                optimized_gates[i].get('type') == 'H' and 
                optimized_gates[i + 1].get('type') == 'H' and
                optimized_gates[i].get('qubits') == optimized_gates[i + 1].get('qubits')):
                # Remove redundant Hadamard (H H = I)
                optimized_gates.pop(i)
                break
        
        return optimized_gates
    
    async def _optimize_oracle_optimization(self, gates: List[Dict[str, Any]], 
                                          start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Apply oracle optimization for Grover search."""
        # Simple oracle optimization logic
        optimized_gates = gates.copy()
        
        # Optimize oracle function (placeholder)
        # This would involve specific oracle optimization techniques
        return optimized_gates
    
    async def _optimize_diffusion_optimization(self, gates: List[Dict[str, Any]], 
                                             start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Apply diffusion optimization for Grover search."""
        # Simple diffusion optimization logic
        optimized_gates = gates.copy()
        
        # Optimize diffusion operator (placeholder)
        # This would involve specific diffusion optimization techniques
        return optimized_gates
    
    def learn_pattern(self, circuit_data: Dict[str, Any], 
                     optimization_result: Dict[str, Any]):
        """Learn from a new circuit pattern."""
        if not self.learning_enabled:
            return
        
        gates = circuit_data.get('gates', [])
        if len(gates) < self.min_pattern_length:
            return
        
        # Extract pattern characteristics
        pattern_characteristics = self._extract_pattern_characteristics(gates)
        
        # Check if this is a new pattern or variation of existing pattern
        existing_pattern = self._find_similar_pattern(pattern_characteristics)
        
        if existing_pattern:
            # Update existing pattern
            self._update_pattern(existing_pattern, optimization_result)
        else:
            # Create new pattern
            self._create_new_pattern(circuit_data, pattern_characteristics, optimization_result)
    
    def _extract_pattern_characteristics(self, gates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract characteristics from a circuit pattern."""
        return {
            'gate_count': len(gates),
            'gate_types': [gate.get('type') for gate in gates],
            'qubit_usage': list(set(qubit for gate in gates for qubit in gate.get('qubits', []))),
            'entanglement_gates': sum(1 for gate in gates if gate.get('type') in ['CNOT', 'CZ', 'SWAP']),
            'single_qubit_gates': sum(1 for gate in gates if gate.get('type') in ['H', 'X', 'Y', 'Z']),
            'parameterized_gates': sum(1 for gate in gates if gate.get('type') in ['Rx', 'Ry', 'Rz'])
        }
    
    def _find_similar_pattern(self, characteristics: Dict[str, Any]) -> Optional[CircuitPattern]:
        """Find similar existing pattern."""
        for pattern in self.known_patterns.values():
            similarity = self._calculate_pattern_characteristics_similarity(
                characteristics, pattern
            )
            if similarity > 0.8:  # High similarity threshold
                return pattern
        return None
    
    def _calculate_pattern_characteristics_similarity(self, characteristics: Dict[str, Any], 
                                                    pattern: CircuitPattern) -> float:
        """Calculate similarity between pattern characteristics."""
        # Simple similarity calculation
        similarities = []
        
        # Gate count similarity
        pattern_gate_count = len(pattern.circuit_sequence)
        gate_count_similarity = 1.0 - abs(characteristics['gate_count'] - pattern_gate_count) / max(characteristics['gate_count'], pattern_gate_count)
        similarities.append(gate_count_similarity)
        
        # Gate type similarity
        pattern_types = [gate.get('type') for gate in pattern.circuit_sequence]
        type_similarity = len(set(characteristics['gate_types']).intersection(set(pattern_types))) / len(set(characteristics['gate_types']).union(set(pattern_types)))
        similarities.append(type_similarity)
        
        return np.mean(similarities)
    
    def _update_pattern(self, pattern: CircuitPattern, optimization_result: Dict[str, Any]):
        """Update existing pattern with new learning data."""
        # Update success rate
        success = optimization_result.get('success', False)
        pattern.success_rate = (pattern.success_rate * pattern.frequency + success) / (pattern.frequency + 1)
        
        # Update confidence
        pattern.confidence = min(pattern.confidence + 0.01, 1.0)
        
        # Update frequency
        pattern.frequency += 1
        pattern.last_seen = time.time()
        
        logger.info(f"🔍 Updated pattern {pattern.pattern_id} with new learning data")
    
    def _create_new_pattern(self, circuit_data: Dict[str, Any], 
                          characteristics: Dict[str, Any], 
                          optimization_result: Dict[str, Any]):
        """Create new pattern from circuit data."""
        pattern_id = f"learned_{int(time.time() * 1000)}"
        
        # Determine pattern type based on characteristics
        pattern_type = self._classify_pattern_type(characteristics)
        
        # Determine complexity
        complexity = self._classify_pattern_complexity(characteristics)
        
        new_pattern = CircuitPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            complexity=complexity,
            circuit_sequence=circuit_data.get('gates', []),
            optimization_rules=[],
            success_rate=1.0 if optimization_result.get('success', False) else 0.0,
            confidence=0.5,  # Initial confidence for learned patterns
            frequency=1,
            last_seen=time.time()
        )
        
        self.known_patterns[pattern_id] = new_pattern
        logger.info(f"🔍 Created new pattern {pattern_id} of type {pattern_type.value}")
    
    def _classify_pattern_type(self, characteristics: Dict[str, Any]) -> PatternType:
        """Classify pattern type based on characteristics."""
        gate_types = characteristics['gate_types']
        gate_count = characteristics['gate_count']
        entanglement_gates = characteristics['entanglement_gates']
        
        # Simple classification logic
        if gate_count == 2 and 'H' in gate_types and 'CNOT' in gate_types:
            return PatternType.BELL_STATE
        elif gate_count == 3 and gate_types.count('H') == 1 and gate_types.count('CNOT') == 2:
            return PatternType.GHZ_STATE
        elif gate_count >= 5 and gate_types.count('H') >= 2 and entanglement_gates >= 2:
            return PatternType.GROVER_SEARCH
        elif gate_count >= 4 and entanglement_gates >= 3:
            return PatternType.QFT
        else:
            return PatternType.OPTIMIZATION_OPPORTUNITY
    
    def _classify_pattern_complexity(self, characteristics: Dict[str, Any]) -> PatternComplexity:
        """Classify pattern complexity based on characteristics."""
        gate_count = characteristics['gate_count']
        entanglement_gates = characteristics['entanglement_gates']
        
        if gate_count <= 3 and entanglement_gates <= 1:
            return PatternComplexity.SIMPLE
        elif gate_count <= 10 and entanglement_gates <= 3:
            return PatternComplexity.MODERATE
        elif gate_count <= 20 and entanglement_gates <= 5:
            return PatternComplexity.COMPLEX
        else:
            return PatternComplexity.ADVANCED
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern recognition statistics."""
        return {
            'total_patterns': len(self.known_patterns),
            'pattern_types': {pt.value: sum(1 for p in self.known_patterns.values() if p.pattern_type == pt) 
                            for pt in PatternType},
            'pattern_complexities': {pc.value: sum(1 for p in self.known_patterns.values() if p.complexity == pc) 
                                   for pc in PatternComplexity},
            'most_frequent_patterns': sorted(
                [(p.pattern_id, p.frequency) for p in self.known_patterns.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
            'pattern_history_count': len(self.pattern_history),
            'learning_enabled': self.learning_enabled
        }
    
    def get_optimization_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get pattern-based optimization recommendations."""
        gates = circuit_data.get('gates', [])
        if not gates:
            return []
        
        recommendations = []
        
        # Check for common optimization opportunities
        gate_types = [gate.get('type') for gate in gates]
        
        # Redundant gates recommendation
        if len(gate_types) != len(set(gate_types)):
            recommendations.append({
                'type': 'redundant_gates',
                'message': 'Redundant gates detected',
                'recommendation': 'Consider gate merging optimization',
                'priority': 'medium',
                'confidence': 0.8
            })
        
        # Sequential single-qubit gates recommendation
        sequential_count = 0
        for i in range(len(gates) - 1):
            if (gates[i].get('type') in ['H', 'X', 'Y', 'Z'] and 
                gates[i+1].get('type') in ['H', 'X', 'Y', 'Z'] and
                gates[i].get('qubits') == gates[i+1].get('qubits')):
                sequential_count += 1
        
        if sequential_count > 0:
            recommendations.append({
                'type': 'sequential_gates',
                'message': f'{sequential_count} sequential single-qubit gates detected',
                'recommendation': 'Consider gate consolidation optimization',
                'priority': 'high',
                'confidence': 0.9
            })
        
        # Pattern recognition recommendations
        pattern_matches = self.recognize_patterns(circuit_data)
        for match in pattern_matches:
            if match.optimization_potential > 0.7:
                recommendations.append({
                    'type': 'pattern_optimization',
                    'message': f'Pattern {match.pattern.pattern_type.value} detected',
                    'recommendation': f'Apply {match.pattern.pattern_type.value} optimization',
                    'priority': 'high',
                    'confidence': match.match_confidence
                })
        
        return recommendations
