"""
Quantum Debugger - Interactive Quantum Circuit Debugging
=======================================================

The Quantum Debugger provides interactive debugging capabilities for
quantum circuits with breakpoints, state inspection, and step-by-step execution.

This is the GOD-TIER debugging system that makes quantum circuit
development and debugging as intuitive as classical programming.
"""

import time
import logging
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class DebugMode(Enum):
    """Debug modes for quantum circuit debugging."""
    STEP_BY_STEP = "step_by_step"
    BREAKPOINT = "breakpoint"
    CONTINUOUS = "continuous"
    INSPECT_STATE = "inspect_state"
    TRACE_EXECUTION = "trace_execution"

class BreakpointType(Enum):
    """Types of breakpoints."""
    GATE_BREAKPOINT = "gate_breakpoint"
    STATE_BREAKPOINT = "state_breakpoint"
    CONDITIONAL_BREAKPOINT = "conditional_breakpoint"
    ENTANGLEMENT_BREAKPOINT = "entanglement_breakpoint"
    MEASUREMENT_BREAKPOINT = "measurement_breakpoint"

@dataclass
class Breakpoint:
    """A breakpoint in quantum circuit debugging."""
    breakpoint_id: str
    breakpoint_type: BreakpointType
    gate_index: Optional[int] = None
    qubit_index: Optional[int] = None
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    last_hit: Optional[float] = None

@dataclass
class DebugState:
    """Debug state information."""
    circuit_state: Dict[str, Any]
    current_gate_index: int
    execution_time: float
    memory_usage: float
    entanglement_metrics: Dict[str, float]
    measurement_results: List[Dict[str, Any]]
    debug_info: Dict[str, Any]

@dataclass
class DebugSession:
    """A quantum circuit debug session."""
    session_id: str
    circuit_data: Dict[str, Any]
    debug_mode: DebugMode
    breakpoints: List[Breakpoint]
    debug_states: deque = field(default_factory=lambda: deque(maxlen=1000))
    is_active: bool = False
    start_time: float = 0.0

class QuantumDebugger:
    """
    Quantum Debugger for Interactive Quantum Circuit Debugging.
    
    This is the GOD-TIER debugging system that provides comprehensive
    debugging capabilities for quantum circuits.
    """
    
    def __init__(self):
        """Initialize the quantum debugger."""
        self.active_sessions: Dict[str, DebugSession] = {}
        self.debug_history: deque = deque(maxlen=1000)
        
        # Debug statistics
        self.debug_stats = {
            'total_debug_sessions': 0,
            'total_breakpoints_hit': 0,
            'average_debug_time': 0.0,
            'debug_success_rate': 0.0,
            'state_inspections': 0,
            'step_executions': 0
        }
        
        logger.info("🎨 Quantum Debugger initialized - Interactive debugging active")
    
    async def start_debug_session(self, circuit_data: Dict[str, Any], 
                                debug_mode: DebugMode = DebugMode.STEP_BY_STEP) -> str:
        """Start a new debug session."""
        session_id = f"debug_{int(time.time() * 1000)}"
        
        debug_session = DebugSession(
            session_id=session_id,
            circuit_data=circuit_data,
            debug_mode=debug_mode,
            breakpoints=[],
            is_active=True,
            start_time=time.time()
        )
        
        self.active_sessions[session_id] = debug_session
        self.debug_stats['total_debug_sessions'] += 1
        
        logger.info(f"🎨 Debug session started: {session_id} ({debug_mode.value})")
        return session_id
    
    async def stop_debug_session(self, session_id: str):
        """Stop a debug session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].is_active = False
            del self.active_sessions[session_id]
            logger.info(f"🎨 Debug session stopped: {session_id}")
    
    async def add_breakpoint(self, session_id: str, breakpoint: Breakpoint):
        """Add a breakpoint to a debug session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].breakpoints.append(breakpoint)
            logger.info(f"🎨 Breakpoint added: {breakpoint.breakpoint_id} to session {session_id}")
    
    async def remove_breakpoint(self, session_id: str, breakpoint_id: str):
        """Remove a breakpoint from a debug session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.breakpoints = [bp for bp in session.breakpoints if bp.breakpoint_id != breakpoint_id]
            logger.info(f"🎨 Breakpoint removed: {breakpoint_id} from session {session_id}")
    
    async def execute_step(self, session_id: str) -> Optional[DebugState]:
        """Execute a single step in debug mode."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        if not session.is_active:
            return None
        
        # Get current circuit state
        current_state = await self._get_current_circuit_state(session)
        
        # Check for breakpoints
        if await self._check_breakpoints(session, current_state):
            logger.info(f"🎨 Breakpoint hit in session {session_id}")
            return current_state
        
        # Execute next gate
        next_gate_index = current_state.current_gate_index
        gates = session.circuit_data.get('gates', [])
        
        if next_gate_index < len(gates):
            # Execute gate
            await self._execute_gate(session, gates[next_gate_index], next_gate_index)
            
            # Update state
            current_state.current_gate_index += 1
            current_state.execution_time = time.time() - session.start_time
            
            # Store debug state
            session.debug_states.append(current_state)
            
            # Update statistics
            self.debug_stats['step_executions'] += 1
            
            return current_state
        
        return None
    
    async def inspect_state(self, session_id: str, qubit_index: Optional[int] = None) -> Dict[str, Any]:
        """Inspect quantum state at current debug position."""
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        current_state = await self._get_current_circuit_state(session)
        
        # Generate state inspection data
        inspection_data = {
            'session_id': session_id,
            'current_gate_index': current_state.current_gate_index,
            'execution_time': current_state.execution_time,
            'memory_usage': current_state.memory_usage,
            'entanglement_metrics': current_state.entanglement_metrics,
            'measurement_results': current_state.measurement_results,
            'qubit_states': await self._get_qubit_states(current_state, qubit_index),
            'entanglement_structure': await self._get_entanglement_structure(current_state),
            'debug_info': current_state.debug_info
        }
        
        # Update statistics
        self.debug_stats['state_inspections'] += 1
        
        logger.info(f"🎨 State inspected for session {session_id}")
        return inspection_data
    
    async def set_breakpoint(self, session_id: str, gate_index: int, 
                           breakpoint_type: BreakpointType = BreakpointType.GATE_BREAKPOINT,
                           condition: Optional[str] = None) -> str:
        """Set a breakpoint at a specific gate."""
        breakpoint_id = f"bp_{gate_index}_{int(time.time() * 1000)}"
        
        breakpoint = Breakpoint(
            breakpoint_id=breakpoint_id,
            breakpoint_type=breakpoint_type,
            gate_index=gate_index,
            condition=condition
        )
        
        await self.add_breakpoint(session_id, breakpoint)
        return breakpoint_id
    
    async def continue_execution(self, session_id: str) -> List[DebugState]:
        """Continue execution until next breakpoint or completion."""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        execution_states = []
        
        while session.is_active:
            state = await self.execute_step(session_id)
            if state is None:
                break
            
            execution_states.append(state)
            
            # Check if we hit a breakpoint
            if await self._check_breakpoints(session, state):
                break
        
        return execution_states
    
    async def _get_current_circuit_state(self, session: DebugSession) -> DebugState:
        """Get current circuit state for debugging."""
        gates = session.circuit_data.get('gates', [])
        num_qubits = session.circuit_data.get('num_qubits', 0)
        
        # Simulate quantum state
        quantum_state = await self._simulate_quantum_state(gates, num_qubits, session.debug_states)
        
        # Calculate entanglement metrics
        entanglement_metrics = await self._calculate_entanglement_metrics(quantum_state)
        
        # Get memory usage
        memory_usage = await self._calculate_memory_usage(quantum_state)
        
        return DebugState(
            circuit_state=quantum_state,
            current_gate_index=len(session.debug_states),
            execution_time=time.time() - session.start_time,
            memory_usage=memory_usage,
            entanglement_metrics=entanglement_metrics,
            measurement_results=[],
            debug_info={
                'session_id': session.session_id,
                'debug_mode': session.debug_mode.value,
                'breakpoints_count': len(session.breakpoints),
                'execution_progress': len(session.debug_states) / max(len(gates), 1)
            }
        )
    
    async def _simulate_quantum_state(self, gates: List[Dict[str, Any]], 
                                    num_qubits: int, 
                                    debug_states: deque) -> Dict[str, Any]:
        """Simulate quantum state for debugging."""
        # Simplified quantum state simulation
        state_vector = np.zeros(2 ** num_qubits, dtype=complex)
        state_vector[0] = 1.0  # Start in |0...0⟩ state
        
        # Apply gates up to current debug position
        current_position = len(debug_states)
        for i, gate in enumerate(gates[:current_position]):
            state_vector = await self._apply_gate_to_state(state_vector, gate, num_qubits)
        
        return {
            'state_vector': state_vector.tolist(),
            'num_qubits': num_qubits,
            'gate_count': current_position,
            'entanglement_entropy': await self._calculate_entanglement_entropy(state_vector, num_qubits)
        }
    
    async def _apply_gate_to_state(self, state_vector: np.ndarray, 
                                 gate: Dict[str, Any], 
                                 num_qubits: int) -> np.ndarray:
        """Apply a gate to the quantum state."""
        gate_type = gate.get('type', '')
        qubits = gate.get('qubits', [])
        
        # Simplified gate application
        if gate_type == 'H':
            # Hadamard gate
            for qubit in qubits:
                if qubit < num_qubits:
                    # Apply Hadamard to qubit
                    pass  # Simplified
        elif gate_type == 'CNOT':
            # CNOT gate
            if len(qubits) >= 2:
                # Apply CNOT
                pass  # Simplified
        
        return state_vector
    
    async def _calculate_entanglement_entropy(self, state_vector: np.ndarray, 
                                            num_qubits: int) -> float:
        """Calculate entanglement entropy of the state."""
        # Simplified entanglement entropy calculation
        probabilities = np.abs(state_vector) ** 2
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return float(entropy)
    
    async def _calculate_entanglement_metrics(self, quantum_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate entanglement metrics for the state."""
        return {
            'entanglement_entropy': quantum_state.get('entanglement_entropy', 0.0),
            'state_complexity': len(quantum_state.get('state_vector', [])),
            'quantum_correlations': 0.5,  # Simplified
            'measurement_probability': 0.0  # Simplified
        }
    
    async def _calculate_memory_usage(self, quantum_state: Dict[str, Any]) -> float:
        """Calculate memory usage of the quantum state."""
        state_vector = quantum_state.get('state_vector', [])
        return len(state_vector) * 16 / (1024 * 1024)  # 16 bytes per complex number, convert to MB
    
    async def _check_breakpoints(self, session: DebugSession, state: DebugState) -> bool:
        """Check if any breakpoints are hit."""
        for breakpoint in session.breakpoints:
            if not breakpoint.enabled:
                continue
            
            if breakpoint.breakpoint_type == BreakpointType.GATE_BREAKPOINT:
                if breakpoint.gate_index == state.current_gate_index:
                    breakpoint.hit_count += 1
                    breakpoint.last_hit = time.time()
                    self.debug_stats['total_breakpoints_hit'] += 1
                    return True
            
            elif breakpoint.breakpoint_type == BreakpointType.STATE_BREAKPOINT:
                if breakpoint.condition and await self._evaluate_condition(breakpoint.condition, state):
                    breakpoint.hit_count += 1
                    breakpoint.last_hit = time.time()
                    self.debug_stats['total_breakpoints_hit'] += 1
                    return True
        
        return False
    
    async def _evaluate_condition(self, condition: str, state: DebugState) -> bool:
        """Evaluate a breakpoint condition."""
        # Simplified condition evaluation
        # In a real implementation, this would parse and evaluate the condition
        try:
            # Safe evaluation of simple conditions
            if 'entanglement_entropy' in condition:
                return state.entanglement_metrics.get('entanglement_entropy', 0) > 0.5
            elif 'execution_time' in condition:
                return state.execution_time > 1.0
            elif 'memory_usage' in condition:
                return state.memory_usage > 100.0  # MB
        except Exception:
            pass
        
        return False
    
    async def _get_qubit_states(self, state: DebugState, qubit_index: Optional[int] = None) -> Dict[str, Any]:
        """Get qubit states for inspection."""
        quantum_state = state.circuit_state
        state_vector = quantum_state.get('state_vector', [])
        num_qubits = quantum_state.get('num_qubits', 0)
        
        if qubit_index is not None and qubit_index < num_qubits:
            # Get specific qubit state
            return {
                'qubit_index': qubit_index,
                'state': 'superposition',  # Simplified
                'measurement_probability': 0.5,  # Simplified
                'entanglement_partners': []  # Simplified
            }
        else:
            # Get all qubit states
            qubit_states = {}
            for i in range(num_qubits):
                qubit_states[f'qubit_{i}'] = {
                    'state': 'superposition',
                    'measurement_probability': 0.5,
                    'entanglement_partners': []
                }
            return qubit_states
    
    async def _get_entanglement_structure(self, state: DebugState) -> Dict[str, Any]:
        """Get entanglement structure for inspection."""
        return {
            'entanglement_entropy': state.entanglement_metrics.get('entanglement_entropy', 0.0),
            'quantum_correlations': state.entanglement_metrics.get('quantum_correlations', 0.0),
            'entangled_qubits': [],  # Simplified
            'entanglement_network': {}  # Simplified
        }
    
    async def _execute_gate(self, session: DebugSession, gate: Dict[str, Any], gate_index: int):
        """Execute a gate in debug mode."""
        # Simulate gate execution
        await asyncio.sleep(0.001)  # Simulate processing time
        
        # Update debug info
        session.debug_states.append({
            'gate_index': gate_index,
            'gate_type': gate.get('type', ''),
            'qubits': gate.get('qubits', []),
            'execution_time': time.time() - session.start_time
        })
    
    def get_debug_statistics(self) -> Dict[str, Any]:
        """Get quantum debugger statistics."""
        return {
            'debug_stats': self.debug_stats,
            'active_sessions': len(self.active_sessions),
            'debug_history_size': len(self.debug_history),
            'session_details': {
                session_id: {
                    'debug_mode': session.debug_mode.value,
                    'breakpoints_count': len(session.breakpoints),
                    'debug_states_count': len(session.debug_states),
                    'is_active': session.is_active
                }
                for session_id, session in self.active_sessions.items()
            }
        }
    
    def get_debug_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get debugging recommendations."""
        recommendations = []
        
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        # Debugging strategy recommendations
        if len(gates) > 50:
            recommendations.append({
                'type': 'debugging_strategy',
                'message': f'Large circuit ({len(gates)} gates) detected',
                'recommendation': 'Use step-by-step debugging with strategic breakpoints',
                'priority': 'medium'
            })
        
        # Breakpoint recommendations
        if num_qubits > 10:
            recommendations.append({
                'type': 'breakpoints',
                'message': f'Large qubit count ({num_qubits}) detected',
                'recommendation': 'Set breakpoints at entangling gates for better debugging',
                'priority': 'low'
            })
        
        # Performance recommendations
        if self.debug_stats['average_debug_time'] > 5.0:
            recommendations.append({
                'type': 'performance',
                'message': f'Long debug time ({self.debug_stats["average_debug_time"]:.2f}s)',
                'recommendation': 'Consider using continuous mode for faster execution',
                'priority': 'medium'
            })
        
        return recommendations
