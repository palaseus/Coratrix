"""
Interactive Controls - User Interface for Quantum Visualization
==============================================================

The Interactive Controls provide user interface elements for
interacting with quantum circuit visualizations.

This is the GOD-TIER interactive control system that makes
quantum circuit visualization intuitive and user-friendly.
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

class ControlType(Enum):
    """Types of interactive controls."""
    BUTTON = "button"
    SLIDER = "slider"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    TEXT_INPUT = "text_input"
    RANGE_SLIDER = "range_slider"
    COLOR_PICKER = "color_picker"
    TOGGLE = "toggle"

class ControlPanel:
    """Interactive control panel for quantum visualization."""
    
    def __init__(self):
        """Initialize the control panel."""
        self.controls: Dict[str, Any] = {}
        self.control_callbacks: Dict[str, Callable] = {}
        self.control_values: Dict[str, Any] = {}
        
        # Control statistics
        self.control_stats = {
            'total_interactions': 0,
            'controls_created': 0,
            'callback_executions': 0,
            'user_sessions': 0
        }
        
        logger.info("🎨 Interactive Controls initialized - User interface active")
    
    def create_control(self, control_id: str, control_type: ControlType, 
                      properties: Dict[str, Any], callback: Optional[Callable] = None):
        """Create an interactive control."""
        control = {
            'id': control_id,
            'type': control_type.value,
            'properties': properties,
            'callback': callback,
            'created_at': time.time()
        }
        
        self.controls[control_id] = control
        if callback:
            self.control_callbacks[control_id] = callback
        
        self.control_stats['controls_created'] += 1
        logger.info(f"🎨 Created control: {control_id} ({control_type.value})")
    
    def update_control_value(self, control_id: str, value: Any):
        """Update a control value."""
        if control_id in self.controls:
            self.control_values[control_id] = value
            
            # Execute callback if available
            if control_id in self.control_callbacks:
                try:
                    self.control_callbacks[control_id](value)
                    self.control_stats['callback_executions'] += 1
                except Exception as e:
                    logger.error(f"❌ Control callback error: {e}")
            
            self.control_stats['total_interactions'] += 1
            logger.info(f"🎨 Updated control {control_id}: {value}")
    
    def get_control_value(self, control_id: str) -> Any:
        """Get a control value."""
        return self.control_values.get(control_id)
    
    def get_all_controls(self) -> Dict[str, Any]:
        """Get all controls."""
        return {
            control_id: {
                'control': control,
                'value': self.control_values.get(control_id),
                'has_callback': control_id in self.control_callbacks
            }
            for control_id, control in self.controls.items()
        }
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """Get control statistics."""
        return {
            'control_stats': self.control_stats,
            'total_controls': len(self.controls),
            'active_callbacks': len(self.control_callbacks)
        }

class InteractiveControls:
    """
    Interactive Controls for Quantum Visualization.
    
    This is the GOD-TIER interactive control system that makes
    quantum circuit visualization intuitive and user-friendly.
    """
    
    def __init__(self):
        """Initialize the interactive controls."""
        self.control_panel = ControlPanel()
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Control statistics
        self.control_stats = {
            'total_user_interactions': 0,
            'active_sessions': 0,
            'control_panels_created': 0,
            'user_engagement_score': 0.0
        }
        
        logger.info("🎨 Interactive Controls initialized - User interface active")
    
    def create_visualization_controls(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create controls for quantum circuit visualization."""
        logger.info(f"🎨 Creating visualization controls for circuit: {circuit_data.get('name', 'Unknown')}")
        
        controls = {}
        
        # Create render mode control
        self.control_panel.create_control(
            control_id="render_mode",
            control_type=ControlType.DROPDOWN,
            properties={
                'label': 'Render Mode',
                'options': ['wireframe', 'solid', 'particle', 'holographic', 'quantum'],
                'default': 'quantum',
                'description': 'Select the rendering mode for the circuit'
            },
            callback=self._on_render_mode_change
        )
        controls['render_mode'] = self.control_panel.get_control_value('render_mode')
        
        # Create quality control
        self.control_panel.create_control(
            control_id="quality",
            control_type=ControlType.SLIDER,
            properties={
                'label': 'Render Quality',
                'min': 0.1,
                'max': 1.0,
                'step': 0.1,
                'default': 0.8,
                'description': 'Adjust the rendering quality'
            },
            callback=self._on_quality_change
        )
        controls['quality'] = self.control_panel.get_control_value('quality')
        
        # Create animation speed control
        self.control_panel.create_control(
            control_id="animation_speed",
            control_type=ControlType.SLIDER,
            properties={
                'label': 'Animation Speed',
                'min': 0.1,
                'max': 5.0,
                'step': 0.1,
                'default': 1.0,
                'description': 'Control the speed of circuit animations'
            },
            callback=self._on_animation_speed_change
        )
        controls['animation_speed'] = self.control_panel.get_control_value('animation_speed')
        
        # Create entanglement visualization control
        self.control_panel.create_control(
            control_id="show_entanglement",
            control_type=ControlType.CHECKBOX,
            properties={
                'label': 'Show Entanglement',
                'default': True,
                'description': 'Toggle entanglement visualization'
            },
            callback=self._on_entanglement_toggle
        )
        controls['show_entanglement'] = self.control_panel.get_control_value('show_entanglement')
        
        # Create state visualization control
        self.control_panel.create_control(
            control_id="show_state",
            control_type=ControlType.CHECKBOX,
            properties={
                'label': 'Show Quantum State',
                'default': True,
                'description': 'Toggle quantum state visualization'
            },
            callback=self._on_state_toggle
        )
        controls['show_state'] = self.control_panel.get_control_value('show_state')
        
        # Create performance monitoring control
        self.control_panel.create_control(
            control_id="show_performance",
            control_type=ControlType.CHECKBOX,
            properties={
                'label': 'Show Performance Metrics',
                'default': False,
                'description': 'Toggle performance monitoring display'
            },
            callback=self._on_performance_toggle
        )
        controls['show_performance'] = self.control_panel.get_control_value('show_performance')
        
        # Create camera control
        self.control_panel.create_control(
            control_id="camera_angle",
            control_type=ControlType.RANGE_SLIDER,
            properties={
                'label': 'Camera Angle',
                'min': 0,
                'max': 360,
                'step': 1,
                'default': [45, 135],
                'description': 'Control the camera viewing angle'
            },
            callback=self._on_camera_angle_change
        )
        controls['camera_angle'] = self.control_panel.get_control_value('camera_angle')
        
        # Create color scheme control
        self.control_panel.create_control(
            control_id="color_scheme",
            control_type=ControlType.DROPDOWN,
            properties={
                'label': 'Color Scheme',
                'options': ['viridis', 'plasma', 'inferno', 'magma', 'quantum'],
                'default': 'quantum',
                'description': 'Select the color scheme for visualization'
            },
            callback=self._on_color_scheme_change
        )
        controls['color_scheme'] = self.control_panel.get_control_value('color_scheme')
        
        # Create debug mode control
        self.control_panel.create_control(
            control_id="debug_mode",
            control_type=ControlType.TOGGLE,
            properties={
                'label': 'Debug Mode',
                'default': False,
                'description': 'Enable debug mode for circuit inspection'
            },
            callback=self._on_debug_mode_toggle
        )
        controls['debug_mode'] = self.control_panel.get_control_value('debug_mode')
        
        # Create step control for debugging
        self.control_panel.create_control(
            control_id="step_execution",
            control_type=ControlType.BUTTON,
            properties={
                'label': 'Step Forward',
                'description': 'Execute next gate in debug mode'
            },
            callback=self._on_step_forward
        )
        controls['step_execution'] = self.control_panel.get_control_value('step_execution')
        
        # Create reset control
        self.control_panel.create_control(
            control_id="reset_circuit",
            control_type=ControlType.BUTTON,
            properties={
                'label': 'Reset Circuit',
                'description': 'Reset circuit to initial state'
            },
            callback=self._on_reset_circuit
        )
        controls['reset_circuit'] = self.control_panel.get_control_value('reset_circuit')
        
        # Create export control
        self.control_panel.create_control(
            control_id="export_visualization",
            control_type=ControlType.BUTTON,
            properties={
                'label': 'Export Visualization',
                'description': 'Export current visualization'
            },
            callback=self._on_export_visualization
        )
        controls['export_visualization'] = self.control_panel.get_control_value('export_visualization')
        
        # Update statistics
        self.control_stats['control_panels_created'] += 1
        
        logger.info("✅ Visualization controls created")
        return controls
    
    def _on_render_mode_change(self, value: str):
        """Handle render mode change."""
        logger.info(f"🎨 Render mode changed to: {value}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_quality_change(self, value: float):
        """Handle quality change."""
        logger.info(f"🎨 Quality changed to: {value}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_animation_speed_change(self, value: float):
        """Handle animation speed change."""
        logger.info(f"🎨 Animation speed changed to: {value}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_entanglement_toggle(self, value: bool):
        """Handle entanglement toggle."""
        logger.info(f"🎨 Entanglement visualization: {'enabled' if value else 'disabled'}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_state_toggle(self, value: bool):
        """Handle state visualization toggle."""
        logger.info(f"🎨 State visualization: {'enabled' if value else 'disabled'}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_performance_toggle(self, value: bool):
        """Handle performance monitoring toggle."""
        logger.info(f"🎨 Performance monitoring: {'enabled' if value else 'disabled'}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_camera_angle_change(self, value: List[float]):
        """Handle camera angle change."""
        logger.info(f"🎨 Camera angle changed to: {value}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_color_scheme_change(self, value: str):
        """Handle color scheme change."""
        logger.info(f"🎨 Color scheme changed to: {value}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_debug_mode_toggle(self, value: bool):
        """Handle debug mode toggle."""
        logger.info(f"🎨 Debug mode: {'enabled' if value else 'disabled'}")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_step_forward(self, value: Any):
        """Handle step forward action."""
        logger.info("🎨 Step forward executed")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_reset_circuit(self, value: Any):
        """Handle circuit reset."""
        logger.info("🎨 Circuit reset")
        self.control_stats['total_user_interactions'] += 1
    
    def _on_export_visualization(self, value: Any):
        """Handle visualization export."""
        logger.info("🎨 Visualization export requested")
        self.control_stats['total_user_interactions'] += 1
    
    def create_user_session(self, user_id: str) -> str:
        """Create a user session."""
        session_id = f"session_{user_id}_{int(time.time() * 1000)}"
        
        self.user_sessions[session_id] = {
            'user_id': user_id,
            'created_at': time.time(),
            'interactions': 0,
            'controls_used': set()
        }
        
        self.control_stats['active_sessions'] += 1
        logger.info(f"🎨 Created user session: {session_id}")
        return session_id
    
    def update_user_interaction(self, session_id: str, control_id: str):
        """Update user interaction."""
        if session_id in self.user_sessions:
            self.user_sessions[session_id]['interactions'] += 1
            self.user_sessions[session_id]['controls_used'].add(control_id)
            self.control_stats['total_user_interactions'] += 1
    
    def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get user session information."""
        return self.user_sessions.get(session_id)
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """Get interactive controls statistics."""
        return {
            'control_stats': self.control_stats,
            'control_panel_stats': self.control_panel.get_control_statistics(),
            'user_sessions': len(self.user_sessions),
            'active_sessions': self.control_stats['active_sessions']
        }
    
    def get_control_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get control recommendations."""
        recommendations = []
        
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        # Control complexity recommendations
        if len(gates) > 50:
            recommendations.append({
                'type': 'control_complexity',
                'message': f'Large circuit ({len(gates)} gates) detected',
                'recommendation': 'Consider using step-by-step controls for better navigation',
                'priority': 'medium'
            })
        
        # User interaction recommendations
        if self.control_stats['total_user_interactions'] > 1000:
            recommendations.append({
                'type': 'user_interaction',
                'message': f'High interaction count ({self.control_stats["total_user_interactions"]})',
                'recommendation': 'Consider adding keyboard shortcuts for power users',
                'priority': 'low'
            })
        
        # Performance recommendations
        if num_qubits > 15:
            recommendations.append({
                'type': 'performance',
                'message': f'Large qubit count ({num_qubits}) detected',
                'recommendation': 'Consider using simplified controls for better performance',
                'priority': 'medium'
            })
        
        return recommendations
