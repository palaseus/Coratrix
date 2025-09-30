"""
Circuit Renderer - 3D Quantum Circuit Visualization
=================================================

The Circuit Renderer provides 3D visualization of quantum circuits
with WebGL-based rendering and interactive manipulation.

This is the GOD-TIER circuit visualization system that transforms
quantum circuits into immersive 3D experiences.
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

logger = logging.getLogger(__name__)

class CircuitStyle(Enum):
    """Visual styles for circuit rendering."""
    CLASSIC = "classic"
    MODERN = "modern"
    QUANTUM = "quantum"
    HOLOGRAPHIC = "holographic"
    MINIMAL = "minimal"

class RenderQuality(Enum):
    """Rendering quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class GateRenderer:
    """Renderer for individual quantum gates."""
    gate_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    color: Tuple[float, float, float, float]
    material: Dict[str, Any]
    animation: Dict[str, Any]

@dataclass
class CircuitRenderer:
    """Circuit renderer for 3D quantum circuit visualization."""
    
    def __init__(self, style: CircuitStyle = CircuitStyle.QUANTUM, 
                 quality: RenderQuality = RenderQuality.HIGH):
        """Initialize the circuit renderer."""
        self.style = style
        self.quality = quality
        self.gate_renderers: Dict[str, GateRenderer] = {}
        self.circuit_geometry: Dict[str, Any] = {}
        
        # Rendering statistics
        self.render_stats = {
            'total_circuits_rendered': 0,
            'average_render_time': 0.0,
            'total_gates_rendered': 0,
            'geometry_updates': 0,
            'shader_compilations': 0
        }
        
        logger.info(f"🎨 Circuit Renderer initialized - {style.value} style, {quality.value} quality")
    
    async def render_circuit(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a quantum circuit in 3D."""
        logger.info(f"🎨 Rendering circuit: {circuit_data.get('name', 'Unknown')}")
        
        start_time = time.time()
        
        try:
            # Generate circuit geometry
            geometry = await self._generate_circuit_geometry(circuit_data)
            
            # Create gate renderers
            gate_renderers = await self._create_gate_renderers(circuit_data)
            
            # Generate materials and textures
            materials = await self._generate_materials(circuit_data)
            
            # Generate lighting setup
            lighting = await self._generate_lighting(circuit_data)
            
            # Generate shaders
            shaders = await self._generate_shaders(circuit_data)
            
            # Create render output
            render_output = {
                'geometry': geometry,
                'gate_renderers': gate_renderers,
                'materials': materials,
                'lighting': lighting,
                'shaders': shaders,
                'style': self.style.value,
                'quality': self.quality.value,
                'metadata': {
                    'circuit_name': circuit_data.get('name', 'Unknown'),
                    'num_qubits': circuit_data.get('num_qubits', 0),
                    'num_gates': len(circuit_data.get('gates', [])),
                    'render_time': time.time() - start_time
                }
            }
            
            # Update statistics
            self._update_render_stats(time.time() - start_time)
            
            logger.info(f"✅ Circuit rendered in {time.time() - start_time:.4f}s")
            return render_output
            
        except Exception as e:
            logger.error(f"❌ Circuit rendering failed: {e}")
            raise
    
    async def _generate_circuit_geometry(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D geometry for the circuit."""
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        geometry = {
            'qubit_lines': [],
            'gate_objects': [],
            'connection_lines': [],
            'measurement_points': [],
            'bounding_box': self._calculate_bounding_box(gates, num_qubits)
        }
        
        # Generate qubit lines
        for qubit in range(num_qubits):
            qubit_line = await self._generate_qubit_line(qubit, len(gates))
            geometry['qubit_lines'].append(qubit_line)
        
        # Generate gate objects
        for i, gate in enumerate(gates):
            gate_object = await self._generate_gate_object(gate, i, num_qubits)
            geometry['gate_objects'].append(gate_object)
            
            # Generate connections for multi-qubit gates
            if len(gate.get('qubits', [])) > 1:
                connections = await self._generate_gate_connections(gate, i)
                geometry['connection_lines'].extend(connections)
        
        # Generate measurement points
        measurement_points = await self._generate_measurement_points(gates, num_qubits)
        geometry['measurement_points'] = measurement_points
        
        self.render_stats['geometry_updates'] += 1
        return geometry
    
    async def _generate_qubit_line(self, qubit: int, num_gates: int) -> Dict[str, Any]:
        """Generate geometry for a qubit line."""
        y_position = qubit * 2.0
        
        return {
            'id': f"qubit_line_{qubit}",
            'type': 'line',
            'start': [0.0, y_position, 0.0],
            'end': [num_gates * 2.0, y_position, 0.0],
            'thickness': 0.1,
            'color': [0.3, 0.3, 0.3, 1.0],
            'material': 'qubit_line_material',
            'style': self.style.value
        }
    
    async def _generate_gate_object(self, gate: Dict[str, Any], 
                                  gate_index: int, 
                                  num_qubits: int) -> Dict[str, Any]:
        """Generate geometry for a gate object."""
        gate_type = gate.get('type', '')
        qubits = gate.get('qubits', [])
        
        # Calculate position
        x_position = gate_index * 2.0
        y_position = qubits[0] * 2.0 if qubits else 0.0
        z_position = 0.0
        
        # Get gate geometry
        gate_geometry = self._get_gate_geometry(gate_type)
        
        # Get gate material
        gate_material = self._get_gate_material(gate_type)
        
        # Get gate animation
        gate_animation = self._get_gate_animation(gate_type)
        
        return {
            'id': f"gate_{gate_index}",
            'type': gate_type,
            'position': [x_position, y_position, z_position],
            'rotation': [0.0, 0.0, 0.0],
            'scale': [1.0, 1.0, 1.0],
            'geometry': gate_geometry,
            'material': gate_material,
            'animation': gate_animation,
            'qubits': qubits,
            'style': self.style.value
        }
    
    def _get_gate_geometry(self, gate_type: str) -> Dict[str, Any]:
        """Get geometry data for a specific gate type."""
        geometry_map = {
            'H': {
                'type': 'box',
                'size': [0.8, 0.8, 0.8],
                'vertices': self._generate_box_vertices(0.8),
                'normals': self._generate_box_normals(),
                'uvs': self._generate_box_uvs()
            },
            'X': {
                'type': 'box',
                'size': [0.8, 0.8, 0.8],
                'vertices': self._generate_box_vertices(0.8),
                'normals': self._generate_box_normals(),
                'uvs': self._generate_box_uvs()
            },
            'Y': {
                'type': 'box',
                'size': [0.8, 0.8, 0.8],
                'vertices': self._generate_box_vertices(0.8),
                'normals': self._generate_box_normals(),
                'uvs': self._generate_box_uvs()
            },
            'Z': {
                'type': 'box',
                'size': [0.8, 0.8, 0.8],
                'vertices': self._generate_box_vertices(0.8),
                'normals': self._generate_box_normals(),
                'uvs': self._generate_box_uvs()
            },
            'CNOT': {
                'type': 'cylinder',
                'size': [0.6, 0.6, 0.6],
                'vertices': self._generate_cylinder_vertices(0.6, 0.6),
                'normals': self._generate_cylinder_normals(),
                'uvs': self._generate_cylinder_uvs()
            },
            'CZ': {
                'type': 'cylinder',
                'size': [0.6, 0.6, 0.6],
                'vertices': self._generate_cylinder_vertices(0.6, 0.6),
                'normals': self._generate_cylinder_normals(),
                'uvs': self._generate_cylinder_uvs()
            },
            'SWAP': {
                'type': 'sphere',
                'size': [0.7, 0.7, 0.7],
                'vertices': self._generate_sphere_vertices(0.7),
                'normals': self._generate_sphere_normals(),
                'uvs': self._generate_sphere_uvs()
            }
        }
        
        return geometry_map.get(gate_type, {
            'type': 'box',
            'size': [0.8, 0.8, 0.8],
            'vertices': self._generate_box_vertices(0.8),
            'normals': self._generate_box_normals(),
            'uvs': self._generate_box_uvs()
        })
    
    def _generate_box_vertices(self, size: float) -> List[List[float]]:
        """Generate vertices for a box."""
        half_size = size / 2.0
        return [
            [-half_size, -half_size, -half_size],
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, half_size],
            [-half_size, half_size, half_size]
        ]
    
    def _generate_box_normals(self) -> List[List[float]]:
        """Generate normals for a box."""
        return [
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0],
            [-1, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]
        ]
    
    def _generate_box_uvs(self) -> List[List[float]]:
        """Generate UV coordinates for a box."""
        return [
            [0, 0], [1, 0], [1, 1], [0, 1],
            [0, 0], [1, 0], [1, 1], [0, 1],
            [0, 0], [1, 0], [1, 1], [0, 1],
            [0, 0], [1, 0], [1, 1], [0, 1]
        ]
    
    def _generate_cylinder_vertices(self, radius: float, height: float) -> List[List[float]]:
        """Generate vertices for a cylinder."""
        vertices = []
        segments = 16
        
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            vertices.append([x, -height/2, z])
            vertices.append([x, height/2, z])
        
        return vertices
    
    def _generate_cylinder_normals(self) -> List[List[float]]:
        """Generate normals for a cylinder."""
        normals = []
        segments = 16
        
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x = np.cos(angle)
            z = np.sin(angle)
            
            normals.append([x, 0, z])
            normals.append([x, 0, z])
        
        return normals
    
    def _generate_cylinder_uvs(self) -> List[List[float]]:
        """Generate UV coordinates for a cylinder."""
        uvs = []
        segments = 16
        
        for i in range(segments + 1):
            u = i / segments
            uvs.append([u, 0])
            uvs.append([u, 1])
        
        return uvs
    
    def _generate_sphere_vertices(self, radius: float) -> List[List[float]]:
        """Generate vertices for a sphere."""
        vertices = []
        segments = 16
        rings = 8
        
        for ring in range(rings + 1):
            v = ring / rings
            phi = v * np.pi
            
            for segment in range(segments + 1):
                u = segment / segments
                theta = u * 2 * np.pi
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)
                
                vertices.append([x, y, z])
        
        return vertices
    
    def _generate_sphere_normals(self) -> List[List[float]]:
        """Generate normals for a sphere."""
        normals = []
        segments = 16
        rings = 8
        
        for ring in range(rings + 1):
            v = ring / rings
            phi = v * np.pi
            
            for segment in range(segments + 1):
                u = segment / segments
                theta = u * 2 * np.pi
                
                x = np.sin(phi) * np.cos(theta)
                y = np.cos(phi)
                z = np.sin(phi) * np.sin(theta)
                
                normals.append([x, y, z])
        
        return normals
    
    def _generate_sphere_uvs(self) -> List[List[float]]:
        """Generate UV coordinates for a sphere."""
        uvs = []
        segments = 16
        rings = 8
        
        for ring in range(rings + 1):
            v = ring / rings
            
            for segment in range(segments + 1):
                u = segment / segments
                uvs.append([u, v])
        
        return uvs
    
    def _get_gate_material(self, gate_type: str) -> Dict[str, Any]:
        """Get material properties for a specific gate type."""
        material_map = {
            'H': {
                'color': [1.0, 1.0, 1.0, 1.0],
                'emission': [0.2, 0.2, 0.2],
                'shininess': 100,
                'transparency': 0.8,
                'texture': 'hadamard_texture'
            },
            'X': {
                'color': [1.0, 0.0, 0.0, 1.0],
                'emission': [0.1, 0.0, 0.0],
                'shininess': 50,
                'transparency': 0.9,
                'texture': 'pauli_x_texture'
            },
            'Y': {
                'color': [0.0, 1.0, 0.0, 1.0],
                'emission': [0.0, 0.1, 0.0],
                'shininess': 50,
                'transparency': 0.9,
                'texture': 'pauli_y_texture'
            },
            'Z': {
                'color': [0.0, 0.0, 1.0, 1.0],
                'emission': [0.0, 0.0, 0.1],
                'shininess': 50,
                'transparency': 0.9,
                'texture': 'pauli_z_texture'
            },
            'CNOT': {
                'color': [1.0, 0.5, 0.0, 1.0],
                'emission': [0.2, 0.1, 0.0],
                'shininess': 75,
                'transparency': 0.7,
                'texture': 'cnot_texture'
            },
            'CZ': {
                'color': [0.5, 0.0, 1.0, 1.0],
                'emission': [0.1, 0.0, 0.2],
                'shininess': 75,
                'transparency': 0.7,
                'texture': 'cz_texture'
            },
            'SWAP': {
                'color': [1.0, 1.0, 0.0, 1.0],
                'emission': [0.2, 0.2, 0.0],
                'shininess': 25,
                'transparency': 0.6,
                'texture': 'swap_texture'
            }
        }
        
        return material_map.get(gate_type, {
            'color': [0.5, 0.5, 0.5, 1.0],
            'emission': [0.1, 0.1, 0.1],
            'shininess': 50,
            'transparency': 0.8,
            'texture': 'default_texture'
        })
    
    def _get_gate_animation(self, gate_type: str) -> Dict[str, Any]:
        """Get animation data for a specific gate type."""
        animation_map = {
            'H': {
                'type': 'rotation',
                'speed': 1.0,
                'axis': [0, 1, 0],
                'amplitude': 0.5,
                'frequency': 1.0
            },
            'X': {
                'type': 'pulse',
                'speed': 2.0,
                'amplitude': 0.1,
                'frequency': 2.0
            },
            'Y': {
                'type': 'pulse',
                'speed': 2.0,
                'amplitude': 0.1,
                'frequency': 2.0
            },
            'Z': {
                'type': 'pulse',
                'speed': 2.0,
                'amplitude': 0.1,
                'frequency': 2.0
            },
            'CNOT': {
                'type': 'rotation',
                'speed': 0.5,
                'axis': [1, 0, 0],
                'amplitude': 0.3,
                'frequency': 0.5
            },
            'CZ': {
                'type': 'rotation',
                'speed': 0.5,
                'axis': [0, 0, 1],
                'amplitude': 0.3,
                'frequency': 0.5
            },
            'SWAP': {
                'type': 'bounce',
                'speed': 1.5,
                'amplitude': 0.2,
                'frequency': 1.5
            }
        }
        
        return animation_map.get(gate_type, {
            'type': 'static',
            'speed': 0.0,
            'amplitude': 0.0,
            'frequency': 0.0
        })
    
    async def _generate_gate_connections(self, gate: Dict[str, Any], gate_index: int) -> List[Dict[str, Any]]:
        """Generate connection lines for multi-qubit gates."""
        connections = []
        qubits = gate.get('qubits', [])
        
        if len(qubits) > 1:
            x_position = gate_index * 2.0
            
            for i in range(len(qubits) - 1):
                connection = {
                    'id': f"connection_{gate_index}_{i}",
                    'type': 'line',
                    'start': [x_position, qubits[i] * 2.0, 0.0],
                    'end': [x_position, qubits[i + 1] * 2.0, 0.0],
                    'thickness': 0.05,
                    'color': [1.0, 0.5, 0.0, 1.0],
                    'material': 'connection_material',
                    'style': self.style.value
                }
                connections.append(connection)
        
        return connections
    
    async def _generate_measurement_points(self, gates: List[Dict[str, Any]], 
                                        num_qubits: int) -> List[Dict[str, Any]]:
        """Generate measurement points for the circuit."""
        measurement_points = []
        
        # Find measurement gates
        for i, gate in enumerate(gates):
            if gate.get('type', '') == 'MEASURE':
                qubits = gate.get('qubits', [])
                for qubit in qubits:
                    measurement_point = {
                        'id': f"measurement_{i}_{qubit}",
                        'type': 'measurement',
                        'position': [i * 2.0, qubit * 2.0, 0.0],
                        'size': 0.3,
                        'color': [1.0, 0.0, 0.0, 1.0],
                        'material': 'measurement_material',
                        'style': self.style.value
                    }
                    measurement_points.append(measurement_point)
        
        return measurement_points
    
    def _calculate_bounding_box(self, gates: List[Dict[str, Any]], num_qubits: int) -> Dict[str, Any]:
        """Calculate bounding box for the circuit."""
        width = len(gates) * 2.0
        height = num_qubits * 2.0
        depth = 2.0
        
        return {
            'min': [0.0, 0.0, -depth/2],
            'max': [width, height, depth/2],
            'center': [width/2, height/2, 0.0],
            'size': [width, height, depth]
        }
    
    async def _create_gate_renderers(self, circuit_data: Dict[str, Any]) -> Dict[str, GateRenderer]:
        """Create gate renderers for the circuit."""
        gate_renderers = {}
        gates = circuit_data.get('gates', [])
        
        for i, gate in enumerate(gates):
            gate_type = gate.get('type', '')
            qubits = gate.get('qubits', [])
            
            # Calculate position
            x_position = i * 2.0
            y_position = qubits[0] * 2.0 if qubits else 0.0
            z_position = 0.0
            
            # Get gate properties
            geometry = self._get_gate_geometry(gate_type)
            material = self._get_gate_material(gate_type)
            animation = self._get_gate_animation(gate_type)
            
            # Create gate renderer
            gate_renderer = GateRenderer(
                gate_type=gate_type,
                position=(x_position, y_position, z_position),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                color=material['color'],
                material=material,
                animation=animation
            )
            
            gate_renderers[f"gate_{i}"] = gate_renderer
        
        return gate_renderers
    
    async def _generate_materials(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate materials for circuit rendering."""
        return {
            'qubit_line_material': {
                'type': 'line_material',
                'color': [0.3, 0.3, 0.3, 1.0],
                'thickness': 0.1,
                'shininess': 0
            },
            'gate_materials': {
                gate_type: self._get_gate_material(gate_type)
                for gate in circuit_data.get('gates', [])
                for gate_type in [gate.get('type', '')]
            },
            'connection_material': {
                'type': 'line_material',
                'color': [1.0, 0.5, 0.0, 1.0],
                'thickness': 0.05,
                'shininess': 0
            },
            'measurement_material': {
                'type': 'point_material',
                'color': [1.0, 0.0, 0.0, 1.0],
                'size': 0.3,
                'shininess': 100
            }
        }
    
    async def _generate_lighting(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lighting setup for circuit rendering."""
        return {
            'ambient_light': {
                'color': [0.3, 0.3, 0.3],
                'intensity': 0.5
            },
            'directional_light': {
                'color': [1.0, 1.0, 1.0],
                'intensity': 0.8,
                'direction': [0, -1, -1]
            },
            'point_lights': [
                {
                    'position': [0, 5, 5],
                    'color': [0.8, 0.8, 1.0],
                    'intensity': 0.6,
                    'distance': 20.0
                },
                {
                    'position': [10, 5, 5],
                    'color': [1.0, 0.8, 0.8],
                    'intensity': 0.4,
                    'distance': 20.0
                }
            ],
            'quantum_glow': {
                'intensity': 0.3,
                'color': [0.2, 0.8, 1.0],
                'radius': 2.0
            }
        }
    
    async def _generate_shaders(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate shaders for circuit rendering."""
        return {
            'vertex_shader': self._get_vertex_shader(),
            'fragment_shader': self._get_fragment_shader(),
            'geometry_shader': self._get_geometry_shader(),
            'quantum_shader': self._get_quantum_shader()
        }
    
    def _get_vertex_shader(self) -> str:
        """Get vertex shader code."""
        return """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        out float Time;
        
        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            TexCoord = aTexCoord;
            Time = time;
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
    
    def _get_fragment_shader(self) -> str:
        """Get fragment shader code."""
        return """
        #version 330 core
        out vec4 FragColor;
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        in float Time;
        
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;
        uniform float quantumIntensity;
        
        void main() {
            // Basic lighting
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Quantum effects
            float quantumEffect = sin(Time * 2.0 + FragPos.x * 10.0) * 0.5 + 0.5;
            vec3 quantumColor = objectColor * quantumEffect * quantumIntensity;
            
            vec3 result = (diffuse + quantumColor) * objectColor;
            FragColor = vec4(result, 1.0);
        }
        """
    
    def _get_geometry_shader(self) -> str:
        """Get geometry shader code."""
        return """
        #version 330 core
        layout (triangles) in;
        layout (triangle_strip, max_vertices = 3) out;
        
        in vec3 FragPos[];
        in vec3 Normal[];
        in vec2 TexCoord[];
        in float Time[];
        
        out vec3 FragPosOut;
        out vec3 NormalOut;
        out vec2 TexCoordOut;
        out float TimeOut;
        
        void main() {
            for(int i = 0; i < 3; i++) {
                FragPosOut = FragPos[i];
                NormalOut = Normal[i];
                TexCoordOut = TexCoord[i];
                TimeOut = Time[i];
                
                gl_Position = gl_in[i].gl_Position;
                EmitVertex();
            }
            EndPrimitive();
        }
        """
    
    def _get_quantum_shader(self) -> str:
        """Get quantum-specific shader code."""
        return """
        #version 330 core
        // Quantum circuit visualization shader
        uniform float entanglement;
        uniform float superposition;
        uniform float measurement;
        
        vec3 quantumColor(vec3 baseColor, float quantumState) {
            float phase = quantumState * 2.0 * 3.14159;
            vec3 quantumShift = vec3(
                sin(phase),
                sin(phase + 2.094),
                sin(phase + 4.188)
            );
            return baseColor + quantumShift * 0.3;
        }
        """
    
    def _update_render_stats(self, render_time: float):
        """Update rendering statistics."""
        self.render_stats['total_circuits_rendered'] += 1
        
        # Update average render time
        total = self.render_stats['total_circuits_rendered']
        current_avg = self.render_stats['average_render_time']
        self.render_stats['average_render_time'] = (current_avg * (total - 1) + render_time) / total
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """Get circuit renderer statistics."""
        return {
            'render_stats': self.render_stats,
            'style': self.style.value,
            'quality': self.quality.value,
            'gate_renderers_count': len(self.gate_renderers)
        }
    
    def get_render_recommendations(self, circuit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get rendering recommendations."""
        recommendations = []
        
        gates = circuit_data.get('gates', [])
        num_qubits = circuit_data.get('num_qubits', 0)
        
        # Performance recommendations
        if len(gates) > 100:
            recommendations.append({
                'type': 'performance',
                'message': f'Large circuit ({len(gates)} gates) detected',
                'recommendation': 'Consider using lower quality settings for better performance',
                'priority': 'medium'
            })
        
        # Style recommendations
        if num_qubits > 15:
            recommendations.append({
                'type': 'style',
                'message': f'Large qubit count ({num_qubits}) detected',
                'recommendation': 'Consider using minimal style for better clarity',
                'priority': 'low'
            })
        
        # Quality recommendations
        if self.quality == RenderQuality.LOW:
            recommendations.append({
                'type': 'quality',
                'message': 'Low quality rendering enabled',
                'recommendation': 'Consider upgrading to higher quality for better visualization',
                'priority': 'low'
            })
        
        return recommendations
