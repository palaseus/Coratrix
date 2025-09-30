"""
Reproducibility and security utilities for Coratrix.

This module provides deterministic seeds, metadata tracking, and security
features for reproducible quantum computing experiments.
"""

import os
import sys
import time
import json
import hashlib
import platform
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import random
import uuid

# Optional GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class SystemMetadata:
    """System metadata for reproducibility."""
    timestamp: str
    python_version: str
    platform: str
    architecture: str
    cpu_count: int
    memory_total_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    numpy_version: str
    scipy_version: str
    cupy_version: Optional[str]
    git_commit_hash: Optional[str]
    git_branch: Optional[str]
    working_directory: str
    environment_variables: Dict[str, str]
    random_seed: int
    session_id: str


@dataclass
class ExperimentMetadata:
    """Metadata for a specific experiment."""
    experiment_id: str
    session_id: str
    timestamp: str
    experiment_type: str
    parameters: Dict[str, Any]
    system_metadata: SystemMetadata
    reproducibility_hash: str
    output_files: List[str]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class ReproducibilityManager:
    """Manager for reproducibility and security features."""
    
    def __init__(self, session_id: Optional[str] = None, random_seed: Optional[int] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.random_seed = random_seed or 42
        self.experiments = []
        
        # Set deterministic seeds
        self._set_deterministic_seeds()
        
        # Collect system metadata
        self.system_metadata = self._collect_system_metadata()
    
    def _set_deterministic_seeds(self):
        """Set deterministic seeds for reproducibility."""
        # Set Python random seed
        random.seed(self.random_seed)
        
        # Set NumPy random seed
        np.random.seed(self.random_seed)
        
        # Set GPU random seed if available
        if GPU_AVAILABLE:
            cp.random.seed(self.random_seed)
        
        # Set environment variable for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
    
    def _collect_system_metadata(self) -> SystemMetadata:
        """Collect system metadata for reproducibility."""
        import psutil
        
        # Get system information
        timestamp = datetime.now().isoformat()
        python_version = sys.version
        platform_info = platform.platform()
        architecture = platform.architecture()[0]
        cpu_count = psutil.cpu_count()
        memory_total_gb = psutil.virtual_memory().total / (1024**3)
        
        # Get GPU information
        gpu_available = GPU_AVAILABLE
        gpu_memory_gb = 0.0
        if gpu_available:
            try:
                gpu_memory_gb = cp.cuda.Device().mem_info[1] / (1024**3)
            except:
                gpu_memory_gb = 0.0
        
        # Get package versions
        numpy_version = np.__version__
        scipy_version = self._get_package_version('scipy')
        cupy_version = self._get_package_version('cupy') if gpu_available else None
        
        # Get Git information
        git_commit_hash = self._get_git_commit_hash()
        git_branch = self._get_git_branch()
        
        # Get working directory
        working_directory = os.getcwd()
        
        # Get relevant environment variables
        env_vars = {
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', ''),
            'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', '')
        }
        
        return SystemMetadata(
            timestamp=timestamp,
            python_version=python_version,
            platform=platform_info,
            architecture=architecture,
            cpu_count=cpu_count,
            memory_total_gb=memory_total_gb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            numpy_version=numpy_version,
            scipy_version=scipy_version,
            cupy_version=cupy_version,
            git_commit_hash=git_commit_hash,
            git_branch=git_branch,
            working_directory=working_directory,
            environment_variables=env_vars,
            random_seed=self.random_seed,
            session_id=self.session_id
        )
    
    def _get_package_version(self, package_name: str) -> str:
        """Get version of a Python package."""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except:
            try:
                # Use importlib.metadata as fallback for older Python versions
                import importlib.metadata as metadata
                return metadata.version(package_name)
            except:
                return "unknown"
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """Get Git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return None
    
    def _get_git_branch(self) -> Optional[str]:
        """Get Git branch name."""
        try:
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return None
    
    def create_experiment(self, experiment_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new experiment with metadata."""
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create reproducibility hash
        # Exclude timestamp from system metadata for reproducibility comparison
        system_metadata_dict = asdict(self.system_metadata)
        system_metadata_dict.pop('timestamp', None)  # Remove timestamp for comparison
        
        reproducibility_data = {
            'experiment_type': experiment_type,
            'parameters': parameters,
            'system_metadata': system_metadata_dict,
            'timestamp': timestamp
        }
        reproducibility_hash = self._calculate_hash(reproducibility_data)
        
        # Create experiment metadata
        experiment_metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            session_id=self.session_id,
            timestamp=timestamp,
            experiment_type=experiment_type,
            parameters=parameters,
            system_metadata=self.system_metadata,
            reproducibility_hash=reproducibility_hash,
            output_files=[],
            execution_time=0.0,
            success=False
        )
        
        self.experiments.append(experiment_metadata)
        return experiment_id
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash for reproducibility verification."""
        # Convert data to JSON string for hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def update_experiment(self, experiment_id: str, execution_time: float, 
                         success: bool, output_files: List[str] = None,
                         error_message: Optional[str] = None):
        """Update experiment with results."""
        for experiment in self.experiments:
            if experiment.experiment_id == experiment_id:
                experiment.execution_time = execution_time
                experiment.success = success
                experiment.output_files = output_files or []
                experiment.error_message = error_message
                break
    
    def get_experiment_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Get experiment metadata by ID."""
        for experiment in self.experiments:
            if experiment.experiment_id == experiment_id:
                return experiment
        return None
    
    def export_session_metadata(self, filename: str):
        """Export session metadata to file."""
        session_data = {
            'session_id': self.session_id,
            'random_seed': self.random_seed,
            'system_metadata': asdict(self.system_metadata),
            'experiments': [asdict(exp) for exp in self.experiments]
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def verify_reproducibility(self, experiment_id: str) -> Tuple[bool, str]:
        """Verify experiment reproducibility."""
        experiment = self.get_experiment_metadata(experiment_id)
        if not experiment:
            return False, "Experiment not found"
        
        # Recalculate hash with current system state
        current_system_metadata = self._collect_system_metadata()
        
        # Check if system metadata matches
        if current_system_metadata.git_commit_hash != experiment.system_metadata.git_commit_hash:
            return False, "Git commit hash mismatch"
        
        if current_system_metadata.python_version != experiment.system_metadata.python_version:
            return False, "Python version mismatch"
        
        if current_system_metadata.numpy_version != experiment.system_metadata.numpy_version:
            return False, "NumPy version mismatch"
        
        # Recalculate reproducibility hash
        # Exclude timestamp from system metadata for reproducibility comparison
        system_metadata_dict = asdict(current_system_metadata)
        system_metadata_dict.pop('timestamp', None)  # Remove timestamp for comparison
        
        reproducibility_data = {
            'experiment_type': experiment.experiment_type,
            'parameters': experiment.parameters,
            'system_metadata': system_metadata_dict,
            'timestamp': experiment.timestamp
        }
        current_hash = self._calculate_hash(reproducibility_data)
        
        if current_hash != experiment.reproducibility_hash:
            return False, "Reproducibility hash mismatch"
        
        return True, "Reproducibility verified"
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        total_experiments = len(self.experiments)
        successful_experiments = sum(1 for exp in self.experiments if exp.success)
        total_execution_time = sum(exp.execution_time for exp in self.experiments)
        
        return {
            'session_id': self.session_id,
            'random_seed': self.random_seed,
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / max(total_experiments, 1),
            'total_execution_time': total_execution_time,
            'system_metadata': asdict(self.system_metadata)
        }


class SecurityManager:
    """Security manager for sensitive data handling."""
    
    def __init__(self, privacy_mode: bool = False):
        self.privacy_mode = privacy_mode
        self.sensitive_fields = [
            'git_commit_hash',
            'working_directory',
            'environment_variables',
            'session_id',
            'experiment_id'
        ]
    
    def redact_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive data from metadata."""
        if not self.privacy_mode:
            return data
        
        redacted_data = data.copy()
        
        for field in self.sensitive_fields:
            if field in redacted_data:
                redacted_data[field] = "[REDACTED]"
        
        # Redact nested sensitive data
        if 'system_metadata' in redacted_data:
            system_metadata = redacted_data['system_metadata'].copy()
            for field in self.sensitive_fields:
                if field in system_metadata:
                    system_metadata[field] = "[REDACTED]"
            redacted_data['system_metadata'] = system_metadata
        
        return redacted_data
    
    def create_privacy_report(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create privacy-safe report."""
        redacted_metadata = self.redact_sensitive_data(metadata)
        
        # Add privacy notice
        privacy_notice = {
            'privacy_mode': True,
            'redacted_fields': self.sensitive_fields,
            'redaction_timestamp': datetime.now().isoformat()
        }
        
        return {
            'privacy_notice': privacy_notice,
            'metadata': redacted_metadata
        }


class DeterministicRandom:
    """Deterministic random number generator for reproducibility."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.python_random = random.Random(seed)
        
        if GPU_AVAILABLE:
            cp.random.seed(seed)
    
    def get_random_state(self) -> Dict[str, Any]:
        """Get current random state for reproducibility."""
        return {
            'numpy_state': self.np_random.get_state(),
            'python_seed': self.python_random.getstate(),
            'cupy_seed': self.seed if GPU_AVAILABLE else None
        }
    
    def set_random_state(self, state: Dict[str, Any]):
        """Set random state for reproducibility."""
        if 'numpy_state' in state:
            self.np_random.set_state(state['numpy_state'])
        
        if 'python_seed' in state:
            self.python_random.setstate(state['python_seed'])
        
        if 'cupy_seed' in state and GPU_AVAILABLE:
            cp.random.seed(state['cupy_seed'])
    
    def random(self) -> float:
        """Generate random float."""
        return self.np_random.random()
    
    def randint(self, low: int, high: int) -> int:
        """Generate random integer."""
        return self.np_random.randint(low, high)
    
    def choice(self, choices: List[Any]) -> Any:
        """Choose random element from list."""
        return self.np_random.choice(choices)
    
    def shuffle(self, array: List[Any]) -> List[Any]:
        """Shuffle array in place."""
        shuffled = array.copy()
        self.np_random.shuffle(shuffled)
        return shuffled


class ReproducibilityChecker:
    """Checker for reproducibility verification."""
    
    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance
    
    def check_numerical_reproducibility(self, values1: List[float], 
                                      values2: List[float]) -> Tuple[bool, float]:
        """Check if two sets of numerical values are reproducible."""
        if len(values1) != len(values2):
            return False, float('inf')
        
        differences = [abs(v1 - v2) for v1, v2 in zip(values1, values2)]
        max_difference = max(differences)
        
        is_reproducible = max_difference <= self.tolerance
        return is_reproducible, max_difference
    
    def check_quantum_state_reproducibility(self, state1, state2) -> Tuple[bool, float]:
        """Check if two quantum states are reproducible."""
        if hasattr(state1, 'to_dense'):
            vec1 = state1.to_dense()
        else:
            vec1 = state1.state_vector
        
        if hasattr(state2, 'to_dense'):
            vec2 = state2.to_dense()
        else:
            vec2 = state2.state_vector
        
        if len(vec1) != len(vec2):
            return False, float('inf')
        
        differences = np.abs(vec1 - vec2)
        max_difference = np.max(differences)
        
        is_reproducible = max_difference <= self.tolerance
        return is_reproducible, max_difference
    
    def check_circuit_reproducibility(self, circuit1, circuit2) -> Tuple[bool, str]:
        """Check if two circuits are reproducible."""
        if circuit1.num_qubits != circuit2.num_qubits:
            return False, "Different number of qubits"
        
        if len(circuit1.gates) != len(circuit2.gates):
            return False, "Different number of gates"
        
        for i, ((gate1, qubits1), (gate2, qubits2)) in enumerate(zip(circuit1.gates, circuit2.gates)):
            if gate1.name != gate2.name:
                return False, f"Different gate at position {i}"
            
            if qubits1 != qubits2:
                return False, f"Different qubit indices at position {i}"
        
        return True, "Circuits are reproducible"
