"""
CLI interface for hardware backends.

This module provides command-line interface for running quantum circuits
on different hardware backends.
"""

import argparse
import sys
import json
import time
from typing import List, Dict, Any, Optional

from .backend_interface import BackendManager, BackendResult
from .openqasm_interface import OpenQASMInterface
from core.circuit import QuantumCircuit


class HardwareCLI:
    """Command-line interface for hardware operations."""
    
    def __init__(self):
        self.backend_manager = BackendManager()
        self.openqasm_interface = OpenQASMInterface()
    
    def list_backends(self) -> List[Dict[str, Any]]:
        """List available backends with their capabilities."""
        backends = []
        
        for name in self.backend_manager.list_backends():
            backend = self.backend_manager.get_backend(name)
            if backend:
                capabilities = backend.capabilities
                backends.append({
                    'name': name,
                    'type': backend.backend_type.value,
                    'max_qubits': capabilities.max_qubits,
                    'supported_gates': capabilities.supported_gates,
                    'supports_noise': capabilities.supports_noise,
                    'supports_parameterized_gates': capabilities.supports_parameterized_gates,
                    'max_circuit_depth': capabilities.max_circuit_depth,
                    'available': backend.is_available()
                })
        
        return backends
    
    def execute_circuit(self, circuit: QuantumCircuit, backend_name: str, 
                       shots: int = 1024, verbose: bool = False) -> BackendResult:
        """Execute a circuit on a specific backend."""
        if verbose:
            print(f"Executing circuit on backend: {backend_name}")
            print(f"Circuit: {len(circuit.gates)} gates, {circuit.num_qubits} qubits")
            print(f"Shots: {shots}")
        
        result = self.backend_manager.execute_circuit(circuit, backend_name, shots)
        
        if verbose:
            if result.success:
                print(f"Execution successful in {result.execution_time:.3f}s")
                print(f"Results: {result.counts}")
            else:
                print(f"Execution failed: {result.error_message}")
        
        return result
    
    def import_and_execute(self, qasm_file: str, backend_name: str, 
                          shots: int = 1024, verbose: bool = False) -> BackendResult:
        """Import OpenQASM file and execute on backend."""
        if verbose:
            print(f"Importing circuit from: {qasm_file}")
        
        try:
            circuit = self.openqasm_interface.import_circuit(qasm_file)
            return self.execute_circuit(circuit, backend_name, shots, verbose)
        except Exception as e:
            return BackendResult(
                success=False,
                counts={},
                execution_time=0.0,
                backend_info={'backend_name': backend_name},
                error_message=f"Import error: {str(e)}"
            )
    
    def export_circuit(self, circuit: QuantumCircuit, output_file: str, verbose: bool = False):
        """Export circuit to OpenQASM file."""
        if verbose:
            print(f"Exporting circuit to: {output_file}")
        
        self.openqasm_interface.export_circuit(circuit, output_file)
        
        if verbose:
            print("Export completed successfully")
    
    def benchmark_backends(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """Benchmark circuit execution across all available backends."""
        results = {}
        
        for backend_name in self.backend_manager.list_backends():
            print(f"Benchmarking {backend_name}...")
            
            start_time = time.time()
            result = self.execute_circuit(circuit, backend_name, shots, verbose=False)
            end_time = time.time()
            
            results[backend_name] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'total_time': end_time - start_time,
                'shots': shots,
                'counts': result.counts,
                'error': result.error_message
            }
        
        return results


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Coratrix Hardware CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List backends command
    list_parser = subparsers.add_parser('list', help='List available backends')
    list_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute a circuit')
    execute_parser.add_argument('--backend', required=True, help='Backend name')
    execute_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    execute_parser.add_argument('--qasm', help='OpenQASM file to execute')
    execute_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export circuit to OpenQASM')
    export_parser.add_argument('--output', required=True, help='Output file')
    export_parser.add_argument('--qasm', help='OpenQASM file to export')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark backends')
    benchmark_parser.add_argument('--qasm', required=True, help='OpenQASM file to benchmark')
    benchmark_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    benchmark_parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = HardwareCLI()
    
    if args.command == 'list':
        backends = cli.list_backends()
        
        if args.json:
            print(json.dumps(backends, indent=2))
        else:
            print("Available Backends:")
            print("-" * 40)
            for backend in backends:
                print(f"Name: {backend['name']}")
                print(f"Type: {backend['type']}")
                print(f"Max Qubits: {backend['max_qubits']}")
                print(f"Supports Noise: {backend['supports_noise']}")
                print(f"Available: {backend['available']}")
                print()
    
    elif args.command == 'execute':
        if args.qasm:
            result = cli.import_and_execute(args.qasm, args.backend, args.shots, args.verbose)
        else:
            print("Error: --qasm required for execute command")
            return
        
        if args.verbose:
            if result.success:
                print(f"Success! Execution time: {result.execution_time:.3f}s")
                print(f"Results: {result.counts}")
            else:
                print(f"Failed: {result.error_message}")
    
    elif args.command == 'export':
        if args.qasm:
            try:
                circuit = cli.openqasm_interface.import_circuit(args.qasm)
                cli.export_circuit(circuit, args.output, verbose=True)
            except Exception as e:
                print(f"Export failed: {e}")
        else:
            print("Error: --qasm required for export command")
    
    elif args.command == 'benchmark':
        try:
            circuit = cli.openqasm_interface.import_circuit(args.qasm)
            results = cli.benchmark_backends(circuit, args.shots)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Benchmark results saved to {args.output}")
            else:
                print("Benchmark Results:")
                print("-" * 40)
                for backend_name, result in results.items():
                    print(f"{backend_name}:")
                    print(f"  Success: {result['success']}")
                    print(f"  Execution Time: {result['execution_time']:.3f}s")
                    if result['error']:
                        print(f"  Error: {result['error']}")
                    print()
        
        except Exception as e:
            print(f"Benchmark failed: {e}")


if __name__ == '__main__':
    main()
