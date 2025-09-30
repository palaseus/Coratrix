"""
Enhanced CLI interface for Coratrix quantum computer.

This module provides the enhanced CLI functionality with advanced features
including visualization, entanglement analysis, and algorithm execution.
"""

import argparse
import sys
import os
from typing import List, Optional, Dict, Any
from vm.enhanced_parser import EnhancedQuantumParser
from vm.executor import QuantumExecutor
from vm.enhanced_instructions import QuantumInstruction
from core.entanglement_analysis import EntanglementAnalyzer
from visualization.circuit_diagram import CircuitDiagram
from algorithms.quantum_algorithms import (
    GroverAlgorithm, QuantumFourierTransform, 
    QuantumTeleportation, GHZState, WState
)


def main():
    """Main entry point for the enhanced Coratrix CLI."""
    parser = argparse.ArgumentParser(
        description="Coratrix: Research-Grade Virtual Quantum Computer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  coratrix --interactive                    # Start interactive mode
  coratrix --script bell_state.qasm        # Run a quantum script
  coratrix --script bell_state.qasm --visualize # Run with visualization
  coratrix --algorithm grover --qubits 3   # Run Grover's algorithm
  coratrix --entanglement-analysis         # Analyze entanglement
  coratrix --help                          # Show this help message
        """
    )
    
    parser.add_argument(
        '--script', '-s',
        type=str,
        help='Path to quantum script file to execute'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--qubits', '-q',
        type=int,
        default=2,
        help='Number of qubits (default: 2)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--visualize', '--viz',
        action='store_true',
        help='Enable visualization output'
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['grover', 'qft', 'teleportation', 'ghz', 'w-state'],
        help='Run a quantum algorithm'
    )
    
    parser.add_argument(
        '--entanglement-analysis',
        action='store_true',
        help='Perform entanglement analysis'
    )
    
    parser.add_argument(
        '--circuit-diagram',
        action='store_true',
        help='Generate circuit diagram'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration (requires CuPy)'
    )
    
    parser.add_argument(
        '--sparse',
        action='store_true',
        help='Use sparse matrix representation'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Coratrix 2.0.0 (Research Edition)'
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.qubits, args.verbose, args.visualize, args.gpu, args.sparse)
    elif args.script:
        run_script(args.script, args.qubits, args.verbose, args.visualize, args.gpu, args.sparse)
    elif args.algorithm:
        run_algorithm(args.algorithm, args.qubits, args.verbose, args.visualize, args.gpu, args.sparse)
    else:
        parser.print_help()


def run_script(script_path: str, num_qubits: int = 2, verbose: bool = False, 
               visualize: bool = False, use_gpu: bool = False, use_sparse: bool = False):
    """Run a quantum script from a file."""
    try:
        # Parse the script
        parser = EnhancedQuantumParser()
        instructions = parser.parse_file(script_path)
        
        # Validate instructions
        parser.validate_instructions(instructions, num_qubits)
        
        # Create executor
        executor = QuantumExecutor(num_qubits)
        executor.parser = parser  # For enhanced features
        
        if verbose:
            print(f"Running quantum script: {script_path}")
            print(f"Number of qubits: {num_qubits}")
            print(f"Number of instructions: {len(instructions)}")
            print("-" * 50)
        
        # Execute instructions
        results = executor.execute_instructions(instructions)
        
        # Display results
        if verbose:
            print(f"Final state: {executor.get_state_string()}")
            print(f"Probabilities: {executor.get_probabilities()}")
            
            # Entanglement analysis
            if num_qubits >= 2:
                analyzer = EntanglementAnalyzer()
                analysis = analyzer.analyze_entanglement(executor.get_state())
                print(f"Entanglement analysis: {analysis}")
        
        # Visualization
        if visualize:
            generate_visualizations(executor, instructions, num_qubits)
        
        # Show measurement results if any
        measurement_history = executor.get_measurement_history()
        if measurement_history:
            print(f"Measurement results: {measurement_history}")
        
    except FileNotFoundError:
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def run_algorithm(algorithm_name: str, num_qubits: int = 2, verbose: bool = False,
                 visualize: bool = False, use_gpu: bool = False, use_sparse: bool = False):
    """Run a quantum algorithm."""
    try:
        # Create executor
        executor = QuantumExecutor(num_qubits)
        
        # Get algorithm
        algorithm_map = {
            'grover': GroverAlgorithm(),
            'qft': QuantumFourierTransform(),
            'teleportation': QuantumTeleportation(),
            'ghz': GHZState(),
            'w-state': WState()
        }
        
        algorithm = algorithm_map[algorithm_name]
        
        if verbose:
            print(f"Running {algorithm.name}")
            print(f"Description: {algorithm.get_description()}")
            print(f"Number of qubits: {num_qubits}")
            print("-" * 50)
        
        # Execute algorithm
        parameters = {'num_qubits': num_qubits}
        result = algorithm.execute(executor, parameters)
        
        if verbose:
            print(f"Algorithm result: {result}")
            print(f"Final state: {executor.get_state_string()}")
        
        # Visualization
        if visualize:
            generate_algorithm_visualization(algorithm_name, executor, num_qubits)
        
    except Exception as e:
        print(f"Error running algorithm: {e}")
        sys.exit(1)


def interactive_mode(num_qubits: int = 2, verbose: bool = False, visualize: bool = False,
                    use_gpu: bool = False, use_sparse: bool = False):
    """Start interactive mode for quantum programming."""
    print("Coratrix Research-Grade Quantum Computer")
    print("=" * 50)
    print(f"Number of qubits: {num_qubits}")
    print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Sparse matrices: {'Enabled' if use_sparse else 'Disabled'}")
    print("Type 'help' for available commands, 'quit' to exit")
    print()
    
    executor = QuantumExecutor(num_qubits)
    parser = EnhancedQuantumParser()
    executor.parser = parser
    analyzer = EntanglementAnalyzer()
    diagram_generator = CircuitDiagram()
    
    while True:
        try:
            command = input("coratrix> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif command.lower() == 'help':
                show_enhanced_help()
            
            elif command.lower() == 'state':
                show_state(executor, verbose)
            
            elif command.lower() == 'reset':
                executor.reset()
                print("Quantum state reset to |00...0⟩")
            
            elif command.lower() == 'history':
                show_history(executor, verbose)
            
            elif command.lower() == 'entanglement':
                show_entanglement_analysis(executor, analyzer)
            
            elif command.lower() == 'visualize':
                show_visualization_options(executor, diagram_generator, num_qubits)
            
            elif command.lower() == 'algorithms':
                show_available_algorithms()
            
            elif command.lower().startswith('algorithm '):
                algorithm_name = command[10:].strip()
                run_interactive_algorithm(algorithm_name, executor, num_qubits)
            
            elif command.lower().startswith('run '):
                script_path = command[4:].strip()
                if os.path.exists(script_path):
                    run_script(script_path, num_qubits, verbose, visualize, use_gpu, use_sparse)
                else:
                    print(f"Script file not found: {script_path}")
            
            else:
                # Try to parse as quantum instruction
                try:
                    instruction = parser.parse_line(command)
                    if instruction:
                        result = executor.execute_instruction(instruction)
                        print(f"Executed: {instruction}")
                        if result is not None:
                            print(f"Result: {result}")
                    else:
                        print("Empty instruction")
                except ValueError as e:
                    print(f"Parse error: {e}")
                except Exception as e:
                    print(f"Execution error: {e}")
        
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except EOFError:
            print("\nGoodbye!")
            break


def show_enhanced_help():
    """Display enhanced help information."""
    print("Available commands:")
    print("  help                    - Show this help message")
    print("  state                   - Show current quantum state")
    print("  reset                   - Reset to |00...0⟩ state")
    print("  history                 - Show execution history")
    print("  entanglement            - Show entanglement analysis")
    print("  visualize               - Show visualization options")
    print("  algorithms              - Show available algorithms")
    print("  algorithm <name>        - Run a quantum algorithm")
    print("  run <file>              - Run a quantum script file")
    print("  quit/exit/q            - Exit the program")
    print()
    print("Enhanced quantum instructions:")
    print("  X q0                    - Apply X gate to qubit 0")
    print("  Y q0                    - Apply Y gate to qubit 0")
    print("  Z q0                    - Apply Z gate to qubit 0")
    print("  H q0                    - Apply Hadamard gate to qubit 0")
    print("  CNOT q0,q1              - Apply CNOT gate (control q0, target q1)")
    print("  Toffoli q0,q1,q2        - Apply Toffoli gate")
    print("  SWAP q0,q1              - Apply SWAP gate")
    print("  Rx(theta) q0            - Apply rotation around X-axis")
    print("  Ry(theta) q0            - Apply rotation around Y-axis")
    print("  Rz(theta) q0            - Apply rotation around Z-axis")
    print("  CPhase(phi) q0,q1       - Apply controlled phase gate")
    print("  S q0                    - Apply S gate")
    print("  T q0                    - Apply T gate")
    print("  Fredkin q0,q1,q2        - Apply Fredkin gate")
    print("  MEASURE                 - Measure all qubits")
    print("  MEASURE q0              - Measure qubit 0")
    print("  LOOP n: <instructions>  - Repeat instructions n times")
    print("  SUBROUTINE name: <body>  - Define subroutine")
    print("  CALL name               - Call subroutine")
    print("  IF var=val: <body>      - Conditional execution")
    print("  SET var=value           - Set variable")
    print("  # comment               - Add a comment")
    print()


def show_state(executor: QuantumExecutor, verbose: bool):
    """Display the current quantum state."""
    print(f"Current state: {executor.get_state_string()}")
    if verbose:
        print(f"State vector: {executor.get_state_vector()}")
        print(f"Probabilities: {executor.get_probabilities()}")


def show_history(executor: QuantumExecutor, verbose: bool):
    """Display the execution history."""
    history = executor.get_execution_history()
    if not history:
        print("No instructions executed yet")
        return
    
    print("Execution history:")
    for i, (instruction, result) in enumerate(history, 1):
        print(f"  {i}. {instruction}")
        if result is not None and verbose:
            print(f"     Result: {result}")


def show_entanglement_analysis(executor: QuantumExecutor, analyzer: EntanglementAnalyzer):
    """Display entanglement analysis."""
    analysis = analyzer.analyze_entanglement(executor.get_state())
    
    print("Entanglement Analysis:")
    print(f"  Entangled: {analysis['is_entangled']}")
    print(f"  Separable: {analysis['is_separable']}")
    print(f"  Entanglement Entropy: {analysis['entanglement_entropy']:.4f}")
    
    if analysis['is_bell_state']:
        print(f"  Bell State: {analysis['bell_state_type']}")
    
    if analysis['concurrence'] > 0:
        print(f"  Concurrence: {analysis['concurrence']:.4f}")
    
    if analysis['negativity'] > 0:
        print(f"  Negativity: {analysis['negativity']:.4f}")
    
    print(f"  Entanglement Rank: {analysis['entanglement_rank']}")


def show_visualization_options(executor: QuantumExecutor, diagram_generator: CircuitDiagram, num_qubits: int):
    """Show visualization options."""
    print("Visualization Options:")
    print("1. Circuit diagram")
    print("2. State visualization")
    print("3. Probability heatmap")
    print("4. Bloch sphere (single qubit)")
    
    choice = input("Select visualization (1-4): ").strip()
    
    if choice == '1':
        # Generate circuit diagram
        instructions = executor.get_execution_history()
        if instructions:
            print("\nCircuit Diagram:")
            # Convert execution history to diagram format
            diagram_instructions = []
            for instruction, result in instructions:
                if hasattr(instruction, 'gate_name') and hasattr(instruction, 'target_qubits'):
                    diagram_instructions.append({
                        'gate': instruction.gate_name,
                        'qubits': instruction.target_qubits
                    })
            if diagram_instructions:
                print(diagram_generator.generate_diagram(diagram_instructions, num_qubits))
            else:
                print("No circuit operations to visualize")
        else:
            print("No circuit to visualize")
    
    elif choice == '2':
        print(f"\nState: {executor.get_state_string()}")
        print(f"Probabilities: {executor.get_probabilities()}")
    
    elif choice == '3':
        print("\nProbability Heatmap:")
        probs = executor.get_probabilities()
        for i, prob in enumerate(probs):
            binary = format(i, f'0{num_qubits}b')
            bar = '█' * int(prob * 20)
            print(f"|{binary}⟩: {bar} {prob:.3f}")
    
    elif choice == '4':
        if num_qubits == 1:
            print("\nBloch sphere visualization would go here")
        else:
            print("Bloch sphere visualization only available for single qubit")


def show_available_algorithms():
    """Show available quantum algorithms."""
    print("Available Quantum Algorithms:")
    print("1. Grover's Search Algorithm")
    print("2. Quantum Fourier Transform")
    print("3. Quantum Teleportation Protocol")
    print("4. GHZ State Preparation")
    print("5. W State Preparation")


def run_interactive_algorithm(algorithm_name: str, executor: QuantumExecutor, num_qubits: int):
    """Run an algorithm interactively."""
    algorithm_map = {
        'grover': GroverAlgorithm(),
        'qft': QuantumFourierTransform(),
        'teleportation': QuantumTeleportation(),
        'ghz': GHZState(),
        'w-state': WState()
    }
    
    if algorithm_name in algorithm_map:
        algorithm = algorithm_map[algorithm_name]
        print(f"Running {algorithm.name}")
        print(f"Description: {algorithm.get_description()}")
        
        parameters = {'num_qubits': num_qubits}
        result = algorithm.execute(executor, parameters)
        
        print(f"Algorithm result: {result}")
        print(f"Final state: {executor.get_state_string()}")
    else:
        print(f"Unknown algorithm: {algorithm_name}")


def generate_visualizations(executor: QuantumExecutor, instructions: List[QuantumInstruction], num_qubits: int):
    """Generate visualizations for the quantum circuit."""
    print("\n" + "="*50)
    print("VISUALIZATION OUTPUT")
    print("="*50)
    
    # Convert instructions to diagram format
    diagram_instructions = []
    for instruction in instructions:
        if hasattr(instruction, 'gate') and hasattr(instruction, 'target_qubits'):
            diagram_instructions.append({
                'gate': instruction.gate_name if hasattr(instruction, 'gate_name') else 'UNKNOWN',
                'qubits': instruction.target_qubits
            })
    
    # Circuit diagram
    if diagram_instructions:
        diagram_generator = CircuitDiagram()
        print("\nCircuit Diagram:")
        print(diagram_generator.generate_diagram(diagram_instructions, num_qubits))
    else:
        print("\nNo circuit operations to visualize")
    
    # Probability heatmap
    print("\nProbability Distribution:")
    probs = executor.get_probabilities()
    for i, prob in enumerate(probs):
        binary = format(i, f'0{num_qubits}b')
        bar = '█' * int(prob * 20)
        print(f"|{binary}⟩: {bar} {prob:.3f}")


def generate_algorithm_visualization(algorithm_name: str, executor: QuantumExecutor, num_qubits: int):
    """Generate visualizations for quantum algorithms."""
    print("\n" + "="*50)
    print(f"ALGORITHM VISUALIZATION: {algorithm_name.upper()}")
    print("="*50)
    
    # State visualization
    print(f"\nFinal State: {executor.get_state_string()}")
    
    # Probability distribution
    print("\nProbability Distribution:")
    probs = executor.get_probabilities()
    for i, prob in enumerate(probs):
        binary = format(i, f'0{num_qubits}b')
        bar = '█' * int(prob * 20)
        print(f"|{binary}⟩: {bar} {prob:.3f}")


if __name__ == "__main__":
    main()
