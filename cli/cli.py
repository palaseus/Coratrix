"""
Command-line interface implementation for Coratrix.

This module provides the main CLI functionality for running quantum scripts,
interactive mode, and displaying quantum state information.
"""

import argparse
import sys
import os
from typing import List, Optional
from vm.parser import QuantumParser
from vm.executor import QuantumExecutor
from vm.instructions import QuantumInstruction


def main():
    """Main entry point for the Coratrix CLI."""
    parser = argparse.ArgumentParser(
        description="Coratrix: A Modular Virtual Quantum Computer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  coratrix --interactive                    # Start interactive mode
  coratrix --script bell_state.qasm         # Run a quantum script
  coratrix --script bell_state.qasm --verbose # Run with detailed output
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
        '--version',
        action='version',
        version='Coratrix 1.0.0'
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.qubits, args.verbose)
    elif args.script:
        run_script(args.script, args.qubits, args.verbose)
    else:
        parser.print_help()


def run_script(script_path: str, num_qubits: int = 2, verbose: bool = False):
    """
    Run a quantum script from a file.
    
    Args:
        script_path: Path to the quantum script file
        num_qubits: Number of qubits in the system
        verbose: Whether to show detailed output
    """
    try:
        # Parse the script
        parser = QuantumParser()
        instructions = parser.parse_file(script_path)
        
        # Validate instructions
        parser.validate_instructions(instructions, num_qubits)
        
        # Create executor and run
        executor = QuantumExecutor(num_qubits)
        
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
            
            # Check for entanglement
            entanglement_info = executor.get_entanglement_info()
            if entanglement_info['is_bell_state']:
                print(f"Bell state detected: {entanglement_info['bell_state']}")
            elif entanglement_info['entanglement'] == 'maximal':
                print("Entangled state detected")
            else:
                print("No entanglement detected")
        
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


def interactive_mode(num_qubits: int = 2, verbose: bool = False):
    """
    Start interactive mode for quantum programming.
    
    Args:
        num_qubits: Number of qubits in the system
        verbose: Whether to show detailed output
    """
    print("Coratrix Interactive Quantum Computer")
    print("=" * 40)
    print(f"Number of qubits: {num_qubits}")
    print("Type 'help' for available commands, 'quit' to exit")
    print()
    
    executor = QuantumExecutor(num_qubits)
    parser = QuantumParser()
    
    while True:
        try:
            command = input("coratrix> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif command.lower() == 'help':
                show_help()
            
            elif command.lower() == 'state':
                show_state(executor, verbose)
            
            elif command.lower() == 'reset':
                executor.reset()
                print("Quantum state reset to |00...0⟩")
            
            elif command.lower() == 'history':
                show_history(executor, verbose)
            
            elif command.lower() == 'entanglement':
                show_entanglement(executor)
            
            elif command.lower().startswith('run '):
                script_path = command[4:].strip()
                if os.path.exists(script_path):
                    run_script(script_path, num_qubits, verbose)
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


def show_help():
    """Display help information for interactive mode."""
    print("Available commands:")
    print("  help                    - Show this help message")
    print("  state                   - Show current quantum state")
    print("  reset                   - Reset to |00...0⟩ state")
    print("  history                 - Show execution history")
    print("  entanglement            - Show entanglement information")
    print("  run <file>              - Run a quantum script file")
    print("  quit/exit/q            - Exit the program")
    print()
    print("Quantum instructions:")
    print("  X q0                    - Apply X gate to qubit 0")
    print("  Y q0                    - Apply Y gate to qubit 0")
    print("  Z q0                    - Apply Z gate to qubit 0")
    print("  H q0                    - Apply Hadamard gate to qubit 0")
    print("  CNOT q0,q1              - Apply CNOT gate (control q0, target q1)")
    print("  MEASURE                 - Measure all qubits")
    print("  MEASURE q0              - Measure qubit 0")
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


def show_entanglement(executor: QuantumExecutor):
    """Display entanglement information."""
    entanglement_info = executor.get_entanglement_info()
    
    if entanglement_info['is_bell_state']:
        print(f"Bell state detected: {entanglement_info['bell_state']}")
        print("This is a maximally entangled 2-qubit state")
    elif entanglement_info['entanglement'] == 'maximal':
        print("Entangled state detected")
        print("The qubits are in an entangled superposition")
    else:
        print("No entanglement detected")
        print("The qubits are in a separable state")
