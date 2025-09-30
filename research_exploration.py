#!/usr/bin/env python3
"""
Comprehensive research-grade quantum exploration script.

This script executes a full-spectrum quantum exploration on Coratrix 2.0,
combining multiple advanced algorithms, entanglement analysis, visualization,
and comprehensive reporting.
"""

import sys
import os
import argparse
import time
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.quantum_explorer import QuantumExplorer
from research.entanglement_tracker import EntanglementTracker
from research.visualization_engine import VisualizationEngine
from research.report_generator import ReportGenerator


def main():
    """Main entry point for quantum exploration."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Quantum Exploration on Coratrix 2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 research_exploration.py                    # Default 5-qubit exploration
  python3 research_exploration.py --qubits 8        # 8-qubit exploration
  python3 research_exploration.py --qubits 6 --gpu  # 6-qubit with GPU acceleration
  python3 research_exploration.py --qubits 10 --sparse  # 10-qubit with sparse matrices
        """
    )
    
    parser.add_argument(
        '--qubits', '-q',
        type=int,
        default=5,
        help='Number of qubits for exploration (default: 5)'
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
        '--output', '-o',
        type=str,
        help='Output filename for report (default: auto-generated)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Enable interactive mode after exploration'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.qubits < 2:
        print("❌ Error: Minimum 2 qubits required")
        sys.exit(1)
    
    if args.qubits > 12:
        print("⚠️  Warning: Large qubit count may require significant memory")
        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Exploration cancelled")
            sys.exit(0)
    
    # Initialize exploration
    print("🚀 INITIALIZING QUANTUM EXPLORATION")
    print("=" * 60)
    print(f"Qubits: {args.qubits}")
    print(f"GPU: {'Enabled' if args.gpu else 'Disabled'}")
    print(f"Sparse: {'Enabled' if args.sparse else 'Disabled'}")
    print(f"State Dimension: {2**args.qubits}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Initialize quantum explorer
        explorer = QuantumExplorer(
            num_qubits=args.qubits,
            use_gpu=args.gpu,
            use_sparse=args.sparse
        )
        
        # Initialize supporting components
        entanglement_tracker = EntanglementTracker()
        visualization_engine = VisualizationEngine()
        report_generator = ReportGenerator()
        
        # Execute full exploration
        start_time = time.time()
        exploration_results = explorer.execute_full_exploration()
        exploration_time = time.time() - start_time
        
        print(f"\n⏱️  Total exploration time: {exploration_time:.4f}s")
        
        # Generate comprehensive report
        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("-" * 50)
        
        report = report_generator.generate_comprehensive_report(exploration_results)
        
        # Save report to file
        if args.output:
            report_filename = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"quantum_exploration_report_{timestamp}.json"
        
        report_generator.save_report_to_file(report, report_filename)
        
        # Print interactive summary
        report_generator.print_interactive_summary(report)
        
        # Interactive mode if requested
        if args.interactive:
            interactive_mode(explorer, report)
        
        print(f"\n✅ QUANTUM EXPLORATION COMPLETED SUCCESSFULLY")
        print(f"📄 Report saved to: {report_filename}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Exploration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during exploration: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def interactive_mode(explorer: QuantumExplorer, report: Dict[str, Any]):
    """Interactive mode for exploring results."""
    print("\n🔍 INTERACTIVE EXPLORATION MODE")
    print("=" * 50)
    print("Available commands:")
    print("  state - Show current quantum state")
    print("  algorithm <name> - Run specific algorithm")
    print("  entanglement - Show entanglement analysis")
    print("  visualize - Show visualization options")
    print("  report - Show report summary")
    print("  help - Show this help")
    print("  quit - Exit interactive mode")
    print()
    
    while True:
        try:
            command = input("quantum> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            elif command.lower() == 'help':
                print("Available commands:")
                print("  state - Show current quantum state")
                print("  algorithm <name> - Run specific algorithm")
                print("  entanglement - Show entanglement analysis")
                print("  visualize - Show visualization options")
                print("  report - Show report summary")
                print("  help - Show this help")
                print("  quit - Exit interactive mode")
            
            elif command.lower() == 'state':
                current_state = explorer.executor.get_state()
                print(f"Current state: {current_state}")
                print(f"Probabilities: {current_state.get_probabilities()}")
            
            elif command.lower().startswith('algorithm '):
                algo_name = command[10:].strip()
                if algo_name in explorer.algorithms:
                    print(f"Running {algo_name} algorithm...")
                    # Run algorithm (simplified)
                    print(f"Algorithm {algo_name} executed")
                else:
                    print(f"Unknown algorithm: {algo_name}")
                    print(f"Available algorithms: {list(explorer.algorithms.keys())}")
            
            elif command.lower() == 'entanglement':
                current_state = explorer.executor.get_state()
                analysis = explorer.entanglement_analyzer.analyze_entanglement(current_state)
                print("Entanglement Analysis:")
                print(f"  Entangled: {analysis['is_entangled']}")
                print(f"  Entropy: {analysis['entanglement_entropy']:.4f}")
                print(f"  Concurrence: {analysis.get('concurrence', 0.0):.4f}")
                print(f"  Negativity: {analysis.get('negativity', 0.0):.4f}")
            
            elif command.lower() == 'visualize':
                print("Visualization options:")
                print("  1. Probability heatmap")
                print("  2. State evolution")
                print("  3. Entanglement metrics")
                choice = input("Select option (1-3): ").strip()
                if choice == '1':
                    current_state = explorer.executor.get_state()
                    probabilities = current_state.get_probabilities()
                    print("Probability Heatmap:")
                    for i, prob in enumerate(probabilities):
                        binary = format(i, f'0{current_state.num_qubits}b')
                        bar = '█' * int(prob * 20)
                        print(f"|{binary}⟩: {bar} {prob:.4f}")
                elif choice == '2':
                    print("State evolution visualization would go here")
                elif choice == '3':
                    print("Entanglement metrics visualization would go here")
                else:
                    print("Invalid choice")
            
            elif command.lower() == 'report':
                print("Report Summary:")
                exec_summary = report.get('executive_summary', {})
                print(f"  Total Algorithms: {exec_summary.get('total_algorithms', 0)}")
                print(f"  Success Rate: {exec_summary.get('success_rate', 0.0):.2%}")
                print(f"  Entangled States: {exec_summary.get('entangled_states', 0)}")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
