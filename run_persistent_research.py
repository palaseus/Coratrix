#!/usr/bin/env python3
"""
Run Persistent Quantum Research Engine

This script runs the persistent quantum research engine that maintains
state, runs continuous research cycles, and accumulates meaningful results.

Usage:
    python run_persistent_research.py [--duration HOURS] [--output DIR]

Author: Quantum Research Engine - Coratrix 4.0
"""

import asyncio
import logging
import argparse
import signal
import sys
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from quantum_research.persistent_research_engine import (
    PersistentResearchEngine,
    run_persistent_research,
    setup_signal_handlers
)
from quantum_research.quantum_research_engine import ResearchConfig, ResearchMode

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('persistent_research.log')
        ]
    )

async def main():
    """Main function to run persistent research."""
    parser = argparse.ArgumentParser(
        description="Run persistent quantum research engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_persistent_research.py                    # Run for 24 hours
  python run_persistent_research.py --duration 1      # Run for 1 hour
  python run_persistent_research.py --output ./research  # Custom output directory
        """
    )
    
    parser.add_argument(
        "--duration", 
        type=float, 
        default=24.0,
        help="Session duration in hours (default: 24.0)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="research_outputs",
        help="Output directory for research results (default: research_outputs)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["continuous", "exploratory", "focused"],
        default="continuous",
        help="Research mode (default: continuous)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Convert research mode
    mode_map = {
        "continuous": ResearchMode.CONTINUOUS,
        "exploratory": ResearchMode.EXPLORATORY,
        "focused": ResearchMode.FOCUSED
    }
    research_mode = mode_map[args.mode]
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Persistent Quantum Research Engine")
    logger.info(f"📊 Configuration:")
    logger.info(f"  • Duration: {args.duration} hours")
    logger.info(f"  • Output Directory: {output_path.absolute()}")
    logger.info(f"  • Research Mode: {args.mode}")
    logger.info(f"  • Verbose Logging: {args.verbose}")
    
    try:
        # Create and configure research engine
        config = ResearchConfig(research_mode=research_mode)
        engine = PersistentResearchEngine(
            config=config,
            session_duration_hours=args.duration,
            output_dir=str(output_path)
        )
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(engine)
        
        # Start research session
        session_id = await engine.start_research_session()
        logger.info(f"🎯 Research session started: {session_id}")
        
        # Keep running until shutdown
        while engine.running and not engine.shutdown_requested:
            await asyncio.sleep(1.0)
        
        logger.info("✅ Research session completed")
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"❌ Error in persistent research: {e}")
        raise
    finally:
        if 'engine' in locals():
            await engine.shutdown()
        logger.info("🏁 Persistent research engine shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
