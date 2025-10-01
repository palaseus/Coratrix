"""
Persistent Quantum Research Engine

This module provides a persistent, long-running quantum research engine that
maintains state, runs continuous research cycles, and accumulates knowledge
over time.

Author: Quantum Research Engine - Coratrix 4.0
"""

import asyncio
import time
import logging
import json
import signal
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import uuid
from pathlib import Path

from .quantum_research_engine import QuantumResearchEngine, ResearchConfig, ResearchMode
from .quantum_algorithm_generator import QuantumAlgorithm
from .autonomous_experimenter import ExperimentResult
from .self_evolving_optimizer import OptimizationResult
from .quantum_strategy_advisor import StrategicRecommendation
from .knowledge_expander import KnowledgeEntry
from .continuous_evolver import EvolutionResult

logger = logging.getLogger(__name__)

@dataclass
class ResearchSession:
    """Research session data."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    algorithms_generated: int = 0
    experiments_completed: int = 0
    optimizations_performed: int = 0
    breakthroughs_detected: int = 0
    knowledge_entries: int = 0
    total_research_time: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    status: str = "active"

@dataclass
class ResearchMetrics:
    """Research metrics over time."""
    timestamp: float
    algorithms_generated: int
    experiments_completed: int
    optimizations_performed: int
    breakthroughs_detected: int
    knowledge_entries: int
    memory_usage_mb: float
    cpu_usage_percent: float
    active_research_count: int
    queued_research_count: int

class PersistentResearchEngine:
    """
    Persistent Quantum Research Engine that runs continuously.
    
    This engine maintains state across research cycles, accumulates knowledge,
    and provides meaningful research metrics and outputs.
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None, 
                 session_duration_hours: float = 24.0,
                 output_dir: str = "research_outputs"):
        """Initialize the persistent research engine."""
        self.config = config or ResearchConfig()
        self.session_duration = session_duration_hours * 3600  # Convert to seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Session management
        self.current_session: Optional[ResearchSession] = None
        self.session_history: List[ResearchSession] = []
        self.research_metrics: List[ResearchMetrics] = []
        
        # Persistent state
        self.accumulated_algorithms: List[QuantumAlgorithm] = []
        self.accumulated_experiments: List[ExperimentResult] = []
        self.accumulated_optimizations: List[OptimizationResult] = []
        self.accumulated_advisories: List[StrategicRecommendation] = []
        self.accumulated_knowledge: List[KnowledgeEntry] = []
        self.accumulated_evolutions: List[EvolutionResult] = []
        
        # Research engine
        self.engine: Optional[QuantumResearchEngine] = None
        self.running = False
        self.shutdown_requested = False
        
        # Metrics tracking
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        
        logger.info(f"Persistent Research Engine initialized")
    
    async def start_research_session(self) -> str:
        """Start a new research session."""
        session_id = f"session_{int(time.time() * 1000)}"
        
        self.current_session = ResearchSession(
            session_id=session_id,
            start_time=time.time()
        )
        
        # Initialize research engine
        self.engine = QuantumResearchEngine(self.config)
        await self.engine.start()
        
        self.running = True
        logger.info(f"🚀 Started research session: {session_id}")
        
        # Start background tasks
        asyncio.create_task(self._research_cycle())
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._state_persister())
        asyncio.create_task(self._session_monitor())
        
        return session_id
    
    async def stop_research_session(self):
        """Stop the current research session."""
        if not self.running:
            logger.warning("No active research session to stop")
            return
        
        self.running = False
        
        if self.engine:
            await self.engine.stop()
            self.engine = None
        
        if self.current_session:
            self.current_session.end_time = time.time()
            self.current_session.total_research_time = (
                self.current_session.end_time - self.current_session.start_time
            )
            self.current_session.status = "completed"
            self.session_history.append(self.current_session)
            
            # Save session data
            await self._save_session_data()
            
            logger.info(f"🛑 Stopped research session: {self.current_session.session_id}")
            logger.info(f"📊 Session Summary:")
            logger.info(f"  • Duration: {self.current_session.total_research_time:.2f}s")
            logger.info(f"  • Algorithms Generated: {self.current_session.algorithms_generated}")
            logger.info(f"  • Experiments Completed: {self.current_session.experiments_completed}")
            logger.info(f"  • Optimizations Performed: {self.current_session.optimizations_performed}")
            logger.info(f"  • Breakthroughs Detected: {self.current_session.breakthroughs_detected}")
            logger.info(f"  • Knowledge Entries: {self.current_session.knowledge_entries}")
            
            self.current_session = None
    
    async def _research_cycle(self):
        """Main research cycle that runs continuously."""
        cycle_count = 0
        
        while self.running and not self.shutdown_requested:
            try:
                cycle_count += 1
                cycle_start = time.time()
                
                logger.info(f"🔄 Research Cycle {cycle_count} starting...")
                
                # Collect current state
                if self.engine:
                    # Get statistics from engine
                    engine_stats = self.engine.get_research_statistics()
                    
                    # Update session metrics
                    if self.current_session:
                        self.current_session.algorithms_generated = len(self.engine.algorithm_generator.generated_algorithms)
                        self.current_session.experiments_completed = len(self.engine.experimenter.completed_experiments)
                        self.current_session.optimizations_performed = len(self.engine.optimizer.completed_optimizations)
                        self.current_session.breakthroughs_detected = len(self.engine.breakthrough_detections)
                        self.current_session.knowledge_entries = len(self.engine.knowledge_expander.knowledge_base)
                    
                    # Accumulate results
                    await self._accumulate_results()
                    
                    # Generate research report
                    await self._generate_cycle_report(cycle_count)
                
                cycle_duration = time.time() - cycle_start
                logger.info(f"✅ Research Cycle {cycle_count} completed in {cycle_duration:.2f}s")
                
                # Wait before next cycle
                await asyncio.sleep(10.0)  # 10 second cycles
                
            except Exception as e:
                logger.error(f"❌ Error in research cycle {cycle_count}: {e}")
                await asyncio.sleep(5.0)
    
    async def _accumulate_results(self):
        """Accumulate results from the research engine."""
        if not self.engine:
            return
        
        # Accumulate algorithms
        new_algorithms = self.engine.algorithm_generator.generated_algorithms
        for algo in new_algorithms:
            if algo not in self.accumulated_algorithms:
                self.accumulated_algorithms.append(algo)
        
        # Accumulate experiments
        new_experiments = self.engine.experimenter.completed_experiments
        for exp in new_experiments:
            if exp not in self.accumulated_experiments:
                self.accumulated_experiments.append(exp)
        
        # Accumulate optimizations
        new_optimizations = self.engine.optimizer.completed_optimizations
        for opt in new_optimizations:
            if opt not in self.accumulated_optimizations:
                self.accumulated_optimizations.append(opt)
        
        # Accumulate advisories
        new_advisories = self.engine.strategy_advisor.completed_advisories
        for adv in new_advisories:
            if adv not in self.accumulated_advisories:
                self.accumulated_advisories.append(adv)
        
        # Accumulate knowledge
        new_knowledge = self.engine.knowledge_expander.knowledge_base
        for know in new_knowledge:
            if know not in self.accumulated_knowledge:
                self.accumulated_knowledge.append(know)
        
        # Accumulate evolutions
        new_evolutions = self.engine.continuous_evolver.completed_evolutions
        for evol in new_evolutions:
            if evol not in self.accumulated_evolutions:
                self.accumulated_evolutions.append(evol)
    
    async def _metrics_collector(self):
        """Collect research metrics periodically."""
        while self.running and not self.shutdown_requested:
            try:
                import psutil
                process = psutil.Process()
                
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                # Update peaks
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
                
                if self.current_session:
                    self.current_session.peak_memory_mb = self.peak_memory_mb
                    self.current_session.peak_cpu_percent = self.peak_cpu_percent
                
                # Create metrics entry
                metrics = ResearchMetrics(
                    timestamp=time.time(),
                    algorithms_generated=len(self.accumulated_algorithms),
                    experiments_completed=len(self.accumulated_experiments),
                    optimizations_performed=len(self.accumulated_optimizations),
                    breakthroughs_detected=len(self.engine.breakthrough_detections) if self.engine else 0,
                    knowledge_entries=len(self.accumulated_knowledge),
                    memory_usage_mb=memory_mb,
                    cpu_usage_percent=cpu_percent,
                    active_research_count=len(self.engine.active_research) if self.engine else 0,
                    queued_research_count=len(self.engine.research_queue) if self.engine else 0
                )
                
                self.research_metrics.append(metrics)
                
                # Keep only last 1000 metrics entries
                if len(self.research_metrics) > 1000:
                    self.research_metrics = self.research_metrics[-1000:]
                
                await asyncio.sleep(5.0)  # Collect metrics every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Error collecting metrics: {e}")
                await asyncio.sleep(10.0)
    
    async def _state_persister(self):
        """Persist research state periodically."""
        while self.running and not self.shutdown_requested:
            try:
                await self._save_research_state()
                await asyncio.sleep(30.0)  # Save state every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error persisting state: {e}")
                await asyncio.sleep(60.0)
    
    async def _session_monitor(self):
        """Monitor session duration and health."""
        while self.running and not self.shutdown_requested:
            try:
                if self.current_session:
                    elapsed = time.time() - self.current_session.start_time
                    
                    # Check if session duration exceeded
                    if elapsed >= self.session_duration:
                        logger.info(f"⏰ Session duration reached ({self.session_duration/3600:.1f}h), stopping...")
                        await self.stop_research_session()
                        break
                    
                    # Check engine health
                    if self.engine and not self.engine.running:
                        logger.warning("⚠️ Research engine stopped unexpectedly, restarting...")
                        await self.engine.start()
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                logger.error(f"❌ Error in session monitor: {e}")
                await asyncio.sleep(30.0)
    
    async def _generate_cycle_report(self, cycle_number: int):
        """Generate a research cycle report."""
        if not self.current_session:
            return
        
        report = {
            "cycle_number": cycle_number,
            "session_id": self.current_session.session_id,
            "timestamp": time.time(),
            "duration_seconds": time.time() - self.current_session.start_time,
            "algorithms_generated": len(self.accumulated_algorithms),
            "experiments_completed": len(self.accumulated_experiments),
            "optimizations_performed": len(self.accumulated_optimizations),
            "breakthroughs_detected": len(self.engine.breakthrough_detections) if self.engine else 0,
            "knowledge_entries": len(self.accumulated_knowledge),
            "memory_usage_mb": self.peak_memory_mb,
            "cpu_usage_percent": self.peak_cpu_percent,
            "active_research": len(self.engine.active_research) if self.engine else 0,
            "queued_research": len(self.engine.research_queue) if self.engine else 0
        }
        
        # Save cycle report
        report_file = self.output_dir / f"cycle_report_{cycle_number:04d}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Cycle {cycle_number} Report:")
        logger.info(f"  • Algorithms: {report['algorithms_generated']}")
        logger.info(f"  • Experiments: {report['experiments_completed']}")
        logger.info(f"  • Optimizations: {report['optimizations_performed']}")
        logger.info(f"  • Breakthroughs: {report['breakthroughs_detected']}")
        logger.info(f"  • Knowledge: {report['knowledge_entries']}")
        logger.info(f"  • Memory: {report['memory_usage_mb']:.1f}MB")
        logger.info(f"  • CPU: {report['cpu_usage_percent']:.1f}%")
    
    async def _save_research_state(self):
        """Save current research state to disk."""
        state = {
            "session_id": self.current_session.session_id if self.current_session else None,
            "start_time": self.current_session.start_time if self.current_session else None,
            "algorithms_count": len(self.accumulated_algorithms),
            "experiments_count": len(self.accumulated_experiments),
            "optimizations_count": len(self.accumulated_optimizations),
            "advisories_count": len(self.accumulated_advisories),
            "knowledge_count": len(self.accumulated_knowledge),
            "evolutions_count": len(self.accumulated_evolutions),
            "metrics_count": len(self.research_metrics),
            "peak_memory_mb": self.peak_memory_mb,
            "peak_cpu_percent": self.peak_cpu_percent,
            "timestamp": time.time()
        }
        
        state_file = self.output_dir / "research_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    async def _save_session_data(self):
        """Save session data to disk."""
        if not self.current_session:
            return
        
        session_data = {
            "session": {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time,
                "end_time": self.current_session.end_time,
                "total_research_time": self.current_session.total_research_time,
                "algorithms_generated": self.current_session.algorithms_generated,
                "experiments_completed": self.current_session.experiments_completed,
                "optimizations_performed": self.current_session.optimizations_performed,
                "breakthroughs_detected": self.current_session.breakthroughs_detected,
                "knowledge_entries": self.current_session.knowledge_entries,
                "peak_memory_mb": self.current_session.peak_memory_mb,
                "peak_cpu_percent": self.current_session.peak_cpu_percent,
                "status": self.current_session.status
            },
            "accumulated_results": {
                "algorithms": len(self.accumulated_algorithms),
                "experiments": len(self.accumulated_experiments),
                "optimizations": len(self.accumulated_optimizations),
                "advisories": len(self.accumulated_advisories),
                "knowledge": len(self.accumulated_knowledge),
                "evolutions": len(self.accumulated_evolutions)
            },
            "metrics_summary": {
                "total_metrics": len(self.research_metrics),
                "avg_memory_mb": sum(m.memory_usage_mb for m in self.research_metrics) / max(len(self.research_metrics), 1),
                "avg_cpu_percent": sum(m.cpu_usage_percent for m in self.research_metrics) / max(len(self.research_metrics), 1),
                "peak_memory_mb": max(m.memory_usage_mb for m in self.research_metrics) if self.research_metrics else 0,
                "peak_cpu_percent": max(m.cpu_usage_percent for m in self.research_metrics) if self.research_metrics else 0
            }
        }
        
        session_file = self.output_dir / f"session_{self.current_session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get current research summary."""
        return {
            "running": self.running,
            "current_session": {
                "session_id": self.current_session.session_id if self.current_session else None,
                "duration": time.time() - self.current_session.start_time if self.current_session else 0,
                "algorithms_generated": self.current_session.algorithms_generated if self.current_session else 0,
                "experiments_completed": self.current_session.experiments_completed if self.current_session else 0,
                "optimizations_performed": self.current_session.optimizations_performed if self.current_session else 0,
                "breakthroughs_detected": self.current_session.breakthroughs_detected if self.current_session else 0,
                "knowledge_entries": self.current_session.knowledge_entries if self.current_session else 0
            } if self.current_session else None,
            "accumulated_results": {
                "algorithms": len(self.accumulated_algorithms),
                "experiments": len(self.accumulated_experiments),
                "optimizations": len(self.accumulated_optimizations),
                "advisories": len(self.accumulated_advisories),
                "knowledge": len(self.accumulated_knowledge),
                "evolutions": len(self.accumulated_evolutions)
            },
            "session_history": len(self.session_history),
            "metrics_count": len(self.research_metrics),
            "peak_memory_mb": self.peak_memory_mb,
            "peak_cpu_percent": self.peak_cpu_percent
        }
    
    async def shutdown(self):
        """Gracefully shutdown the research engine."""
        logger.info("🛑 Shutting down persistent research engine...")
        self.shutdown_requested = True
        
        if self.running:
            await self.stop_research_session()
        
        logger.info("✅ Persistent research engine shutdown complete")

# Signal handlers for graceful shutdown
def setup_signal_handlers(engine: PersistentResearchEngine):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(engine.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def run_persistent_research(session_duration_hours: float = 24.0, 
                                output_dir: str = "research_outputs"):
    """Run persistent research for the specified duration."""
    config = ResearchConfig(research_mode=ResearchMode.CONTINUOUS)
    engine = PersistentResearchEngine(config, session_duration_hours, output_dir)
    
    # Setup signal handlers
    setup_signal_handlers(engine)
    
    try:
        # Start research session
        session_id = await engine.start_research_session()
        logger.info(f"🚀 Research session {session_id} started")
        
        # Keep running until shutdown
        while engine.running and not engine.shutdown_requested:
            await asyncio.sleep(1.0)
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Error in persistent research: {e}")
    finally:
        await engine.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run persistent quantum research")
    parser.add_argument("--duration", type=float, default=24.0, 
                       help="Session duration in hours (default: 24.0)")
    parser.add_argument("--output", type=str, default="research_outputs",
                       help="Output directory (default: research_outputs)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run persistent research
    asyncio.run(run_persistent_research(args.duration, args.output))
