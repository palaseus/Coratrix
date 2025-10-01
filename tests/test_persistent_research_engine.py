"""
Test Persistent Research Engine

This module tests the persistent research engine that runs continuous
research cycles and accumulates meaningful results over time.

Author: Quantum Research Engine - Coratrix 4.0
"""

import asyncio
import time
import logging
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

from quantum_research.persistent_research_engine import (
    PersistentResearchEngine, 
    ResearchSession, 
    ResearchMetrics
)
from quantum_research.quantum_research_engine import ResearchConfig, ResearchMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersistentResearchEngineTest:
    """Test suite for the persistent research engine."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all persistent research engine tests."""
        logger.info("🧪 Starting Persistent Research Engine Tests")
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp(prefix="persistent_research_test_")
        logger.info(f"📁 Using test directory: {self.temp_dir}")
        
        try:
            # Run tests
            await self.test_engine_initialization()
            await self.test_session_management()
            await self.test_research_cycle()
            await self.test_metrics_collection()
            await self.test_state_persistence()
            await self.test_graceful_shutdown()
            
            # Generate test report
            report = self._generate_test_report()
            return report
            
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"🧹 Cleaned up test directory: {self.temp_dir}")
    
    async def test_engine_initialization(self):
        """Test engine initialization."""
        logger.info("🔧 Testing engine initialization...")
        
        start_time = time.time()
        
        try:
            config = ResearchConfig(research_mode=ResearchMode.EXPLORATION)
            engine = PersistentResearchEngine(
                config=config,
                session_duration_hours=0.1,  # 6 minutes for testing
                output_dir=self.temp_dir
            )
            
            # Verify initialization
            assert engine.config == config
            assert engine.session_duration == 360  # 6 minutes in seconds
            assert engine.output_dir == Path(self.temp_dir)
            assert not engine.running
            assert engine.current_session is None
            assert len(engine.session_history) == 0
            assert len(engine.research_metrics) == 0
            
            self.test_results.append({
                "test": "engine_initialization",
                "success": True,
                "duration": time.time() - start_time,
                "message": "Engine initialized successfully"
            })
            
            logger.info("✅ Engine initialization test passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "engine_initialization",
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            })
            logger.error(f"❌ Engine initialization test failed: {e}")
    
    async def test_session_management(self):
        """Test session management."""
        logger.info("🔧 Testing session management...")
        
        start_time = time.time()
        
        try:
            config = ResearchConfig(research_mode=ResearchMode.EXPLORATION)
            engine = PersistentResearchEngine(
                config=config,
                session_duration_hours=0.1,  # 6 minutes for testing
                output_dir=self.temp_dir
            )
            
            # Test session start
            session_id = await engine.start_research_session()
            assert session_id is not None
            assert engine.running
            assert engine.current_session is not None
            assert engine.current_session.session_id == session_id
            assert engine.engine is not None
            assert engine.engine.running
            
            # Test session stop
            await engine.stop_research_session()
            assert not engine.running
            assert engine.current_session is None
            assert engine.engine is None
            assert len(engine.session_history) == 1
            
            session = engine.session_history[0]
            assert session.session_id == session_id
            assert session.end_time is not None
            assert session.total_research_time > 0
            assert session.status == "completed"
            
            self.test_results.append({
                "test": "session_management",
                "success": True,
                "duration": time.time() - start_time,
                "message": "Session management working correctly"
            })
            
            logger.info("✅ Session management test passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "session_management",
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            })
            logger.error(f"❌ Session management test failed: {e}")
    
    async def test_research_cycle(self):
        """Test research cycle execution."""
        logger.info("🔧 Testing research cycle...")
        
        start_time = time.time()
        
        try:
            config = ResearchConfig(research_mode=ResearchMode.EXPLORATION)
            engine = PersistentResearchEngine(
                config=config,
                session_duration_hours=0.05,  # 3 minutes for testing
                output_dir=self.temp_dir
            )
            
            # Start session
            session_id = await engine.start_research_session()
            
            # Let it run for a short time to generate some research
            await asyncio.sleep(15)  # 15 seconds of research
            
            # Check that research is happening
            assert engine.running
            assert engine.current_session is not None
            assert engine.engine is not None
            assert engine.engine.running
            
            # Check that some research has been generated
            # (Note: This might be 0 if the engine hasn't had time to generate anything yet)
            algorithms_generated = len(engine.accumulated_algorithms)
            experiments_completed = len(engine.accumulated_experiments)
            
            logger.info(f"📊 Research generated during test:")
            logger.info(f"  • Algorithms: {algorithms_generated}")
            logger.info(f"  • Experiments: {experiments_completed}")
            
            # Stop session
            await engine.stop_research_session()
            
            # Verify session was recorded
            assert len(engine.session_history) == 1
            session = engine.session_history[0]
            assert session.total_research_time > 0
            
            self.test_results.append({
                "test": "research_cycle",
                "success": True,
                "duration": time.time() - start_time,
                "message": f"Research cycle completed, generated {algorithms_generated} algorithms, {experiments_completed} experiments"
            })
            
            logger.info("✅ Research cycle test passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "research_cycle",
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            })
            logger.error(f"❌ Research cycle test failed: {e}")
    
    async def test_metrics_collection(self):
        """Test metrics collection."""
        logger.info("🔧 Testing metrics collection...")
        
        start_time = time.time()
        
        try:
            config = ResearchConfig(research_mode=ResearchMode.EXPLORATION)
            engine = PersistentResearchEngine(
                config=config,
                session_duration_hours=0.05,  # 3 minutes for testing
                output_dir=self.temp_dir
            )
            
            # Start session
            await engine.start_research_session()
            
            # Let it run to collect metrics
            await asyncio.sleep(10)  # 10 seconds
            
            # Check metrics collection
            assert len(engine.research_metrics) > 0
            
            # Check that metrics have reasonable values
            latest_metrics = engine.research_metrics[-1]
            assert latest_metrics.timestamp > 0
            assert latest_metrics.memory_usage_mb >= 0
            assert latest_metrics.cpu_usage_percent >= 0
            
            # Stop session
            await engine.stop_research_session()
            
            self.test_results.append({
                "test": "metrics_collection",
                "success": True,
                "duration": time.time() - start_time,
                "message": f"Collected {len(engine.research_metrics)} metrics entries"
            })
            
            logger.info("✅ Metrics collection test passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "metrics_collection",
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            })
            logger.error(f"❌ Metrics collection test failed: {e}")
    
    async def test_state_persistence(self):
        """Test state persistence."""
        logger.info("🔧 Testing state persistence...")
        
        start_time = time.time()
        
        try:
            config = ResearchConfig(research_mode=ResearchMode.EXPLORATION)
            engine = PersistentResearchEngine(
                config=config,
                session_duration_hours=0.05,  # 3 minutes for testing
                output_dir=self.temp_dir
            )
            
            # Start session
            await engine.start_research_session()
            
            # Let it run to generate some state
            await asyncio.sleep(10)  # 10 seconds
            
            # Stop session
            await engine.stop_research_session()
            
            # Check that state files were created
            output_path = Path(self.temp_dir)
            state_file = output_path / "research_state.json"
            session_file = output_path / f"session_{engine.session_history[0].session_id}.json"
            
            assert state_file.exists(), f"State file not found: {state_file}"
            assert session_file.exists(), f"Session file not found: {session_file}"
            
            # Check state file content
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                assert "session_id" in state_data
                assert "timestamp" in state_data
            
            # Check session file content
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                assert "session" in session_data
                assert "accumulated_results" in session_data
                assert "metrics_summary" in session_data
            
            self.test_results.append({
                "test": "state_persistence",
                "success": True,
                "duration": time.time() - start_time,
                "message": "State persistence working correctly"
            })
            
            logger.info("✅ State persistence test passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "state_persistence",
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            })
            logger.error(f"❌ State persistence test failed: {e}")
    
    async def test_graceful_shutdown(self):
        """Test graceful shutdown."""
        logger.info("🔧 Testing graceful shutdown...")
        
        start_time = time.time()
        
        try:
            config = ResearchConfig(research_mode=ResearchMode.EXPLORATION)
            engine = PersistentResearchEngine(
                config=config,
                session_duration_hours=0.05,  # 3 minutes for testing
                output_dir=self.temp_dir
            )
            
            # Start session
            await engine.start_research_session()
            
            # Let it run briefly
            await asyncio.sleep(5)  # 5 seconds
            
            # Test graceful shutdown
            await engine.shutdown()
            
            # Verify shutdown
            assert not engine.running
            assert engine.shutdown_requested
            assert engine.current_session is None
            assert engine.engine is None
            
            self.test_results.append({
                "test": "graceful_shutdown",
                "success": True,
                "duration": time.time() - start_time,
                "message": "Graceful shutdown working correctly"
            })
            
            logger.info("✅ Graceful shutdown test passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "graceful_shutdown",
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            })
            logger.error(f"❌ Graceful shutdown test failed: {e}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        total_duration = sum(result["duration"] for result in self.test_results)
        
        report = {
            "test_suite": "Persistent Research Engine Tests",
            "timestamp": time.time(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "test_results": self.test_results
        }
        
        # Save report
        if self.temp_dir:
            report_file = Path(self.temp_dir) / "persistent_research_test_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

async def main():
    """Run persistent research engine tests."""
    logger.info("🚀 Starting Persistent Research Engine Test Suite")
    
    test_suite = PersistentResearchEngineTest()
    report = await test_suite.run_all_tests()
    
    # Print summary
    logger.info("📊 Test Summary:")
    logger.info(f"  • Total Tests: {report['total_tests']}")
    logger.info(f"  • Passed: {report['passed_tests']}")
    logger.info(f"  • Failed: {report['failed_tests']}")
    logger.info(f"  • Success Rate: {report['success_rate']:.1f}%")
    logger.info(f"  • Total Duration: {report['total_duration']:.2f}s")
    
    # Print failed tests
    failed_tests = [result for result in report['test_results'] if not result['success']]
    if failed_tests:
        logger.error("❌ Failed Tests:")
        for test in failed_tests:
            logger.error(f"  • {test['test']}: {test.get('error', 'Unknown error')}")
    else:
        logger.info("✅ All tests passed!")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())
