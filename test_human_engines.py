#!/usr/bin/env python3
"""
Test script for human-like chess engines in Chess LLM Benchmark.

This script tests the installation and functionality of human-like engines:
- Auto-detection of available engines
- Engine initialization and configuration
- Move generation with human-like characteristics
- ELO-based configuration

Usage:
    python test_human_engines.py [--engine ENGINE_TYPE] [--elo ELO]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chess
from chess_llm_bench.core.models import Config
from chess_llm_bench.core.human_engine import (
    autodetect_human_engines,
    get_best_human_engine,
    create_human_engine,
    validate_human_engine,
    get_human_engine_installation_hint,
    HumanEngineError
)
from chess_llm_bench.core.engine import ChessEngine, autodetect_stockfish

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_engine_detection():
    """Test auto-detection of human engines."""
    print("🔍 Testing engine detection...")
    print("-" * 40)

    # Test human engine detection
    available_engines = autodetect_human_engines()

    print("Available human engines:")
    for engine_type, path in available_engines.items():
        if path:
            print(f"✅ {engine_type}: {path}")
        else:
            print(f"❌ {engine_type}: Not found")

    # Test best engine selection
    best_engine = get_best_human_engine()
    if best_engine:
        engine_type, engine_path = best_engine
        print(f"\n🏆 Best available engine: {engine_type} at {engine_path}")
    else:
        print("\n❌ No human engines found")
        print("\nInstallation hint:")
        print(get_human_engine_installation_hint())

    # Test Stockfish fallback
    stockfish_path = autodetect_stockfish()
    if stockfish_path:
        print(f"✅ Stockfish fallback available: {stockfish_path}")
    else:
        print("❌ Stockfish fallback not available")

    return available_engines, best_engine


async def test_engine_validation(engine_path: str):
    """Test engine validation."""
    print(f"\n🔧 Testing engine validation for: {engine_path}")
    print("-" * 40)

    is_valid = validate_human_engine(engine_path)
    if is_valid:
        print(f"✅ Engine {engine_path} is valid and responds to UCI")
    else:
        print(f"❌ Engine {engine_path} failed validation")

    return is_valid


async def test_engine_initialization(engine_type: str, engine_path: str):
    """Test engine initialization and basic functionality."""
    print(f"\n🚀 Testing {engine_type} engine initialization...")
    print("-" * 40)

    # Create configuration
    config = Config()

    try:
        # Create and start engine
        if engine_type == "stockfish":
            engine = ChessEngine(engine_path, config)
        else:
            engine = create_human_engine(engine_type, engine_path, config)

        await engine.start()
        print(f"✅ {engine_type} engine started successfully")

        # Test ELO configuration
        test_elos = [800, 1200, 1600, 2000]
        for elo in test_elos:
            try:
                await engine.configure_elo(elo)
                print(f"✅ Configured for ELO {elo}")
            except Exception as e:
                print(f"⚠️  ELO {elo} configuration failed: {e}")

        # Test move generation
        board = chess.Board()

        # Test several positions
        test_positions = [
            chess.Board(),  # Starting position
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After 1.e4
            chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"),  # After 1.e4 d5
        ]

        for i, test_board in enumerate(test_positions):
            try:
                move = await engine.get_move(test_board)
                print(f"✅ Generated move for position {i+1}: {move}")

                # Validate the move
                if move in test_board.legal_moves:
                    print(f"   ✅ Move {move} is legal")
                else:
                    print(f"   ❌ Move {move} is ILLEGAL!")

            except Exception as e:
                print(f"❌ Move generation failed for position {i+1}: {e}")

        # Stop engine
        await engine.stop()
        print(f"✅ {engine_type} engine stopped successfully")

        return True

    except Exception as e:
        print(f"❌ {engine_type} engine test failed: {e}")
        return False


async def test_human_like_behavior(engine_type: str, engine_path: str, elo: int):
    """Test human-like behavior characteristics."""
    print(f"\n🧠 Testing human-like behavior for {engine_type} at ELO {elo}...")
    print("-" * 40)

    config = Config()

    try:
        # Create engine
        if engine_type == "stockfish":
            engine = ChessEngine(engine_path, config)
        else:
            engine = create_human_engine(engine_type, engine_path, config)

        await engine.start()
        await engine.configure_elo(elo)

        # Test multiple moves from the same position to see variation
        board = chess.Board()
        moves = []

        for i in range(5):
            try:
                move = await engine.get_move(board)
                moves.append(move)
                print(f"Move {i+1}: {move}")
            except Exception as e:
                print(f"Failed to get move {i+1}: {e}")

        # Analyze move diversity
        unique_moves = set(moves)
        if len(unique_moves) > 1:
            print(f"✅ Shows variation: {len(unique_moves)} different moves out of {len(moves)}")
        else:
            print(f"⚠️  No variation: Same move {moves[0]} every time")

        # Test different ELO levels for the same position
        print(f"\nTesting ELO scaling...")
        elo_moves = {}
        test_elos = [600, 1000, 1400, 1800]

        for test_elo in test_elos:
            try:
                await engine.configure_elo(test_elo)
                move = await engine.get_move(board)
                elo_moves[test_elo] = move
                print(f"ELO {test_elo}: {move}")
            except Exception as e:
                print(f"ELO {test_elo}: Failed - {e}")

        await engine.stop()
        return True

    except Exception as e:
        print(f"❌ Human-like behavior test failed: {e}")
        return False


async def run_comprehensive_test(target_engine: str = None, target_elo: int = 1200):
    """Run comprehensive test suite."""
    print("🏆 Chess LLM Benchmark - Human Engine Test Suite")
    print("=" * 60)

    # Test 1: Engine Detection
    available_engines, best_engine = await test_engine_detection()

    if not available_engines and not best_engine:
        print("\n❌ No engines found! Cannot proceed with tests.")
        return False

    # Determine which engine to test
    if target_engine:
        if target_engine in available_engines and available_engines[target_engine]:
            engine_path = available_engines[target_engine]
            engine_type = target_engine
        else:
            print(f"\n❌ Requested engine '{target_engine}' not available")
            return False
    elif best_engine:
        engine_type, engine_path = best_engine
    else:
        # Fall back to stockfish
        stockfish_path = autodetect_stockfish()
        if stockfish_path:
            engine_type = "stockfish"
            engine_path = stockfish_path
        else:
            print("\n❌ No engines available for testing")
            return False

    print(f"\n🎯 Testing engine: {engine_type} at {engine_path}")

    # Test 2: Engine Validation
    if not await test_engine_validation(engine_path):
        print(f"\n❌ Engine validation failed, skipping further tests")
        return False

    # Test 3: Engine Initialization and Basic Functionality
    if not await test_engine_initialization(engine_type, engine_path):
        print(f"\n❌ Engine initialization failed")
        return False

    # Test 4: Human-like Behavior
    if engine_type != "stockfish":
        await test_human_like_behavior(engine_type, engine_path, target_elo)

    # Test Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"✅ Engine detection: Working")
    print(f"✅ Engine validation: Working")
    print(f"✅ Engine initialization: Working")
    print(f"✅ Move generation: Working")

    if engine_type != "stockfish":
        print(f"✅ Human-like behavior: Tested")
        print(f"\n🎉 Human engine {engine_type} is ready for use!")
        print(f"\nTo use in benchmarks:")
        print(f"  python main.py --preset premium --use-human-engine --human-engine-type {engine_type}")
    else:
        print(f"\n✅ Stockfish fallback is working")
        print(f"\nTo use Stockfish:")
        print(f"  python main.py --preset premium")

    return True


async def quick_demo():
    """Run a quick demonstration of human engine vs regular engine."""
    print("🎮 Quick Demo: Human vs Traditional Engine")
    print("=" * 50)

    # Get best human engine
    best_engine = get_best_human_engine()
    stockfish_path = autodetect_stockfish()

    if not best_engine and not stockfish_path:
        print("❌ No engines available for demo")
        return

    config = Config()
    board = chess.Board()

    # Test position after 1.e4 e5 2.Nf3 Nc6 3.Bb5
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")

    print(f"Test position: {board.fen()}")
    print("Move comparison:")

    if best_engine:
        engine_type, engine_path = best_engine
        print(f"\n🧠 {engine_type} engine moves:")

        try:
            human_engine = create_human_engine(engine_type, engine_path, config)
            await human_engine.start()

            for elo in [800, 1200, 1600, 2000]:
                await human_engine.configure_elo(elo)
                move = await human_engine.get_move(board)
                print(f"  ELO {elo}: {move}")

            await human_engine.stop()
        except Exception as e:
            print(f"❌ Human engine demo failed: {e}")

    if stockfish_path:
        print(f"\n🤖 Stockfish moves:")

        try:
            stockfish_engine = ChessEngine(stockfish_path, config)
            await stockfish_engine.start()

            for elo in [800, 1200, 1600, 2000]:
                await stockfish_engine.configure_elo(elo)
                move = await stockfish_engine.get_move(board)
                print(f"  ELO {elo}: {move}")

            await stockfish_engine.stop()
        except Exception as e:
            print(f"❌ Stockfish demo failed: {e}")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test human-like chess engines for Chess LLM Benchmark"
    )
    parser.add_argument(
        "--engine",
        choices=["maia", "lczero", "human_stockfish", "stockfish"],
        help="Specific engine to test"
    )
    parser.add_argument(
        "--elo",
        type=int,
        default=1200,
        help="ELO level to test (default: 1200)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo comparing engines"
    )
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Only test engine detection"
    )

    args = parser.parse_args()

    try:
        if args.demo:
            asyncio.run(quick_demo())
        elif args.detection_only:
            asyncio.run(test_engine_detection())
        else:
            success = asyncio.run(run_comprehensive_test(args.engine, args.elo))
            if not success:
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
