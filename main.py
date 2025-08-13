#!/usr/bin/env python3
"""
Chess LLM Benchmark - Main Entry Point

A tool for testing Large Language Models with chess games and assessing their ELOs
by running them against Stockfish at various rating levels.

This is the main entry point for the chess LLM benchmark tool. It imports and
runs the CLI interface from the chess_llm_bench package.

Quick Examples:
    # Run demo with random bots
    python main.py --demo

    # Test OpenAI models (requires OPENAI_API_KEY)
    export OPENAI_API_KEY=your-api-key
    python main.py --bots "openai:gpt-4o-mini:gpt4o,random::baseline"

    # Custom ELO ladder
    python main.py --bots "random::test" --start-elo 800 --max-elo 1600 --elo-step 200

Requirements:
    - Python 3.8+
    - Stockfish chess engine installed and in PATH
    - Required Python packages (see requirements.txt)

Installation:
    pip install -r requirements.txt
    # Or install Stockfish:
    # macOS:    brew install stockfish
    # Ubuntu:   sudo apt-get install stockfish
    # Windows:  choco install stockfish
"""

import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional
    pass

# Add the project root to the Python path so we can import our package
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # Import the main CLI function from our package
    from chess_llm_bench.cli import main
except ImportError as e:
    print(f"Error importing chess_llm_bench package: {e}", file=sys.stderr)
    print("\nMake sure you have installed the required dependencies:", file=sys.stderr)
    print("  pip install -r requirements.txt", file=sys.stderr)
    print("\nOr install the package in development mode:", file=sys.stderr)
    print("  pip install -e .", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    # Run the main CLI function
    sys.exit(main())
