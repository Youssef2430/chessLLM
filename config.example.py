#!/usr/bin/env python3
"""
Example configuration file for Chess LLM Benchmark.

This file demonstrates how to configure the chess LLM benchmark tool
with various settings, bot specifications, and API configurations.

Copy this file to config.py and modify as needed, or use it as a reference
for command-line arguments and environment variable setup.
"""

import os
from chess_llm_bench.core.models import Config

# =============================================================================
# API Keys Configuration
# =============================================================================

# OpenAI API Key (required for OpenAI models)
# Get your key from: https://platform.openai.com/account/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Anthropic API Key (required for Anthropic models)
# Get your key from: https://console.anthropic.com/
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# =============================================================================
# Engine Configuration
# =============================================================================

# Stockfish path - auto-detected if not specified
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", None)

# Alternative paths to try (modify for your system)
STOCKFISH_CANDIDATES = [
    "/opt/homebrew/bin/stockfish",  # macOS Homebrew
    "/usr/local/bin/stockfish",     # macOS/Linux
    "/usr/bin/stockfish",           # Linux
    "C:/Program Files/Stockfish/stockfish.exe",  # Windows
]

# =============================================================================
# Bot Configurations
# =============================================================================

# Demo configuration (no API keys required)
DEMO_BOTS = "random::demo1,random::demo2"

# OpenAI bot configurations
OPENAI_BOTS = [
    "openai:gpt-4o:gpt4o",
    "openai:gpt-4o-mini:gpt4o-mini",
    "openai:gpt-4-turbo:gpt4-turbo",
    "openai:gpt-3.5-turbo:gpt35",
]

# Anthropic bot configurations
ANTHROPIC_BOTS = [
    "anthropic:claude-3-opus-20240229:claude-opus",
    "anthropic:claude-3-sonnet-20240229:claude-sonnet",
    "anthropic:claude-3-haiku-20240307:claude-haiku",
]

# Mixed provider configurations
MIXED_BOTS = [
    "openai:gpt-4o-mini:gpt4o-mini",
    "anthropic:claude-3-haiku-20240307:claude",
    "random::baseline",
]

# Research comparison setup
RESEARCH_BOTS = [
    "openai:gpt-4o:gpt4o-main",
    "openai:gpt-4o-mini:gpt4o-mini",
    "anthropic:claude-3-sonnet-20240229:claude-sonnet",
    "anthropic:claude-3-haiku-20240307:claude-haiku",
    "random::random-baseline",
]

# =============================================================================
# Benchmark Configurations
# =============================================================================

# Quick test configuration
QUICK_CONFIG = Config(
    bots=",".join(DEMO_BOTS),
    start_elo=600,
    elo_step=200,
    max_elo=1200,
    think_time=0.1,
    escalate_on="always"
)

# Standard benchmark configuration
STANDARD_CONFIG = Config(
    bots=",".join(OPENAI_BOTS[:2]),  # First 2 OpenAI bots
    start_elo=600,
    elo_step=100,
    max_elo=2000,
    think_time=0.3,
    escalate_on="always",
    llm_timeout=30.0,
    llm_temperature=0.0
)

# Research configuration (comprehensive)
RESEARCH_CONFIG = Config(
    bots=",".join(RESEARCH_BOTS),
    start_elo=800,
    elo_step=50,
    max_elo=2400,
    think_time=1.0,
    escalate_on="on_win",  # Harder progression
    llm_timeout=60.0,
    llm_temperature=0.0,
    max_plies=200
)

# Speed test configuration
SPEED_CONFIG = Config(
    bots=",".join(DEMO_BOTS),
    start_elo=600,
    elo_step=400,  # Large steps
    max_elo=1400,
    think_time=0.1,  # Fast games
    escalate_on="always",
    llm_timeout=10.0
)

# =============================================================================
# Preset Scenarios
# =============================================================================

# Scenario 1: Model comparison
MODEL_COMPARISON = {
    "name": "GPT Models Comparison",
    "config": Config(
        bots="openai:gpt-4o:gpt4o,openai:gpt-4o-mini:gpt4o-mini,openai:gpt-3.5-turbo:gpt35",
        start_elo=800,
        elo_step=100,
        max_elo=1800,
        think_time=0.5
    )
}

# Scenario 2: Provider comparison
PROVIDER_COMPARISON = {
    "name": "OpenAI vs Anthropic",
    "config": Config(
        bots="openai:gpt-4o-mini:openai,anthropic:claude-3-haiku-20240307:anthropic",
        start_elo=600,
        elo_step=100,
        max_elo=1600,
        think_time=0.3
    )
}

# Scenario 3: Baseline establishment
BASELINE_TEST = {
    "name": "Random Baseline",
    "config": Config(
        bots="random::baseline1,random::baseline2,random::baseline3",
        start_elo=400,
        elo_step=100,
        max_elo=1000,
        think_time=0.1
    )
}

# =============================================================================
# Advanced Settings
# =============================================================================

# Custom prompting strategies (for future extension)
PROMPT_STRATEGIES = {
    "standard": "You are a strong chess player. Choose the best move.",
    "tactical": "You are a tactical chess expert. Look for tactics and combinations.",
    "positional": "You are a positional chess master. Focus on long-term advantages.",
    "aggressive": "You are an aggressive attacking player. Look for sharp, forcing moves.",
}

# Time control presets
TIME_CONTROLS = {
    "blitz": 0.1,       # Very fast
    "rapid": 0.3,       # Standard
    "classical": 1.0,   # Slower, more accurate
    "correspondence": 5.0  # Very slow
}

# ELO progression presets
ELO_PROGRESSIONS = {
    "beginner": {"start": 400, "step": 100, "max": 1200},
    "intermediate": {"start": 800, "step": 100, "max": 1800},
    "advanced": {"start": 1200, "step": 50, "max": 2200},
    "expert": {"start": 1600, "step": 25, "max": 2600},
}

# =============================================================================
# Environment Setup Helpers
# =============================================================================

def setup_openai_environment():
    """Set up OpenAI environment variables."""
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. OpenAI bots will not work.")
        print("Set with: export OPENAI_API_KEY=your-key-here")
    else:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def setup_anthropic_environment():
    """Set up Anthropic environment variables."""
    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set. Anthropic bots will not work.")
        print("Set with: export ANTHROPIC_API_KEY=your-key-here")
    else:
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

def setup_stockfish_path():
    """Set up Stockfish path."""
    if STOCKFISH_PATH:
        os.environ["STOCKFISH_PATH"] = STOCKFISH_PATH
    else:
        # Try to find Stockfish in common locations
        import shutil
        stockfish = shutil.which("stockfish")
        if not stockfish:
            for candidate in STOCKFISH_CANDIDATES:
                if os.path.exists(candidate):
                    os.environ["STOCKFISH_PATH"] = candidate
                    break

def setup_all_environments():
    """Set up all environment variables."""
    setup_openai_environment()
    setup_anthropic_environment()
    setup_stockfish_path()

# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    print("Chess LLM Benchmark - Configuration Examples")
    print("=" * 50)

    # Setup environment
    setup_all_environments()

    print(f"OpenAI API Key set: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"Anthropic API Key set: {'Yes' if ANTHROPIC_API_KEY else 'No'}")
    print(f"Stockfish path: {os.getenv('STOCKFISH_PATH', 'Auto-detect')}")

    print("\nAvailable configurations:")
    print("- QUICK_CONFIG: Fast demo run")
    print("- STANDARD_CONFIG: Standard benchmark")
    print("- RESEARCH_CONFIG: Comprehensive research setup")
    print("- SPEED_CONFIG: Quick performance test")

    print("\nAvailable scenarios:")
    for scenario_name in ["MODEL_COMPARISON", "PROVIDER_COMPARISON", "BASELINE_TEST"]:
        scenario = globals()[scenario_name]
        print(f"- {scenario['name']}")

    print("\nTo use a configuration:")
    print("from config import STANDARD_CONFIG")
    print("# Then use STANDARD_CONFIG in your code")
