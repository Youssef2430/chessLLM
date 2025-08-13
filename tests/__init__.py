"""
Test package for Chess LLM Benchmark.

This package contains unit tests for all components of the chess LLM benchmark
tool, including bot parsing, engine management, game logic, and UI components.
"""

# Import test modules for easier discovery
from . import test_bots
from . import test_engine

__all__ = [
    "test_bots",
    "test_engine",
]
