"""
LLM package for Chess LLM Benchmark.

This package contains components for interfacing with Large Language Models
to generate chess moves, including support for multiple providers like
OpenAI, Anthropic, and random baseline.
"""

from .client import (
    LLMClient,
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    RandomProvider,
    LLMProviderError,
    parse_bot_spec
)

__all__ = [
    # Main client
    "LLMClient",

    # Provider base and implementations
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "RandomProvider",

    # Exceptions
    "LLMProviderError",

    # Utilities
    "parse_bot_spec",
]
