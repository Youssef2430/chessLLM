"""
AG2-based Chess Agents Module

This module provides agent-based chess playing capabilities using AG2 (AutoGen)
as the communication backbone. It includes various agent types for different
playing styles and game management.

Key Components:
- ChessAgent: Base class for all chess-playing agents
- GameAgent: Manages game state and coordinates between players
- RandomPlayerAgent: Plays random legal moves
- LLMChessAgent: Uses LLM providers through AG2 for move generation
- AutoReplyAgent: Automated response agent for specific scenarios

The agents maintain compatibility with the existing tracking, statistics,
and ELO ladder functionality while providing a more modular and extensible
architecture through AG2's multi-agent framework.
"""

from .base import ChessAgent, ChessAgentConfig
from .game_agent import GameAgent
from .random_agent import RandomPlayerAgent
from .llm_agent import LLMChessAgent
from .auto_reply_agent import AutoReplyAgent, AutoReplyRule
from .factory import (
    AgentFactory,
    AgentCreationError,
    create_agent,
    create_agents,
    create_random_agent,
    create_llm_agent,
    create_openai_agent,
    create_anthropic_agent,
    create_gemini_agent,
    get_supported_providers,
    is_provider_supported,
    validate_bot_spec,
    register_custom_agent,
    get_factory_stats,
)

__all__ = [
    # Base classes
    "ChessAgent",
    "ChessAgentConfig",

    # Specific agent types
    "GameAgent",
    "RandomPlayerAgent",
    "LLMChessAgent",
    "AutoReplyAgent",
    "AutoReplyRule",

    # Factory classes and functions
    "AgentFactory",
    "AgentCreationError",
    "create_agent",
    "create_agents",
    "create_random_agent",
    "create_llm_agent",
    "create_openai_agent",
    "create_anthropic_agent",
    "create_gemini_agent",
    "get_supported_providers",
    "is_provider_supported",
    "validate_bot_spec",
    "register_custom_agent",
    "get_factory_stats",
]
