"""
Agent-based chess playing module.

This module provides tool-based agents that can analyze chess positions
and make decisions using reasoning workflows instead of simple prompting.
"""

from .base_agent import (
    ChessAgent,
    ThinkingStrategy,
    AgentThought,
    AgentDecision
)

from .chess_tools import (
    ChessAnalysisTools,
    MoveAnalysis,
    MoveCategory
)

from .llm_agent_provider import (
    LLMAgentProvider,
    LLMChessAgent,
    create_agent_provider
)

__all__ = [
    # Base agent classes
    'ChessAgent',
    'ThinkingStrategy',
    'AgentThought',
    'AgentDecision',

    # Chess analysis tools
    'ChessAnalysisTools',
    'MoveAnalysis',
    'MoveCategory',

    # LLM agent providers
    'LLMAgentProvider',
    'LLMChessAgent',
    'create_agent_provider'
]
