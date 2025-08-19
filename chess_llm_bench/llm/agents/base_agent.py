"""
Base agent class for tool-based chess playing.

This module provides the foundation for agent-based chess players that use
tools and reasoning workflows instead of simple prompting.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import chess
import chess.pgn

from .chess_tools import ChessAnalysisTools, MoveAnalysis, MoveCategory


logger = logging.getLogger(__name__)


class ThinkingStrategy(Enum):
    """Different thinking strategies for the agent."""
    FAST = "fast"  # Quick heuristic evaluation
    BALANCED = "balanced"  # Balance between speed and depth
    DEEP = "deep"  # Thorough analysis with multiple considerations
    ADAPTIVE = "adaptive"  # Adapt based on position complexity


@dataclass
class AgentThought:
    """Represents a single thought in the agent's reasoning process."""
    thought_type: str  # "observation", "analysis", "strategy", "decision"
    content: str
    confidence: float = 0.5  # 0.0 to 1.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    """Final decision made by the agent."""
    move: chess.Move
    san: str
    uci: str
    reasoning: List[AgentThought]
    confidence: float
    alternatives_considered: List[MoveAnalysis]
    time_taken: float = 0.0


class ChessAgent(ABC):
    """
    Base class for chess-playing agents that use tools and reasoning.

    This agent follows a think-analyze-decide workflow using various
    chess analysis tools rather than relying on engine evaluation.
    """

    def __init__(
        self,
        name: str = "ChessAgent",
        strategy: ThinkingStrategy = ThinkingStrategy.BALANCED,
        verbose: bool = False,
        max_thinking_time: float = 10.0
    ):
        """
        Initialize the chess agent.

        Args:
            name: Agent's name for identification
            strategy: Thinking strategy to use
            verbose: Whether to output detailed reasoning
            max_thinking_time: Maximum time allowed for thinking (seconds)
        """
        self.name = name
        self.strategy = strategy
        self.verbose = verbose
        self.max_thinking_time = max_thinking_time

        # Agent state
        self.current_board: Optional[chess.Board] = None
        self.game_history: List[chess.Move] = []
        self.thoughts: List[AgentThought] = []
        self.tools: Optional[ChessAnalysisTools] = None

        # Performance tracking
        self.total_moves = 0
        self.total_thinking_time = 0.0
        self.move_history: List[AgentDecision] = []

    async def make_move(self, board: chess.Board) -> AgentDecision:
        """
        Main entry point for making a move.

        This orchestrates the entire thinking process using tools and reasoning.

        Args:
            board: Current chess board state

        Returns:
            AgentDecision containing the chosen move and reasoning
        """
        import time
        start_time = time.time()

        # Update agent state
        self.current_board = board.copy()
        self.tools = ChessAnalysisTools(self.current_board)
        self.thoughts = []

        try:
            # Step 1: Observe the position
            await self._observe_position()

            # Step 2: Analyze using tools
            await self._analyze_position()

            # Step 3: Generate candidate moves
            candidates = await self._generate_candidates()

            # Step 4: Evaluate candidates
            evaluated_moves = await self._evaluate_candidates(candidates)

            # Step 5: Make decision
            decision = await self._make_decision(evaluated_moves)

            # Track performance
            time_taken = time.time() - start_time
            decision.time_taken = time_taken
            self.total_thinking_time += time_taken
            self.total_moves += 1
            self.move_history.append(decision)

            if self.verbose:
                self._log_decision(decision)

            return decision

        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}")
            # Fallback to a random legal move
            return self._emergency_move()

    async def _observe_position(self) -> None:
        """Observe and understand the current position."""
        board_state = self.tools.get_board_state()

        # Basic observations
        observations = []

        if board_state["is_check"]:
            observations.append("We are in check and must escape")

        if board_state["move_number"] <= 10:
            observations.append("We are in the opening phase")
        elif board_state["move_number"] <= 30:
            observations.append("We are in the middlegame")
        else:
            observations.append("We are in the endgame")

        material = board_state["material_balance"]
        if abs(material["material_difference"]) > 3:
            if (material["material_advantage"] == "white" and self.current_board.turn) or \
               (material["material_advantage"] == "black" and not self.current_board.turn):
                observations.append("We have a material advantage")
            else:
                observations.append("We are behind in material")

        for obs in observations:
            self.thoughts.append(AgentThought(
                thought_type="observation",
                content=obs,
                confidence=0.9,
                supporting_data={"board_state": board_state}
            ))

    async def _analyze_position(self) -> None:
        """Analyze the position using available tools."""
        # Get position evaluation
        position_eval = self.tools.evaluate_position()
        material_eval = self.tools.evaluate_material()

        # Analyze key aspects
        analyses = []

        # Center control
        center = position_eval["center_control"]
        if abs(center["score"]) > 2:
            analyses.append(AgentThought(
                thought_type="analysis",
                content=f"Center control: {center['evaluation']}",
                confidence=0.7,
                supporting_data={"center": center}
            ))

        # King safety
        safety = position_eval["king_safety"]
        if abs(safety["score"]) > 3:
            side = "Our" if (safety["score"] > 0 and self.current_board.turn) else "Enemy"
            analyses.append(AgentThought(
                thought_type="analysis",
                content=f"{side} king is more exposed",
                confidence=0.8,
                supporting_data={"safety": safety}
            ))

        # Development (in opening)
        if self.current_board.fullmove_number <= 15:
            dev = position_eval["development"]
            if dev["score"] != 0:
                analyses.append(AgentThought(
                    thought_type="analysis",
                    content=f"Development: {dev['evaluation']}",
                    confidence=0.6,
                    supporting_data={"development": dev}
                ))

        # Check for endgame
        endgame = self.tools.evaluate_endgame()
        if endgame["is_endgame"]:
            analyses.append(AgentThought(
                thought_type="analysis",
                content=f"Endgame phase: Focus on king activity and pawn advancement",
                confidence=0.8,
                supporting_data={"endgame": endgame}
            ))

        self.thoughts.extend(analyses)

    async def _generate_candidates(self) -> List[MoveAnalysis]:
        """Generate candidate moves using tools - evaluate ALL legal moves."""
        # Get ALL legal moves and analyze each one
        candidates = []

        for move in self.current_board.legal_moves:
            move_analysis = self.tools.analyze_move(move)
            candidates.append(move_analysis)

        # Sort by initial score for better ordering
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Add strategy thought
        self.thoughts.append(AgentThought(
            thought_type="strategy",
            content=f"Evaluating all {len(candidates)} legal moves",
            confidence=0.5,
            supporting_data={"total_moves": len(candidates), "sample_moves": [c.san for c in candidates[:5]]}
        ))

        return candidates

    async def _evaluate_candidates(self, candidates: List[MoveAnalysis]) -> List[Tuple[MoveAnalysis, float]]:
        """
        Evaluate candidate moves based on strategy.

        Returns list of (move_analysis, score) tuples.
        """
        evaluated = []

        for candidate in candidates:
            score = await self._score_move(candidate)
            evaluated.append((candidate, score))

        # Sort by score
        evaluated.sort(key=lambda x: x[1], reverse=True)

        # Add evaluation thought
        if evaluated:
            best = evaluated[0]
            self.thoughts.append(AgentThought(
                thought_type="analysis",
                content=f"Evaluated all {len(evaluated)} legal moves. Leading candidate: {best[0].san}",
                confidence=min(0.9, best[1] / 100),  # Normalize confidence
                supporting_data={"total_evaluated": len(evaluated), "top_scores": [(e[0].san, e[1]) for e in evaluated[:5]]}
            ))

        return evaluated

    async def _score_move(self, move_analysis: MoveAnalysis) -> float:
        """
        Score a move based on various factors.

        This is where strategy-specific scoring happens.
        """
        score = move_analysis.score  # Base heuristic score

        # Bonus for specific move categories based on strategy
        if self.strategy == ThinkingStrategy.FAST:
            # Prefer simple, forcing moves
            if MoveCategory.CHECK in move_analysis.categories:
                score += 5
            if MoveCategory.CAPTURE in move_analysis.categories:
                score += 3

        elif self.strategy == ThinkingStrategy.DEEP:
            # Consider long-term factors
            if MoveCategory.CENTER_CONTROL in move_analysis.categories:
                score += 2
            if MoveCategory.PIECE_DEVELOPMENT in move_analysis.categories:
                score += 2
            if MoveCategory.KING_SAFETY in move_analysis.categories:
                score += 3

        elif self.strategy == ThinkingStrategy.BALANCED:
            # Balance tactical and positional
            if MoveCategory.CHECK in move_analysis.categories:
                score += 3
            if MoveCategory.CAPTURE in move_analysis.categories:
                score += 2
            if MoveCategory.CENTER_CONTROL in move_analysis.categories:
                score += 1

        elif self.strategy == ThinkingStrategy.ADAPTIVE:
            # Adapt based on position
            if self.current_board.fullmove_number <= 10:
                # Opening: prioritize development
                if MoveCategory.PIECE_DEVELOPMENT in move_analysis.categories:
                    score += 4
                if MoveCategory.CENTER_CONTROL in move_analysis.categories:
                    score += 3
            elif self.current_board.fullmove_number > 40:
                # Endgame: prioritize pawn advancement
                if MoveCategory.PAWN_ADVANCE in move_analysis.categories:
                    score += 3
            else:
                # Middlegame: tactical play
                if MoveCategory.ATTACKING in move_analysis.categories:
                    score += 2
                if MoveCategory.TACTICAL in move_analysis.categories:
                    score += 3

        # Penalties
        if MoveCategory.DEFENSIVE in move_analysis.categories and \
           not self.current_board.is_check():
            score -= 1  # Slight penalty for passive play when not forced

        return score

    async def _make_decision(self, evaluated_moves: List[Tuple[MoveAnalysis, float]]) -> AgentDecision:
        """Make the final decision on which move to play."""
        if not evaluated_moves:
            return self._emergency_move()

        # Select best move (with some randomness for variety if desired)
        best_move, best_score = evaluated_moves[0]

        # Calculate confidence based on score difference
        confidence = 0.5
        if len(evaluated_moves) > 1:
            score_diff = best_score - evaluated_moves[1][1]
            confidence = min(0.9, 0.5 + score_diff / 20)

        # Add decision thought
        self.thoughts.append(AgentThought(
            thought_type="decision",
            content=f"Playing {best_move.san} with confidence {confidence:.2f}",
            confidence=confidence,
            supporting_data={"score": best_score, "alternatives": len(evaluated_moves)}
        ))

        return AgentDecision(
            move=best_move.move,
            san=best_move.san,
            uci=best_move.move.uci(),
            reasoning=self.thoughts.copy(),
            confidence=confidence,
            alternatives_considered=[m[0] for m in evaluated_moves]
        )

    def _emergency_move(self) -> AgentDecision:
        """Fallback to a random legal move in case of errors."""
        if not self.current_board:
            raise ValueError("No board state available")

        legal_moves = list(self.current_board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        import random
        move = random.choice(legal_moves)

        return AgentDecision(
            move=move,
            san=self.current_board.san(move),
            uci=move.uci(),
            reasoning=[AgentThought(
                thought_type="decision",
                content="Emergency fallback: random move",
                confidence=0.1
            )],
            confidence=0.1,
            alternatives_considered=[]
        )

    def _log_decision(self, decision: AgentDecision) -> None:
        """Log the agent's decision and reasoning."""
        logger.info(f"{self.name} plays: {decision.san} (confidence: {decision.confidence:.2f})")

        if self.verbose:
            print(f"\n{self.name} Reasoning Process:")
            print("=" * 50)

            for thought in decision.reasoning:
                symbol = {
                    "observation": "👁️ ",
                    "analysis": "🔍",
                    "strategy": "🎯",
                    "decision": "✅"
                }.get(thought.thought_type, "💭")

                print(f"{symbol} [{thought.thought_type}] {thought.content}")
                if thought.confidence < 0.5:
                    print(f"   ⚠️  Low confidence: {thought.confidence:.2f}")

            print(f"\nMove chosen: {decision.san}")
            print(f"Time taken: {decision.time_taken:.2f}s")
            print("=" * 50)

    def reset(self) -> None:
        """Reset the agent's state for a new game."""
        self.current_board = None
        self.game_history = []
        self.thoughts = []
        self.tools = None
        self.total_moves = 0
        self.total_thinking_time = 0.0
        self.move_history = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the agent."""
        avg_time = self.total_thinking_time / max(1, self.total_moves)
        avg_confidence = sum(d.confidence for d in self.move_history) / max(1, len(self.move_history))

        return {
            "total_moves": self.total_moves,
            "total_thinking_time": self.total_thinking_time,
            "average_move_time": avg_time,
            "average_confidence": avg_confidence,
            "strategy": self.strategy.value
        }

    @abstractmethod
    async def customize_evaluation(self, move_analysis: MoveAnalysis) -> float:
        """
        Hook for subclasses to customize move evaluation.

        Override this to implement agent-specific evaluation logic.
        """
        pass
