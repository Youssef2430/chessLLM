"""
LLM-based agent provider that uses tools and reasoning workflows.

This module provides agent-based LLM providers that can be used as drop-in
replacements for the traditional prompting-based providers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn

from ..client import BaseLLMProvider
from ...core.models import BotSpec
from ...core.budget import record_llm_usage
from .base_agent import ChessAgent, AgentDecision, AgentThought, ThinkingStrategy
from .chess_tools import ChessAnalysisTools, MoveAnalysis, MoveCategory


logger = logging.getLogger(__name__)


class LLMAgentProvider(BaseLLMProvider):
    """
    LLM provider that uses agent-based reasoning with tools.

    This provider wraps an LLM agent to work with the existing system,
    providing a plug-and-play replacement for traditional prompting.
    """

    def __init__(
        self,
        spec: BotSpec,
        strategy: ThinkingStrategy = ThinkingStrategy.BALANCED,
        verbose: bool = False,
        use_tools: bool = True,
        max_retries: int = 3,
        temperature: float = 0.0
    ):
        """
        Initialize the LLM agent provider.

        Args:
            spec: Bot specification with provider, model, and name
            strategy: Thinking strategy for the agent
            verbose: Whether to output detailed reasoning
            use_tools: Whether to use chess analysis tools
            max_retries: Maximum retries for LLM calls
            temperature: LLM temperature for generation
        """
        super().__init__(spec)

        self.provider = spec.provider.lower()
        self.model = spec.model
        self.strategy = strategy
        self.verbose = verbose
        self.use_tools = use_tools
        self.max_retries = max_retries
        self.temperature = temperature

        # Create the underlying LLM agent
        self.agent = LLMChessAgent(
            spec=spec,
            strategy=strategy,
            verbose=verbose,
            use_tools=use_tools,
            temperature=temperature
        )

    async def generate_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        move_history: list = []
    ) -> str:
        """
        Generate a move using the agent-based approach.

        Args:
            board: Current chess board
            temperature: Temperature for generation (unused for agents)
            timeout_s: Timeout in seconds
            move_history: List of moves in the game

        Returns:
            Move in UCI format
        """
        start_time = time.time()

        try:
            # Use the agent to make a decision
            decision = await self.agent.make_move(board)

            # Extract UCI move
            uci_move = decision.uci

            # Validate the move
            try:
                move = chess.Move.from_uci(uci_move)
                if move not in board.legal_moves:
                    logger.warning(f"Agent suggested illegal move: {uci_move}")
                    # Try to find the intended move
                    for legal_move in board.legal_moves:
                        if board.san(legal_move) == decision.san:
                            uci_move = legal_move.uci()
                            break
                    else:
                        # Fallback to random
                        return self._fallback_random_move(board)
            except:
                logger.error(f"Invalid move format from agent: {uci_move}")
                return self._fallback_random_move(board)

            if self.verbose:
                logger.info(f"Agent move: {uci_move} (confidence: {decision.confidence:.2f})")

            # Record overall move generation (individual LLM calls are tracked separately)
            # This helps with move counting and statistics
            if self.provider != "random":
                # Create a summary of the agent's reasoning for tracking
                reasoning_summary = f"Agent decision: {decision.san} (confidence: {decision.confidence:.2f})"
                if decision.reasoning:
                    key_thoughts = [t.content for t in decision.reasoning if t.thought_type == "decision"]
                    if key_thoughts:
                        reasoning_summary += f"\nReasoning: {key_thoughts[0][:100]}"

                record_llm_usage(
                    provider=self.provider,
                    model=self.model,
                    bot_name=self.spec.name,
                    prompt=f"Agent move generation for position: {board.fen()[:50]}...",
                    response=reasoning_summary,
                    success=True
                )

            return uci_move

        except Exception as e:
            logger.error(f"Agent error: {e}")

            # Record failed move generation
            if self.provider != "random":
                record_llm_usage(
                    provider=self.provider,
                    model=self.model,
                    bot_name=self.spec.name,
                    prompt=f"Agent move generation for position: {board.fen()[:50]}...",
                    response="",
                    success=False,
                    error_message=str(e)
                )

            return self._fallback_random_move(board)


class LLMChessAgent(ChessAgent):
    """
    Chess agent that uses LLMs for decision making with tool assistance.
    """

    def __init__(
        self,
        spec: BotSpec,
        strategy: ThinkingStrategy = ThinkingStrategy.BALANCED,
        verbose: bool = False,
        use_tools: bool = True,
        temperature: float = 0.0
    ):
        """Initialize the LLM chess agent."""
        super().__init__(
            name=spec.name,
            strategy=strategy,
            verbose=verbose
        )

        self.spec = spec
        self.provider = spec.provider
        self.model = spec.model
        self.temperature = temperature
        self.use_tools = use_tools

        # Initialize the appropriate LLM client
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "openai":
            from ..client import OpenAIProvider
            self.llm_client = OpenAIProvider(self.spec)
        elif self.provider == "anthropic":
            from ..client import AnthropicProvider
            self.llm_client = AnthropicProvider(self.spec)
        elif self.provider == "gemini":
            from ..client import GeminiProvider
            self.llm_client = GeminiProvider(self.spec)
        elif self.provider == "random":
            from ..client import RandomProvider
            self.llm_client = RandomProvider(self.spec)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def customize_evaluation(self, move_analysis: MoveAnalysis) -> float:
        """
        Use LLM to provide additional evaluation of moves.

        This is called during the evaluation phase to get LLM insights.
        """
        if not self.use_tools:
            return 0.0  # No additional evaluation

        # Create a prompt for the LLM to evaluate the move
        prompt = self._create_evaluation_prompt(move_analysis)

        try:
            # Get LLM evaluation
            response = await self._call_llm(prompt, track_costs=True)

            # Parse the evaluation score
            score = self._parse_evaluation_score(response)

            return score

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return 0.0

    async def _analyze_position(self) -> None:
        """
        Analyze position using both tools and LLM insights.
        """
        # First use the standard tool-based analysis
        await super()._analyze_position()

        if not self.use_tools:
            return

        # Then get LLM strategic insights
        try:
            board_state = self.tools.get_board_state()
            prompt = self._create_strategic_prompt(board_state)

            response = await self._call_llm(prompt, track_costs=True)

            # Parse strategic insights
            insights = self._parse_strategic_insights(response)

            for insight in insights:
                self.thoughts.append(AgentThought(
                    thought_type="analysis",
                    content=insight,
                    confidence=0.7,
                    supporting_data={"source": "llm_analysis"}
                ))

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")

    async def _make_decision(self, evaluated_moves: List[Tuple[MoveAnalysis, float]]) -> AgentDecision:
        """
        Make final decision with optional LLM confirmation.
        """
        if not evaluated_moves:
            return self._emergency_move()

        # Get base decision from parent class
        base_decision = await super()._make_decision(evaluated_moves)

        if not self.use_tools or len(evaluated_moves) < 2:
            return base_decision

        # Use LLM to confirm or adjust the decision
        try:
            prompt = self._create_decision_prompt(evaluated_moves)
            response = await self._call_llm(prompt, track_costs=True)

            # Parse LLM's choice
            llm_choice = self._parse_move_choice(response, evaluated_moves)

            if llm_choice and llm_choice != base_decision.move:
                # LLM suggests a different move
                for move_analysis, score in evaluated_moves:
                    if move_analysis.move == llm_choice:
                        # Create new decision with LLM's choice
                        self.thoughts.append(AgentThought(
                            thought_type="decision",
                            content=f"LLM override: Choosing {move_analysis.san} instead",
                            confidence=0.8,
                            supporting_data={"llm_reasoning": response}
                        ))

                        return AgentDecision(
                            move=move_analysis.move,
                            san=move_analysis.san,
                            uci=move_analysis.move.uci(),
                            reasoning=self.thoughts.copy(),
                            confidence=0.8,
                            alternatives_considered=[m[0] for m in evaluated_moves],
                            time_taken=base_decision.time_taken
                        )

        except Exception as e:
            logger.warning(f"LLM decision confirmation failed: {e}")

        return base_decision

    async def _call_llm(self, prompt: str, track_costs: bool = False) -> str:
        """
        Call the LLM with a prompt and return the response.
        """
        # Create a minimal board state for the LLM call
        board_state = str(self.current_board)
        move_history = []  # We don't need full history for agent decisions

        # Use the existing provider's generate_move method with our custom prompt
        # We'll override the prompt creation in the provider
        original_create_prompt = self.llm_client._create_chess_prompt

        try:
            # Temporarily replace the prompt creation
            self.llm_client._create_chess_prompt = lambda *args: prompt

            # Call the LLM
            response = await self.llm_client.generate_move(
                self.current_board,
                temperature=self.temperature,
                timeout_s=20.0,
                move_history=move_history
            )

            # Track costs if requested
            if track_costs:
                record_llm_usage(
                    provider=self.provider,
                    model=self.model,
                    bot_name=self.name,
                    prompt=prompt,
                    response=response,
                    success=True
                )

            return response

        except Exception as e:
            # Track failed usage
            if track_costs:
                record_llm_usage(
                    provider=self.provider,
                    model=self.model,
                    bot_name=self.name,
                    prompt=prompt,
                    response="",
                    success=False,
                    error_message=str(e)
                )
            raise

        finally:
            # Restore original prompt creation
            self.llm_client._create_chess_prompt = original_create_prompt

    def _create_evaluation_prompt(self, move_analysis: MoveAnalysis) -> str:
        """Create a prompt for LLM to evaluate a specific move."""
        return f"""
Evaluate this chess move:
Move: {move_analysis.san}
Categories: {', '.join(cat.value for cat in move_analysis.categories)}
Explanation: {move_analysis.explanation}

Current position:
{self.current_board}

Rate this move on a scale of -10 to +10, where:
- Negative scores indicate bad moves
- 0 is neutral
- Positive scores indicate good moves

Consider:
1. Tactical soundness
2. Strategic value
3. Position improvement
4. Risk vs reward

Respond with just a number between -10 and 10.
"""

    def _create_strategic_prompt(self, board_state: Dict[str, Any]) -> str:
        """Create a prompt for strategic analysis."""
        return f"""
Analyze this chess position strategically:

Position: {self.current_board.fen()}
Move number: {board_state['move_number']}
Material balance: {board_state['material_balance']['material_difference']}
Legal moves available: {board_state['legal_moves_count']}

Provide 2-3 key strategic insights for the {board_state['turn']} player.
Focus on:
- Immediate threats or opportunities
- Piece coordination
- Pawn structure
- King safety
- Strategic plans

Keep each insight to one sentence.
Separate insights with newlines.
"""

    def _create_decision_prompt(self, evaluated_moves: List[Tuple[MoveAnalysis, float]]) -> str:
        """Create a prompt for final move decision."""
        # Show ALL legal moves to the LLM, not just top candidates
        all_moves = []
        for analysis, score in evaluated_moves:
            move_desc = f"• {analysis.san}"
            if analysis.categories:
                categories = [cat.value for cat in analysis.categories[:3]]
                if categories:
                    move_desc += f" ({', '.join(categories)})"
            all_moves.append(move_desc)

        return f"""
Choose the best move from ALL available legal moves:

Current position:
{self.current_board}

All legal moves ({len(evaluated_moves)} total):
{chr(10).join(all_moves)}

Consider:
- Tactical opportunities (captures, checks, threats)
- Strategic goals (center control, development, king safety)
- Position improvement
- Risk vs reward

Respond with just the move in SAN notation (e.g., "Nf3" or "e4").
"""

    def _parse_evaluation_score(self, response: str) -> float:
        """Parse evaluation score from LLM response."""
        try:
            # Extract number from response
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range
                return max(-10, min(10, score))
        except:
            pass

        return 0.0

    def _parse_strategic_insights(self, response: str) -> List[str]:
        """Parse strategic insights from LLM response."""
        insights = []

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                # Clean up common prefixes
                line = line.lstrip('- •·123456789.')
                if line:
                    insights.append(line)

        return insights[:3]  # Limit to 3 insights

    def _parse_move_choice(self, response: str, evaluated_moves: List[Tuple[MoveAnalysis, float]]) -> Optional[chess.Move]:
        """Parse the LLM's move choice from response."""
        try:
            # Try to extract a move in SAN notation
            import re

            # Common chess move patterns
            move_pattern = r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?|O-O-O|O-O)\b'
            matches = re.findall(move_pattern, response)

            if matches:
                san_move = matches[0]

                # Try to match with our candidates
                for move_analysis, _ in evaluated_moves:
                    if move_analysis.san == san_move:
                        return move_analysis.move

                # Try to parse the move directly
                try:
                    move = self.current_board.parse_san(san_move)
                    if move in self.current_board.legal_moves:
                        return move
                except:
                    pass

        except Exception as e:
            logger.warning(f"Failed to parse move choice: {e}")

        return None


# Factory function to create agent providers
def create_agent_provider(
    spec: BotSpec,
    strategy: str = "balanced",
    verbose: bool = False,
    use_tools: bool = True,
    temperature: float = 0.0
) -> LLMAgentProvider:
    """
    Factory function to create an agent-based LLM provider.

    Args:
        spec: Bot specification with provider, model, and name
        strategy: Thinking strategy ("fast", "balanced", "deep", "adaptive")
        verbose: Whether to output reasoning
        use_tools: Whether to use analysis tools
        temperature: Generation temperature

    Returns:
        Configured LLMAgentProvider instance
    """
    strategy_map = {
        "fast": ThinkingStrategy.FAST,
        "balanced": ThinkingStrategy.BALANCED,
        "deep": ThinkingStrategy.DEEP,
        "adaptive": ThinkingStrategy.ADAPTIVE
    }

    thinking_strategy = strategy_map.get(strategy.lower(), ThinkingStrategy.BALANCED)

    return LLMAgentProvider(
        spec=spec,
        strategy=thinking_strategy,
        verbose=verbose,
        use_tools=use_tools,
        temperature=temperature
    )
