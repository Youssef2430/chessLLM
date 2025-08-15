"""
LLM Chess Agent

This module provides LLM-powered chess agents using AG2 (AutoGen) framework
for communication. It integrates with existing LLM providers (OpenAI, Anthropic,
Gemini) while using AG2 as the backbone for agent communication and coordination.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
from autogen import ConversableAgent, LLMConfig

from .base import ChessAgent, ChessAgentConfig
from ..core.models import BotSpec
from ..llm.client import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    LLMProviderError
)

logger = logging.getLogger(__name__)


class LLMChessAgent(ChessAgent):
    """
    LLM-powered chess agent using AG2 framework.

    This agent combines the power of Large Language Models with AG2's
    multi-agent communication capabilities to play chess. It supports
    multiple LLM providers and maintains compatibility with the existing
    ELO ladder and statistics systems.
    """

    def __init__(
        self,
        bot_spec: BotSpec,
        name: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: float = 30.0,
        **kwargs
    ):
        """
        Initialize LLM chess agent with AG2 integration.

        Args:
            bot_spec: Bot specification with provider and model info
            name: Agent name (defaults to bot_spec.name)
            temperature: LLM temperature for move generation
            timeout_seconds: Timeout for move generation
            **kwargs: Additional AG2 ConversableAgent arguments
        """
        # Set up agent configuration
        agent_name = name or bot_spec.name or f"{bot_spec.provider}_{bot_spec.model}"

        config = ChessAgentConfig(
            name=agent_name,
            bot_spec=bot_spec,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retry_attempts=3,
            enable_fallback=True,
        )

        # Create AG2 LLM configuration
        llm_config = self._create_ag2_llm_config(bot_spec, temperature, timeout_seconds)

        # Create chess-specific system message
        system_message = self._create_chess_system_message(bot_spec, agent_name, temperature)

        # Initialize parent ChessAgent
        super().__init__(
            config=config,
            llm_config=llm_config,
            system_message=system_message,
            **kwargs
        )

        # Initialize LLM provider for direct API calls when needed
        self._llm_provider = self._create_llm_provider(bot_spec)

        # Chess-specific prompt templates
        self._prompt_templates = self._create_prompt_templates()

        logger.info(f"Initialized LLM chess agent: {agent_name} ({bot_spec.provider}:{bot_spec.model})")

    def _create_ag2_llm_config(self, bot_spec: BotSpec, temperature: float, timeout_seconds: float) -> LLMConfig:
        """Create AG2 LLM configuration for the specified provider."""
        base_config = {
            "temperature": temperature,
            "timeout": timeout_seconds,
        }

        if bot_spec.provider == "openai":
            return LLMConfig(
                api_type="openai",
                model=bot_spec.model,
                **base_config
            )
        elif bot_spec.provider == "anthropic":
            return LLMConfig(
                api_type="anthropic",
                model=bot_spec.model,
                **base_config
            )
        elif bot_spec.provider == "gemini":
            return LLMConfig(
                api_type="google",
                model=bot_spec.model,
                **base_config
            )
        else:
            # For other providers, use a minimal config
            logger.warning(f"Unknown provider {bot_spec.provider}, using basic LLM config")
            return LLMConfig(
                api_type="openai",  # Fallback
                model="gpt-3.5-turbo",
                **base_config
            )

    def _create_llm_provider(self, bot_spec: BotSpec):
        """Create direct LLM provider for API calls when AG2 isn't suitable."""
        try:
            if bot_spec.provider == "openai":
                return OpenAIProvider(bot_spec.model)
            elif bot_spec.provider == "anthropic":
                return AnthropicProvider(bot_spec.model)
            elif bot_spec.provider == "gemini":
                return GeminiProvider(bot_spec.model)
            else:
                logger.warning(f"No direct provider for {bot_spec.provider}")
                return None
        except Exception as e:
            logger.warning(f"Failed to create direct LLM provider: {e}")
            return None

    def _create_chess_system_message(self, bot_spec: BotSpec, agent_name: str, temperature: float) -> str:
        """Create comprehensive system message for chess playing."""
        return f"""You are {agent_name}, an expert chess-playing AI agent powered by {bot_spec.provider}:{bot_spec.model}.

CORE MISSION:
You are participating in a chess engine benchmark where your performance will be measured against various ELO-rated opponents. Your goal is to play the strongest possible chess moves.

MOVE FORMAT REQUIREMENTS:
- Provide moves in Standard Algebraic Notation (SAN) like "e4", "Nf3", "O-O", "Qxh7+"
- You may also use UCI format like "e2e4", "g1f3" if needed
- Always ensure your move is legal in the current position
- If unsure, explain your reasoning before stating the final move

CHESS ANALYSIS APPROACH:
1. Analyze the current position carefully
2. Look for tactical opportunities (checks, captures, threats)
3. Consider strategic factors (pawn structure, piece activity, king safety)
4. Calculate key variations when possible
5. Choose the move that best improves your position

RESPONSE FORMAT:
When asked for a move, respond with:
1. Brief position analysis (2-3 sentences)
2. Your chosen move clearly stated
3. Brief explanation of why this move is good

Example:
"The position shows White has good central control. I will play e4 to claim space in the center and develop my pieces actively."

IMPORTANT NOTES:
- You are playing at a competitive level - think carefully
- Avoid obvious blunders and tactical mistakes
- Play principled chess focusing on piece development, center control, and king safety
- When ahead, simplify; when behind, create complications

Current game details:
- Provider: {bot_spec.provider}
- Model: {bot_spec.model}
- Temperature: {temperature}

Let's play excellent chess!"""

    def _create_prompt_templates(self) -> Dict[str, str]:
        """Create templates for different types of chess prompts."""
        return {
            "move_request": """Current chess position (FEN): {fen}

Board visualization:
{board_ascii}

Game context:
- Move number: {move_number}
- Your color: {color}
- Last opponent move: {opponent_move}
- Time remaining: {time_remaining}

Please analyze this position and provide your best move. Consider:
1. Immediate tactics (checks, captures, threats)
2. Strategic considerations (piece development, center control)
3. King safety and pawn structure

Respond with your move in standard algebraic notation (SAN) or UCI format.""",

            "opening_move": """You are starting a new chess game as {color}.

Current position (starting position):
{board_ascii}

This is move {move_number} of the game. Please choose your opening move.
Focus on sound opening principles:
- Control the center (e4, d4, Nf3, etc.)
- Develop pieces toward the center
- Ensure king safety

Provide your move in standard algebraic notation.""",

            "endgame_move": """Current endgame position (FEN): {fen}

Board visualization:
{board_ascii}

Material count:
{material_balance}

This is an endgame position. Please analyze carefully and find the best move.
In endgames, focus on:
- King activity and centralization
- Pawn promotion possibilities
- Precise calculation
- Converting advantages

Your move:""",
        }

    async def generate_move(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None,
        opponent_last_move: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> chess.Move:
        """
        Generate chess move using AG2 LLM capabilities.

        Args:
            board: Current chess board position
            time_limit: Maximum time for move generation
            opponent_last_move: Opponent's last move
            temperature: Override default temperature
            **kwargs: Additional move generation parameters

        Returns:
            Legal chess move

        Raises:
            LLMProviderError: If move generation fails
            TimeoutError: If time limit exceeded
        """
        try:
            # Create chess prompt
            prompt = self._create_chess_prompt(board, opponent_last_move)

            # Use AG2 to generate response
            if hasattr(self, 'llm_config') and self.llm_config:
                # Try AG2 approach first
                try:
                    response = await self._generate_with_ag2(prompt, time_limit)
                    move = self._extract_move_from_response(response, board)
                    if move:
                        return move
                except Exception as e:
                    logger.warning(f"AG2 move generation failed: {e}")

            # Fallback to direct LLM provider
            if self._llm_provider:
                move = await self._llm_provider.generate_move(
                    board,
                    temperature or self.chess_config.temperature,
                    time_limit or self.chess_config.timeout_seconds,
                    opponent_last_move
                )
                return move

            # Final fallback - random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                logger.warning("Using random fallback move")
                return legal_moves[0]

            raise LLMProviderError("No legal moves available")

        except Exception as e:
            logger.error(f"Move generation failed: {e}")
            raise LLMProviderError(f"Failed to generate move: {e}")

    async def _generate_with_ag2(self, prompt: str, time_limit: Optional[float]) -> str:
        """Generate move using AG2 ConversableAgent capabilities."""
        try:
            # Set timeout if provided
            timeout = time_limit or self.chess_config.timeout_seconds

            # Use AG2's generate_reply method
            reply = await asyncio.wait_for(
                self._async_generate_reply(prompt),
                timeout=timeout
            )

            return reply

        except asyncio.TimeoutError:
            raise TimeoutError(f"AG2 move generation timed out after {timeout}s")
        except Exception as e:
            raise LLMProviderError(f"AG2 generation failed: {e}")

    async def _async_generate_reply(self, prompt: str) -> str:
        """Async wrapper for AG2 reply generation."""
        # This is a simplified approach - in practice, you might want to use
        # AG2's more sophisticated conversation management
        try:
            # Create a temporary message context
            messages = [{"role": "user", "content": prompt}]

            # Generate reply using the LLM config
            if hasattr(self, 'llm_config') and self.llm_config:
                # Use AG2's LLM capabilities
                response = await self.a_generate_oai_reply(messages)
                if response and hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            else:
                raise LLMProviderError("No LLM config available")

        except Exception as e:
            logger.error(f"AG2 async reply generation failed: {e}")
            raise

    def _create_chess_prompt(self, board: chess.Board, opponent_last_move: Optional[str]) -> str:
        """Create detailed chess prompt for the current position."""
        # Determine game phase and select appropriate template
        if board.ply() <= 10:
            template_key = "opening_move"
        elif len(board.piece_map()) <= 10:
            template_key = "endgame_move"
        else:
            template_key = "move_request"

        template = self._prompt_templates[template_key]

        # Prepare template variables
        color = "White" if board.turn == chess.WHITE else "Black"
        move_number = (board.ply() // 2) + 1
        opponent_move_str = opponent_last_move or "None (game start)"

        # Create material balance for endgames
        material_balance = ""
        if template_key == "endgame_move":
            material_balance = self._calculate_material_balance(board)

        # Format the prompt
        try:
            prompt = template.format(
                fen=board.fen(),
                board_ascii=str(board),
                move_number=move_number,
                color=color,
                opponent_move=opponent_move_str,
                time_remaining="30s",  # Default
                material_balance=material_balance
            )
        except KeyError as e:
            # Fallback to basic prompt if template formatting fails
            logger.warning(f"Template formatting failed: {e}")
            prompt = f"""
Chess position (FEN): {board.fen()}

{board}

You are playing as {color}. Move {move_number}.
Last opponent move: {opponent_move_str}

Please provide your best move in standard algebraic notation.
"""

        return prompt

    def _calculate_material_balance(self, board: chess.Board) -> str:
        """Calculate material balance for endgame analysis."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        white_material = 0
        black_material = 0

        for square, piece in board.piece_map().items():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value

        balance = white_material - black_material
        if balance > 0:
            return f"White +{balance} material advantage"
        elif balance < 0:
            return f"Black +{abs(balance)} material advantage"
        else:
            return "Material is equal"

    def _extract_move_from_response(self, response: str, board: chess.Board) -> Optional[chess.Move]:
        """Extract and validate chess move from LLM response."""
        if not response:
            return None

        # Try to find moves in the response
        potential_moves = self._find_potential_moves(response)

        for move_str in potential_moves:
            move = self._parse_move_string(move_str, board)
            if move and move in board.legal_moves:
                logger.debug(f"Extracted valid move: {move}")
                return move

        logger.warning(f"No valid move found in response: {response[:100]}...")
        return None

    def _find_potential_moves(self, text: str) -> List[str]:
        """Find potential chess moves in text using regex patterns."""
        moves = []

        # SAN patterns (e4, Nf3, O-O, Qxh7+, etc.)
        san_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)\b'
        san_matches = re.findall(san_pattern, text)
        moves.extend(san_matches)

        # UCI patterns (e2e4, g1f3, etc.)
        uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
        uci_matches = re.findall(uci_pattern, text)
        moves.extend(uci_matches)

        return moves

    def _parse_move_string(self, move_str: str, board: chess.Board) -> Optional[chess.Move]:
        """Parse move string into chess.Move object."""
        try:
            # Try SAN first
            move = board.parse_san(move_str)
            return move
        except:
            try:
                # Try UCI
                move = chess.Move.from_uci(move_str)
                return move
            except:
                return None

    def update_temperature(self, new_temperature: float) -> None:
        """Update the agent's temperature setting."""
        self.chess_config.temperature = new_temperature
        if hasattr(self, 'llm_config') and self.llm_config:
            self.llm_config.temperature = new_temperature
        logger.info(f"Updated temperature to {new_temperature} for {self.name}")

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the LLM provider."""
        return {
            "provider": self.bot_spec.provider,
            "model": self.bot_spec.model,
            "temperature": self.chess_config.temperature,
            "timeout": self.chess_config.timeout_seconds,
            "has_ag2_config": hasattr(self, 'llm_config') and self.llm_config is not None,
            "has_direct_provider": self._llm_provider is not None,
        }

    def __str__(self) -> str:
        """String representation of LLM agent."""
        return f"LLMChessAgent({self.name}, {self.bot_spec.provider}:{self.bot_spec.model})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"LLMChessAgent(name='{self.name}', "
            f"provider='{self.bot_spec.provider}', "
            f"model='{self.bot_spec.model}', "
            f"temperature={self.chess_config.temperature}, "
            f"moves_played={self._move_stats['move_count']})"
        )


def create_llm_agent(
    provider: str,
    model: str,
    name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMChessAgent:
    """
    Convenience function to create an LLM chess agent.

    Args:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model name
        name: Agent name (auto-generated if None)
        temperature: LLM temperature
        **kwargs: Additional agent parameters

    Returns:
        Configured LLMChessAgent
    """
    bot_spec = BotSpec(
        provider=provider,
        model=model,
        name=name or f"{provider}_{model}"
    )

    return LLMChessAgent(
        bot_spec=bot_spec,
        name=name,
        temperature=temperature,
        **kwargs
    )
