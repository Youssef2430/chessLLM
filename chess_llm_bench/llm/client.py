"""
LLM client module for chess move generation.

This module provides a unified interface for generating chess moves using various
Large Language Model providers, with robust parsing, validation, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import os

import chess

from ..core.models import BotSpec
from ..core.budget import record_llm_usage

logger = logging.getLogger(__name__)

# Regex for extracting UCI moves from LLM responses
MOVE_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, spec: BotSpec):
        """Initialize the provider with bot specification."""
        self.spec = spec
        self.random = random.Random()

    @abstractmethod
    async def generate_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        move_history: list = []
    ) -> str:
        """
        Generate a move response from the LLM.

        Args:
            board: Current chess position
            temperature: Sampling temperature
            timeout_s: Timeout in seconds
            move_history: List of previous moves in the game

        Returns:
            Raw text response from the LLM

        Raises:
            LLMProviderError: If move generation fails
        """
        pass

    def _create_chess_prompt(self, board: chess.Board, move_history: list = None) -> str:
        """Create a standardized chess prompt for the LLM."""
        legal_moves = " ".join(move.uci() for move in board.legal_moves)
        color = "White" if board.turn == chess.WHITE else "Black"

        # Create a comprehensive but concise prompt
        prompt = (
            "You are a strong chess player. Given the position and legal moves, "
            "choose the best move and respond with ONLY the UCI notation (like e2e4 or a7a8q).\n\n"
        )

        # Add move history if available
        if move_history and len(move_history) > 0:
            # Format move history in standard algebraic notation style
            move_pairs = []

            for i in range(0, len(move_history), 2):
                move_num = (i // 2) + 1
                white_move = move_history[i]
                black_move = move_history[i+1] if i+1 < len(move_history) else ""

                if black_move:
                    move_pairs.append(f"{move_num}. {white_move} {black_move}")
                else:
                    move_pairs.append(f"{move_num}. {white_move}")

            # Join moves and add to prompt
            move_history_text = "\n".join(move_pairs)
            prompt += (
                f"Game history ({len(move_history)} moves so far):\n"
                f"{move_history_text}\n\n"
            )
            logger.debug(f"Added {len(move_history)} moves to prompt history")

        # Add current position information
        prompt += (
            f"Position (FEN): {board.fen()}\n"
            f"Side to move: {color}\n"
            f"Legal moves (UCI): {legal_moves}\n\n"
            "Your response must be exactly one legal UCI move from the list above, nothing else."
        )

        return prompt

    def _fallback_random_move(self, board: chess.Board) -> chess.Move:
        """Generate a random legal move as fallback."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise LLMProviderError("No legal moves available")
        return legal_moves[self.random.randrange(len(legal_moves))]


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider for chess move generation."""

    def __init__(self, spec: BotSpec):
        super().__init__(spec)

        # Import OpenAI only when needed
        try:
            from openai import OpenAI
            self._openai = OpenAI
        except ImportError:
            raise LLMProviderError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        # Initialize client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY environment variable is required for OpenAI provider"
            )

        self.client = self._openai(api_key=api_key)

        # Validate model
        if not spec.model:
            raise LLMProviderError("Model name is required for OpenAI provider")

    async def generate_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        move_history: list = []
    ) -> str:
        """Generate move using OpenAI GPT models."""
        prompt = self._create_chess_prompt(board, move_history)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._call_openai, prompt, temperature),
                timeout=timeout_s
            )

            # Record usage for budget tracking
            record_llm_usage(
                provider="openai",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response=response,
                success=True
            )

            return response

        except asyncio.TimeoutError:
            # Record failed usage
            record_llm_usage(
                provider="openai",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response="",
                success=False,
                error_message=f"Request timed out after {timeout_s}s"
            )
            raise LLMProviderError(f"OpenAI request timed out after {timeout_s}s")
        except Exception as e:
            # Record failed usage
            record_llm_usage(
                provider="openai",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response="",
                success=False,
                error_message=str(e)
            )
            raise LLMProviderError(f"OpenAI API error: {e}")

    def _call_openai(self, prompt: str, temperature: float) -> str:
        """Make synchronous OpenAI API call."""
        try:
            response = self.client.chat.completions.create(
                model=self.spec.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=16,  # Short response expected
                n=1
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("OpenAI returned empty response")

            return content.strip()

        except Exception as e:
            raise LLMProviderError(f"OpenAI completion failed: {e}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider for chess move generation."""

    def __init__(self, spec: BotSpec):
        super().__init__(spec)

        # Import Anthropic only when needed
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise LLMProviderError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        # Initialize client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )

        self.client = self._anthropic.Anthropic(api_key=api_key)

        # Set default model if not specified
        if not spec.model:
            spec.model = "claude-3-haiku-20240307"

    async def generate_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        move_history: list = []
    ) -> str:
        """Generate move using Anthropic Claude models."""
        prompt = self._create_chess_prompt(board, move_history)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._call_anthropic, prompt, temperature),
                timeout=timeout_s
            )

            # Record usage for budget tracking
            record_llm_usage(
                provider="anthropic",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response=response,
                success=True
            )

            return response

        except asyncio.TimeoutError:
            # Record failed usage
            record_llm_usage(
                provider="anthropic",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response="",
                success=False,
                error_message=f"Request timed out after {timeout_s}s"
            )
            raise LLMProviderError(f"Anthropic request timed out after {timeout_s}s")
        except Exception as e:
            # Record failed usage
            record_llm_usage(
                provider="anthropic",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response="",
                success=False,
                error_message=str(e)
            )
            raise LLMProviderError(f"Anthropic API error: {e}")

    def _call_anthropic(self, prompt: str, temperature: float) -> str:
        """Make synchronous Anthropic API call."""
        try:
            response = self.client.messages.create(
                model=self.spec.model,
                max_tokens=16,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            if not response.content:
                raise LLMProviderError("Anthropic returned empty response")

            # Extract text content
            text_content = ""
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    text_content += content_block.text

            if not text_content:
                raise LLMProviderError("No text content in Anthropic response")

            return text_content.strip()

        except Exception as e:
            raise LLMProviderError(f"Anthropic completion failed: {e}")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider for chess move generation."""

    def __init__(self, spec: BotSpec):
        super().__init__(spec)

        # Import Google Generative AI only when needed
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise LLMProviderError(
                "Google Generative AI package not installed. Install with: pip install google-generativeai"
            )

        # Initialize client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "GEMINI_API_KEY environment variable is required for Gemini provider"
            )

        self._genai.configure(api_key=api_key)

        # Set default model if not specified
        if not spec.model:
            spec.model = "gemini-1.5-flash"

        # Create the model
        try:
            self.model = self._genai.GenerativeModel(spec.model)
        except Exception as e:
            raise LLMProviderError(f"Failed to create Gemini model {spec.model}: {e}")

    async def generate_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        move_history: list = []
    ) -> str:
        """Generate move using Google Gemini models."""
        prompt = self._create_chess_prompt(board, move_history)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._call_gemini, prompt, temperature),
                timeout=timeout_s
            )

            # Record usage for budget tracking
            record_llm_usage(
                provider="gemini",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response=response,
                success=True
            )

            return response

        except asyncio.TimeoutError:
            # Record failed usage
            record_llm_usage(
                provider="gemini",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response="",
                success=False,
                error_message=f"Request timed out after {timeout_s}s"
            )
            raise LLMProviderError(f"Gemini request timed out after {timeout_s}s")
        except Exception as e:
            # Record failed usage
            record_llm_usage(
                provider="gemini",
                model=self.spec.model,
                bot_name=self.spec.name,
                prompt=prompt,
                response="",
                success=False,
                error_message=str(e)
            )
            raise LLMProviderError(f"Gemini API error: {e}")

    def _call_gemini(self, prompt: str, temperature: float) -> str:
        """Make synchronous Gemini API call."""
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": 16,
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            if not response.text:
                raise LLMProviderError("Gemini returned empty response")

            return response.text.strip()

        except Exception as e:
            raise LLMProviderError(f"Gemini completion failed: {e}")


class RandomProvider(BaseLLMProvider):
    """Random move provider (baseline)."""

    def __init__(self, spec: BotSpec):
        """Initialize the random move provider."""
        super().__init__(spec)

    async def generate_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        move_history: list = []
    ) -> str:
        """Generate a random valid move."""
        move = self._fallback_random_move(board)
        return move.uci()


class LLMClient:
    """
    Main LLM client that manages different providers and handles move parsing.

    This class provides a unified interface for chess move generation across
    different LLM providers with robust error handling and move validation.
    """

    # Registry of available providers
    PROVIDERS: Dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "random": RandomProvider,
    }

    def __init__(self, spec: BotSpec):
        """
        Initialize LLM client with the specified bot configuration.

        Args:
            spec: Bot specification including provider, model, and name
        """
        self.spec = spec

        # Validate and create provider
        provider_class = self.PROVIDERS.get(spec.provider.lower())
        if not provider_class:
            available = ", ".join(self.PROVIDERS.keys())
            raise LLMProviderError(
                f"Unsupported provider '{spec.provider}'. Available: {available}"
            )

        try:
            self.provider = provider_class(spec)
            logger.info(f"Initialized LLM client: {spec}")
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize provider {spec.provider}: {e}")

        # Initialize move statistics tracking
        self._total_move_time = 0.0
        self._illegal_move_attempts = 0
        self._move_count = 0

        # Initialize move history tracking
        self._move_history = []

    async def pick_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        opponent_move: Optional[str] = None
    ) -> chess.Move:
        """
        Generate a legal chess move for the given position.

        Args:
            board: Current chess position
            temperature: Sampling temperature (0.0 = deterministic)
            timeout_s: Timeout for move generation

        Returns:
            A legal chess move

        Raises:
            LLMProviderError: If move generation fails completely
        """
        if board.is_game_over():
            raise LLMProviderError("Cannot generate move for finished game")

        # Start timing
        start_time = time.time()
        illegal_attempts = 0

        # If opponent just moved, add it to history
        if opponent_move:
            self._move_history.append(opponent_move)
            logger.debug(f"Added opponent move to history: {opponent_move} (moves: {len(self._move_history)})")

        try:
            # Generate move from LLM with complete game history
            logger.debug(f"Generating move with history of {len(self._move_history)} previous moves")
            response = await self.provider.generate_move(board, temperature, timeout_s, self._move_history)
            logger.debug(f"LLM response: {response}")

            # Parse and validate move
            move = self._parse_move(response, board)

            if move and move in board.legal_moves:
                logger.debug(f"Selected move: {move.uci()}")
                # Add the selected move to game history
                move_uci = move.uci()
                self._move_history.append(move_uci)
                logger.debug(f"Added LLM move to history: {move_uci} (moves: {len(self._move_history)})")
                self._record_move_stats(start_time, illegal_attempts)
                return move
            elif move:
                # Move was parsed but is illegal
                illegal_attempts += 1
                logger.debug(f"Illegal move attempted: {move.uci()}")

            # If parsing failed, try SAN notation as fallback
            # Try SAN notation as fallback
            move = self._try_san_parsing(response, board)
            if move and move in board.legal_moves:
                logger.debug(f"Parsed SAN move: {move.uci()}")
                # Add the move to history
                move_uci = move.uci()
                self._move_history.append(move_uci)
                logger.debug(f"Added LLM move (SAN parsed) to history: {move_uci} (moves: {len(self._move_history)})")
                self._record_move_stats(start_time, illegal_attempts)
                return move
            elif move:
                # SAN move was parsed but is illegal
                illegal_attempts += 1
                logger.debug(f"Illegal SAN move attempted: {move.uci()}")
            else:
                # Both parsing attempts failed
                illegal_attempts += 1
                logger.debug("Move parsing failed completely")

        except Exception as e:
            logger.warning(f"LLM move generation failed: {e}")
            illegal_attempts += 1

        # Final fallback to random move
        logger.info("Falling back to random move")
        illegal_attempts += 1  # Count using a random move as an illegal attempt
        fallback_move = self.provider._fallback_random_move(board)

        # Add the fallback move to history
        fallback_uci = fallback_move.uci()
        self._move_history.append(fallback_uci)
        logger.debug(f"Added fallback move to history: {fallback_uci} (moves: {len(self._move_history)})")

        # Record stats with extra illegal attempt for random move
        self._record_move_stats(start_time, illegal_attempts)
        return fallback_move

    def _record_move_stats(self, start_time: float, illegal_attempts: int) -> None:
        """Record timing and illegal move statistics for this move."""
        move_time = time.time() - start_time
        self._total_move_time += move_time
        self._illegal_move_attempts += illegal_attempts
        self._move_count += 1
        logger.debug(f"Move stats: {move_time:.3f}s, {illegal_attempts} illegal attempts")

    def get_move_stats(self) -> Tuple[float, int, float]:
        """
        Get current move statistics.

        Returns:
            Tuple of (total_time, illegal_attempts, average_time_per_move)
        """
        avg_time = self._total_move_time / self._move_count if self._move_count > 0 else 0.0
        return (self._total_move_time, self._illegal_move_attempts, avg_time)

    def reset_move_stats(self) -> None:
        """Reset move statistics for a new game."""
        self._total_move_time = 0.0
        self._illegal_move_attempts = 0
        self._move_count = 0
        self._move_history = []  # Reset move history for new game

    def _parse_move(self, response: str, board: chess.Board) -> Optional[chess.Move]:
        """
        Parse UCI move from LLM response text.

        Args:
            response: Raw LLM response
            board: Current chess position

        Returns:
            Parsed move if successful, None otherwise
        """
        # Extract UCI move with regex
        match = MOVE_REGEX.search(response)
        if not match:
            return None

        uci_str = match.group(1).lower()

        try:
            move = chess.Move.from_uci(uci_str)

            # Handle promotion notation variations
            if move not in board.legal_moves and len(uci_str) == 5:
                # Try without promotion piece (auto-promote to queen)
                try:
                    move = chess.Move.from_uci(uci_str[:4] + 'q')
                except:
                    pass

            return move if move in board.legal_moves else None

        except (ValueError, chess.InvalidMoveError):
            return None

    def _try_san_parsing(self, response: str, board: chess.Board) -> Optional[chess.Move]:
        """
        Attempt to parse Standard Algebraic Notation as fallback.

        Args:
            response: Raw LLM response
            board: Current chess position

        Returns:
            Parsed move if successful, None otherwise
        """
        # Clean up response for SAN parsing
        cleaned = response.strip().split()[0]  # Take first word

        # Try common SAN variations
        san_candidates = [
            cleaned,
            cleaned.rstrip('+#'),  # Remove check/mate symbols
            cleaned.replace('x', ''),  # Remove capture notation
        ]

        for san in san_candidates:
            try:
                move = board.parse_san(san)
                if move in board.legal_moves:
                    return move
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                continue

        return None

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available LLM providers."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """
        Register a custom LLM provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Provider implementation class
        """
        cls.PROVIDERS[name.lower()] = provider_class
        logger.info(f"Registered custom provider: {name}")


def parse_bot_spec(spec_string: str) -> List[BotSpec]:
    """
    Parse bot specification strings into BotSpec objects.

    Format: "provider:model:name" or "provider::name" (empty model)
    Multiple bots can be specified separated by commas.

    Args:
        spec_string: Comma-separated bot specifications

    Returns:
        List of BotSpec objects

    Raises:
        ValueError: If specification format is invalid
    """
    bots: List[BotSpec] = []

    if not spec_string.strip():
        return bots

    for raw_spec in [s.strip() for s in spec_string.split(",") if s.strip()]:
        parts = raw_spec.split(":")

        if len(parts) == 1:
            # Single part: treat as provider name
            provider, model, name = parts[0], "", parts[0]
        elif len(parts) == 2:
            # Two parts: provider:model or provider:name
            provider, second_part = parts
            if second_part:
                model, name = second_part, second_part
            else:
                model, name = "", provider
        else:
            # Three or more parts: provider:model:name (name can have colons)
            provider, model, name = parts[0], parts[1], ":".join(parts[2:])

        # Validate provider
        if provider.lower() not in LLMClient.PROVIDERS:
            available = ", ".join(LLMClient.PROVIDERS.keys())
            raise ValueError(f"Unsupported provider '{provider}'. Available: {available}")

        bots.append(BotSpec(
            provider=provider.lower(),
            model=model,
            name=name
        ))

    return bots
