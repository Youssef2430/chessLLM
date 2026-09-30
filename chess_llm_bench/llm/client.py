"""Async providers, auditable usage, and strict chess move validation."""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess

from ..core.models import BotSpec
from ..core.budget import get_budget_tracker

MOVE_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
AGENTS_AVAILABLE = True  # Agent imports are lazy to avoid the former circular import.


class LLMProviderError(RuntimeError):
    """Infrastructure failure; never substitute a chess move."""


class InvalidMoveError(ValueError):
    """A model response violated the move contract (a game forfeit)."""


@dataclass
class Completion:
    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]  # Includes billed reasoning/thinking tokens.


class BaseLLMProvider(ABC):
    def __init__(self, spec: BotSpec):
        self.spec = spec
        self.random = random.Random(42)
        self.max_output_tokens = 2048
        self.reasoning_effort = "low"
        self.usage_context = {}

    @abstractmethod
    async def generate_move(
        self, board, temperature=0.0, timeout_s=60.0, move_history=None
    ) -> str:
        pass

    def _create_chess_prompt(self, board, move_history=None) -> str:
        # The board's stack is authoritative, including all supplied opening plies.
        history = " ".join(move.uci() for move in board.move_stack)
        return (
            "Choose the best chess move. Reply with exactly one legal UCI move, "
            "such as e2e4 or a7a8q, and no other text.\n"
            f"Position (FEN): {board.fen()}\n"
            f"Side to move: {'White' if board.turn else 'Black'}\n"
            f"History (UCI, from the starting position): {history}\n"
            f"Legal moves (UCI): {' '.join(move.uci() for move in board.legal_moves)}"
        )

    def _fallback_random_move(self, board) -> str:
        """Used only by the explicit random baseline."""
        legal = list(board.legal_moves)
        if not legal:
            raise LLMProviderError("No legal moves available")
        return self.random.choice(legal).uci()

    async def close(self):
        pass


class APIProvider(BaseLLMProvider):
    async def generate_move(
        self, board, temperature=0.0, timeout_s=60.0, move_history=None
    ):
        return await self.complete(
            self._create_chess_prompt(board, move_history), temperature, timeout_s
        )

    async def complete(self, prompt: str, temperature: float, timeout_s: float) -> str:
        tracker = get_budget_tracker()
        input_allowance = len(prompt.encode("utf-8")) + 256
        reservation = tracker.reserve(
            self.spec.provider, self.spec.model, input_allowance, self.max_output_tokens
        )
        try:
            try:
                response = await asyncio.wait_for(
                    self._request(prompt, temperature, timeout_s), timeout_s
                )
            except BaseException as exc:
                # A timeout/cancellation may still be billed. Preserve a conservative
                # charge instead of claiming the request was free or retrying it.
                tracker.record_usage(
                    self.spec.provider,
                    self.spec.model,
                    self.spec.name,
                    prompt,
                    actual_input_tokens=input_allowance,
                    actual_output_tokens=self.max_output_tokens,
                    success=False,
                    error_message=type(exc).__name__,
                    usage_source="uncertain_upper_estimate",
                    **self.usage_context,
                )
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if not isinstance(exc, Exception):
                    raise
                raise LLMProviderError(
                    f"{self.spec.provider} request failed ({type(exc).__name__})"
                ) from exc
            reported = (
                response.input_tokens is not None and response.output_tokens is not None
            )
            tracker.record_usage(
                self.spec.provider,
                self.spec.model,
                self.spec.name,
                prompt,
                response.text,
                actual_input_tokens=(
                    response.input_tokens if reported else input_allowance
                ),
                actual_output_tokens=(
                    response.output_tokens if reported else self.max_output_tokens
                ),
                usage_source="reported" if reported else "uncertain_upper_estimate",
                **self.usage_context,
            )
            # Empty/truncated responses are still billed and count as invalid output.
            return response.text.strip()
        finally:
            tracker.release(reservation)


class OpenAIProvider(APIProvider):
    def __init__(self, spec):
        super().__init__(spec)
        from openai import AsyncOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise LLMProviderError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(max_retries=0)

    async def _request(self, prompt, temperature, timeout_s):
        kwargs = dict(
            model=self.spec.model,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
            timeout=timeout_s,
            store=False,
        )
        if self.spec.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            kwargs["temperature"] = temperature
        response = await self.client.responses.create(**kwargs)
        usage = response.usage
        return Completion(
            response.output_text or "",
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )

    async def close(self):
        await self.client.close()


class AnthropicProvider(APIProvider):
    def __init__(self, spec):
        super().__init__(spec)
        from anthropic import AsyncAnthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMProviderError("ANTHROPIC_API_KEY is required")
        self.client = AsyncAnthropic(max_retries=0)

    async def _request(self, prompt, temperature, timeout_s):
        kwargs = dict(
            model=self.spec.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_output_tokens,
            timeout=timeout_s,
        )
        if self.spec.model.startswith(
            ("claude-sonnet-5", "claude-opus-5", "claude-fable-5")
        ):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.reasoning_effort}
        else:
            kwargs["temperature"] = temperature
        response = await self.client.messages.create(**kwargs)
        usage = response.usage
        inputs = getattr(usage, "input_tokens", None)
        if inputs is not None:
            inputs += getattr(usage, "cache_read_input_tokens", 0) or 0
            inputs += getattr(usage, "cache_creation_input_tokens", 0) or 0
        return Completion(
            "".join(block.text for block in response.content if block.type == "text"),
            inputs,
            getattr(usage, "output_tokens", None),
        )

    async def close(self):
        await self.client.close()


class GeminiProvider(APIProvider):
    def __init__(self, spec):
        super().__init__(spec)
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMProviderError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    async def _request(self, prompt, temperature, timeout_s):
        from google.genai import types

        config = dict(max_output_tokens=self.max_output_tokens)
        if self.spec.model.startswith("gemini-3"):
            config["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.reasoning_effort
            )
        else:
            config["temperature"] = temperature
        response = await self.client.aio.models.generate_content(
            model=self.spec.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        )
        usage = response.usage_metadata
        output = getattr(usage, "candidates_token_count", None)
        if output is not None:
            output += getattr(usage, "thoughts_token_count", 0) or 0
        text = "".join(
            part.text
            for candidate in (response.candidates or [])
            for part in (candidate.content.parts if candidate.content else [])
            if part.text and not part.thought
        )
        return Completion(text, getattr(usage, "prompt_token_count", None), output)

    async def close(self):
        await self.client.aio.aclose()
        self.client.close()


class RandomProvider(BaseLLMProvider):
    async def generate_move(
        self, board, temperature=0.0, timeout_s=60.0, move_history=None
    ):
        return self._fallback_random_move(board)


class LLMClient:
    PROVIDERS: Dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "random": RandomProvider,
    }

    def __init__(
        self,
        spec,
        use_agent=False,
        agent_strategy="balanced",
        verbose_agent=False,
        max_output_tokens=2048,
        reasoning_effort="low",
    ):
        self.spec = spec
        self.use_agent = use_agent and spec.provider != "random"
        self.agent_strategy = agent_strategy
        self.verbose_agent = verbose_agent
        if self.use_agent:
            from .agents.llm_agent_provider import create_agent_provider

            self.provider = create_agent_provider(
                spec, strategy=agent_strategy, verbose=verbose_agent
            )
        else:
            provider = self.PROVIDERS.get(spec.provider)
            if provider is None:
                raise LLMProviderError(f"Unsupported provider {spec.provider}")
            self.provider = provider(spec)
        self.provider.max_output_tokens = max_output_tokens
        self.provider.reasoning_effort = reasoning_effort
        self.reset_move_stats()

    async def pick_move(
        self, board, temperature=0.0, timeout_s=60.0, opponent_move=None
    ):
        if board.is_game_over(claim_draw=True):
            raise LLMProviderError("Cannot generate move for finished game")
        start = time.monotonic()
        self._move_history = [move.uci() for move in board.move_stack]
        try:
            response = await self.provider.generate_move(
                board, temperature, timeout_s, self._move_history
            )
            move = self._parse_move(response, board)
            if move is None:
                self._illegal_move_attempts += 1
                raise InvalidMoveError("Expected exactly one legal UCI move")
            return move
        finally:
            self._total_move_time += time.monotonic() - start
            self._move_count += 1

    def get_move_stats(self) -> Tuple[float, int, float]:
        return (
            self._total_move_time,
            self._illegal_move_attempts,
            self._total_move_time / self._move_count if self._move_count else 0.0,
        )

    def reset_move_stats(self):
        self._total_move_time = 0.0
        self._illegal_move_attempts = 0
        self._move_count = 0
        self._move_history = []

    def _parse_move(self, response, board) -> Optional[chess.Move]:
        match = MOVE_REGEX.fullmatch(response.strip())
        if not match:
            return None
        try:
            move = chess.Move.from_uci(match.group(1).lower())
            return move if move in board.legal_moves else None
        except ValueError:
            return None

    async def close(self):
        await self.provider.close()

    @classmethod
    def get_available_providers(cls):
        return list(cls.PROVIDERS)

    @classmethod
    def register_provider(cls, name, provider_class):
        cls.PROVIDERS[name.lower()] = provider_class


def parse_bot_spec(spec_string: str) -> List[BotSpec]:
    bots = []
    for raw in filter(None, (part.strip() for part in spec_string.split(","))):
        parts = [part.strip() for part in raw.split(":", 2)]
        provider = parts[0].lower()
        model = parts[1] if len(parts) > 1 else ""
        name = parts[2] if len(parts) > 2 else model or provider
        if provider not in LLMClient.PROVIDERS:
            raise ValueError(f"Unsupported provider '{provider}'")
        if provider != "random" and not model:
            raise ValueError(f"Model ID required for {provider}")
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Bot name must be a nonempty filename-safe name")
        if any(bot.name == name for bot in bots):
            raise ValueError(f"Duplicate bot name: {name}")
        bots.append(BotSpec(provider, model, name))
    return bots
