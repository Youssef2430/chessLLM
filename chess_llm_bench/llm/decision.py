"""Auditable, bounded decision loop for protocol 3 subscription games."""

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

from ..core.journal import append_row, atomic_json, read_rows
from ..core.moves import chess_prompt, move_schema, resolve_move
from .client import InvalidMoveError, LLMProviderError
from .subscription import parse_events, normalized_tokens


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def error_category(message):
    text = message.lower()
    for category, words in {
        "quota": (
            "quota",
            "rate limit",
            "rate_limit",
            "usage limit",
            "credit balance",
            "429",
        ),
        "authentication": ("auth", "sign in", "login", "401", "403"),
        "model_mismatch": ("model mismatch", "did not confirm", "unavailable"),
        "tool_use": ("tool use", "tools disabled"),
        "timeout": ("timeout", "timed out"),
        "cancelled": ("cancelled",),
    }.items():
        if any(w in text for w in words):
            return category
    return "transport"


class DecisionClient:
    def __init__(
        self,
        provider,
        directory,
        gate,
        max_attempts=3,
        timeout=180,
        structured=True,
        prompt_style="checklist",
    ):
        self.provider, self.directory, self.gate = provider, directory, gate
        self.max_attempts, self.timeout, self.structured, self.prompt_style = (
            max_attempts,
            timeout,
            structured,
            prompt_style,
        )
        self.records = read_rows(directory / "requests.jsonl")
        self.last_decision = None

    async def choose(self, board, game_id):
        name = self.provider.spec.name
        decision_id = f"{name}-g{game_id}-p{board.ply()+1}"
        prior = [r for r in self.records if r.get("decision_id") == decision_id]
        for row in reversed(prior):
            if row.get("legal"):
                if row["fen"] != board.fen():
                    raise RuntimeError(
                        "Checkpoint position does not match saved decision"
                    )
                move = board.parse_uci(row["move_uci"])
                self.last_decision = row
                return move
        structured = self.structured and self.provider.spec.provider in {
            "codex",
            "claude",
        }
        self.provider.response_schema = move_schema(board) if structured else None
        self.provider.system_prompt = (
            "You are a chess player. Use only the supplied chess position. Never call tools or external engines. "
            + (
                "Return only the requested JSON move object."
                if structured
                else "Return your chosen UCI move alone on the final line."
            )
        )
        base = (
            chess_prompt(board, structured)
            if self.prompt_style == "checklist"
            else self.provider._create_chess_prompt(board)
        )
        if self.prompt_style == "minimal" and structured:
            base = base.replace(
                "Reply with exactly one legal UCI move, such as e2e4 or a7a8q, and no other text.",
                'Return only JSON with one field named "move" containing a legal UCI move.',
            )
        correction = ""
        if prior and not prior[-1].get("error"):
            correction = f"\nPrevious answer was rejected: {prior[-1].get('validation_error')}. Return one legal move in the required format."
        for attempt in range(len(prior) + 1, self.max_attempts + 1):
            request_id = f"{decision_id}-a{attempt}-{uuid.uuid4().hex[:8]}"
            prompt = base + correction
            queue_start = time.monotonic()
            async with self.gate:
                started = time.monotonic()
                row = dict(
                    schema_version=3,
                    request_id=request_id,
                    decision_id=decision_id,
                    attempt=attempt,
                    bot=name,
                    game_id=str(game_id),
                    provider=self.provider.spec.provider,
                    requested_model=self.provider.spec.model,
                    reasoning_effort=self.provider.reasoning_effort,
                    structured_requested=structured,
                    prompt_style=self.prompt_style,
                    started_at=utcnow(),
                    ply=board.ply() + 1,
                    side_to_move="white" if board.turn else "black",
                    fen=board.fen(),
                    history_uci=[m.uci() for m in board.move_stack],
                    legal_moves=[m.uci() for m in board.legal_moves],
                    legal_move_count=board.legal_moves.count(),
                    in_check=board.is_check(),
                    pieces=len(board.piece_map()),
                    halfmove_clock=board.halfmove_clock,
                    prompt=prompt,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_bytes=len(prompt.encode()),
                    queue_seconds=started - queue_start,
                    provider_concurrency=getattr(self.gate, "request_limit", None),
                    billing_route="subscription_cli",
                    incremental_charge_usd=None,
                    output_token_cap=None,
                    legal=None,
                    error=None,
                )
                atomic_json(self.directory / "pending" / f"{name}.json", row)
                try:
                    code, stdout, stderr = await asyncio.wait_for(
                        self.provider.transport(prompt, self.timeout), self.timeout + 5
                    )
                    raw_path = self.directory / "traces" / f"{request_id}.json"
                    atomic_json(raw_path, dict(stdout=stdout, stderr=stderr))
                    row["trace_path"] = str(raw_path.relative_to(self.directory))
                    parsed = parse_events(
                        self.provider.spec.provider,
                        stdout,
                        stderr,
                        self.provider.spec.model,
                        expected_native=getattr(self.provider, "native_model", None),
                        structured=structured,
                    )
                    row.update(parsed, exit_code=code)
                    if code or parsed["error"]:
                        raise LLMProviderError(
                            parsed["error"] or f"CLI exit {code}: {stderr[-1200:]}"
                        )
                    result = resolve_move(parsed["response"], board)
                    row.update(
                        legal=result.move is not None,
                        strict_uci_valid=result.strict_valid,
                        parse_method=result.method,
                        validation_error=result.error,
                        format_recovered=result.move is not None
                        and result.method
                        not in ({"json"} if structured else {"strict"}),
                    )
                    if result.move:
                        move = result.move
                        after = board.copy()
                        after.push(move)
                        row.update(
                            move_uci=move.uci(),
                            move_san=board.san(move),
                            fen_after=after.fen(),
                            capture=board.is_capture(move),
                            castling=board.is_castling(move),
                            gives_check=board.gives_check(move),
                            promotion=move.promotion,
                        )
                except BaseException as exc:
                    message = str(exc) or type(exc).__name__
                    row.update(
                        error=message,
                        error_category=error_category(message),
                        legal=None,
                    )
                    partial = {
                        "stdout": getattr(exc, "stdout", ""),
                        "stderr": getattr(exc, "stderr", ""),
                    }
                    if hasattr(self.provider, "acp"):
                        partial["acp_events"] = self.provider.acp.events
                    if any(partial.values()):
                        partial_path = (
                            self.directory / "traces" / f"{request_id}-partial.json"
                        )
                        atomic_json(partial_path, partial)
                        row["partial_trace_path"] = str(
                            partial_path.relative_to(self.directory)
                        )
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if not isinstance(exc, Exception):
                        raise
                finally:
                    row.update(
                        wall_seconds=time.monotonic() - started, finished_at=utcnow()
                    )
                    row.update(normalized_tokens(row["provider"], row.get("usage", {})))
                    append_row(self.directory / "requests.jsonl", row)
                    self.records.append(row)
                    (self.directory / "pending" / f"{name}.json").unlink(
                        missing_ok=True
                    )
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "bot",
                            "game_id",
                            "ply",
                            "attempt",
                            "move_uci",
                            "parse_method",
                            "wall_seconds",
                            "error",
                        )
                    }
                ),
                flush=True,
            )
            if row.get("legal"):
                self.last_decision = row
                append_row(
                    self.directory / "decisions.jsonl",
                    dict(
                        decision_id=decision_id,
                        bot=name,
                        game_id=str(game_id),
                        ply=board.ply() + 1,
                        request_id=request_id,
                        attempts=attempt,
                        recovered=row["format_recovered"],
                        status="accepted",
                    ),
                )
                return board.parse_uci(row["move_uci"])
            if row.get("error"):
                if row["error_category"] not in {"timeout", "transport"}:
                    raise LLMProviderError(row["error"])
                if hasattr(self.provider, "acp"):
                    from .antigravity_acp import ACPClient

                    await self.provider.acp.close()
                    self.provider.acp = ACPClient()
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2**attempt, 8))
            else:
                correction = (
                    f"\nYour previous response was rejected ({row['validation_error']}).\n"
                    f"Previous response: {row['response'][:2000]}\n"
                    "Choose from the listed legal moves. Return only the requested move format."
                )
        failed = [r for r in self.records if r.get("decision_id") == decision_id]
        if failed and failed[-1].get("error"):
            raise LLMProviderError(failed[-1]["error"])
        raise InvalidMoveError("No valid move after the bounded correction attempts")
