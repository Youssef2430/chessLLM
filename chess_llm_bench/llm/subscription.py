"""Subscription CLI transport. Never imports an API SDK or loads .env files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import chess

from .client import BaseLLMProvider, LLMProviderError

SYSTEM = (
    "You are a chess player in a controlled benchmark. Choose the best move using "
    "only the supplied position, history and legal moves. Do not call tools, use "
    "external resources, inspect files, or delegate. Return exactly one legal "
    "UCI move and nothing else."
)


def subscription_environment():
    # Allow-list essentials: no API keys, provider overrides, inherited SDK
    # sessions, proxies or injected prompts. HOME remains the real auth location.
    allowed = {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SYSTEMROOT",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(NO_COLOR="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    return env


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")
        stream.flush()


async def execute(argv, prompt, cwd, timeout):
    """Kill the entire process group on timeout/cancellation; never retry a turn."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=subscription_environment(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    task = asyncio.create_task(process.communicate(prompt.encode()))
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(task), timeout)
    except BaseException as exc:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        partial_out, partial_err = await task
        if isinstance(exc, asyncio.TimeoutError):
            exc.stdout = partial_out.decode(errors="replace")
            exc.stderr = partial_err.decode(errors="replace")
        raise
    return (
        process.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def parse_events(
    provider, stdout, stderr, requested, expected_native=None, structured=False
):
    events = []
    try:
        whole = json.loads(stdout)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, list):
        events.extend(value for value in whole if isinstance(value, dict))
    elif isinstance(whole, dict):
        events.append(whole)
    for line in ([] if isinstance(whole, (list, dict)) else stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
        except json.JSONDecodeError:
            continue
    text, usage, actual, error = "", {}, None, None
    api_seconds, rate_limits, tool_calls, timing = None, [], [], {}
    if not events:
        error = (
            "No structured client events received; transport output cannot be validated"
        )
    if provider == "codex":
        match = re.search(r"^model:\s*(\S+)", stderr, re.MULTILINE)
        actual = match.group(1) if match else None
        for event in events:
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
            if item.get("type") in {
                "command_execution",
                "mcp_tool_call",
                "web_search",
                "file_change",
            }:
                tool_calls.append(item.get("type"))
            if event.get("type") == "turn.completed":
                usage = event.get("usage", {})
            if event.get("type") in {"error", "turn.failed"}:
                error = str(event.get("message") or event.get("error"))
    else:
        for event in events:
            if event.get("type") == "system" and event.get("subtype") == "init":
                actual = event.get("model")
            if event.get("type") == "assistant":
                message = event.get("message", {})
                actual = message.get("model", actual)
                tool_calls.extend(
                    b.get("name")
                    for b in message.get("content", [])
                    if b.get("type") == "tool_use"
                    and not (structured and b.get("name") == "StructuredOutput")
                )
            if event.get("type") == "rate_limit_event":
                rate_limits.append(event.get("rate_limit_info", {}))
            if event.get("type") == "result":
                text = event.get("result", "")
                if structured and event.get("structured_output") is not None:
                    text = json.dumps(event["structured_output"])
                usage = event.get("usage", {})
                api_seconds = (
                    event["duration_api_ms"] / 1000
                    if event.get("duration_api_ms") is not None
                    else None
                )
                if event.get("is_error"):
                    error = text or str(event.get("errors") or event.get("subtype"))
                # This is an API-equivalent estimate, NOT a subscription charge.
                usage["api_equivalent_usd"] = event.get("total_cost_usd")
                usage["model_usage"] = event.get("modelUsage", {})
                timing = {
                    k: event.get(k)
                    for k in (
                        "ttft_ms",
                        "ttft_stream_ms",
                        "time_to_request_ms",
                        "first_content_frame_ms",
                        "duration_ms",
                        "num_turns",
                        "stop_reason",
                    )
                }
            if event.get("type") == "acp_trace":
                actual = event.get("native_model", actual)
    if tool_calls:
        error = "Tool use invalidates this unaided benchmark: " + str(tool_calls)
    if actual and actual != requested:
        # Claude sometimes appends a dated snapshot to the explicitly selected ID.
        if not (
            re.fullmatch(re.escape(requested) + r"-\d{8}", actual)
            or (
                provider == "antigravity"
                and actual == (expected_native or requested + "-low")
            )
        ):
            error = f"Model mismatch: requested {requested}, reported {actual}"
    return dict(
        response=text.strip(),
        usage=usage,
        reported_model=actual,
        model_verification="client_reported" if actual else "requested_only",
        api_seconds=api_seconds,
        rate_limits=rate_limits,
        tool_calls=tool_calls,
        client_timing=timing,
        error=error,
    )


def request_interrupted(row):
    """Cancellation is a run interruption, not a model/provider failure."""
    return (
        row.get("error_category") in {"cancelled", "interrupted"}
        or row.get("error") == "CancelledError"
    )


def normalized_tokens(provider, usage):
    """Preserve missing metrics and account for each client's cache semantics."""
    inputs = usage.get("input_tokens")
    outputs = usage.get("output_tokens")
    if provider == "claude" and inputs is not None:
        cached = usage.get("cache_read_input_tokens", 0)
        written = usage.get("cache_creation_input_tokens", 0)
        total = inputs + cached + written
        reasoning = usage.get("output_tokens_details", {}).get("thinking_tokens")
    else:
        cached = usage.get("cached_input_tokens")
        written = usage.get("cache_write_input_tokens")
        total = inputs
        reasoning = usage.get("reasoning_output_tokens")
    return dict(
        input_tokens_total=total,
        output_tokens_total=outputs,
        cached_input_tokens=cached,
        cache_write_tokens=written,
        reasoning_tokens=reasoning,
        tokens_reported=inputs is not None and outputs is not None,
    )


class SubscriptionProvider(BaseLLMProvider):
    def __init__(self, spec):
        super().__init__(spec)
        self.output_dir = None
        self.gate = None
        self.request_count = 0
        self.response_schema = None
        self.system_prompt = SYSTEM

    def command(self, directory):
        if self.spec.provider == "claude":
            argv = [
                "claude",
                "--print",
                "--model",
                self.spec.model,
                "--effort",
                self.reasoning_effort,
                "--safe-mode",
                "--restricted",
                "--tools",
                "",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--disable-slash-commands",
                "--no-chrome",
                "--no-session-persistence",
                "--setting-sources",
                "",
                "--system-prompt",
                self.system_prompt,
                "--output-format",
                "json" if self.response_schema else "stream-json",
                "--verbose",
            ]
            if self.response_schema:
                argv += ["--json-schema", json.dumps(self.response_schema)]
            return argv
        instructions = directory / "instructions.txt"
        instructions.write_text(self.system_prompt)
        argv = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--json",
            "--model",
            self.spec.model,
        ]
        config = {
            "forced_login_method": "chatgpt",
            "model_provider": "openai",
            "model_reasoning_effort": self.reasoning_effort,
            "model_instructions_file": str(instructions),
            "web_search": "disabled",
            "approval_policy": "never",
            "project_doc_max_bytes": 0,
            "features.skip_host_skill_discovery": True,
        }
        for feature in (
            "shell_tool",
            "unified_exec",
            "apps",
            "plugins",
            "hooks",
            "multi_agent",
            "multi_agent_v2",
            "browser_use",
            "computer_use",
            "image_generation",
            "code_mode_host",
            "memories",
            "skill_search",
            "in_app_browser",
            "in_app_local_automation",
            "view_image",
            "goals",
            "sleep_tool",
        ):
            config[f"features.{feature}"] = False
        for key, value in config.items():
            argv += ["-c", f"{key}={json.dumps(value)}"]
        if self.response_schema:
            schema_path = directory / "move.schema.json"
            schema_path.write_text(json.dumps(self.response_schema))
            argv += ["--output-schema", str(schema_path)]
        return argv + ["-"]

    async def generate_move(
        self, board, temperature=0.0, timeout_s=120.0, move_history=None
    ):
        queued = time.monotonic()
        if self.gate:
            await self.gate.acquire()
        try:
            return await self._generate(board, timeout_s, time.monotonic() - queued)
        finally:
            if self.gate:
                self.gate.release()

    async def _generate(self, board, timeout_s, queue_seconds):
        self.request_count += 1
        prompt = self._create_chess_prompt(board)
        started = time.monotonic()
        request_id = f"{self.spec.name}-{self.request_count:05d}"
        record = dict(
            schema_version=1,
            request_id=request_id,
            bot=self.spec.name,
            provider=self.spec.provider,
            requested_model=self.spec.model,
            started_at=datetime.now(timezone.utc).isoformat(),
            **self.usage_context,
            fen=board.fen(),
            history_uci=[m.uci() for m in board.move_stack],
            side_to_move="white" if board.turn else "black",
            ply=board.ply() + 1,
            legal_moves=[m.uci() for m in board.legal_moves],
            legal_move_count=board.legal_moves.count(),
            in_check=board.is_check(),
            pieces=len(board.piece_map()),
            halfmove_clock=board.halfmove_clock,
            prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_bytes=len(prompt.encode()),
            reasoning_effort=self.reasoning_effort,
            output_token_cap=None,
            queue_seconds=queue_seconds,
            billing_route="subscription_cli",
            incremental_charge_usd=None,
        )
        try:
            code, stdout, stderr = await self.transport(prompt, timeout_s)
            parsed = parse_events(self.spec.provider, stdout, stderr, self.spec.model)
            record.update(parsed, exit_code=code)
            # Persist diagnostics separately from the flat chart data. CLIs do not
            # print credentials; never save environment or authentication files.
            if self.output_dir:
                raw = self.output_dir / "traces" / f"{request_id}.json"
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_text(
                    json.dumps({"stdout": stdout, "stderr": stderr}, indent=2)
                )
                record["trace_path"] = str(raw.relative_to(self.output_dir))
            if code or parsed["error"]:
                raise LLMProviderError(
                    parsed["error"] or f"CLI exited {code}: {stderr[-1000:]}"
                )
            response = parsed["response"]
            move = None
            try:
                move = chess.Move.from_uci(response)
            except ValueError:
                pass
            record["legal"] = move is not None and move in board.legal_moves
            if record["legal"]:
                record.update(
                    move_uci=move.uci(),
                    move_san=board.san(move),
                    capture=board.is_capture(move),
                    castling=board.is_castling(move),
                    promotion=move.promotion,
                    gives_check=board.gives_check(move),
                )
                after = board.copy()
                after.push(move)
                record["fen_after"] = after.fen()
            return response
        except BaseException as exc:
            record.update(error=str(exc) or type(exc).__name__, legal=None)
            if isinstance(exc, asyncio.CancelledError):
                raise
            if not isinstance(exc, Exception):
                raise
            raise LLMProviderError(record["error"]) from exc
        finally:
            record["wall_seconds"] = time.monotonic() - started
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            if self.output_dir:
                append_jsonl(self.output_dir / "requests.jsonl", record)
            print(
                json.dumps(
                    {
                        k: record.get(k)
                        for k in (
                            "bot",
                            "game_id",
                            "ply",
                            "response",
                            "wall_seconds",
                            "error",
                        )
                    }
                ),
                flush=True,
            )

    async def transport(self, prompt, timeout_s):
        with tempfile.TemporaryDirectory(prefix="chess-subscription-") as tmp:
            return await execute(self.command(Path(tmp)), prompt, tmp, timeout_s)
