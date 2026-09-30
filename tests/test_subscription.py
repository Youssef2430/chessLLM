"""Offline tests: subscription transport must never silently change models/routes."""

import asyncio
import json
import os
import sys

import chess
import pytest

from chess_llm_bench.core.models import BotSpec
from chess_llm_bench.llm.client import LLMProviderError
from chess_llm_bench.llm.subscription import (
    SubscriptionProvider,
    execute,
    parse_events,
    subscription_environment,
    normalized_tokens,
)
from chess_llm_bench.subscription_run import export_data


def test_environment_excludes_all_api_routes(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CODEX_HOME",
    ):
        monkeypatch.setenv(key, "must-not-be-inherited")
    env = subscription_environment()
    assert env["HOME"] == os.environ["HOME"]
    assert "must-not-be-inherited" not in env.values()


def test_token_normalization_preserves_missing_and_cache_semantics():
    assert normalized_tokens("antigravity", {})["input_tokens_total"] is None
    assert (
        normalized_tokens(
            "claude",
            {
                "input_tokens": 2,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 10,
                "output_tokens": 4,
            },
        )["input_tokens_total"]
        == 32
    )
    assert (
        normalized_tokens(
            "codex", {"input_tokens": 32, "cached_input_tokens": 20, "output_tokens": 4}
        )["input_tokens_total"]
        == 32
    )


def test_antigravity_native_variant_is_preserved_and_validated():
    raw = json.dumps({"type": "acp_trace", "native_model": "gemini-3.1-pro-low"})
    r = parse_events("antigravity", raw, "", "gemini-3.1-pro")
    assert r["reported_model"] == "gemini-3.1-pro-low"
    assert r["error"] is None
    assert parse_events("antigravity", raw, "", "gemini-3.8-flash")["error"]


def test_claude_model_mismatch_and_tool_use_rejected():
    data = [
        dict(type="system", subtype="init", model="claude-haiku-4-5"),
        dict(type="result", result="e2e4", usage={}),
    ]
    result = parse_events(
        "claude", "\n".join(map(json.dumps, data)), "", "claude-opus-5-5"
    )
    assert "Model mismatch" in result["error"]
    data[0]["model"] = "claude-opus-5-5"
    data.append(
        dict(
            type="assistant",
            message={"content": [{"type": "tool_use", "name": "Bash"}]},
        )
    )
    result = parse_events(
        "claude", "\n".join(map(json.dumps, data)), "", "claude-opus-5-5"
    )
    assert "Tool use" in result["error"]


def test_codex_missing_reported_model_is_not_fabricated():
    raw = "\n".join(
        map(
            json.dumps,
            [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "e2e4"},
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 100, "output_tokens": 4},
                },
            ],
        )
    )
    result = parse_events("codex", raw, "", "gpt-6-astra")
    assert result["response"] == "e2e4"
    assert result["reported_model"] is None
    assert result["model_verification"] == "requested_only"


@pytest.mark.asyncio
async def test_timeout_kills_child(tmp_path):
    pidfile = tmp_path / "pid"
    script = "import os,time,pathlib;pathlib.Path('pid').write_text(str(os.getpid()));time.sleep(20)"
    with pytest.raises(asyncio.TimeoutError):
        await execute([sys.executable, "-c", script], "", tmp_path, 0.3)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_move_trace_records_illegal_response_without_random_fallback(tmp_path):
    provider = SubscriptionProvider(BotSpec("claude", "claude-opus-5-5", "opus"))
    provider.output_dir = tmp_path

    async def transport(*args):
        return (
            0,
            json.dumps(
                {"type": "result", "result": "a1a8", "usage": {"input_tokens": 5}}
            ),
            "",
        )

    provider.transport = transport
    assert await provider.generate_move(chess.Board()) == "a1a8"
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    assert row["legal"] is False
    assert row["incremental_charge_usd"] is None
    assert row["fen"] == chess.STARTING_FEN
    assert row["legal_move_count"] == 20
    assert row["prompt_sha256"]


@pytest.mark.asyncio
async def test_transport_failure_is_checkpointed(tmp_path):
    provider = SubscriptionProvider(BotSpec("codex", "gpt-6-astra", "astra"))
    provider.output_dir = tmp_path

    async def transport(*args):
        raise asyncio.TimeoutError()

    provider.transport = transport
    with pytest.raises(LLMProviderError):
        await provider.generate_move(chess.Board())
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    assert row["error"] == "TimeoutError"
    assert row["legal"] is None
    summary = export_data(tmp_path)
    assert summary["astra"]["provider_errors"] == 1
    assert summary["astra"]["completed_games"] == 0


def test_claude_disables_tools_but_keeps_subscription_auth(tmp_path):
    command = SubscriptionProvider(
        BotSpec("claude", "claude-sonnet-5-5", "sonnet")
    ).command(tmp_path)
    assert "--bare" not in command  # Bare mode ignores subscription login.
    assert command[command.index("--tools") + 1] == ""
    assert "--safe-mode" in command
    assert "--fallback-model" not in command
