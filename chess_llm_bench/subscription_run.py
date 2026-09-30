"""Run the explicitly selected current models through signed-in subscription CLIs.

Usage: python -m chess_llm_bench.subscription_run --output-dir runs/subscriptions-DATE
This entry point intentionally does not use the API CLI, dotenv, or API pricing.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .core.engine import ChessEngine
from .core.journal import read_rows, latest, atomic_json
from .core.game import GameRunner, LadderRunner
from .core.models import BotSpec, Config, LadderStats, LiveState
from .llm.client import LLMClient
from .llm.subscription import (
    SubscriptionProvider,
    append_jsonl,
    execute,
    normalized_tokens,
    parse_events,
    request_interrupted,
)
from .llm.antigravity_acp import ACPClient, AntigravityProvider, discover_installation

MODELS = {
    "gpt-6-astra": "codex",
    "gpt-6-sol": "codex",
    "gpt-6-luna": "codex",
    "claude-opus-5-5": "claude",
    "claude-sonnet-5-5": "claude",
    "gemini-3.1-pro": "antigravity",
    "gemini-3.8-flash": "antigravity",
    "gpt-oss-120b": "antigravity",
}


class SubscriptionGameRunner(GameRunner):
    def _create_pgn_header(self, elo, llm_white):
        game = super()._create_pgn_header(elo, llm_white)
        game.headers["Event"] = "Subscription chess benchmark"
        game.headers["Transport"] = "subscription_cli"
        return game

    async def _save_pgn(self, game, output_dir, elo):
        # The CLI does not expose the same output-token cap as the raw API.
        game.headers["MaxOutputTokens"] = "client-controlled"
        return await super()._save_pgn(game, output_dir, elo)

    def _execute_move(self, board, move, pgn_node, state, player_name):
        record = dict(
            bot=self.llm.spec.name,
            game_id=str(self._game_counter),
            ply=board.ply() + 1,
            player=player_name,
            fen=board.fen(),
            move_uci=move.uci(),
            move_san=board.san(move),
            capture=board.is_capture(move),
            gives_check=board.gives_check(move),
            castling=board.is_castling(move),
            promotion=move.promotion,
            legal_move_count=board.legal_moves.count(),
        )
        super()._execute_move(board, move, pgn_node, state, player_name)
        record["fen_after"] = board.fen()
        append_jsonl(self.llm.provider.output_dir / "plies.jsonl", record)


def read_jsonl(path):
    rows = read_rows(path)
    if Path(path).name == "games.jsonl" and rows and all("game_id" in r for r in rows):
        return latest(rows, ("bot", "game_id"))
    if Path(path).name == "plies.jsonl":
        return latest(rows, ("bot", "game_id", "ply"))
    return rows


def enriched_requests(directory):
    rows = read_jsonl(directory / "requests.jsonl")
    for row in rows:
        trace = directory / row.get("trace_path", "missing")
        if trace.is_file():
            raw = json.loads(trace.read_text())
            parsed = parse_events(
                row["provider"], raw["stdout"], raw["stderr"], row["requested_model"]
            )
            for key in (
                "client_timing",
                "reported_model",
                "model_verification",
                "rate_limits",
            ):
                row[key] = parsed.get(key)
        row.update(normalized_tokens(row["provider"], row.get("usage", {})))
    return rows


def export_data(directory):
    requests = enriched_requests(directory)
    games = read_jsonl(directory / "games.jsonl")
    for stem in ("requests", "games", "plies"):
        rows = (
            requests if stem == "requests" else read_jsonl(directory / f"{stem}.jsonl")
        )
        if not rows:
            continue
        flat = []
        for row in rows:
            result = {}
            for key, value in row.items():
                if key == "usage":
                    for k, v in value.items():
                        result[f"usage_{k}"] = (
                            json.dumps(v) if isinstance(v, (dict, list)) else v
                        )
                else:
                    result[key] = (
                        json.dumps(value) if isinstance(value, (dict, list)) else value
                    )
            flat.append(result)
        fields = list(dict.fromkeys(k for row in flat for k in row))
        with (directory / f"{stem}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
    summary = {}
    for name in sorted({r["bot"] for r in requests} | {g["bot"] for g in games}):
        rr = [r for r in requests if r["bot"] == name]
        gg = [g for g in games if g["bot"] == name]
        completed = [g for g in gg if g["result"] != "*"]
        wins = sum(
            (g["result"] == "1-0" and g["color_llm_white"])
            or (g["result"] == "0-1" and not g["color_llm_white"])
            for g in completed
        )
        draws = sum(g["result"] == "1/2-1/2" for g in completed)
        timings = [r["wall_seconds"] for r in rr if r.get("wall_seconds") is not None]
        summary[name] = dict(
            requests=len(rr),
            legal_moves=sum(r.get("legal") is True for r in rr),
            provider_errors=sum(
                bool(r.get("error")) and not request_interrupted(r) for r in rr
            ),
            interrupted_requests=sum(request_interrupted(r) for r in rr),
            completed_games=len(completed),
            aborted_games=len(gg) - len(completed),
            wins=wins,
            draws=draws,
            losses=len(completed) - wins - draws,
            median_request_seconds=(statistics.median(timings) if timings else None),
            input_tokens=(
                sum(r["input_tokens_total"] or 0 for r in rr)
                if any(r["tokens_reported"] for r in rr)
                else None
            ),
            output_tokens=(
                sum(r["output_tokens_total"] or 0 for r in rr)
                if any(r["tokens_reported"] for r in rr)
                else None
            ),
            requests_with_reported_tokens=sum(r["tokens_reported"] for r in rr),
            incremental_charge_usd=None,
        )
    atomic_json(directory / "summary.json", summary)
    return summary


async def run(args):
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = Config(
        max_games=args.games,
        max_plies=args.max_plies,
        seed=args.seed,
        llm_timeout=args.timeout,
        reasoning_effort=args.effort,
        fixed_opponent_elo=1320,
        think_time=0.3,
        output_dir=str(directory),
    )
    manifest = dict(
        schema_version=1,
        protocol="2-subscription",
        started_at=datetime.now(timezone.utc).isoformat(),
        requested_models=args.models,
        config=asdict(config),
        python=platform.python_version(),
        platform=platform.platform(),
        billing_route="subscription_cli",
        api_key_fallback=False,
        output_token_cap=None,
        limitations=[
            "Client harnesses differ; this is a subscription-product comparison.",
            "No model temperature or exact output cap is guaranteed by these CLIs.",
            "CLI dollar estimates are API equivalents, not subscription charges.",
            "Subscription quota and actual additional charges may not be exposed.",
            "Engine UCI_Elo is an opponent setting, not a measured rating of the LLM.",
        ],
    )
    statuses = {}
    acp_catalog = {}
    # Read-only auth checks; save only status, never email, IDs, credentials or env.
    for provider in set(MODELS[m] for m in args.models):
        if provider == "antigravity":
            if discover_installation():
                acp = ACPClient()
                try:
                    info = await acp.start()
                    acp_catalog = await acp.new_session()
                    acp_catalog.pop("sessionId", None)
                    statuses[provider] = dict(
                        available=True, version=info.get("agentInfo", {}).get("version")
                    )
                except Exception as exc:
                    statuses[provider] = dict(available=False, reason=str(exc))
                finally:
                    await acp.close()
            else:
                statuses[provider] = {
                    "available": False,
                    "reason": "Antigravity CLI/auth not available; no API fallback",
                }
            continue
        if not shutil.which(provider):
            statuses[provider] = {"available": False, "reason": "CLI not installed"}
            continue
        command = (
            ["codex", "login", "status"]
            if provider == "codex"
            else ["claude", "auth", "status"]
        )
        code, out, err = await execute(command, "", str(directory), 20)
        if provider == "codex":
            valid = code == 0 and "using ChatGPT" in out + err
        else:
            try:
                auth = json.loads(out)
                valid = (
                    code == 0
                    and auth.get("authMethod") == "claude.ai"
                    and auth.get("loggedIn")
                )
            except json.JSONDecodeError:
                valid = False
        _, version, _ = await execute([provider, "--version"], "", str(directory), 20)
        statuses[provider] = dict(
            available=bool(valid),
            version=version.strip(),
            reason=None if valid else "Subscription authentication not confirmed",
        )
    manifest["clients"] = statuses
    manifest["antigravity_catalog"] = acp_catalog
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    gates = {p: asyncio.Semaphore(1) for p in ("codex", "claude", "antigravity")}
    for provider in gates:
        LLMClient.register_provider(
            provider,
            AntigravityProvider if provider == "antigravity" else SubscriptionProvider,
        )
    engine_path = shutil.which("stockfish")
    if not engine_path:
        raise RuntimeError("Stockfish is required for this schedule")

    async def one(model):
        provider = MODELS[model]
        if not statuses[provider]["available"]:
            append_jsonl(
                directory / "status.jsonl",
                dict(model=model, status="unavailable", **statuses[provider]),
            )
            return
        if provider == "antigravity":
            available = {
                m["modelId"]
                for m in acp_catalog.get("models", {}).get("availableModels", [])
            }
            if AntigravityProvider.native_models.get(model) not in available:
                append_jsonl(
                    directory / "status.jsonl",
                    dict(
                        model=model,
                        status="unavailable",
                        reason="Not advertised by authenticated ACP model catalog",
                    ),
                )
                return
        spec = BotSpec(provider, model, model)
        client = LLMClient(spec, reasoning_effort=args.effort)
        client.provider.output_dir = directory
        client.provider.gate = gates[provider]
        engine = ChessEngine(engine_path, config)
        try:
            await engine.start()
            minimum = engine._supported_elo_range[0]
            if minimum != config.fixed_opponent_elo:
                raise RuntimeError(
                    f"Native minimum {minimum} differs from configured {config.fixed_opponent_elo}"
                )
            append_jsonl(
                directory / "status.jsonl",
                dict(
                    model=model,
                    status="running",
                    engine=engine._engine_name,
                    native_elo_range=engine._supported_elo_range,
                ),
            )
            runner = SubscriptionGameRunner(client, engine, config)
            stats = LadderStats()
            await LadderRunner(runner, config).run_ladder(
                directory, LiveState(model), stats
            )
            append_jsonl(
                directory / "status.jsonl",
                dict(
                    model=model,
                    status="finished" if len(stats.games) == args.games else "stopped",
                    games=len(stats.games),
                    completed=stats.total_games,
                    error=stats.games[-1].error if stats.games else None,
                ),
            )
        except Exception as exc:
            append_jsonl(
                directory / "status.jsonl",
                dict(model=model, status="failed", error=str(exc)),
            )
        finally:
            await client.close()
            await engine.stop()
            export_data(directory)

    try:
        await asyncio.gather(*(one(m) for m in args.models))
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(export_data(directory), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
