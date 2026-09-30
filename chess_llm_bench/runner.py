"""Protocol 3: resumable, subscription-only chess benchmark.

python -m chess_llm_bench.runner --output-dir runs/improved --detach
python -m chess_llm_bench.runner --output-dir runs/improved --resume --detach
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn

from .core.engine import ChessEngine
from .core.journal import RunLease, append_row, atomic_json, latest, read_rows
from .core.models import BotSpec, Config
from .core.openings import OpeningBook
from .llm.antigravity_acp import ACPClient, AntigravityProvider, discover_installation
from .llm.client import InvalidMoveError, LLMProviderError
from .llm.decision import DecisionClient, error_category, utcnow
from .llm.subscription import SubscriptionProvider, execute
from .subscription_run import MODELS

DEFAULT_MODELS = [m for m in MODELS if m != "gpt-oss-120b"]


@dataclass
class RunConfig:
    games: int = 2
    max_plies: int = 200
    seed: int = 42
    effort: str = "high"
    timeout: float = 180
    attempts: int = 3
    structured: bool = True
    prompt_style: str = "checklist"
    think_time: float = 0.3
    opponent_elo: int | None = None
    models: list | None = None
    offline: bool = False
    codex_concurrency: int = 3

    def __post_init__(self):
        import math

        if self.games < 2 or self.games % 2:
            raise ValueError("games must be positive and even")
        if self.max_plies < 8:
            raise ValueError("max_plies must be at least 8")
        if not 1 <= self.attempts <= 3:
            raise ValueError("attempts must be between 1 and 3")
        if (
            type(self.codex_concurrency) is not int
            or not 1 <= self.codex_concurrency <= 3
        ):
            raise ValueError("codex_concurrency must be an integer between 1 and 3")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(self.think_time) or self.think_time <= 0:
            raise ValueError("think_time must be finite and positive")
        if self.effort not in {"low", "medium", "high"}:
            raise ValueError("invalid effort")
        if self.prompt_style not in {"minimal", "checklist"}:
            raise ValueError("invalid prompt style")
        if self.models is None:
            self.models = ["random-baseline"] if self.offline else DEFAULT_MODELS.copy()
        if len(self.models) != len(set(self.models)):
            raise ValueError("Duplicate models")
        if self.offline and self.models != ["random-baseline"]:
            raise ValueError("Offline mode only runs the random baseline")
        if not self.offline and any(m not in MODELS for m in self.models):
            raise ValueError("Unknown model")


def request_gates(config, providers):
    gates = {}
    for provider in providers:
        limit = config.codex_concurrency if provider == "codex" else 1
        gate = asyncio.Semaphore(limit)
        gate.request_limit = limit
        gates[provider] = gate
    return gates


def schedule(config):
    book = OpeningBook(config.seed)
    result = []
    for pair in range(config.games // 2):
        eco, name, moves = book.get_random_opening()
        for white in (True, False):
            result.append(
                dict(
                    game_id=str(len(result) + 1),
                    pair_id=pair + 1,
                    eco=eco,
                    opening=name,
                    opening_moves=moves,
                    white=white,
                    seed=config.seed + len(result),
                )
            )
    return result


class OfflineProvider(SubscriptionProvider):
    def __init__(self, spec, seed):
        super().__init__(spec)
        self.rng = random.Random(seed)

    async def transport(self, prompt, timeout_s):
        legal = prompt.split("Legal choices (UCI = SAN): ")[-1].split("\n")[0]
        choices = re.findall(r"([a-h][1-8][a-h][1-8][qrbn]?) =", legal)
        if not choices:
            choices = re.findall(
                r"\b[a-h][1-8][a-h][1-8][qrbn]?\b",
                prompt.split("Legal moves (UCI): ")[-1].split("\n")[0],
            )
        return (
            0,
            json.dumps(dict(type="result", result=self.rng.choice(choices), usage={})),
            "",
        )


async def preflight(directory, config):
    if config.offline:
        return {"random": dict(available=True, version="python random baseline")}, {}
    statuses, catalog = {}, {}
    for provider in sorted({MODELS[m] for m in config.models}):
        if provider == "antigravity":
            acp = ACPClient()
            try:
                if not discover_installation():
                    raise RuntimeError(
                        "Antigravity personal OAuth installation missing"
                    )
                info = await acp.start()
                catalog = await acp.new_session()
                catalog.pop("sessionId", None)
                statuses[provider] = dict(
                    available=True, version=info.get("agentInfo", {}).get("version")
                )
            except Exception as exc:
                statuses[provider] = dict(available=False, reason=str(exc))
            finally:
                await acp.close()
        else:
            try:
                if not shutil.which(provider):
                    raise RuntimeError("CLI not installed")
                cmd = (
                    ["codex", "login", "status"]
                    if provider == "codex"
                    else ["claude", "auth", "status"]
                )
                code, out, err = await execute(cmd, "", directory, 20)
                if provider == "codex":
                    valid = code == 0 and "using ChatGPT" in out + err
                else:
                    auth = json.loads(out)
                    valid = (
                        code == 0
                        and auth.get("loggedIn")
                        and auth.get("authMethod") == "claude.ai"
                        and auth.get("apiProvider") == "firstParty"
                    )
                if not valid:
                    raise RuntimeError(
                        "Subscription auth not confirmed; API fallback disabled"
                    )
                _, version, _ = await execute(
                    [provider, "--version"], "", directory, 20
                )
                statuses[provider] = dict(available=True, version=version.strip())
            except Exception as exc:
                statuses[provider] = dict(available=False, reason=str(exc))
    return statuses, catalog


def board_from_state(state):
    board = chess.Board()
    for move in state["history"]:
        board.push_uci(move)
    if state.get("fen") and board.fen() != state["fen"]:
        raise ValueError("Checkpoint history/FEN mismatch")
    return board


def recover_plies(directory, name, state):
    board = board_from_state(state)
    rows = latest(read_rows(directory / "plies.jsonl"), ("bot", "game_id", "ply"))
    for row in sorted(
        (
            r
            for r in rows
            if r["bot"] == name
            and r["game_id"] == state["game_id"]
            and r["ply"] > board.ply()
        ),
        key=lambda r: r["ply"],
    ):
        if row["ply"] != board.ply() + 1 or row["fen"] != board.fen():
            raise ValueError("Move journal is inconsistent with its checkpoint")
        board.push_uci(row["move_uci"])
        if row["fen_after"] != board.fen():
            raise ValueError("Move journal has an invalid resulting position")
        state["active_seconds"] = state.get("active_seconds", 0) + row.get(
            "wall_seconds", 0
        )
    state.update(history=[m.uci() for m in board.move_stack], fen=board.fen())
    return board


def save_game(
    directory, name, fixture, state, board, config, result, termination, error=None
):
    game = chess.pgn.Game()
    opponent = f"Stockfish ({config.opponent_elo})"
    game.headers.update(
        Event="Chess LLM · protocol 3",
        White=name if fixture["white"] else opponent,
        Black=opponent if fixture["white"] else name,
        Result=result,
        Round=fixture["game_id"],
        Opening=fixture["opening"],
        ECO=fixture["eco"],
        Protocol="3-subscription",
        ReasoningEffort=config.effort,
        Termination=termination,
        Seed=str(fixture["seed"]),
    )
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    path = directory / name / f"game-{fixture['game_id']}.pgn"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(str(game) + "\n")
    temp.replace(path)
    requests = [
        r
        for r in read_rows(directory / "requests.jsonl")
        if r["bot"] == name and r["game_id"] == fixture["game_id"]
    ]
    record = dict(
        bot=name,
        game_id=fixture["game_id"],
        pair_id=fixture["pair_id"],
        elo=config.opponent_elo,
        color_llm_white=fixture["white"],
        result=result,
        termination=termination,
        error=error,
        ply_count=board.ply(),
        path=str(path),
        timestamp=utcnow(),
        opening=fixture["opening"],
        seed=fixture["seed"],
        game_duration=state.get("active_seconds", 0),
        llm_requests=len(requests),
        llm_moves=sum(bool(r.get("legal")) for r in requests),
        format_recoveries=sum(bool(r.get("format_recovered")) for r in requests),
        retry_requests=sum(r.get("attempt", 1) > 1 for r in requests),
    )
    append_row(directory / "games.jsonl", record)
    return record


async def run_model(
    directory, name, config, fixtures, gate, engine_path, clients, catalog, statuses
):
    kind = "random" if config.offline else MODELS[name]
    provider = (
        OfflineProvider(BotSpec(kind, name, name), config.seed)
        if config.offline
        else (AntigravityProvider if kind == "antigravity" else SubscriptionProvider)(
            BotSpec(kind, name, name)
        )
    )
    provider.reasoning_effort = config.effort
    checkpoint = directory / "state" / f"{name}.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    existing = latest(read_rows(directory / "games.jsonl"), ("bot", "game_id"))
    finished = {
        g["game_id"]
        for g in existing
        if g["bot"] == name
        and g["termination"] not in {"provider_error", "interrupted"}
    }
    if len(finished) == config.games:
        statuses[name] = dict(status="finished", games=len(finished))
        return
    if (
        state
        and state.get("phase") == "blocked"
        and error_category(state.get("error", "")) in {"tool_use", "model_mismatch"}
    ):
        statuses[name] = dict(
            status="blocked",
            games=len(finished),
            game_id=state["game_id"],
            error=state["error"],
        )
        return
    if not clients[kind]["available"]:
        statuses[name] = dict(status="unavailable", reason=clients[kind].get("reason"))
        return
    if kind == "antigravity":
        native = {
            m["modelId"] for m in catalog.get("models", {}).get("availableModels", [])
        }
        if provider.native_model not in native:
            statuses[name] = dict(
                status="unavailable",
                reason="Requested model/effort not in authenticated catalog",
            )
            return
    engine_config = Config(
        fixed_opponent_elo=config.opponent_elo, think_time=config.think_time
    )
    engine = ChessEngine(engine_path, engine_config)
    decision = DecisionClient(
        provider,
        directory,
        gate,
        config.attempts,
        config.timeout,
        config.structured,
        config.prompt_style,
    )
    try:
        await engine.start()
        await engine.configure_elo(config.opponent_elo)
        for fixture in fixtures:
            if fixture["game_id"] in finished:
                continue
            if not state or state.get("game_id") != fixture["game_id"]:
                state = dict(
                    bot=name,
                    game_id=fixture["game_id"],
                    history=fixture["opening_moves"].copy(),
                    active_seconds=0,
                    phase="playing",
                )
            board = recover_plies(directory, name, state)
            state["phase"] = "playing"
            statuses[name] = dict(
                status="running",
                games=len(finished),
                game_id=fixture["game_id"],
                ply=board.ply(),
                color="white" if fixture["white"] else "black",
            )
            atomic_json(checkpoint, state)
            result, termination, error = None, None, None
            while (
                not board.is_game_over(claim_draw=True)
                and board.ply() < config.max_plies
            ):
                started = time.monotonic()
                model_turn = board.turn == fixture["white"]
                if model_turn:
                    try:
                        move = await decision.choose(board, fixture["game_id"])
                    except InvalidMoveError as exc:
                        state["active_seconds"] += time.monotonic() - started
                        result = "0-1" if fixture["white"] else "1-0"
                        termination, error = "invalid_move", str(exc)
                        break
                    except LLMProviderError as exc:
                        state["active_seconds"] += time.monotonic() - started
                        state["phase"] = "blocked"
                        state["error"] = str(exc)
                        atomic_json(checkpoint, state)
                        save_game(
                            directory,
                            name,
                            fixture,
                            state,
                            board,
                            config,
                            "*",
                            "provider_error",
                            str(exc),
                        )
                        statuses[name] = dict(
                            status="blocked",
                            games=len(finished),
                            game_id=fixture["game_id"],
                            error=str(exc),
                        )
                        return
                else:
                    move = await engine.get_move(board)
                before = board.fen()
                entry = dict(
                    bot=name,
                    game_id=fixture["game_id"],
                    ply=board.ply() + 1,
                    player=name if model_turn else "Stockfish",
                    fen=before,
                    move_uci=move.uci(),
                    move_san=board.san(move),
                    capture=board.is_capture(move),
                    castling=board.is_castling(move),
                    gives_check=board.gives_check(move),
                    promotion=move.promotion,
                    legal_move_count=board.legal_moves.count(),
                    wall_seconds=time.monotonic() - started,
                )
                if model_turn:
                    entry["request_id"] = decision.last_decision["request_id"]
                board.push(move)
                entry["fen_after"] = board.fen()
                # Checkpoint is authoritative for board state. Stable ply keys
                # make a replay after interrupted writes deduplicable.
                append_row(directory / "plies.jsonl", entry)
                state.update(
                    history=[m.uci() for m in board.move_stack],
                    fen=board.fen(),
                    active_seconds=state.get("active_seconds", 0)
                    + time.monotonic()
                    - started,
                )
                atomic_json(checkpoint, state)
                statuses[name]["ply"] = board.ply()
            if not result:
                outcome = board.outcome(claim_draw=True)
                result = outcome.result() if outcome else "*"
                termination = (
                    outcome.termination.name.lower() if outcome else "max_plies"
                )
            save_game(
                directory,
                name,
                fixture,
                state,
                board,
                config,
                result,
                termination,
                error,
            )
            finished.add(fixture["game_id"])
            state = None
            checkpoint.unlink(missing_ok=True)
        statuses[name] = dict(status="finished", games=len(finished))
    except asyncio.CancelledError:
        if state:
            state["phase"] = "interrupted"
            atomic_json(checkpoint, state)
        statuses[name] = dict(status="interrupted", games=len(finished))
        raise
    except Exception as exc:
        statuses[name] = dict(status="failed", games=len(finished), error=str(exc))
    finally:
        await provider.close()
        await engine.stop()


async def run(directory, config, resume=False):
    directory = Path(directory).resolve()
    if resume:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("protocol") != "3-subscription":
            raise ValueError("Only protocol 3 runs can be resumed")
        if manifest.get("invalidated_reason"):
            raise ValueError(
                "Invalidated experiments cannot be resumed; create a clean cohort"
            )
        # Older manifests used a shared single slot. Preserve that unless an
        # explicit, recorded amendment changes the concurrency setting.
        manifest["settings"].setdefault("codex_concurrency", 1)
        config = RunConfig(**manifest["settings"])
    else:
        directory.mkdir(parents=True, exist_ok=False)
        manifest = dict(
            protocol="3-subscription",
            schema_version=3,
            started_at=utcnow(),
            settings=asdict(config),
            requested_models=config.models,
            platform=platform.platform(),
            python=platform.python_version(),
            billing_route="offline" if config.offline else "subscription_cli",
            api_key_fallback=False,
            output_token_cap=None,
            temperature=None,
            analysis_assistance=False,
        )
    with RunLease(directory):
        engine_path = shutil.which("stockfish")
        if not engine_path:
            raise RuntimeError("Stockfish is required")
        probe = ChessEngine(engine_path, Config())
        await probe.start()
        try:
            native = probe._supported_elo_range
            config.opponent_elo = config.opponent_elo or native[0]
            if not native[0] <= config.opponent_elo <= native[1]:
                raise ValueError("Opponent Elo outside native engine range")
            engine_info = dict(
                name=probe._engine_name,
                path=engine_path,
                native_elo_range=native,
                sha256=hashlib.sha256(Path(engine_path).read_bytes()).hexdigest(),
            )
        finally:
            await probe.stop()
        if resume and manifest["engine"]["sha256"] != engine_info["sha256"]:
            raise ValueError("Engine binary changed; start a new cohort")
        manifest.update(
            settings=asdict(config),
            engine=engine_info,
            config=dict(
                max_games=config.games,
                max_plies=config.max_plies,
                seed=config.seed,
                reasoning_effort=config.effort,
                fixed_opponent_elo=config.opponent_elo,
                think_time=config.think_time,
                codex_concurrency=config.codex_concurrency,
            ),
        )
        manifest["schedule"] = manifest.get("schedule") or schedule(config)
        source_hashes = {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in Path(__file__).parent.rglob("*.py")
        }
        manifest.setdefault("source_hashes", source_hashes)
        manifest.setdefault("executions", []).append(
            dict(
                started_at=utcnow(),
                pid=os.getpid(),
                resume=resume,
                source_hashes=source_hashes,
            )
        )
        manifest.pop("finished_at", None)
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(
            directory / "run_status.json",
            dict(
                pid=os.getpid(),
                updated_at=utcnow(),
                status="running",
                models={m: dict(status="preflight") for m in config.models},
            ),
        )
        clients, catalog = await preflight(directory, config)
        manifest.update(clients=clients, antigravity_catalog=catalog)
        atomic_json(directory / "manifest.json", manifest)
        # Preserve uncertain in-flight calls rather than pretending they never happened.
        known = {r["request_id"] for r in read_rows(directory / "requests.jsonl")}
        for p in (
            (directory / "pending").glob("*.json")
            if (directory / "pending").exists()
            else []
        ):
            row = json.loads(p.read_text())
            if row["request_id"] not in known:
                row.update(
                    error="Interrupted before response was recorded; usage unknown",
                    error_category="interrupted",
                    legal=None,
                    wall_seconds=None,
                    finished_at=utcnow(),
                )
                append_row(directory / "requests.jsonl", row)
            p.unlink()
        statuses = {m: dict(status="queued", games=0) for m in config.models}
        gates = request_gates(config, clients)
        status_path = directory / "run_status.json"
        started = utcnow()

        def heartbeat(status="running"):
            atomic_json(
                status_path,
                dict(
                    pid=os.getpid(),
                    started_at=started,
                    updated_at=utcnow(),
                    status=status,
                    models=statuses,
                ),
            )

        async def ticker():
            while True:
                heartbeat()
                await asyncio.sleep(3)

        timer = asyncio.create_task(ticker())
        tasks = [
            asyncio.create_task(
                run_model(
                    directory,
                    m,
                    config,
                    manifest["schedule"],
                    gates["random" if config.offline else MODELS[m]],
                    engine_path,
                    clients,
                    catalog,
                    statuses,
                )
            )
            for m in config.models
        ]
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: [task.cancel() for task in tasks])
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(config.models, results):
                if isinstance(result, Exception):
                    statuses[name] = dict(status="failed", error=str(result))
        finally:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
            final = (
                "finished"
                if all(
                    v["status"] in {"finished", "unavailable"}
                    for v in statuses.values()
                )
                else "incomplete"
            )
            heartbeat(final)
            for model, status in statuses.items():
                append_row(
                    directory / "status.jsonl",
                    dict(model=model, **status, timestamp=utcnow()),
                )
            manifest["last_stopped_at"] = utcnow()
            if final == "finished":
                manifest["finished_at"] = utcnow()
            atomic_json(directory / "manifest.json", manifest)
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--offline", action="store_true", default=None)
    parser.add_argument("--games", type=int)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--effort", choices=["low", "medium", "high"])
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--attempts", type=int)
    parser.add_argument("--codex-concurrency", type=int)
    parser.add_argument("--prompt-style", choices=["minimal", "checklist"])
    parser.add_argument(
        "--no-structured", dest="structured", action="store_false", default=None
    )
    parser.add_argument("--models", nargs="+", choices=list(MODELS))
    args = parser.parse_args()
    values = {
        k: v
        for k, v in vars(args).items()
        if k in RunConfig.__dataclass_fields__ and v is not None
    }
    if args.resume:
        saved = json.loads((Path(args.output_dir) / "manifest.json").read_text())[
            "settings"
        ]
        if any(saved[k] != v for k, v in values.items()):
            parser.error("Resume cannot change experiment settings; create another run")
    config = RunConfig(**values)
    if args.detach:
        logs = Path("logs")
        logs.mkdir(exist_ok=True)
        log = logs / ("benchmark-" + str(time.time_ns()) + ".log")
        with log.open("w") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "chess_llm_bench.runner",
                    *[a for a in sys.argv[1:] if a != "--detach"],
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        print(
            json.dumps(
                dict(
                    pid=process.pid,
                    log=str(log.resolve()),
                    directory=str(Path(args.output_dir).resolve()),
                )
            )
        )
        return
    asyncio.run(run(args.output_dir, config, args.resume))


if __name__ == "__main__":
    main()
