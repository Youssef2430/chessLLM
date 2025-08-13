#!/usr/bin/env python3
"""
LLM Chess ELO Ladder — CLI Benchmark (TUI)
-----------------------------------------
Run multiple LLM "bots" concurrently against a UCI chess engine (Stockfish)
with increasing ELO (e.g., 600 → 700 → 800 …), stopping when a bot loses.

Fixes in this version
---------------------
• Prevents `SystemExit: 2` by providing sensible **defaults** (no required flags).
• Auto-discovers Stockfish via PATH/`STOCKFISH_PATH` and shows a friendly hint if missing.
• Proper Rich Live rendering (no stray prints) for a smooth real‑time dashboard.
• Adds a `--demo` mode and a `--self-test` suite with unit tests.

Quick start (zero-setup demo, no API keys; requires Stockfish installed):
  python llm_chess_bench.py --demo

Or specify bots explicitly:
  python llm_chess_bench.py \
    --bots random::rand1,random::rand2 \
    --start-elo 600 --max-elo 1600 --elo-step 200

OpenAI example (set OPENAI_API_KEY first):
  export OPENAI_API_KEY=sk-...
  python llm_chess_bench.py \
    --bots openai:gpt-4o-mini:4o-mini,openai:gpt-4.1-mini:4.1m \
    --start-elo 600 --elo-step 100 --max-elo 2200

Bots format: "provider:model:name"
  • provider ∈ {openai, random}
  • model: (random: blank) (openai: e.g., gpt-4o-mini)
  • name: friendly display name
Example: random::baseline, openai:gpt-4o-mini:4o-mini

Notes
-----
• Requires Stockfish (any UCI engine with UCI_LimitStrength/UCI_Elo). Install:
  - macOS:  brew install stockfish
  - Ubuntu: sudo apt-get install stockfish
  - Windows: choco install stockfish
• Python deps: python-chess[engine], rich, openai (only if provider=openai)
• PGNs saved under ./runs/<timestamp>/<bot_name>/elo_<ELO>.pgn
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn as chess_pgn
import chess.engine as chess_engine
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Optional import of OpenAI client — only needed if provider=openai
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - import is optional
    OpenAI = None  # type: ignore

console = Console()
MOVE_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


@dataclass
class BotSpec:
    provider: str  # "openai" | "random"
    model: str
    name: str


def parse_bots(spec: str) -> List[BotSpec]:
    """Parse a comma-separated list of provider:model:name specs."""
    bots: List[BotSpec] = []
    if not spec:
        return bots
    for raw in [s.strip() for s in spec.split(",") if s.strip()]:
        parts = raw.split(":")
        if len(parts) == 1:
            provider, model, name = parts[0], "", parts[0]
        elif len(parts) == 2:
            provider, model = parts
            name = model or provider
        else:
            provider, model, name = parts[0], parts[1], ":".join(parts[2:])
        provider = provider.lower()
        if provider not in {"openai", "random"}:
            raise ValueError(f"Unsupported provider: {provider}")
        bots.append(BotSpec(provider=provider, model=model, name=name))
    return bots


class LLMClient:
    def __init__(self, spec: BotSpec):
        self.spec = spec
        self.random = random.Random()
        self._openai_client = None
        if spec.provider == "openai":
            if OpenAI is None:
                raise RuntimeError(
                    "openai package not installed. `pip install openai` or use provider=random"
                )
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY env var is required for provider=openai")
            self._openai_client = OpenAI(api_key=api_key)

    async def pick_move(self, board: chess.Board, temperature: float = 0.0, timeout_s: float = 20.0) -> chess.Move:
        """Return a *legal* move for current board. Fallback to random legal move on failure or timeout."""
        try:
            return await asyncio.wait_for(self._pick_move_inner(board, temperature), timeout=timeout_s)
        except Exception:
            legal = list(board.legal_moves)
            if not legal:
                raise
            return legal[self.random.randrange(len(legal))]

    async def _pick_move_inner(self, board: chess.Board, temperature: float) -> chess.Move:
        if self.spec.provider == "random":
            legal = list(board.legal_moves)
            return legal[self.random.randrange(len(legal))]
        elif self.spec.provider == "openai":
            assert self._openai_client is not None
            # Build a minimal, robust prompt asking for a single UCI move
            legal_moves = " ".join(m.uci() for m in board.legal_moves)
            color = "White" if board.turn == chess.WHITE else "Black"
            prompt = (
                "You are a strong chess assistant. Given the position FEN and LEGAL MOVES (UCI), "
                "choose the best move and reply with ONLY one UCI move like e2e4 or a7a8q.\n\n"
                f"FEN: {board.fen()}\n"
                f"Side to move: {color}\n"
                f"LEGAL MOVES (UCI): {legal_moves}\n"
                "Your reply MUST be exactly one legal UCI move from the list above, nothing else."
            )
            move_txt = await asyncio.to_thread(self._openai_chat_once, prompt, temperature)
            # Extract UCI move
            match = MOVE_RE.search(move_txt or "")
            if not match:
                # Try to parse SAN as a fallback
                try:
                    mv = board.parse_san((move_txt or "").strip())
                    if mv in board.legal_moves:
                        return mv
                except Exception:
                    pass
                # Last resort: random legal move
                legal = list(board.legal_moves)
                return legal[self.random.randrange(len(legal))]
            uci = match.group(1).lower()
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                # Repair under-promotions: convert e7e8=Q style
                try:
                    mv = chess.Move.from_uci(uci.replace("=", ""))
                except Exception:
                    pass
            if mv in board.legal_moves:
                return mv
            # Fallback random
            legal = list(board.legal_moves)
            return legal[self.random.randrange(len(legal))]
        else:
            raise RuntimeError(f"Unsupported provider: {self.spec.provider}")

    def _openai_chat_once(self, prompt: str, temperature: float) -> str:
        assert self._openai_client is not None
        resp = self._openai_client.chat.completions.create(
            model=self.spec.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=8,
            n=1,
        )
        return (resp.choices[0].message.content or "").strip()


@dataclass
class GameRecord:
    elo: int
    color_llm_white: bool
    result: str  # "1-0", "0-1", "1/2-1/2"
    ply_count: int
    path: Path


@dataclass
class LadderStats:
    max_elo_reached: int = 0
    games: List[GameRecord] = field(default_factory=list)
    losses: int = 0
    draws: int = 0
    wins: int = 0


@dataclass
class LiveState:
    title: str
    ladder: List[int] = field(default_factory=list)
    current_elo: int = 0
    status: str = "waiting"
    board_ascii: str = ""
    last_move_uci: str = ""
    color_llm_white: bool = True
    moves_made: int = 0
    final_result: Optional[str] = None


async def play_one_game(
    llm: LLMClient,
    engine: chess_engine.SimpleEngine,
    elo: int,
    think_time_s: float,
    out_dir: Path,
    state: LiveState,
    escalate_on: str = "always",  # or "on_win"
) -> Tuple[str, Path, int, bool]:
    """Play one game at a given engine ELO. Return (result, pgn_path, ply_count, llm_was_white)."""
    # Configure engine strength
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    except chess_engine.EngineError:
        # Some engines require different options (best-effort)
        try:
            engine.configure({"Skill Level": max(0, min(20, (elo - 1000) // 50))})
        except Exception:
            pass

    board = chess.Board()
    llm_white = (len(state.ladder) % 2 == 0)  # alternate colors each rung

    # PGN setup
    game = chess_pgn.Game()
    game.headers["Event"] = "LLM Chess ELO Ladder"
    game.headers["Site"] = "local"
    game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
    game.headers["White"] = llm.spec.name if llm_white else f"Stockfish({elo})"
    game.headers["Black"] = f"Stockfish({elo})" if llm_white else llm.spec.name
    node = game

    state.status = f"vs {elo} (playing...)"
    state.current_elo = elo
    state.color_llm_white = llm_white
    state.board_ascii = str(board)
    state.moves_made = 0

    def push_and_update(move: chess.Move, who: str):
        nonlocal node
        board.push(move)
        node = node.add_variation(move)
        state.board_ascii = str(board)
        state.moves_made += 1
        state.last_move_uci = move.uci()
        state.status = f"vs {elo} (last: {who} {move.uci()})"

    # Play until game over or move cap
    MAX_PLIES = 300  # hard cap to avoid marathons

    while not board.is_game_over() and board.ply() < MAX_PLIES:
        if (board.turn == chess.WHITE and llm_white) or (board.turn == chess.BLACK and not llm_white):
            mv = await llm.pick_move(board)
            push_and_update(mv, who=llm.spec.name)
        else:
            # Engine move
            try:
                res = await asyncio.to_thread(engine.play, board, chess_engine.Limit(time=think_time_s))
                mv = res.move
                push_and_update(mv, who=f"SF{elo}")
            except Exception:
                # Engine hiccup: resign
                break

    result = board.result(claim_draw=True)
    game.headers["Result"] = result

    # Save PGN
    bot_dir = out_dir / llm.spec.name
    bot_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = bot_dir / f"elo_{elo}.pgn"
    with pgn_path.open("w", encoding="utf-8") as f:
        print(game, file=f)

    state.final_result = result
    state.status = f"finished {result} vs {elo}"
    return result, pgn_path, board.ply(), llm_white


def render_dashboard(all_states: Dict[str, LiveState], stats: Dict[str, LadderStats]) -> Panel:
    rows: List[Panel] = []
    for bot_name, st in all_states.items():
        ladder_txt = " → ".join(map(str, st.ladder)) if st.ladder else "—"
        hdr = Text(f"{bot_name}  |  Ladder: {ladder_txt}")
        hdr.stylize("bold")
        tbl = Table.grid(expand=True)
        tbl.add_row(Text(st.board_ascii))
        meta = Text(
            f"status: {st.status}\n"
            f"current ELO: {st.current_elo}  |  color: {'White' if st.color_llm_white else 'Black'}\n"
            f"moves: {st.moves_made}  |  last: {st.last_move_uci or '—'}\n"
        )
        s = stats.get(bot_name)
        if s:
            meta.append(
                f"wins: {s.wins}  draws: {s.draws}  losses: {s.losses}  |  max ELO reached: {s.max_elo_reached}\n"
            )
        tbl.add_row(meta)
        rows.append(Panel(Group(hdr, tbl), title=f"{bot_name}", border_style="cyan"))
    return Panel(Group(*rows), title="LLM Chess ELO Ladder", border_style="magenta")


async def run_ladder_for_bot(
    spec: BotSpec,
    stockfish_path: str,
    start_elo: int,
    max_elo: int,
    elo_step: int,
    think_time_s: float,
    out_root: Path,
    state: LiveState,
    stats: LadderStats,
    escalate_on: str,
):
    llm = LLMClient(spec)
    engine = chess_engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        elo = start_elo
        while elo <= max_elo:
            state.ladder.append(elo)
            res, pgn_path, plies, llm_white = await play_one_game(
                llm=llm,
                engine=engine,
                elo=elo,
                think_time_s=think_time_s,
                out_dir=out_root,
                state=state,
                escalate_on=escalate_on,
            )
            stats.max_elo_reached = max(stats.max_elo_reached, elo)
            if res == "1-0":
                # White won
                if llm_white:
                    stats.wins += 1
                    go_up = True
                else:
                    stats.losses += 1
                    go_up = False
            elif res == "0-1":
                # Black won
                if llm_white:
                    stats.losses += 1
                    go_up = False
                else:
                    stats.wins += 1
                    go_up = True
            else:  # draw
                stats.draws += 1
                go_up = (escalate_on == "always")

            # Stop on loss
            if not go_up:
                break
            elo += elo_step
            state.final_result = None
            state.status = "advancing..."
    finally:
        try:
            engine.quit()
        except Exception:
            pass


async def main_async(args):
    bots = parse_bots(args.bots)
    if not bots:
        # If no bots parsed (empty string), default to a single random bot
        bots = [BotSpec(provider="random", model="", name="random-demo")]

    # Output root directory for PGNs
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = Path("runs") / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    all_states: Dict[str, LiveState] = {b.name: LiveState(title=b.name) for b in bots}
    stats: Dict[str, LadderStats] = {b.name: LadderStats() for b in bots}

    tasks = [
        asyncio.create_task(
            run_ladder_for_bot(
                spec=b,
                stockfish_path=args.stockfish,
                start_elo=args.start_elo,
                max_elo=args.max_elo,
                elo_step=args.elo_step,
                think_time_s=args.think_time,
                out_root=out_root,
                state=all_states[b.name],
                stats=stats[b.name],
                escalate_on=args.escalate_on,
            )
        )
        for b in bots
    ]

    # Live dashboard
    with Live(render_dashboard(all_states, stats), refresh_per_second=6, console=console) as live:
        while any(not t.done() for t in tasks):
            await asyncio.sleep(0.2)
            live.update(render_dashboard(all_states, stats))

    # Final render
    live = render_dashboard(all_states, stats)
    console.print(live)
    console.print(f"PGNs saved under: [bold]{out_root}[/bold]")


def _autodetect_stockfish(cli_path: Optional[str]) -> Optional[str]:
    """Return a usable path to stockfish if found, else None.
    Order: CLI arg → env STOCKFISH_PATH → shutil.which("stockfish")."""
    if cli_path and os.path.exists(cli_path):
        return cli_path
    env_path = os.getenv("STOCKFISH_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    which = shutil.which("stockfish")
    return which


def _friendly_stockfish_hint():
    console.print(
        "[red]Stockfish not found.[/red] Install it and try again:\n"
        "• macOS:  brew install stockfish\n"
        "• Ubuntu: sudo apt-get install stockfish\n"
        "• Windows: choco install stockfish\n"
        "Or set env var STOCKFISH_PATH=/full/path/to/stockfish",
    )


def run_unit_tests() -> int:
    import unittest

    class ParseBotsTests(unittest.TestCase):
        def test_parse_bots_random_and_openai(self):
            bots = parse_bots("random::foo,openai:gpt-4o-mini:bar")
            self.assertEqual(len(bots), 2)
            self.assertEqual(bots[0].provider, "random")
            self.assertEqual(bots[0].name, "foo")
            self.assertEqual(bots[1].provider, "openai")
            self.assertEqual(bots[1].model, "gpt-4o-mini")
            self.assertEqual(bots[1].name, "bar")

        def test_parse_bots_invalid_provider(self):
            with self.assertRaises(ValueError):
                parse_bots("weird::x")

        def test_parse_bots_empty(self):
            self.assertEqual(parse_bots(""), [])

    class MoveRegexTests(unittest.TestCase):
        def test_move_regex_simple(self):
            m = MOVE_RE.search("best: e2e4")
            self.assertIsNotNone(m)
            self.assertEqual(m.group(1), "e2e4")

        def test_move_regex_promotion(self):
            m = MOVE_RE.search("go a7a8q now")
            self.assertIsNotNone(m)
            self.assertEqual(m.group(1), "a7a8q")

        def test_move_regex_no_match(self):
            self.assertIsNone(MOVE_RE.search("hello world"))

    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(ParseBotsTests))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(MoveRegexTests))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description="LLM Chess ELO Ladder CLI benchmark")
    # No positional command required; everything via flags for simplicity
    parser.add_argument(
        "--bots",
        type=str,
        default="random::bot1,random::bot2",
        help=(
            "Comma-separated bots as provider:model:name (e.g., 'openai:gpt-4o-mini:4o-mini,random::baseline'). "
            "Default runs two random bots."
        ),
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to stockfish executable. If omitted, tries STOCKFISH_PATH or PATH lookup.",
    )
    parser.add_argument("--start-elo", type=int, default=600)
    parser.add_argument("--elo-step", type=int, default=100)
    parser.add_argument("--max-elo", type=int, default=2400)
    parser.add_argument("--think-time", type=float, default=0.3, help="Engine time per move (seconds)")
    parser.add_argument(
        "--escalate-on",
        type=str,
        default="always",
        choices=["always", "on_win"],
        help="Advance ELO after any result (always) or only after wins",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a short demo (random bots; 600→800 by 100). Ignores --bots if set.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal unit tests and exit.",
    )

    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_unit_tests())

    # Demo presets
    if args.demo:
        args.bots = "random::demo1,random::demo2"
        args.start_elo = 600
        args.elo_step = 100
        args.max_elo = 800

    # Resolve Stockfish
    sf_path = _autodetect_stockfish(args.stockfish)
    if not sf_path:
        _friendly_stockfish_hint()
        sys.exit(2)
    args.stockfish = sf_path

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user[/bold yellow]")


if __name__ == "__main__":
    main()
