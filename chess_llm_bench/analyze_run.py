"""Incremental, offline Stockfish analysis of recorded subscription decisions.

No provider requests. Fixed node budgets, one engine thread, fresh hash per
decision. Scores are from the mover's perspective; mate scores stay separate.
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import chess
import chess.engine

from .llm.subscription import append_jsonl
from .subscription_run import read_jsonl, export_data


def score_fields(info, color, ply):
    score = info["score"].pov(color)
    wdl = score.wdl(model="sf", ply=ply)
    return dict(
        cp=score.score(),
        mate=score.mate(),
        expected_score=wdl.expectation(),
        wdl=list(wdl),
        depth=info.get("depth"),
        seldepth=info.get("seldepth"),
        nodes=info.get("nodes"),
        time=info.get("time"),
        pv=[m.uci() for m in info.get("pv", [])],
    )


def position_metrics(board):
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    material = {
        color: sum(
            values[p.piece_type] for p in board.piece_map().values() if p.color == color
        )
        for color in (chess.WHITE, chess.BLACK)
    }
    nonpawn = sum(
        values[p.piece_type]
        for p in board.piece_map().values()
        if p.piece_type != chess.PAWN
    )
    return dict(
        material_white=material[chess.WHITE],
        material_black=material[chess.BLACK],
        material_advantage=(material[board.turn] - material[not board.turn]),
        nonpawn_material=nonpawn,
        phase=(
            "opening"
            if board.ply() <= 20
            else "endgame" if nonpawn <= 26 else "middlegame"
        ),
        pawn_count=len(board.pieces(chess.PAWN, True))
        + len(board.pieces(chess.PAWN, False)),
        attacked_own_pieces=sum(
            board.is_attacked_by(not board.turn, s)
            for s, p in board.piece_map().items()
            if p.color == board.turn
        ),
        legal_captures=len(list(board.generate_legal_captures())),
        legal_checks=sum(board.gives_check(m) for m in board.legal_moves),
    )


def analyze(directory, nodes=50000):
    directory = Path(directory)
    destination = directory / "analysis.jsonl"
    existing = read_jsonl(destination)
    if existing and any(r.get("node_budget") != nodes for r in existing):
        raise ValueError(
            "Existing analysis uses another node budget; use a separate artifact"
        )
    done = {r["request_id"] for r in existing}
    requests = [
        r
        for r in read_jsonl(directory / "requests.jsonl")
        if r.get("legal") and r["request_id"] not in done
    ]
    if requests:
        engine_path = shutil.which("stockfish")
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        try:
            engine.configure({"Threads": 1, "Hash": 32, "UCI_LimitStrength": False})
            binary_hash = hashlib.sha256(Path(engine_path).read_bytes()).hexdigest()
            for row in requests:
                board = chess.Board()
                for uci in row["history_uci"]:
                    board.push_uci(uci)
                if board.fen() != row["fen"]:
                    raise ValueError(f"History/FEN mismatch for {row['request_id']}")
                move = board.parse_uci(row["move_uci"])
                engine.configure({"Clear Hash": None})
                best_info = engine.analyse(
                    board, chess.engine.Limit(nodes=nodes), game=row["request_id"]
                )
                best = score_fields(best_info, board.turn, board.ply())
                engine.configure({"Clear Hash": None})
                chosen_info = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=nodes),
                    root_moves=[move],
                    game=row["request_id"],
                )
                chosen = score_fields(chosen_info, board.turn, board.ply())
                raw_loss = (
                    best["cp"] - chosen["cp"]
                    if best["cp"] is not None and chosen["cp"] is not None
                    else None
                )
                loss = max(0, raw_loss) if raw_loss is not None else None
                expectation_loss = max(
                    0, best["expected_score"] - chosen["expected_score"]
                )
                analysis = dict(
                    schema_version=1,
                    request_id=row["request_id"],
                    bot=row["bot"],
                    game_id=row["game_id"],
                    ply=row["ply"],
                    side_to_move=row["side_to_move"],
                    fen=row["fen"],
                    move_uci=move.uci(),
                    move_san=row["move_san"],
                    node_budget=nodes,
                    engine=engine.id,
                    engine_sha256=binary_hash,
                    best=best,
                    chosen=chosen,
                    best_move_match=best["pv"][0] == move.uci(),
                    centipawn_loss=loss,
                    raw_centipawn_loss=raw_loss,
                    expected_score_loss=expectation_loss,
                    missed_forced_mate=best["mate"] is not None
                    and best["mate"] > 0
                    and not (chosen["mate"] is not None and chosen["mate"] > 0),
                    cp_loss_bucket=(
                        "mate_line"
                        if loss is None
                        else (
                            "under_50"
                            if loss < 50
                            else (
                                "50_to_99"
                                if loss < 100
                                else "100_to_299" if loss < 300 else "300_plus"
                            )
                        )
                    ),
                    **position_metrics(board),
                )
                append_jsonl(destination, analysis)
        finally:
            engine.quit()
    rows = read_jsonl(destination)
    if rows:
        flat = [
            {
                **{k: v for k, v in r.items() if k not in {"best", "chosen", "engine"}},
                **{
                    f"{prefix}_{k}": json.dumps(v) if isinstance(v, list) else v
                    for prefix in ("best", "chosen")
                    for k, v in r[prefix].items()
                },
            }
            for r in rows
        ]
        with (directory / "analysis.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    export_data(directory)
    print(
        f"{directory}: analyzed {len(requests)} new decisions, {len(rows)} total",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--nodes", type=int, default=50000)
    args = parser.parse_args()
    if args.nodes < 1:
        parser.error("--nodes must be positive")
    for directory in args.directories:
        analyze(directory, args.nodes)


if __name__ == "__main__":
    main()
