"""Deterministic move recovery without choosing between chess alternatives."""

from dataclasses import dataclass
import json
import re

import chess

UCI = r"[a-h][1-8][a-h][1-8][qrbn]?"
MARKED = re.compile(
    rf"(?:final(?: move)?|best move|move|uci)\s*[:=]\s*`?({UCI})`?\.?", re.I
)


@dataclass(frozen=True)
class MoveResolution:
    move: chess.Move | None
    method: str
    strict_valid: bool
    error: str | None = None


def resolve_move(response: str, board: chess.Board, strict=False) -> MoveResolution:
    text = response.strip()
    candidate, method = None, "invalid"
    if re.fullmatch(UCI, text, re.I):
        candidate, method = text.lower(), "strict"
    elif not strict:
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("move"), str):
            candidate, method = obj["move"].strip().lower(), "json"
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            # A final fenced block may contain exactly one move, not a list of
            # candidates. Never pick the first/last legal token from arbitrary prose.
            if lines and lines[-1] == "```":
                opening = next(
                    (
                        i
                        for i in range(len(lines) - 2, -1, -1)
                        if lines[i].startswith("```")
                    ),
                    None,
                )
                if opening is not None and len(lines) - opening == 3:
                    candidate, method = lines[-2].strip("`"), "code_block"
            elif lines:
                final = lines[-1]
                match = MARKED.fullmatch(final)
                if match:
                    candidate, method = match.group(1).lower(), "marked_final"
                elif re.fullmatch(rf"`?({UCI})`?", final, re.I):
                    candidate, method = final.strip("`").lower(), "final_line"
    if candidate is None or not re.fullmatch(UCI, candidate, re.I):
        return MoveResolution(None, "invalid", False, "missing_or_ambiguous_move")
    move = chess.Move.from_uci(candidate.lower())
    if move not in board.legal_moves:
        return MoveResolution(None, method, False, "illegal_move")
    return MoveResolution(move, method, method == "strict")


def move_schema(board):
    return {
        "type": "object",
        "properties": {
            "move": {"type": "string", "enum": [m.uci() for m in board.legal_moves]}
        },
        "required": ["move"],
        "additionalProperties": False,
    }


def chess_prompt(board, structured=False):
    pieces = []
    for color, label in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        pieces.append(
            label
            + ": "
            + ", ".join(
                f"{p.symbol().upper()}@{chess.square_name(s)}"
                for s, p in sorted(board.piece_map().items())
                if p.color == color
            )
        )
    output = (
        'Return only JSON: {"move":"e2e4"} using your chosen legal move.'
        if structured
        else "Put your chosen UCI move alone on the final line."
    )
    instruction = (
        "Choose the strongest chess move. Privately check checks, captures and threats; "
        "compare candidate moves against the opponent's strongest forcing reply. "
        "Check king safety, hanging pieces, material and any forced mate before deciding. "
        "Do not use tools or external chess engines."
    )
    history = " ".join(m.uci() for m in board.move_stack)
    choices = "; ".join(f"{m.uci()} = {board.san(m)}" for m in board.legal_moves)
    return "\n".join(
        [
            instruction,
            output,
            f"Side to move: {'White' if board.turn else 'Black'}",
            f"FEN: {board.fen()}",
            *pieces,
            f"Board (rank 8 at top):\n{board}",
            f"History UCI: {history}",
            f"Legal choices (UCI = SAN): {choices}",
        ]
    )
