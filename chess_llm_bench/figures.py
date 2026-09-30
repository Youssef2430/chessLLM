"""Export standalone, interactive research figures from recorded benchmark data.

Install the optional charts extra. No model calls; no remote chart assets.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.svg

from .subscription_run import enriched_requests, read_jsonl
from .llm.subscription import request_interrupted

LABELS = {
    "gpt-6-astra": "GPT-6 Astra",
    "gpt-6-sol": "GPT-6 Sol",
    "gpt-6-luna": "GPT-6 Luna",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
}
COLORS = ["#275dad", "#547b9f", "#83a4c0", "#ae6c20", "#d4a052", "#6e7748", "#a4aa7b"]


def build(directories, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requests, games, quality, statuses = [], [], [], []
    seen = set()
    for directory in map(Path, directories):
        rr = enriched_requests(directory)
        models = {r["bot"] for r in rr}
        if seen & models:
            raise ValueError(
                "Do not merge multiple runs of the same model without selecting a comparable cohort"
            )
        seen |= models
        requests.extend(rr)
        games.extend(read_jsonl(directory / "games.jsonl"))
        quality.extend(read_jsonl(directory / "analysis.jsonl"))
        statuses.extend(read_jsonl(directory / "status.jsonl"))
    names = [m for m in LABELS if m in seen]
    lookup = {r["request_id"]: r for r in requests}
    labels = [LABELS[n] for n in names]
    figure = make_subplots(
        rows=4,
        cols=2,
        horizontal_spacing=0.19,
        vertical_spacing=0.095,
        subplot_titles=[
            "Request latency · seconds",
            "Game outcomes · counts",
            "Move quality · centipawn loss",
            "Output tokens per request",
            "Decision time and score loss",
            "Mean score loss by game phase",
            "Response contract · counts",
            "Input tokens · reported totals",
        ],
    )
    for index, name in enumerate(names):
        rr = [r for r in requests if r["bot"] == name]
        qq = [r for r in quality if r["bot"] == name]
        label, color = LABELS[name], COLORS[index % len(COLORS)]
        figure.add_trace(
            go.Box(
                x=[r["wall_seconds"] for r in rr],
                name=label,
                marker_color=color,
                boxpoints="outliers",
                showlegend=False,
                hovertemplate=label + "<br>%{x:.2f} s<extra></extra>",
            ),
            row=1,
            col=1,
        )
        cp = [r["centipawn_loss"] for r in qq if r["centipawn_loss"] is not None]
        figure.add_trace(
            go.Box(
                x=cp,
                name=label,
                marker_color=color,
                boxpoints="outliers",
                showlegend=False,
                hovertemplate=label
                + f"<br>n={len(cp)}<br>"
                + "%{x:.0f} cp<extra></extra>",
            ),
            row=2,
            col=1,
        )
        tokens = [
            r["output_tokens_total"] for r in rr if r["output_tokens_total"] is not None
        ]
        figure.add_trace(
            go.Box(
                x=tokens,
                name=label,
                marker_color=color,
                boxpoints="outliers",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        if not tokens:
            figure.add_trace(
                go.Scatter(
                    x=[0],
                    y=[label],
                    text=["unreported"],
                    mode="text",
                    textposition="middle right",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=2,
            )
        figure.add_trace(
            go.Scatter(
                x=[lookup[r["request_id"]]["wall_seconds"] for r in qq],
                y=[r["expected_score_loss"] * 100 for r in qq],
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, symbol=index, opacity=0.7),
                text=[
                    f"{label} · game {r['game_id']} · ply {r['ply']} · {r['move_san']}"
                    for r in qq
                ],
                hovertemplate="%{text}<br>%{x:.2f} s<br>%{y:.1f} percentage points<extra></extra>",
            ),
            row=3,
            col=1,
        )
    categories = {"Win": [], "Draw": [], "Loss": [], "Aborted": []}
    for name in names:
        gg = [g for g in games if g["bot"] == name]
        wins = sum(
            (g["result"] == "1-0" and g["color_llm_white"])
            or (g["result"] == "0-1" and not g["color_llm_white"])
            for g in gg
        )
        draws = sum(g["result"] == "1/2-1/2" for g in gg)
        aborted = sum(g["result"] == "*" for g in gg)
        for k, v in zip(
            categories, (wins, draws, len(gg) - wins - draws - aborted, aborted)
        ):
            categories[k].append(v)
    for (key, vals), color in zip(
        categories.items(), ("#275dad", "#c4c9d2", "#b77a35", "#5d6470")
    ):
        figure.add_trace(
            go.Bar(
                y=labels,
                x=vals,
                name=key,
                orientation="h",
                marker_color=color,
                showlegend=False,
                hovertemplate=key + ": %{x}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    phases = ["opening", "middlegame", "endgame"]
    z, counts = [], []
    for name in names:
        values, nn = [], []
        for phase in phases:
            data = [
                r["expected_score_loss"]
                for r in quality
                if r["bot"] == name and r["phase"] == phase
            ]
            values.append(100 * sum(data) / len(data) if data else None)
            nn.append(len(data))
        z.append(values)
        counts.append(nn)
    figure.add_trace(
        go.Heatmap(
            z=z,
            x=phases,
            y=labels,
            customdata=counts,
            zmin=0,
            zmax=100,
            colorscale=[[0, "#f2f5f8"], [1, "#275dad"]],
            showscale=False,
            hovertemplate="%{y} · %{x}<br>%{z:.1f} percentage points<br>n=%{customdata}<extra></extra>",
        ),
        row=3,
        col=2,
    )
    for key, color in (
        ("Legal", "#275dad"),
        ("Invalid output", "#b77a35"),
        ("Service error", "#5d6470"),
        ("Interrupted", "#a4aa7b"),
    ):
        values = []
        for name in names:
            rr = [r for r in requests if r["bot"] == name]
            values.append(
                sum(
                    (
                        r.get("legal") is True
                        if key == "Legal"
                        else (
                            r.get("legal") is False
                            if key == "Invalid output"
                            else (
                                request_interrupted(r)
                                if key == "Interrupted"
                                else bool(r.get("error")) and not request_interrupted(r)
                            )
                        )
                    )
                    for r in rr
                )
            )
        figure.add_trace(
            go.Bar(
                y=labels,
                x=values,
                name=key,
                orientation="h",
                marker_color=color,
                showlegend=False,
                hovertemplate=key + ": %{x}<extra></extra>",
            ),
            row=4,
            col=1,
        )
    token_totals = []
    for name in names:
        values = [
            r["input_tokens_total"]
            for r in requests
            if r["bot"] == name and r["input_tokens_total"] is not None
        ]
        token_totals.append(sum(values) if values else None)
    figure.add_trace(
        go.Bar(
            y=labels,
            x=token_totals,
            orientation="h",
            marker_color="#275dad",
            showlegend=False,
        ),
        row=4,
        col=2,
    )
    missing_labels = [
        label for label, total in zip(labels, token_totals) if total is None
    ]
    figure.add_trace(
        go.Scatter(
            x=[0] * len(missing_labels),
            y=missing_labels,
            text=["unreported"] * len(missing_labels),
            mode="text",
            textposition="middle right",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=4,
        col=2,
    )
    figure.update_layout(
        height=1800,
        width=1250,
        template="plotly_white",
        barmode="stack",
        font=dict(family="Arial, sans-serif", size=12, color="#273345"),
        title=dict(
            text="Chess LLM benchmark · observed decisions", font_size=26, y=0.99
        ),
        margin=dict(l=150, r=45, t=150, b=180),
        legend=dict(orientation="h", y=-0.04, x=0),
    )
    for row in (1, 2, 4):
        for col in (1, 2):
            figure.update_xaxes(rangemode="tozero", row=row, col=col)
            figure.update_yaxes(
                categoryorder="array", categoryarray=labels[::-1], row=row, col=col
            )
    figure.update_xaxes(
        title_text="Request time (seconds, excluding queue)", row=3, col=1
    )
    figure.update_yaxes(
        title_text="Estimated score loss (percentage points)",
        range=[0, 100],
        row=3,
        col=1,
    )
    figure.update_yaxes(categoryorder="array", categoryarray=labels[::-1], row=3, col=2)
    figure.update_annotations(font_size=16)
    finished = {
        s["model"]: s["status"] for s in statuses if s["status"] != "unavailable"
    }
    running = any(finished.get(n) == "running" for n in names)
    subtitle = f"{'IN PROGRESS' if running else 'RECORDED SNAPSHOT'} · {len(requests)} requests · {len(games)} recorded games · {len(quality)} analyzed legal moves"
    figure.add_annotation(
        text=subtitle,
        xref="paper",
        yref="paper",
        x=0,
        y=1.065,
        showarrow=False,
        xanchor="left",
    )
    footnote = (
        "Game colors: blue=win, light gray=draw, gold=loss, dark gray=aborted. Invalid output counts as a loss.<br>"
        "Missing metrics are omitted, not zero. Antigravity token usage is unreported. Mate lines are excluded from centipawn loss.<br>"
        "Stockfish estimates use 50,000 nodes per search. Phase is a heuristic. Small, unfinished samples do not establish model rankings."
    )
    figure.add_annotation(
        text=footnote,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.075,
        showarrow=False,
        xanchor="left",
        align="left",
        font_size=11,
    )
    figure.write_html(
        output / "benchmark-figures.html",
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "svg"},
        },
    )
    figure.write_json(output / "benchmark-figures.plotly.json")

    # Select one game at a time; every model shares a white-perspective [-20,20]
    # pawn scale. Mate lines stay visible in hover text but not fabricated as cp.
    timeline = go.Figure()
    pairs = sorted({(r["bot"], r["game_id"]) for r in quality})
    for i, (name, gid) in enumerate(pairs):
        rows = sorted(
            [r for r in quality if r["bot"] == name and r["game_id"] == gid],
            key=lambda r: r["ply"],
        )
        timeline.add_trace(
            go.Scatter(
                x=[r["ply"] for r in rows],
                y=[
                    (
                        None
                        if r["chosen"]["cp"] is None
                        else r["chosen"]["cp"]
                        / 100
                        * (1 if r["side_to_move"] == "white" else -1)
                    )
                    for r in rows
                ],
                mode="lines+markers",
                visible=i == 0,
                name=f"{LABELS[name]} · game {gid}",
                text=[
                    f"{r['move_san']} · mate={r['chosen']['mate']} · CP loss={r['centipawn_loss']}"
                    for r in rows
                ],
                hovertemplate="Ply %{x}: %{text}<br>%{y:.2f} pawns (White)<extra></extra>",
            )
        )
    buttons = [
        dict(
            label=f"{LABELS[n]} · game {g}",
            method="update",
            args=[{"visible": [j == i for j in range(len(pairs))]}],
        )
        for i, (n, g) in enumerate(pairs)
    ]
    timeline.update_layout(
        template="plotly_white",
        height=650,
        title="Position evaluation after each model decision",
        xaxis_title="Game ply",
        yaxis_title="Pawns · positive favors White",
        yaxis=dict(range=[-20, 20]),
        updatemenus=[dict(buttons=buttons, x=0, y=1.16)],
        margin=dict(t=140),
        showlegend=False,
    )
    timeline.write_html(
        output / "game-evaluations.html",
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    diagrams = output / "positions"
    diagrams.mkdir(exist_ok=True)
    notable = sorted(quality, key=lambda r: r["expected_score_loss"], reverse=True)[:12]
    for row in notable:
        board = chess.Board(row["fen"])
        played = chess.Move.from_uci(row["move_uci"])
        best = chess.Move.from_uci(row["best"]["pv"][0])
        svg = chess.svg.board(
            board,
            arrows=[
                chess.svg.Arrow(
                    played.from_square, played.to_square, color="#bb793acc"
                ),
                chess.svg.Arrow(best.from_square, best.to_square, color="#275dadcc"),
            ],
            orientation=board.turn,
            size=480,
        )
        (diagrams / f"{row['request_id']}.svg").write_text(svg)
    (output / "figure-data.json").write_text(
        json.dumps(
            dict(
                generated_at=datetime.now(timezone.utc).isoformat(),
                source_directories=[str(Path(d).resolve()) for d in directories],
                requests=requests,
                games=games,
                analysis=quality,
                statuses=statuses,
                notable_positions=notable,
            ),
            indent=2,
        )
    )
    print(
        f"Saved {output.resolve()} · {len(requests)} requests / {len(quality)} analyzed decisions"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build(args.directories, args.output_dir)


if __name__ == "__main__":
    main()
