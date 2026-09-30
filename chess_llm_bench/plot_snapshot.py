"""Render a compact, reproducible static summary of an exported snapshot."""

import argparse
import json
from pathlib import Path


def plot_snapshot(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "chess-pilot",
            "font.family": "DejaVu Sans",
        }
    )
    directory = Path(directory)
    models = json.loads((directory / "summary.json").read_text())["models"]
    names = {
        "gpt-6-astra": "GPT-6 Astra",
        "gpt-6-sol": "GPT-6 Sol",
        "gpt-6-luna": "GPT-6 Luna",
        "claude-opus-5-5": "Claude Opus 5.5",
        "claude-sonnet-5-5": "Claude Sonnet 5.5",
        "gemini-3.1-pro": "Gemini 3.1 Pro",
        "gemini-3.8-flash": "Gemini 3.8 Flash",
    }
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    fig.set_facecolor("#f4f2eb")
    for ax in (left, right):
        ax.set_facecolor("#f4f2eb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dfe3d8", linewidth=0.6)
        ax.invert_yaxis()
        ax.tick_params(axis="both", labelsize=9)
    offset = [0] * len(models)
    for field, label, color in [
        ("wins", "Win", "#315b42"),
        ("draws", "Draw", "#aaa078"),
        ("losses", "Loss", "#b8785f"),
        ("incomplete", "Unfinished", "#a8b1ba"),
    ]:
        values = [m[field] for m in models]
        left.barh(
            range(len(models)),
            values,
            left=offset,
            label=label,
            color=color,
            height=0.58,
        )
        offset = [a + b for a, b in zip(offset, values)]
    left.set_yticks(
        range(len(models)), [names.get(m["model"], m["model"]) for m in models]
    )
    left.set_xticks([0, 1, 2])
    left.set_xlim(0, 2.15)
    left.set_xlabel("Recorded games")
    left.set_title("Observed outcomes", loc="left", fontweight="bold", pad=18)
    left.legend(
        loc="upper left",
        bbox_to_anchor=(-0.05, -0.16),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    values = [m["acpl"] or 0 for m in models]
    right.barh(range(len(models)), values, color="#80936b", height=0.58)
    right.set_yticks(range(len(models)), [""] * len(models))
    right.set_xlabel("Mean centipawn loss · lower is better")
    right.set_title(
        "Move quality, with sample coverage", loc="left", fontweight="bold", pad=18
    )
    right.set_xlim(0, max(values) * 1.45)
    for i, m in enumerate(models):
        right.text(
            values[i] + 2,
            i,
            f"{values[i]:.1f} · n={m['cp_samples']}",
            va="center",
            fontsize=9,
        )
    fig.suptitle(
        "Chess Observatory · September 2026 pilot",
        x=0.02,
        ha="left",
        fontsize=18,
        color="#193d31",
    )
    fig.text(
        0.02,
        0.02,
        "Two scheduled games per model; one paired opening. Mate lines excluded from CP means. Gemini Pro: one blocked game, one unplayed.",
        fontsize=8,
        color="#516253",
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.84, bottom=0.24, wspace=0.17)
    fig.savefig(directory / "overview.svg", metadata={"Date": None})
    svg = directory / "overview.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(directory / "overview.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot_snapshot(parser.parse_args().directory)


if __name__ == "__main__":
    main()
