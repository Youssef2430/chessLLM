# API runner · protocol 2

This is the optional API-billed workflow. It is separate from the published
subscription pilot and does not resume games. Prices below are a historical
planning snapshot checked on 2026-09-29, not a subscription bill.

Install with `python -m pip install -e '.[all]'`, then configure keys using
[.env.example](../.env.example). These keys are never used by protocol 3.

## Start with a free estimate

```sh
python main.py --preset latest --max-games 10 --estimate-cost
python main.py --preset budget --max-games 10 --estimate-cost

# Free end-to-end smoke test
python main.py --bots random::baseline --max-games 2 --max-plies 12
```

`--estimate-cost` performs no API requests and initializes no provider clients.
Its JSON scenarios expose the assumptions and per-model prices. Adjust them with
`--estimate-input-tokens`, `--estimate-output-tokens` (including reasoning), and
`--estimate-moves` (model decisions per game).

## Run a benchmark

These commands incur API charges:

```sh
# Small pilot: one opening, both colors, three inexpensive models
python main.py --preset budget --max-games 2 --budget-limit 5 --show-costs

# Six models, five opening pairs each
python main.py --preset latest --max-games 10 --budget-limit 40 --show-costs

# Same schedule against the engine's advertised minimum UCI_Elo
python main.py --preset budget --max-games 10 --opponent lowest-elo --budget-limit 10

# Specific native Stockfish strength (must be within its advertised range)
python main.py --preset budget --opponent-elo 1600 --max-games 10 --budget-limit 10

# Model chooses a move with local heuristic observations added to the prompt
python main.py --preset budget --use-agent --max-games 2 --budget-limit 5
```

The budget is a conservative **local request guard**, not a guaranteed billing
cap. It reserves the next request's input allowance plus maximum output before
sending it, including concurrent bots' reservations. This can stop a run below
the limit. Provider timeouts/cancellations may still be billed: they retain a
marked upper estimate, and stop that bot. SDK retries are disabled. Unknown model
pricing is an error, never a zero-cost assumption. No API benchmark was run as
part of the September 2026 refresh.

## Model presets and cost scenarios

Catalog and standard paid text rates checked **2026-09-29**. Model availability
for your account is not verified by an offline estimate. Aliases can change;
recorded model IDs and settings must accompany published results.

| Preset | Models |
| --- | --- |
| `latest` | GPT-6 Astra, GPT-6 Luna, Claude Sonnet 5.5, Claude Haiku 4.5, Gemini 3.8 Flash, Gemini 3.5 Flash Lite |
| `budget` | GPT-6 Luna, Claude Haiku 4.5, Gemini 3.5 Flash Lite |
| `legacy` | Original legacy IDs for historical reference; some may be retired |

Ten games per model, 60 model decisions/game, 800 input tokens and 512 total
output/reasoning tokens per decision:

| Model | Input / output USD per million | Scenario USD | High scenario USD |
| --- | ---: | ---: | ---: |
| GPT-6 Astra | 10 / 50 | 20.16 | 118.40 |
| GPT-6 Luna | 0.10 / 0.50 | 0.20 | 1.18 |
| Claude Sonnet 5.5 | 2 / 10 | 4.03 | 23.68 |
| Claude Haiku 4.5 | 1 / 5 | 2.02 | 11.84 |
| Gemini 3.8 Flash | 0.75 / 3.75 | 1.51 | 8.88 |
| Gemini 3.5 Flash Lite | 0.30 / 2.50 | 0.91 | 5.60 |
| **Six-model total (60 games)** | | **28.83** | **169.58** |
| **Budget preset total (30 games)** | | **3.13** | **18.62** |

The high scenario assumes 100 decisions/game, 1,600 input tokens, and the full
2,048-token output cap on every request. It is a sensitivity scenario, not a
statistical confidence interval or guaranteed maximum. All estimates exclude
cache discounts, taxes, credits, currency conversion, and regional premiums.
Gemini 3.8 Flash's current promotional rate ends December 31, 2026. Recheck rates
before later runs. A six-model two-game pilot is about **$5.77**, or **$33.92** in
the high scenario. A three-model budget pilot is about **$0.63–$3.72**.

Sources: [OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing),
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing).

## Rules

- Each opening is played twice, reversing the model's color. Every bot gets the
  same seeded sequence; `--max-games` must be positive and even (default 10).
- The prompt includes FEN, all legal UCI moves, and complete UCI move history,
  including the opening. This measures move selection with legal-move assistance.
- Exactly one legal UCI response is accepted. Invalid, empty, or truncated output
  forfeits the game. There is no repair request, SAN conversion, or random fallback.
- API/engine failures and budget stops are incomplete games (`*`), not losses.
  The bot stops on these failures. Other bots can finish their own schedules.
- Draw claims are applied consistently. Games reaching `--max-plies` are incomplete,
  not draws. Inspect completion rates as well as scores; frequent truncation means
  the schedule needs a longer limit and a revised estimate.
- Score is `(wins + 0.5 × draws) / completed games`. Reports include W/D/L,
  incomplete counts, termination reasons and a descriptive Wilson interval for
  win rate. The interval assumes independent games; opening pairs violate that
  assumption to some degree. It is not a significance test of model rankings.
- Defaults: 200 total plies including openings, 2,048 output tokens per request,
  low reasoning effort where supported, 60-second request timeout, seed 42.
  Reasoning effort levels are not equivalent compute budgets across providers.
  Haiku uses ordinary generation. A short UCI answer can still incur many billed
  reasoning tokens; truncation may require a higher cap and a new estimate.
- `--use-agent` is a separate tool-assisted condition: one request per move with
  local material/position observations and all legal moves. The former agent made
  many calls, double-counted usage, and could select heuristic fallback moves.
  Legacy direct agent classes remain experimental; the CLI uses the v2 wrapper.
- `lowest-elo` queries the installed engine's native UCI_Elo minimum (1320 on the
  machine used for this refresh). No invented 600-Elo skill mapping is used.
  Stockfish version and thinking time still affect results. A seed does not make
  provider sampling or Stockfish search deterministic.

Ten games per model are useful for a pilot, not strong ranking claims. For a
larger study, increase opening pairs, repeat seeds, report incomplete/forfeit
rates, and compare identical opponent, token, timeout, and assistance conditions.

## Results and reproducibility

Each run writes:

```text
runs/<UTC timestamp with microseconds>/
  config.json        # actual configuration, including resolved opponent setting
  environment.json   # Python, SDK versions, engine identity and native Elo range
  summary.json       # outcomes, intervals, timing, errors, per-game records
  budget.json        # tokens, prices, final response, prompt hash, usage source
  games.jsonl        # append-only checkpoints after each game
  run.log
  <bot name>/*.pgn   # seed, opening, protocol, result, termination, duration
```

Provider-reported usage is charged once per request at stored list prices;
OpenAI/Anthropic output counts include reasoning and Gemini thought tokens are
added explicitly. Cache discounts are not applied. Missing usage and failed
requests are clearly marked conservative estimates. Interrupted runs keep
finished-game checkpoints and finalize available usage on orderly cancellation;
a process kill can lose in-memory data. Automatic resume is not implemented.

New runs use `data/results-v2.db`. The old `data/results.db` and saved runs are
preserved. Protocol v1 had fallback moves, mislabeled outcomes, incomplete billing,
and different assistance; its scores are not directly comparable.

```sh
python main.py --leaderboard 10
python main.py --results-db data/results-v2.db --provider-stats
python main.py --analyze-model openai:gpt-6-luna
python main.py --list-models
python main.py --list-presets
python main.py --help
```

The leaderboard shows the latest single run rather than mixing each model's best
or latest result from incompatible conditions. Read its run summary for incomplete
games. Historical provider summaries are descriptive aggregates only.
