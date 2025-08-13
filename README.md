# 🏆 Chess LLM Benchmark

**A comprehensive tool for evaluating Large Language Models through chess gameplay against Stockfish at various ELO ratings.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991.svg)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude--3.5-8B5A2B.svg)](https://anthropic.com/)
[![Google](https://img.shields.io/badge/Google-Gemini--1.5-4285F4.svg)](https://ai.google.dev/)

## ✨ Features

- 🤖 **Multi-Provider Support**: OpenAI GPT models, Anthropic Claude, Google Gemini
- 🎯 **ELO Ladder System**: Bots climb ratings by defeating Stockfish at increasing difficulty levels
- 🎨 **Beautiful Terminal UI**: Real-time chess board visualization with rich formatting
- 📊 **Comprehensive Analytics**: Detailed statistics, win rates, and performance tracking
- ⏱️ **Move Timing Analysis**: Track response times and speed metrics for each model
- 🚫 **Illegal Move Detection**: Monitor rule understanding and invalid move attempts
- 🎮 **Preset Configurations**: Ready-made model lineups for different use cases
- 💾 **Game Recording**: All games saved in PGN format with full analysis
- ⚡ **Concurrent Testing**: Run multiple bots simultaneously for efficient benchmarking
- 💰 **Budget Tracking**: Real-time cost monitoring with spending limits and alerts
- 🏆 **Performance Ranking**: Historical leaderboards and model comparison system

## 🎰 Run Demo
![Current State](assets/demo.gif)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-llm-benchmark.git
cd chess-llm-benchmark

# Install dependencies
pip install -r requirements.txt

# Install Stockfish (required chess engine)
# macOS:
brew install stockfish

# Ubuntu/Debian:
sudo apt-get install stockfish

# Windows:
choco install stockfish

# Optional: Install human-like engines for more realistic opponents
# Maia (most human-like, manual installation required):
#   Visit: https://github.com/CSSLab/maia-chess

# LCZero (neural network engine):
# macOS:
brew install lc0

# Ubuntu/Debian:
sudo apt-get install lc0

# Or use the installation helper:
python install_human_engines.py --engine all
```


# Or install specific human engines:
python install_human_engines.py --engine maia  # Most human-like (recommended)
python install_human_engines.py --engine lczero  # Neural network based
```

### Set Up API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
GEMINI_API_KEY="your-gemini-api-key"
```

### Run Your First Benchmark

```bash
# Quick demo with random bots
python main.py --demo

# Premium models from all providers
python main.py --preset premium

# Use human-like engines for more realistic opponents
python main.py --preset premium --use-human-engine

# Budget-friendly models
python main.py --preset budget

# Track spending with $5 budget limit
python main.py --preset premium --budget-limit 5.0 --show-costs

# OpenAI models only
python main.py --preset openai
```

## 🤖 Available Models

### 📡 OpenAI Models
- **GPT-4o** - Latest flagship model with enhanced reasoning
- **GPT-4o Mini** - Faster, cost-effective variant
- **GPT-4 Turbo** - High-performance with large context
- **GPT-3.5 Turbo** - Fast and efficient legacy model

### 📡 Anthropic Models
- **Claude 3.5 Sonnet** - Most intelligent Claude model
- **Claude 3.5 Haiku** - Fast and efficient
- **Claude 3 Opus** - Most capable legacy model
- **Claude 3 Haiku** - Budget-friendly option

### 📡 Google Gemini Models
- **Gemini 1.5 Pro** - Most capable with 1M token context
- **Gemini 1.5 Flash** - Fast and efficient
- **Gemini 1.0 Pro** - Legacy model

## 🎯 Preset Configurations

| Preset | Description | Models |
|--------|-------------|---------|
| `premium` | Top-tier models from each provider | GPT-4o, GPT-4o Mini, Claude 3.5 Sonnet, Claude 3.5 Haiku, Gemini 1.5 Pro, Gemini 1.5 Flash |
| `budget` | Cost-effective models with good performance | GPT-4o Mini, GPT-3.5 Turbo, Claude 3.5 Haiku, Claude 3 Haiku, Gemini 1.5 Flash |
| `recommended` | All recommended models across providers | All ⭐ starred models |
| `openai` | OpenAI's best models | GPT-4o, GPT-4o Mini |
| `anthropic` | Anthropic's best models | Claude 3.5 Sonnet, Claude 3.5 Haiku |
| `gemini` | Google's best models | Gemini 1.5 Pro, Gemini 1.5 Flash |

## 🧠 Human-like Chess Engines

**NEW FEATURE**: Use human-like chess engines instead of traditional Stockfish for more realistic opponents!

Traditional chess engines like Stockfish play perfectly mechanical chess, which may not reflect how human players actually perform. Human-like engines provide more realistic opponents that make human-like decisions and mistakes.

### Available Human Engines

| Engine | Description | Human-likeness | Installation |
|--------|-------------|----------------|--------------|
| **🧠 Maia** | Neural network trained on human games | ⭐⭐⭐⭐⭐ Most realistic | [Maia Chess](https://github.com/CSSLab/maia-chess) |
| **♟️ LCZero** | Neural network with human-like settings | ⭐⭐⭐⭐ Very good | `brew install lc0` (macOS) |
| **🤖 Human Stockfish** | Traditional Stockfish with human settings | ⭐⭐⭐ Good fallback | Built-in with Stockfish |

### Quick Start with Human Engines

```bash
# Use human-like engines (auto-detected)
python main.py --preset premium --use-human-engine

# Specify engine type explicitly
python main.py --preset budget --use-human-engine --human-engine-type maia

# Use Leela Chess Zero for human-like play
python main.py --preset openai --use-human-engine --human-engine-type lczero

# Compare human vs traditional engines
python examples/human_engine_demo.py --mode comparison
```

### Installation Helper

```bash
# Install human engines automatically
python install_human_engines.py --engine all

# Install specific engine
python install_human_engines.py --engine maia

# Test your installation
python test_human_engines.py --demo
```

### Why Use Human Engines?

- **🎯 More Realistic Assessment**: Human engines make mistakes and play like real players
- **📈 Better ELO Scaling**: Strength scales more naturally with human-like characteristics
- **🎲 Move Variation**: Unlike mechanical engines, they show variation in move selection
- **🧠 Human-like Thinking**: Trained on human games, not perfect engine analysis
- **⚖️ Fair Evaluation**: Better reflects how LLMs would perform against human opponents

### Comparison: Human vs Traditional

```bash
# See the difference in action
python examples/human_engine_demo.py --mode comparison

# Results show human engines vary moves 50-75% more than traditional engines
# providing much more realistic opposition for LLM evaluation
```

## 📋 Usage Examples

### Basic Usage

```bash
# List all available models
python main.py --list-models

# List all presets
python main.py --list-presets

# Run with specific preset
python main.py --preset premium

# Custom bot lineup
python main.py --bots "openai:gpt-4o:GPT-4o,anthropic:claude-3-5-sonnet-20241022:Claude-3.5-Sonnet"
```

### Advanced Configuration

```bash
# Custom ELO range
python main.py --preset budget --start-elo 800 --max-elo 1600 --elo-step 200

# Faster games with shorter thinking time
python main.py --preset openai --think-time 0.1 --max-plies 100

# High-temperature creative play
python main.py --bots "openai:gpt-4o:Creative-GPT" --llm-temperature 0.8

# Cost-controlled testing
python main.py --preset budget --budget-limit 2.0 --show-costs
```

### Demo Modes

```bash
# Robot battle visualization
python main.py --robot-demo

# Quick robot demo
python main.py --quick-robot-demo

# Standard demo with random bots
python main.py --demo
```

## 🎮 Live Dashboard

The benchmark features a beautiful real-time terminal dashboard with comprehensive performance tracking:

### 📊 Enhanced Performance Metrics

**NEW**: The dashboard now tracks detailed performance metrics including:

- **⏱️ Move Timing**: Real-time average move generation time per model
- **🚫 Illegal Moves**: Count of invalid move attempts (indicates rule understanding)
- **📈 Live Updates**: Real-time progress tracking during gameplay with immediate state updates
- **💰 Cost Analysis**: Time-based cost estimation for API usage
- **🔄 Responsive Dashboard**: Live display updates every move with no lag or freezing

Example enhanced summary table:
```
┃ Bot               ┃ Status      ┃ Max ELO ┃ Games ┃ Win Rate ┃ Record    ┃ Avg Time ┃ Illegal Moves ┃
┃ Claude 3.5 Sonnet ┃ ✅ Finished ┃    1400 ┃     8 ┃    62.5% ┃ 5W-0D-3L  ┃    2.34s ┃           12  ┃
┃ GPT-4o            ┃ 🤔 Thinking ┃    1200 ┃     6 ┃    50.0% ┃ 3W-0D-3L  ┃    1.87s ┃            8  ┃
```

### 🎯 Benefits of Enhanced Metrics

**Move Timing Analysis** helps you:
- **Cost Optimization**: Estimate API costs based on response times
- **Performance Comparison**: Compare model speed vs accuracy trade-offs
- **Efficiency Ranking**: Identify fastest models for time-critical applications

**Illegal Move Detection** reveals:
- **Rule Understanding**: How well models comprehend chess rules
- **Model Quality**: Lower illegal moves indicate better training
- **Reliability Assessment**: Models with fewer errors are more dependable

```
╭─────── 🏆 Chess LLM ELO Ladder Benchmark ────────╮
│  ╭──────────────── 📊 Summary ────────────────╮  │
│  │ ┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓ │  │
│  │ ┃ Bot              ┃ Max ELO ┃ Win Rate  ┃ │  │
│  │ ┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━┩ │  │
│  │ │ GPT-4o           │    1400 │    67.5%  │ │  │
│  │ │ Claude 3.5       │    1200 │    55.2%  │ │  │
│  │ │ Gemini 1.5 Pro   │    1100 │    48.3%  │ │  │
│  │ └──────────────────┴─────────┴───────────┘ │  │
│  ╰────────────────────────────────────────────╯  │
│  ╭──────────────── GPT-4o ────────────────────╮  │
│  │ Ladder: 600 → 800 → 1000 → 1200 → 1400     │  │
│  │ ╭─ Move 24 | Last: Nf6+ ──╮Status: vs 1400│  │
│  │ │  a b c d e f g h       │Current ELO:    │  │
│  │ │8 ♜   ♝ ♛ ♚ ♝   ♜ 8     │1400           │  │
│  │ │7 ♟ ♟   ♟   ♟ ♟ ♟ 7     │Color: White   │  │
│  │ │6       ♞           6     │Moves: 47      │  │
│  │ │5         ♙         5     │               │  │
│  │ │4   ♙     ♙         4     │Win Rate: 67.5%│  │
│  │ │3     ♘   ♗         3     │Games: 8       │  │
│  │ │2 ♙     ♙   ♙ ♙ ♙   2     │Record: 5W-1D- │  │
│  │ │1 ♖ ♘ ♗ ♕ ♔     ♖   1     │2L            │  │
│  │ ╰─────────────────────────╯               │  │
│  ╰────────────────────────────────────────────╯  │
╰──────────────────────────────────────────────────╯
```

## 📊 Results and Analysis

### Output Structure

```
runs/
└── YYYYMMDD_HHMMSS/
    ├── summary.json          # Benchmark results summary
    ├── config.json           # Configuration used
    ├── games/                # Individual game PGN files
    │   ├── gpt4o_vs_stockfish_600.pgn
    │   └── claude_vs_stockfish_800.pgn
    └── analysis/             # Performance analysis
        └── statistics.json
```

### Performance Metrics

Each bot is evaluated on:
- **Maximum ELO Reached**: Highest Stockfish rating defeated
- **Win Rate**: Percentage of games won
- **Average Game Length**: Moves per game
- **Move Timing Analysis**: Average time per move and response speed metrics
- **Move Quality Assessment**: Illegal move attempts and rule comprehension scoring
- **Opening Performance**: Success with different openings
- **Endgame Strength**: Performance in late-game positions
- **Cost Efficiency**: Performance per dollar spent
- **Consistency Score**: Standard deviation of ELO performance

## ⚙️ Configuration Options

### Command Line Arguments

```bash
# Bot Configuration
--bots "provider:model:name,..."     # Custom bot specification
--preset PRESET                      # Use predefined bot lineup

# Budget & Cost Tracking
--budget-limit AMOUNT                # Set spending limit in USD
--show-costs                         # Display real-time cost tracking

# Analysis & Ranking
--leaderboard [N]                    # Show top N models (default: 20)
--provider-stats                     # Compare provider performance
--analyze-model MODEL_ID             # Deep analysis of specific model

# ELO Ladder Settings
--start-elo ELO                      # Starting ELO rating (default: 600)
--elo-step STEP                      # ELO increment per rung (default: 100)
--max-elo ELO                        # Maximum ELO to attempt (default: 2400)

# Game Settings
--think-time SECONDS                 # Thinking time per move (default: 0.3)
--max-plies COUNT                    # Maximum moves per game (default: 300)
--escalate-on MODE                   # When to advance: "always" or "on_win"

# LLM Settings
--llm-timeout SECONDS                # LLM response timeout (default: 20.0)
--llm-temperature TEMP               # Sampling temperature (default: 0.0)

# Output Settings
--output-dir PATH                    # Results directory (default: "runs")
--save-pgn                          # Save games in PGN format

# Engine Settings
--stockfish PATH                     # Custom Stockfish executable path

# Display Options
--verbose                           # Verbose logging
--debug                             # Debug mode
--refresh-rate HZ                   # UI refresh rate (default: 6)
```

### Bot Specification Format

```
provider:model:name

Examples:
openai:gpt-4o:GPT-4o
anthropic:claude-3-5-sonnet-20241022:Claude-3.5-Sonnet
gemini:gemini-1.5-pro:Gemini-1.5-Pro
random::Baseline
```

## 🔬 Advanced Features

### Custom Prompting

The tool uses optimized prompts for chess move generation:

```python
def _create_chess_prompt(self, board: chess.Board) -> str:
    """Create a standardized chess prompt for the LLM."""
    legal_moves = " ".join(move.uci() for move in board.legal_moves)
    color = "White" if board.turn == chess.WHITE else "Black"

    prompt = (
        "You are a strong chess player. Given the position and legal moves, "
        "choose the best move and respond with ONLY the UCI notation.\n\n"
        f"Position (FEN): {board.fen()}\n"
        f"Side to move: {color}\n"
        f"Legal moves: {legal_moves}\n\n"
        "Your response must be exactly one legal UCI move."
    )
    return prompt
```

### Error Handling and Fallbacks

- **Timeout Protection**: LLM requests timeout after 20 seconds
- **Move Validation**: All moves validated against legal move list
- **Fallback System**: Random legal moves when LLM fails
- **Robust Parsing**: Handles various move notation formats

### Concurrent Execution

Multiple bots run simultaneously with proper async handling:

```python
# Run all bots concurrently on the ELO ladder
tasks = []
for bot_spec in bot_specs:
    task = asyncio.create_task(
        self._run_bot_ladder(bot_spec, engine, output_dir)
    )
    tasks.append(task)

results = await asyncio.gather(*tasks, return_exceptions=True)
```

## 📈 Benchmarking Best Practices

### Model Selection
- Start with **recommended models** for reliable results
- Use **budget preset** for cost-effective testing
- Try **premium preset** for cutting-edge performance

### ELO Configuration
- Begin with **600-1000 ELO** range for initial assessment
- Use **100 ELO steps** for balanced progression
- Extend to **2400 ELO** for comprehensive evaluation

### Statistical Significance
- Run multiple iterations for reliable statistics
- Consider different random seeds
- Analyze both win rates and maximum ELO reached

## 🛠️ Development

### Project Structure

```
chess-llm-benchmark/
├── chess_llm_bench/          # Main package
│   ├── core/                 # Core game logic
│   ├── llm/                  # LLM providers and models
│   ├── ui/                   # Terminal UI components
│   └── cli.py                # Command-line interface
├── tests/                    # Test suite
├── runs/                     # Benchmark results
├── main.py                   # Entry point
└── requirements.txt          # Dependencies
```

### Adding New Providers

1. Create a new provider class in `chess_llm_bench/llm/client.py`
2. Inherit from `BaseLLMProvider`
3. Implement `generate_move()` method
4. Register in `LLMClient.PROVIDERS`
5. Add model definitions to `chess_llm_bench/llm/models.py`

### Running Tests

```bash
# Run built-in tests
python main.py --test

# Run with pytest
pytest tests/ -v

# Type checking
mypy chess_llm_bench/

# Code formatting
black chess_llm_bench/
isort chess_llm_bench/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stockfish Team** - For the amazing chess engine
- **OpenAI, Anthropic, Google** - For providing powerful language models
- **python-chess** - For excellent chess programming library
- **Rich** - For beautiful terminal formatting

## 💰 Budget Tracking & Cost Management

### Real-Time Cost Monitoring
```bash
# Set $10 budget limit with cost display
python main.py --preset premium --budget-limit 10.0 --show-costs

# Track costs for specific models
python main.py --bots "openai:gpt-4o-mini:Test" --budget-limit 1.0
```

### Cost Management Features
- **Real-time tracking**: See costs as they accumulate
- **Budget limits**: Set spending caps with automatic warnings
- **Cost breakdown**: Detailed analysis by provider, model, and bot
- **Efficiency metrics**: Cost per game, cost per ELO point
- **Historical tracking**: Track spending trends over time

### Pricing Information
Current approximate costs (per 1K tokens):
- **GPT-4o Mini**: $0.00015 input, $0.0006 output (~$0.004/game)
- **Claude 3.5 Haiku**: $0.0008 input, $0.004 output (~$0.019/game)
- **Gemini 1.5 Flash**: $0.000075 input, $0.0003 output (~$0.002/game)
- **GPT-4o**: $0.0025 input, $0.01 output (~$0.065/game)

## 🏆 Ranking & Analysis System

### Leaderboard Commands
```bash
# Show top 10 performing models
python main.py --leaderboard 10

# Compare all providers
python main.py --provider-stats

# Deep-dive analysis of specific model
python main.py --analyze-model openai:gpt-4o
```

### Performance Tracking
- **Historical leaderboards**: Track best performers over time
- **Efficiency rankings**: Best performance per dollar spent
- **Trend analysis**: Identify improving vs declining models
- **Statistical insights**: Mean, median, consistency scores
- **Provider comparisons**: Cross-provider performance analysis

### Database Storage
All results are automatically stored in SQLite database for:
- Historical analysis and trends
- Performance comparisons across runs
- Cost tracking and efficiency analysis
- Statistical significance testing

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Youssef2430/chessLLM/issues)
- 📧 **Email**: youssefchouay30@gmail.com

---

**Made with ♟️ and 🤖 by the Youssef**
