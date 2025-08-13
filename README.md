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
- 🎮 **Preset Configurations**: Ready-made model lineups for different use cases
- 💾 **Game Recording**: All games saved in PGN format with full analysis
- ⚡ **Concurrent Testing**: Run multiple bots simultaneously for efficient benchmarking

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

# Budget-friendly models
python main.py --preset budget

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

The benchmark features a beautiful real-time terminal dashboard:

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
- **Opening Performance**: Success with different openings
- **Endgame Strength**: Performance in late-game positions

## ⚙️ Configuration Options

### Command Line Arguments

```bash
# Bot Configuration
--bots "provider:model:name,..."     # Custom bot specification
--preset PRESET                      # Use predefined bot lineup

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

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/chess-llm-benchmark/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/chess-llm-benchmark/discussions)  
- 📧 **Email**: your.email@example.com

---

**Made with ♟️ and 🤖 by the Chess LLM Benchmark Team**