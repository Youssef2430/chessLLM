# 🏆 Chess LLM Benchmark

A comprehensive tool for testing Large Language Models (LLMs) with chess games and assessing their ELO ratings by running them against Stockfish at various difficulty levels.

## 🌟 Features

- **Multi-Provider LLM Support**: OpenAI GPT, Anthropic Claude, and random baseline
- **ELO Ladder System**: Progressive difficulty testing from beginner to master level
- **Real-Time Dashboard**: Live terminal UI showing game states, statistics, and progress
- **Concurrent Testing**: Run multiple bots simultaneously for efficient benchmarking
- **Robust Move Parsing**: Handles various chess notation formats and LLM response styles
- **Comprehensive Logging**: Detailed game records with PGN file generation
- **Flexible Configuration**: Customizable ELO ranges, time controls, and advancement rules

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Stockfish chess engine** installed and accessible

```bash
# Install Stockfish
# macOS
brew install stockfish

# Ubuntu/Debian
sudo apt-get install stockfish

# Windows (with Chocolatey)
choco install stockfish

# Or download from https://stockfishchess.org/
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-llm-bench.git
cd chess-llm-bench

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Demo Run

```bash
# Run a quick demo with random bots (no API keys needed)
python main.py --demo
```

## 📖 Usage Examples

### Basic Usage

```bash
# Test with random bots
python main.py --bots "random::bot1,random::bot2"

# OpenAI GPT models (set OPENAI_API_KEY first)
export OPENAI_API_KEY=your-api-key-here
python main.py --bots "openai:gpt-4o-mini:gpt4o,openai:gpt-3.5-turbo:gpt35"

# Anthropic Claude models (set ANTHROPIC_API_KEY first)
export ANTHROPIC_API_KEY=your-api-key-here
python main.py --bots "anthropic:claude-3-haiku:claude"

# Mixed providers
python main.py --bots "openai:gpt-4o-mini:gpt4o,random::baseline,anthropic:claude-3-haiku:claude"
```

### Advanced Configuration

```bash
# Custom ELO range and progression
python main.py \
  --bots "openai:gpt-4o-mini:test-bot" \
  --start-elo 800 \
  --max-elo 2000 \
  --elo-step 200 \
  --think-time 1.0

# Only advance on wins (harder progression)
python main.py \
  --bots "openai:gpt-4o-mini:challenger" \
  --escalate-on on_win \
  --start-elo 1000 \
  --elo-step 100

# Custom Stockfish path
python main.py \
  --bots "random::test" \
  --stockfish /path/to/your/stockfish
```

### Bot Specification Format

Bots are specified using the format: `provider:model:name`

- **provider**: `openai`, `anthropic`, `random`
- **model**: Model name (can be empty for random provider)
- **name**: Display name for the bot

Examples:
- `random::baseline` - Random bot named "baseline"
- `openai:gpt-4o-mini:gpt4o` - OpenAI GPT-4o-mini named "gpt4o"
- `anthropic:claude-3-haiku:claude` - Anthropic Claude-3-haiku named "claude"

## ⚙️ Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bots` | `random::bot1,random::bot2` | Comma-separated bot specifications |
| `--start-elo` | `600` | Starting ELO rating |
| `--elo-step` | `100` | ELO increment per ladder rung |
| `--max-elo` | `2400` | Maximum ELO rating |
| `--think-time` | `0.3` | Engine thinking time per move (seconds) |
| `--escalate-on` | `always` | Advance on `always` or `on_win` only |
| `--llm-timeout` | `20.0` | LLM response timeout (seconds) |
| `--llm-temperature` | `0.0` | LLM sampling temperature |
| `--output-dir` | `runs` | Output directory for results |
| `--no-pgn` | `False` | Skip saving PGN files |

## 🤖 Supported LLM Providers

### OpenAI
- **Models**: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Setup**: Set `OPENAI_API_KEY` environment variable
- **Installation**: `pip install openai`

### Anthropic
- **Models**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Setup**: Set `ANTHROPIC_API_KEY` environment variable
- **Installation**: `pip install anthropic`

### Random (Baseline)
- **Purpose**: Baseline comparison and testing
- **Setup**: No API key required
- **Behavior**: Selects random legal moves

## 📊 Understanding Results

The benchmark produces several types of output:

### Real-time Dashboard
- Live board positions and move sequences
- Current ELO level and game status
- Win/draw/loss statistics
- Maximum ELO reached

### Final Results
- **Max ELO Reached**: Highest ELO level achieved
- **Games Played**: Total number of games
- **Win Rate**: Percentage of games won
- **Performance Rating**: 
  - 🏆 Excellent (1800+)
  - ⭐ Good (1400-1799)
  - 👍 Fair (1000-1399)
  - 📚 Learning (<1000)

### Game Records
- PGN files saved in `runs/<timestamp>/<bot_name>/`
- Complete game notation with metadata
- Engine ELO and game settings recorded

## 🏗️ Project Structure

```
chess-llm-bench/
├── chess_llm_bench/           # Main package
│   ├── core/                  # Core functionality
│   │   ├── models.py         # Data models and configuration
│   │   ├── engine.py         # Chess engine management
│   │   └── game.py           # Game running logic
│   ├── llm/                  # LLM providers
│   │   └── client.py         # LLM client implementations
│   ├── ui/                   # User interface
│   │   └── dashboard.py      # Terminal dashboard
│   └── cli.py                # Command-line interface
├── tests/                    # Unit tests
│   ├── test_bots.py         # Bot parsing tests
│   └── test_engine.py       # Engine functionality tests
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── setup.py                # Package installation
└── README.md               # This file
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/chess-llm-bench.git
cd chess-llm-bench
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Run self-tests
python main.py --self-test

# Format code
black chess_llm_bench/ tests/
isort chess_llm_bench/ tests/

# Type checking
mypy chess_llm_bench/
```

### Adding New LLM Providers

1. Create a new provider class inheriting from `BaseLLMProvider`
2. Implement the `generate_move` method
3. Register the provider in `LLMClient.PROVIDERS`
4. Add tests and documentation

Example:
```python
from chess_llm_bench.llm.client import BaseLLMProvider, LLMClient

class CustomProvider(BaseLLMProvider):
    async def generate_move(self, board, temperature=0.0, timeout_s=20.0):
        # Your implementation here
        pass

# Register the provider
LLMClient.register_provider("custom", CustomProvider)
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_bots.py

# Run with coverage
python -m pytest --cov=chess_llm_bench

# Run built-in self-tests
python main.py --self-test
```

## 📋 Requirements

- Python 3.8 or higher
- Stockfish chess engine
- Required Python packages:
  - `python-chess[engine]>=1.999`
  - `rich>=13.0.0`
- Optional packages for LLM providers:
  - `openai>=1.0.0` (for OpenAI models)
  - `anthropic>=0.3.0` (for Anthropic models)

## 🐛 Troubleshooting

### Common Issues

**Stockfish not found**
```bash
# Set explicit path
export STOCKFISH_PATH=/path/to/stockfish
python main.py --stockfish /path/to/stockfish
```

**API Key errors**
```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Set API keys
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
```

**Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Debugging

Enable verbose logging:
```bash
python main.py --verbose --bots "your:bot:spec"

# Or debug level
python main.py --debug --bots "your:bot:spec"
```

## 📈 Performance Tips

1. **Concurrent Testing**: Run multiple bots simultaneously for faster benchmarking
2. **Appropriate Time Controls**: Balance speed vs. accuracy with `--think-time`
3. **ELO Step Size**: Larger steps for faster runs, smaller for precision
4. **Temperature Settings**: Use 0.0 for deterministic results, higher for variety

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings for public functions and classes
- Write tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stockfish](https://stockfishchess.org/) - The powerful chess engine
- [python-chess](https://github.com/niklasf/python-chess) - Excellent chess library
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal formatting
- The chess and AI communities for inspiration and feedback

## 📬 Contact

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/chess-llm-bench/issues)
- Discussions: [Community discussions](https://github.com/yourusername/chess-llm-bench/discussions)

---

*Happy benchmarking! May your LLMs play like grandmasters! ♟️🤖*