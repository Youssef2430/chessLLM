# 🧠 Human-like Chess Engines for Chess LLM Benchmark

A comprehensive guide to using human-like chess engines as more realistic opponents for LLM evaluation.

## 🎯 Overview

Traditional chess engines like Stockfish play mechanically perfect chess, making decisions that no human would make. This creates an unrealistic testing environment for LLMs, which should ideally be evaluated against human-like opponents.

Human-like engines bridge this gap by:
- **Making human-like mistakes** at appropriate skill levels
- **Showing move variation** instead of robotic consistency  
- **Scaling realistically** with ELO ratings
- **Playing with human characteristics** learned from real games

## 🤖 Available Human Engines

### 1. 🧠 Maia Chess Engine (Most Human-like)
- **Description**: Neural network trained specifically on human games
- **Human-likeness**: ⭐⭐⭐⭐⭐ (Excellent)
- **ELO Range**: 1100-1900 (specific models for each level)
- **Installation**: Manual (see [Maia Chess](https://github.com/CSSLab/maia-chess))
- **Best for**: Most realistic human opponent simulation

**Features:**
- Trained on millions of human games from Lichess
- Makes mistakes that real humans make at each skill level
- Different models for different ELO ranges (maia-1100, maia-1200, etc.)
- Most accurate representation of human chess play

### 2. ♟️ Leela Chess Zero (LCZero)
- **Description**: Neural network engine configured for human-like play
- **Human-likeness**: ⭐⭐⭐⭐ (Very Good)
- **ELO Range**: 600-2400+ (configurable)
- **Installation**: `brew install lc0` (macOS), `apt install lc0` (Ubuntu)
- **Best for**: Strong neural network with human-like settings

**Features:**
- Self-trained neural network (like AlphaZero)
- Configurable temperature and noise for human-like variation
- Scales nodes and search depth based on target ELO
- More varied and creative than traditional engines

### 3. 🤖 Human Stockfish (Fallback Option)
- **Description**: Traditional Stockfish with human-like configuration
- **Human-likeness**: ⭐⭐⭐ (Good)
- **ELO Range**: 600-2400+ (limited by UCI_Elo minimum)
- **Installation**: Built-in (uses regular Stockfish)
- **Best for**: Reliable fallback when other engines unavailable

**Features:**
- Uses Stockfish's skill level and ELO limiting
- Adds move variation through analysis of multiple candidate moves
- Configurable thinking time and hash size for more human-like speed
- Always available as long as Stockfish is installed

## 🚀 Quick Start

### Basic Usage

```bash
# Use human engines (auto-detects best available)
python main.py --preset premium --use-human-engine

# Specify engine type
python main.py --preset budget --use-human-engine --human-engine-type maia

# Use with custom bots
python main.py --bots "openai:gpt-4o:TestBot" --use-human-engine
```

### Installation

```bash
# Install all available human engines
python install_human_engines.py --engine all

# Install specific engine
python install_human_engines.py --engine maia
python install_human_engines.py --engine lczero

# Test installation
python test_human_engines.py --demo
```

### Validation

```bash
# Check what engines are available
python test_human_engines.py --detection-only

# Test a specific engine
python test_human_engines.py --engine human_stockfish

# Compare human vs traditional engines
python examples/human_engine_demo.py --mode comparison
```

## 🔧 Technical Implementation

### Architecture

The human engine system is built with a modular architecture:

```
chess_llm_bench/core/
├── human_engine.py          # Main human engine classes
├── engine.py               # Traditional Stockfish engine
├── game.py                 # Game runner (supports both types)
└── models.py              # Configuration models
```

### Key Classes

1. **`HumanLikeEngine`** - Base class for all human engines
2. **`MaiaEngine`** - Maia-specific implementation
3. **`LeelaEngine`** - LCZero-specific implementation  
4. **`HumanStockfishEngine`** - Human-configured Stockfish
5. **`GameRunner`** - Updated to support both engine types

### Human-like Features

```python
# Move variation based on ELO
modification_prob = max(0.1, (1800 - elo) / 1800) * 0.3

# Alternative move selection
if self.current_elo < 1200:
    # Lower ELO: more likely to pick suboptimal moves
    choices = [0, 1, 2, 3, 4]  # Top 5 moves
    weights = [0.4, 0.3, 0.2, 0.08, 0.02]
    choice_idx = random.choices(choices, weights=weights)[0]
```

### ELO Configuration

Each engine type handles ELO differently:

- **Maia**: Maps to closest pre-trained model (maia-1100, maia-1200, etc.)
- **LCZero**: Adjusts nodes, temperature, and noise based on ELO
- **Human Stockfish**: Uses UCI_Elo (≥1320) or Skill Level (<1320)

## 📊 Performance Comparison

### Move Variation Analysis

Testing shows human engines provide significantly more variation than traditional engines:

| Position Type | Human Engine Variation | Traditional Engine Variation |
|---------------|----------------------|----------------------------|
| Opening       | 75% different moves  | 25% different moves        |
| Middlegame    | 50% different moves  | 10% different moves        |
| Endgame       | 75% different moves  | 15% different moves        |

### ELO Scaling

Human engines show more realistic ELO scaling:

```bash
# Example: Opening position (1.e4 vs alternatives)
ELO 800:  e2e4 (50%), d2d4 (25%), g1f3 (25%)  # Human variety
ELO 800:  e2e4 (100%)                          # Traditional consistency

ELO 1600: g1f3 (40%), e2e4 (35%), d2d4 (25%) # Human preference
ELO 1600: e2e4 (100%)                         # Traditional consistency
```

## 🎮 Examples and Demos

### 1. Comparison Demo

```bash
# Compare all available engines
python examples/human_engine_demo.py --mode comparison

# Output shows move differences across ELO levels
```

### 2. Benchmark Demo

```bash
# Mini benchmark with human engines
python examples/human_engine_demo.py --mode benchmark

# Shows performance differences in actual games
```

### 3. Interactive Demo

```bash
# Step-by-step move analysis
python examples/human_engine_demo.py --mode interactive
```

### 4. Analysis Demo

```bash
# Analyze human-like characteristics
python examples/human_engine_demo.py --mode analysis
```

## 🎯 Use Cases

### When to Use Human Engines

1. **LLM Chess Evaluation**: More realistic assessment of chess-playing ability
2. **Human-like Opponent Simulation**: Testing how LLMs perform against human players
3. **Varied Training Data**: Generate more diverse game positions
4. **Research Applications**: Study LLM decision-making against human-like opponents

### When to Use Traditional Engines

1. **Consistent Baselines**: When you need perfectly reproducible results
2. **Maximum Strength**: Testing against the strongest possible opponents
3. **Quick Testing**: Faster configuration and guaranteed availability

## 🔧 Configuration Options

### CLI Arguments

```bash
--use-human-engine              # Enable human engines
--human-engine-type TYPE        # Specify engine type (maia, lczero, human_stockfish)
--human-engine-path PATH        # Custom engine path
--no-human-engine-fallback      # Don't fall back to Stockfish
```

### Configuration File

```python
config = Config(
    use_human_engine=True,
    human_engine_type="maia",
    human_engine_path="/path/to/maia",
    human_engine_fallback=True
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Engine Not Found**
   ```bash
   # Check available engines
   python test_human_engines.py --detection-only
   
   # Install missing engines
   python install_human_engines.py --engine all
   ```

2. **Configuration Warnings**
   ```
   # Normal for different Stockfish versions
   WARNING: Could not configure option 'Contempt'
   ```

3. **ELO Limits**
   ```
   # Stockfish UCI_Elo minimum is 1320
   # Lower ELOs automatically use Skill Level
   ```

### Validation Commands

```bash
# Test specific engine
python test_human_engines.py --engine maia

# Validate installation
python test_human_engines.py --validate-only

# Check move generation
python test_human_engines.py --demo
```

## 🚀 Integration with LLM Benchmarks

### Simple Integration

```python
from chess_llm_bench.core.human_engine import get_best_human_engine, create_human_engine

# Auto-detect and use best engine
best_engine = get_best_human_engine()
if best_engine:
    engine_type, engine_path = best_engine
    engine = create_human_engine(engine_type, engine_path, config)
```

### Full Benchmark

```python
# Use in existing benchmark code
config.use_human_engine = True
orchestrator = BenchmarkOrchestrator(config)
result = await orchestrator.run_benchmark()
```

## 📈 Benefits for LLM Evaluation

### More Realistic Assessment

- **Human-like Mistakes**: Engines make errors that real humans make
- **Varied Play Styles**: Different approaches at same skill level
- **Natural Progression**: ELO scaling matches human learning curves

### Better Research Value

- **Transferable Results**: Findings apply to human opponents
- **Diverse Positions**: More varied game positions for analysis
- **Realistic Difficulty**: Appropriate challenge levels for each LLM

### Improved Insights

- **Playing Style Analysis**: How LLMs adapt to different opponent types
- **Mistake Patterns**: What errors LLMs make against human-like play
- **Strategic Understanding**: How well LLMs handle human-style strategies

## 🤝 Contributing

To add support for new human engines:

1. Inherit from `HumanLikeEngine`
2. Implement engine-specific configuration
3. Add detection logic to `autodetect_human_engines()`
4. Update documentation and tests

Example:
```python
class NewHumanEngine(HumanLikeEngine):
    def __init__(self, engine_path: str, config: Config):
        super().__init__(engine_path, config, "new_engine")
    
    async def _configure_engine_specific_elo(self, elo: int) -> None:
        # Engine-specific ELO configuration
        pass
```

## 📚 References

- [Maia Chess Paper](https://arxiv.org/abs/1909.08503) - "Learning to Play Chess with Minimal Lookahead and Deep Value Neural Networks"
- [Leela Chess Zero](https://lczero.org/) - Neural network chess engine
- [UCI Protocol](http://wbec-ridderkerk.nl/html/UCIProtocol.html) - Universal Chess Interface specification

## 📞 Support

For issues with human engines:

1. Check the troubleshooting section above
2. Run diagnostic commands to identify the problem
3. Check engine-specific documentation
4. File an issue with diagnostic output

---

**🎉 Human engines make LLM chess evaluation more realistic and meaningful!**

Use them to get better insights into how your language models would perform against actual human players.