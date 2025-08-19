# Chess Agent System

## Overview

The Chess Agent System provides an advanced, tool-based approach to chess playing for LLMs. Instead of simple prompting, agents use structured reasoning workflows with chess analysis tools to make informed decisions.

## Key Features

### 🔧 Tool-Based Analysis
- **Position Evaluation**: Comprehensive board state analysis
- **Material Calculation**: Track piece values and exchanges
- **Positional Factors**: Center control, king safety, pawn structure
- **Move Analysis**: Categorize and score candidate moves
- **Endgame Evaluation**: Specialized endgame analysis

### 🧠 Reasoning Workflow
1. **Observe**: Understand the current position
2. **Analyze**: Use tools to evaluate position factors
3. **Generate**: Identify candidate moves
4. **Evaluate**: Score moves based on multiple criteria
5. **Decide**: Select the best move with confidence scoring

### 📊 Multiple Strategies
- **Fast**: Quick tactical decisions, focuses on immediate threats
- **Balanced**: Balances tactical and positional considerations
- **Deep**: Thorough analysis with long-term planning
- **Adaptive**: Adapts strategy based on game phase

## Architecture

```
agents/
├── base_agent.py       # Core agent framework
├── chess_tools.py      # Chess analysis tools
└── llm_agent_provider.py  # LLM integration
```

### Components

#### ChessAgent (base_agent.py)
Base class for all chess agents with reasoning workflow:
- Manages thinking process
- Coordinates tool usage
- Tracks decision confidence
- Provides performance metrics

#### ChessAnalysisTools (chess_tools.py)
Comprehensive chess analysis toolkit:
- Board state evaluation
- Move generation and categorization
- Position scoring heuristics
- Tactical and strategic analysis

#### LLMAgentProvider (llm_agent_provider.py)
Integration with LLM providers:
- Wraps agents for plug-and-play usage
- Supports OpenAI, Anthropic, and Gemini
- Combines LLM insights with tool analysis

## Usage

### Command Line Interface

Enable agent mode with the `--use-agent` flag:

```bash
# Basic agent usage
python main.py --use-agent --bots "openai:gpt-4:AgentBot"

# Specify strategy
python main.py --use-agent --agent-strategy deep --bots "anthropic:claude-3:DeepThinker"

# Verbose reasoning output
python main.py --use-agent --verbose-agent --bots "gemini:gemini-pro:Reasoner"

# Compare strategies
python main.py --use-agent --agent-strategy adaptive --preset premium
```

### Programmatic Usage

```python
from chess_llm_bench.llm.agents import create_agent_provider, ThinkingStrategy
import chess

# Create an agent provider
agent = create_agent_provider(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    strategy="balanced",
    verbose=True,
    use_tools=True
)

# Use it to make moves
board = chess.Board()
move_uci, time_taken = await agent.generate_move(
    board=board,
    game_state=str(board),
    move_history=[]
)
```

### Direct Agent Usage

```python
from chess_llm_bench.llm.agents import LLMChessAgent, ThinkingStrategy
import chess

# Create an agent
agent = LLMChessAgent(
    provider="openai",
    model="gpt-4",
    strategy=ThinkingStrategy.DEEP,
    verbose=True
)

# Make a move with full reasoning
board = chess.Board()
decision = await agent.make_move(board)

print(f"Move: {decision.san}")
print(f"Confidence: {decision.confidence:.2%}")
for thought in decision.reasoning:
    print(f"  {thought.thought_type}: {thought.content}")
```

## Tool Categories

### Position Analysis Tools

- **get_board_state()**: Complete position information
- **evaluate_material()**: Material balance and piece counts
- **evaluate_position()**: Positional factors scoring
- **evaluate_endgame()**: Specialized endgame evaluation

### Move Analysis Tools

- **get_legal_moves()**: All legal moves with metadata
- **analyze_move()**: Detailed single move analysis
- **suggest_candidate_moves()**: Top N moves by heuristic
- **categorize_move()**: Classify move types (capture, check, etc.)

### Evaluation Factors

#### Material
- Piece values (P=1, N=3, B=3, R=5, Q=9)
- Material imbalances
- Piece trades evaluation

#### Position
- Center control (e4, d4, e5, d5 squares)
- Piece mobility and activity
- King safety and pawn shield
- Pawn structure quality

#### Tactical
- Captures and threats
- Checks and checkmates
- Pins, forks, and skewers
- Defensive necessities

## Thinking Strategies

### Fast Strategy
```python
strategy=ThinkingStrategy.FAST
```
- Prioritizes forcing moves (checks, captures)
- Quick tactical evaluation
- Minimal positional consideration
- Best for: Blitz games, tactical puzzles

### Balanced Strategy
```python
strategy=ThinkingStrategy.BALANCED
```
- Equal weight to tactics and position
- Moderate analysis depth
- Good general performance
- Best for: Standard games, most situations

### Deep Strategy
```python
strategy=ThinkingStrategy.DEEP
```
- Comprehensive position evaluation
- Long-term strategic planning
- Careful consideration of alternatives
- Best for: Complex positions, correspondence

### Adaptive Strategy
```python
strategy=ThinkingStrategy.ADAPTIVE
```
- Opening: Focus on development
- Middlegame: Tactical opportunities
- Endgame: King activity and pawns
- Best for: Full games, varying positions

## Agent Thoughts System

Agents generate structured thoughts during reasoning:

```python
@dataclass
class AgentThought:
    thought_type: str  # "observation", "analysis", "strategy", "decision"
    content: str       # The thought content
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]
```

### Thought Types

- **Observation**: Initial position understanding
- **Analysis**: Tool-based position evaluation
- **Strategy**: Move generation and planning
- **Decision**: Final move selection

## Performance Comparison

### Traditional Prompting
- Direct LLM query for moves
- No structured analysis
- Limited position understanding
- Inconsistent move format

### Agent-Based Approach
- Systematic position analysis
- Tool-assisted evaluation
- Confidence scoring
- Structured reasoning chain

## Examples

### Run Agent Demo

```bash
# Run the comprehensive agent demo
python examples/agent_demo.py
```

This demonstrates:
- Traditional vs Agent comparison
- Different thinking strategies
- Full agent vs agent games
- Detailed reasoning output

### Benchmark with Agents

```bash
# Premium models with agent reasoning
python main.py --use-agent --preset premium --verbose-agent

# Fast strategy for quick games
python main.py --use-agent --agent-strategy fast --start-elo 1000 --max-elo 1400

# Deep analysis for strong play
python main.py --use-agent --agent-strategy deep --bots "openai:gpt-4:DeepBlue"
```

## Advanced Configuration

### Custom Tool Weights

Modify scoring in `_score_move()` method:

```python
async def _score_move(self, move_analysis: MoveAnalysis) -> float:
    score = move_analysis.score
    
    # Custom scoring logic
    if MoveCategory.CAPTURE in move_analysis.categories:
        score += self.capture_bonus
    
    return score
```

### LLM-Enhanced Analysis

Agents can use LLMs for additional insights:

```python
# In LLMChessAgent
async def customize_evaluation(self, move_analysis: MoveAnalysis) -> float:
    # Get LLM's opinion on the move
    prompt = self._create_evaluation_prompt(move_analysis)
    response = await self._call_llm(prompt)
    llm_score = self._parse_evaluation_score(response)
    
    return llm_score
```

## Debugging and Monitoring

### Verbose Output

Enable detailed reasoning output:

```bash
python main.py --use-agent --verbose-agent
```

Shows:
- Observations about position
- Analysis results from tools
- Strategy considerations
- Decision reasoning
- Confidence scores

### Performance Metrics

Access agent statistics:

```python
stats = agent.get_statistics()
print(f"Average move time: {stats['average_move_time']:.2f}s")
print(f"Average confidence: {stats['average_confidence']:.2%}")
```

## Integration with Existing System

The agent system is fully compatible with the existing benchmark infrastructure:

1. **Drop-in Replacement**: Use `--use-agent` flag to enable
2. **Same Interface**: Works with all existing commands
3. **Budget Tracking**: Integrated cost tracking
4. **PGN Output**: Standard game recording
5. **Dashboard**: Live visualization support

## Limitations and Considerations

### Current Limitations
- No deep search tree exploration
- Heuristic evaluation only (no engine)
- Limited tactical vision
- Pattern recognition depends on tools

### Best Practices
1. Use appropriate strategy for game type
2. Enable verbose mode for debugging
3. Monitor confidence scores
4. Compare with traditional approach

### Performance Tips
- Fast strategy for real-time play
- Deep strategy for analysis
- Adaptive for full games
- Balance tools vs LLM calls

## Future Enhancements

Potential improvements:
- [ ] Monte Carlo tree search
- [ ] Neural network evaluation
- [ ] Opening book integration
- [ ] Endgame tablebase support
- [ ] Multi-agent collaboration
- [ ] Learning from game history

## Contributing

To add new tools or strategies:

1. Extend `ChessAnalysisTools` for new analysis
2. Override `customize_evaluation()` for agent variants
3. Add new `ThinkingStrategy` enum values
4. Implement strategy in `_score_move()`

## Support

For issues or questions:
- Check the main project README
- Run the agent demo for examples
- Enable verbose mode for debugging
- Review agent reasoning output