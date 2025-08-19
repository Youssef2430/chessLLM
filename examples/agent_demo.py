#!/usr/bin/env python3
"""
Chess Agent Demo - Demonstrates agent-based chess playing with tools and reasoning.

This script showcases the difference between traditional prompting and agent-based
chess playing, where agents use analysis tools and reasoning workflows.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any
import chess
import chess.pgn
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_llm_bench.llm.agents import (
    LLMAgentProvider,
    LLMChessAgent,
    ChessAnalysisTools,
    ThinkingStrategy,
    create_agent_provider
)
from chess_llm_bench.llm.client import LLMClient, parse_bot_spec
from chess_llm_bench.core.models import BotSpec
from chess_llm_bench.core.budget import get_budget_tracker
from chess_llm_bench.ui.board import ChessBoardRenderer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import print as rprint


console = Console()
logger = logging.getLogger(__name__)


class AgentDemoRunner:
    """Runs demonstration games with agent-based chess players."""

    def __init__(self, verbose: bool = True):
        """Initialize the demo runner."""
        self.verbose = verbose
        self.board = chess.Board()
        self.board_renderer = ChessBoardRenderer()
        self.move_history = []
        self.game_pgn = None

    async def run_agent_game(
        self,
        white_agent: LLMChessAgent,
        black_agent: LLMChessAgent,
        max_moves: int = 100
    ) -> Dict[str, Any]:
        """
        Run a game between two agents.

        Args:
            white_agent: Agent playing white
            black_agent: Agent playing black
            max_moves: Maximum number of moves before declaring draw

        Returns:
            Dictionary with game results and statistics
        """
        self.board.reset()
        self.move_history = []
        move_count = 0

        # Create PGN game
        game = chess.pgn.Game()
        game.headers["Event"] = "Agent Demo Game"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = white_agent.name
        game.headers["Black"] = black_agent.name

        node = game

        console.print(f"\n[bold cyan]Starting Game:[/bold cyan] {white_agent.name} vs {black_agent.name}\n")

        while not self.board.is_game_over() and move_count < max_moves:
            # Determine current player
            current_agent = white_agent if self.board.turn else black_agent
            color = "White" if self.board.turn else "Black"

            # Display board
            if self.verbose:
                self._display_board(color, current_agent.name)

            # Get agent's move
            try:
                decision = await current_agent.make_move(self.board)

                # Apply the move
                self.board.push(decision.move)
                self.move_history.append(decision.san)
                node = node.add_variation(decision.move)
                move_count += 1

                # Display move and reasoning
                if self.verbose:
                    self._display_move_decision(color, decision)

                # Brief pause for readability
                await asyncio.sleep(0.5 if self.verbose else 0)

            except Exception as e:
                logger.error(f"Error during move generation: {e}")
                console.print(f"[red]Error: {e}[/red]")
                break

        # Determine result
        result = self._get_game_result()
        game.headers["Result"] = result["pgn_result"]
        self.game_pgn = game

        # Display final position
        if self.verbose:
            self._display_final_position(result)

        # Collect statistics
        white_stats = white_agent.get_statistics()
        black_stats = black_agent.get_statistics()

        return {
            "result": result,
            "move_count": move_count,
            "white_stats": white_stats,
            "black_stats": black_stats,
            "pgn": str(game),
            "final_fen": self.board.fen()
        }

    def _display_board(self, color: str, player_name: str):
        """Display the current board position."""
        console.print(f"\n[bold]{color} to move ({player_name})[/bold]")

        # Create board visualization
        board_str = self.board_renderer.render_board(
            self.board,
            flip=not self.board.turn,  # Show from current player's perspective
            highlight_last_move=len(self.move_history) > 0
        )

        console.print(Panel(board_str, title="Current Position", border_style="blue"))

    def _display_move_decision(self, color: str, decision):
        """Display the agent's move and reasoning."""
        console.print(f"\n[bold green]{color} plays: {decision.san}[/bold green]")
        console.print(f"Confidence: {decision.confidence:.2%}")

        if self.verbose and decision.reasoning:
            # Show key reasoning points
            console.print("\n[dim]Reasoning:[/dim]")

            # Group thoughts by type
            observations = [t for t in decision.reasoning if t.thought_type == "observation"]
            analyses = [t for t in decision.reasoning if t.thought_type == "analysis"]
            strategies = [t for t in decision.reasoning if t.thought_type == "strategy"]
            decisions = [t for t in decision.reasoning if t.thought_type == "decision"]

            if observations:
                console.print("  [yellow]Observations:[/yellow]")
                for thought in observations[:2]:  # Limit output
                    console.print(f"    • {thought.content}")

            if analyses:
                console.print("  [cyan]Analysis:[/cyan]")
                for thought in analyses[:3]:
                    console.print(f"    • {thought.content}")

            if strategies:
                console.print("  [magenta]Strategy:[/magenta]")
                for thought in strategies[:2]:
                    console.print(f"    • {thought.content}")

            if decision.alternatives_considered:
                console.print(f"  [dim]Considered {len(decision.alternatives_considered)} alternatives[/dim]")

    def _display_final_position(self, result):
        """Display the final game position and result."""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Game Over![/bold cyan]")
        console.print(f"Result: [bold]{result['description']}[/bold]")

        # Final board
        board_str = self.board_renderer.render_board(self.board, flip=False)
        console.print(Panel(board_str, title="Final Position", border_style="green"))

        # Move list
        if self.move_history:
            console.print("\n[bold]Move History:[/bold]")
            for i in range(0, len(self.move_history), 2):
                move_num = i // 2 + 1
                white_move = self.move_history[i]
                black_move = self.move_history[i + 1] if i + 1 < len(self.move_history) else ""
                console.print(f"  {move_num}. {white_move} {black_move}")

    def _get_game_result(self) -> Dict[str, str]:
        """Get the game result."""
        if self.board.is_checkmate():
            if self.board.turn:  # White to move = Black won
                return {"pgn_result": "0-1", "winner": "black", "description": "Black wins by checkmate"}
            else:
                return {"pgn_result": "1-0", "winner": "white", "description": "White wins by checkmate"}
        elif self.board.is_stalemate():
            return {"pgn_result": "1/2-1/2", "winner": "draw", "description": "Draw by stalemate"}
        elif self.board.is_insufficient_material():
            return {"pgn_result": "1/2-1/2", "winner": "draw", "description": "Draw by insufficient material"}
        elif self.board.can_claim_draw():
            return {"pgn_result": "1/2-1/2", "winner": "draw", "description": "Draw by repetition or 50-move rule"}
        else:
            return {"pgn_result": "*", "winner": "unknown", "description": "Game incomplete"}


async def compare_approaches():
    """Compare traditional prompting vs agent-based approaches."""
    console.print(Panel.fit(
        "[bold cyan]Chess Playing Approach Comparison[/bold cyan]\n\n"
        "Traditional Prompting vs Agent-Based Reasoning",
        border_style="cyan"
    ))

    # Create traditional client (random for demo)
    traditional_bot = BotSpec(provider="random", model="", name="Traditional")
    traditional_client = LLMClient(traditional_bot, use_agent=False)

    # Create agent-based client
    agent_bot = BotSpec(provider="random", model="", name="Agent-Based")
    agent_client = LLMClient(agent_bot, use_agent=True, agent_strategy="balanced")

    # Demonstrate analysis on a position
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4")

    console.print("\n[bold]Test Position:[/bold] Italian Opening")
    console.print(f"FEN: {board.fen()}\n")

    # Traditional approach
    console.print("[bold yellow]Traditional Approach:[/bold yellow]")
    console.print("• Sends board position to LLM")
    console.print("• Asks for next move in specific format")
    console.print("• Parses response for UCI notation")
    console.print("• No structured reasoning process")

    start_time = asyncio.get_event_loop().time()
    trad_move = await traditional_client.pick_move(board)
    trad_time = asyncio.get_event_loop().time() - start_time
    console.print(f"• Move: {trad_move.uci()} (Time: {trad_time:.2f}s)\n")

    # Agent approach
    console.print("[bold green]Agent-Based Approach:[/bold green]")
    console.print("• Analyzes position with chess tools")
    console.print("• Evaluates material, position, threats")
    console.print("• Considers multiple candidate moves")
    console.print("• Makes decision based on scoring")

    # Show detailed analysis
    if hasattr(agent_client, 'provider') and hasattr(agent_client.provider, 'agent'):
        agent = agent_client.provider.agent
        agent.current_board = board.copy()
        agent.tools = ChessAnalysisTools(agent.current_board)

        # Get board analysis
        board_state = agent.tools.get_board_state()
        console.print(f"\n[dim]Board Analysis:[/dim]")
        console.print(f"  Material: White {board_state['material_balance']['white_material']} - "
                     f"Black {board_state['material_balance']['black_material']}")
        console.print(f"  Position score: {board_state['position_evaluation']['total_score']:.1f}")

        # Get all analyzed moves
        all_moves = agent.tools.get_all_move_analyses()
        console.print(f"\n[dim]Legal Moves:[/dim] {len(all_moves)} total")
        console.print(f"\n[dim]Top Scoring Moves:[/dim]")
        for i, candidate in enumerate(all_moves[:3], 1):
            console.print(f"  {i}. {candidate.san}: {candidate.explanation} (score: {candidate.score:.1f})")

    start_time = asyncio.get_event_loop().time()
    agent_move = await agent_client.pick_move(board)
    agent_time = asyncio.get_event_loop().time() - start_time
    console.print(f"\n• Move: {agent_move.uci()} (Time: {agent_time:.2f}s)")

    # Show costs if available
    tracker = get_budget_tracker()
    if tracker.total_cost > 0:
        console.print(f"• Total cost so far: ${tracker.total_cost:.4f}")


async def demo_strategies():
    """Demonstrate different agent thinking strategies."""
    console.print(Panel.fit(
        "[bold cyan]Agent Thinking Strategies[/bold cyan]\n\n"
        "Comparing different reasoning approaches",
        border_style="cyan"
    ))

    strategies = [
        ("fast", "Quick tactical decisions, focuses on immediate threats"),
        ("balanced", "Balances tactical and positional considerations"),
        ("deep", "Thorough analysis with long-term planning"),
        ("adaptive", "Adapts strategy based on game phase")
    ]

    # Test position - middlegame with tactical opportunities
    board = chess.Board("r1bqk2r/pp2bppp/2n2n2/3p4/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 9")

    console.print(f"\n[bold]Test Position:[/bold]\nFEN: {board.fen()}\n")

    results_table = Table(title="Strategy Comparison")
    results_table.add_column("Strategy", style="cyan")
    results_table.add_column("Move", style="green")
    results_table.add_column("Time (s)", style="yellow")
    results_table.add_column("Confidence", style="magenta")
    results_table.add_column("Focus", style="white")

    for strategy_name, description in strategies:
        console.print(f"\n[bold]{strategy_name.upper()} Strategy:[/bold] {description}")

        # Create agent with specific strategy
        agent = LLMChessAgent(
            provider="random",  # Using random for demo
            model="",
            strategy=ThinkingStrategy[strategy_name.upper()],
            verbose=False
        )

        # Make move
        import time
        start = time.time()
        decision = await agent.make_move(board)
        elapsed = time.time() - start

        # Extract main focus from reasoning
        focus_thoughts = [t for t in decision.reasoning if t.thought_type == "strategy"]
        focus = focus_thoughts[0].content if focus_thoughts else "General play"

        results_table.add_row(
            strategy_name.capitalize(),
            decision.san,
            f"{elapsed:.2f}",
            f"{decision.confidence:.0%}",
            focus[:40] + "..." if len(focus) > 40 else focus
        )

    console.print("\n")
    console.print(results_table)


async def play_agent_game():
    """Play a full game between two agents with different strategies."""
    console.print(Panel.fit(
        "[bold cyan]Agent vs Agent Game[/bold cyan]\n\n"
        "Fast Strategy (White) vs Deep Strategy (Black)",
        border_style="cyan"
    ))

    # Create agents with different strategies
    white_agent = LLMChessAgent(
        provider="random",
        model="",
        strategy=ThinkingStrategy.FAST,
        verbose=False,
        name="FastBot (White)"
    )

    black_agent = LLMChessAgent(
        provider="random",
        model="",
        strategy=ThinkingStrategy.DEEP,
        verbose=False,
        name="DeepBot (Black)"
    )

    # Run the game
    runner = AgentDemoRunner(verbose=True)
    result = await runner.run_agent_game(white_agent, black_agent, max_moves=60)

    # Display statistics
    console.print("\n" + "="*60)
    console.print("[bold cyan]Game Statistics[/bold cyan]\n")

    stats_table = Table()
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("White (Fast)", style="white")
    stats_table.add_column("Black (Deep)", style="white")

    stats_table.add_row(
        "Total Moves",
        str(result["white_stats"]["total_moves"]),
        str(result["black_stats"]["total_moves"])
    )
    stats_table.add_row(
        "Avg Move Time",
        f"{result['white_stats']['average_move_time']:.2f}s",
        f"{result['black_stats']['average_move_time']:.2f}s"
    )
    stats_table.add_row(
        "Avg Confidence",
        f"{result['white_stats']['average_confidence']:.0%}",
        f"{result['black_stats']['average_confidence']:.0%}"
    )
    stats_table.add_row(
        "Strategy",
        result["white_stats"]["strategy"],
        result["black_stats"]["strategy"]
    )

    console.print(stats_table)

    # Save PGN if game completed
    if runner.game_pgn:
        pgn_file = Path("agent_demo_game.pgn")
        with open(pgn_file, "w") as f:
            f.write(str(runner.game_pgn))
        console.print(f"\n[green]Game saved to {pgn_file}[/green]")

    # Show total costs if any were incurred
    tracker = get_budget_tracker()
    if tracker.total_cost > 0:
        console.print(f"\n[yellow]Total game costs: ${tracker.total_cost:.4f}[/yellow]")
        console.print(f"Requests made: {tracker.total_requests}")


async def main():
    """Main demo function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Welcome message
        console.print(Panel.fit(
            "[bold cyan]Chess Agent Demo[/bold cyan]\n\n"
            "Demonstrating agent-based chess playing with\n"
            "analysis tools and reasoning workflows",
            border_style="cyan"
        ))

        # Run demos
        console.print("\n[bold]1. Comparing Approaches[/bold]")
        console.print("-" * 40)
        await compare_approaches()

        console.print("\n[bold]2. Agent Strategies[/bold]")
        console.print("-" * 40)
        await demo_strategies()

        console.print("\n[bold]3. Full Agent Game[/bold]")
        console.print("-" * 40)
        await play_agent_game()

        console.print("\n[bold green]Demo completed successfully![/bold green]")

        # Show final cost summary
        tracker = get_budget_tracker()
        if tracker.total_cost > 0:
            console.print("\n[bold cyan]Cost Summary:[/bold cyan]")
            console.print(f"Total cost: ${tracker.total_cost:.4f}")
            console.print(f"Total requests: {tracker.total_requests}")
            console.print(f"Total tokens: {tracker.total_tokens}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Demo failed")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
