"""
Results database and ranking system for Chess LLM Benchmark.

This module provides comprehensive results tracking, historical data management,
performance ranking, and statistical analysis for chess LLM benchmarks.

Features:
- SQLite database for persistent results storage
- Performance ranking and comparison systems
- Historical data analysis and trends
- Statistical metrics and insights
- Results visualization and reporting
- Export capabilities for further analysis
"""

from __future__ import annotations

import sqlite3
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from statistics import mean, median, stdev
import math

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import track

from .models import BotSpec, GameRecord, LadderStats, BenchmarkResult
from .budget import BudgetSummary


@dataclass
class ModelPerformance:
    """Performance metrics for a specific model."""

    model_id: str
    provider: str
    model_name: str
    display_name: str

    # Core performance metrics
    max_elo: int = 0
    avg_elo: float = 0.0
    total_games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    # Advanced metrics
    win_rate: float = 0.0
    draw_rate: float = 0.0
    loss_rate: float = 0.0
    consistency_score: float = 0.0  # Lower std dev = higher consistency
    improvement_rate: float = 0.0  # ELO gained per game

    # Time and cost metrics
    avg_time_per_game: float = 0.0
    total_cost: float = 0.0
    cost_per_game: float = 0.0
    cost_per_elo: float = 0.0  # Cost per ELO point gained

    # Historical data
    benchmark_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    elo_history: List[int] = field(default_factory=list)

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (ELO per dollar)."""
        if self.total_cost > 0:
            return self.max_elo / self.total_cost
        return float("inf") if self.max_elo > 0 else 0.0

    @property
    def performance_rating(self) -> str:
        return "Unrated (opponent setting only)"


@dataclass
class BenchmarkRecord:
    """Complete record of a benchmark run."""

    run_id: str
    timestamp: datetime
    config: Dict[str, Any]

    # Results
    models: List[ModelPerformance]
    total_games: int
    total_cost: float
    duration: timedelta

    # Metadata
    version: str
    environment: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


class ResultsDatabase:
    """SQLite database for storing and retrieving benchmark results."""

    def __init__(self, db_path: Path):
        """Initialize the results database."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    config TEXT,
                    total_games INTEGER,
                    total_cost REAL,
                    duration_seconds REAL,
                    version TEXT,
                    environment TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS model_performances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    model_id TEXT,
                    provider TEXT,
                    model_name TEXT,
                    display_name TEXT,
                    max_elo INTEGER,
                    avg_elo REAL,
                    total_games INTEGER,
                    wins INTEGER,
                    draws INTEGER,
                    losses INTEGER,
                    win_rate REAL,
                    draw_rate REAL,
                    loss_rate REAL,
                    consistency_score REAL,
                    improvement_rate REAL,
                    avg_time_per_game REAL,
                    total_cost REAL,
                    cost_per_game REAL,
                    cost_per_elo REAL,
                    elo_history TEXT,
                    FOREIGN KEY (run_id) REFERENCES benchmarks (run_id)
                );

                CREATE TABLE IF NOT EXISTS game_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    model_id TEXT,
                    elo INTEGER,
                    color_llm_white BOOLEAN,
                    result TEXT,
                    ply_count INTEGER,
                    timestamp DATETIME,
                    game_path TEXT,
                    cost REAL,
                    FOREIGN KEY (run_id) REFERENCES benchmarks (run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_model_elo ON model_performances (model_id, max_elo);
                CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp ON benchmarks (timestamp);
                CREATE INDEX IF NOT EXISTS idx_game_model ON game_records (model_id, elo);
            """)

    def store_benchmark(self, benchmark: BenchmarkRecord) -> None:
        """Store a complete benchmark record."""
        with sqlite3.connect(self.db_path) as conn:
            # Store benchmark metadata
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmarks
                (run_id, timestamp, config, total_games, total_cost, duration_seconds, version, environment, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    benchmark.run_id,
                    benchmark.timestamp.isoformat(),
                    json.dumps(benchmark.config),
                    benchmark.total_games,
                    benchmark.total_cost,
                    benchmark.duration.total_seconds(),
                    benchmark.version,
                    json.dumps(benchmark.environment),
                    benchmark.notes,
                ),
            )

            conn.execute(
                "DELETE FROM model_performances WHERE run_id = ?", (benchmark.run_id,)
            )
            # Store model performances
            for model in benchmark.models:
                conn.execute(
                    """
                    INSERT INTO model_performances
                    (run_id, model_id, provider, model_name, display_name, max_elo, avg_elo,
                     total_games, wins, draws, losses, win_rate, draw_rate, loss_rate,
                     consistency_score, improvement_rate, avg_time_per_game, total_cost,
                     cost_per_game, cost_per_elo, elo_history)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        benchmark.run_id,
                        model.model_id,
                        model.provider,
                        model.model_name,
                        model.display_name,
                        model.max_elo,
                        model.avg_elo,
                        model.total_games,
                        model.wins,
                        model.draws,
                        model.losses,
                        model.win_rate,
                        model.draw_rate,
                        model.loss_rate,
                        model.consistency_score,
                        model.improvement_rate,
                        model.avg_time_per_game,
                        model.total_cost,
                        model.cost_per_game,
                        model.cost_per_elo,
                        json.dumps(model.elo_history),
                    ),
                )

    def get_model_history(self, model_id: str) -> List[ModelPerformance]:
        """Get historical performance for a specific model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM model_performances
                WHERE model_id = ?
                ORDER BY (SELECT timestamp FROM benchmarks WHERE benchmarks.run_id = model_performances.run_id)
            """,
                (model_id,),
            )

            performances = []
            for row in cursor.fetchall():
                perf = ModelPerformance(
                    model_id=row["model_id"],
                    provider=row["provider"],
                    model_name=row["model_name"],
                    display_name=row["display_name"],
                    max_elo=row["max_elo"],
                    avg_elo=row["avg_elo"],
                    total_games=row["total_games"],
                    wins=row["wins"],
                    draws=row["draws"],
                    losses=row["losses"],
                    win_rate=row["win_rate"],
                    draw_rate=row["draw_rate"],
                    loss_rate=row["loss_rate"],
                    consistency_score=row["consistency_score"],
                    improvement_rate=row["improvement_rate"],
                    avg_time_per_game=row["avg_time_per_game"],
                    total_cost=row["total_cost"],
                    cost_per_game=row["cost_per_game"],
                    cost_per_elo=row["cost_per_elo"],
                    elo_history=(
                        json.loads(row["elo_history"]) if row["elo_history"] else []
                    ),
                )
                performances.append(perf)

            return performances

    def store_games(self, result, bots):
        mapping = {bot.name: f"{bot.provider}:{bot.model}" for bot in bots}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM game_records WHERE run_id = ?", (result.run_id,))
            for name, stats in result.bot_results.items():
                for game in stats.games:
                    conn.execute(
                        """INSERT INTO game_records
                        (run_id, model_id, elo, color_llm_white, result, ply_count, timestamp, game_path, cost)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            result.run_id,
                            mapping[name],
                            game.elo,
                            game.color_llm_white,
                            game.result,
                            game.ply_count,
                            game.timestamp.isoformat(),
                            str(game.path),
                            None,
                        ),
                    )

    def get_leaderboard(
        self, limit: int = 50, metric: str = "win_rate"
    ) -> List[ModelPerformance]:
        """Get leaderboard ranked by specified metric."""
        allowed = {"win_rate", "total_games", "max_elo", "total_cost"}
        if metric not in allowed:
            raise ValueError("Unsupported leaderboard metric")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get latest performance for each model
            cursor = conn.execute(
                f"""
                SELECT mp.* FROM model_performances mp
                WHERE mp.run_id = (SELECT run_id FROM benchmarks ORDER BY timestamp DESC LIMIT 1)
                ORDER BY mp.{metric} DESC, mp.total_games DESC
                LIMIT ?
            """,
                (limit,),
            )

            leaderboard = []
            for row in cursor.fetchall():
                perf = ModelPerformance(
                    model_id=row["model_id"],
                    provider=row["provider"],
                    model_name=row["model_name"],
                    display_name=row["display_name"],
                    max_elo=row["max_elo"],
                    avg_elo=row["avg_elo"],
                    total_games=row["total_games"],
                    wins=row["wins"],
                    draws=row["draws"],
                    losses=row["losses"],
                    win_rate=row["win_rate"],
                    draw_rate=row["draw_rate"],
                    loss_rate=row["loss_rate"],
                    consistency_score=row["consistency_score"],
                    improvement_rate=row["improvement_rate"],
                    avg_time_per_game=row["avg_time_per_game"],
                    total_cost=row["total_cost"],
                    cost_per_game=row["cost_per_game"],
                    cost_per_elo=row["cost_per_elo"],
                    elo_history=(
                        json.loads(row["elo_history"]) if row["elo_history"] else []
                    ),
                )
                leaderboard.append(perf)

            return leaderboard

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated statistics by provider."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    provider,
                    COUNT(*) as model_count,
                    AVG(max_elo) as avg_max_elo,
                    MAX(max_elo) as best_elo,
                    AVG(total_cost) as avg_cost,
                    AVG(win_rate) as avg_win_rate
                FROM model_performances
                GROUP BY provider
            """)

            stats = {}
            for row in cursor.fetchall():
                stats[row[0]] = {
                    "model_count": row[1],
                    "avg_max_elo": row[2],
                    "best_elo": row[3],
                    "avg_cost": row[4],
                    "avg_win_rate": row[5],
                }

            return stats


class RankingSystem:
    """Advanced ranking and analysis system for model performance."""

    def __init__(self, db: ResultsDatabase):
        self.db = db
        self.console = Console()

    def calculate_elo_rating(
        self, performances: List[ModelPerformance]
    ) -> Dict[str, float]:
        """Calculate ELO ratings based on head-to-head performance (simplified)."""
        raise NotImplementedError(
            "Opponent settings cannot be converted to model Elo ratings"
        )

    def get_comprehensive_ranking(self):
        return [
            (i + 1, perf, {})
            for i, perf in enumerate(self.db.get_leaderboard(limit=100))
        ]

    def _calculate_trend(self, perf: ModelPerformance) -> str:
        """Calculate performance trend from ELO history."""
        if len(perf.elo_history) < 2:
            return "📊 Insufficient Data"

        if len(perf.elo_history) >= 3:
            recent_avg = mean(perf.elo_history[-3:])
            early_avg = mean(perf.elo_history[:3])

            if recent_avg > early_avg * 1.1:
                return "📈 Strong Improvement"
            elif recent_avg > early_avg * 1.05:
                return "📊 Improving"
            elif recent_avg < early_avg * 0.9:
                return "📉 Declining"
            elif recent_avg < early_avg * 0.95:
                return "📊 Slight Decline"
            else:
                return "➡️  Stable"

        return "📊 Limited Data"

    def create_leaderboard_table(self, limit: int = 20) -> Table:
        """Create a beautiful leaderboard table."""
        table = Table(title="Latest run only — descriptive results, not Elo ratings")
        for column in (
            "Model",
            "Opponent setting",
            "Completed",
            "W / D / L",
            "Score",
            "Cost (USD)",
        ):
            table.add_column(column)
        for _, perf, _ in self.get_comprehensive_ranking()[:limit]:
            score = (
                (perf.wins + 0.5 * perf.draws) / perf.total_games
                if perf.total_games
                else None
            )
            table.add_row(
                perf.display_name,
                str(perf.max_elo),
                str(perf.total_games),
                f"{perf.wins} / {perf.draws} / {perf.losses}",
                f"{score:.1%}" if score is not None else "—",
                f"${perf.total_cost:.4f}",
            )
        return table

    def create_provider_comparison_table(self) -> Table:
        """Create provider comparison table."""
        provider_stats = self.db.get_provider_stats()

        table = Table(title="📊 Provider Performance Comparison")
        table.add_column("Provider", style="cyan", width=15)
        table.add_column("Models", style="blue", justify="right", width=8)
        table.add_column("Highest opponent", style="yellow", justify="right", width=10)
        table.add_column("Mean opponent", style="green", justify="right", width=10)
        table.add_column("Avg Cost", style="red", justify="right", width=10)
        table.add_column("Avg Win Rate", style="magenta", justify="right", width=12)

        # Sort by best ELO
        sorted_providers = sorted(
            provider_stats.items(), key=lambda x: x[1]["best_elo"], reverse=True
        )

        for provider, stats in sorted_providers:
            table.add_row(
                provider.title(),
                str(stats["model_count"]),
                str(int(stats["best_elo"])),
                f"{stats['avg_max_elo']:.0f}",
                f"${stats['avg_cost']:.3f}",
                f"{stats['avg_win_rate']:.1%}",
            )

        return table

    def analyze_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Perform detailed analysis of a specific model."""
        history = self.db.get_model_history(model_id)
        if not history:
            return {"error": "No data found for this model"}

        latest = history[-1]

        analysis = {
            "model_info": {
                "id": model_id,
                "provider": latest.provider,
                "display_name": latest.display_name,
            },
            "current_performance": {
                "max_elo": latest.max_elo,
                "win_rate": latest.win_rate,
                "total_games": latest.total_games,
                "performance_rating": latest.performance_rating,
            },
            "efficiency": {
                "total_cost": latest.total_cost,
                "cost_per_game": latest.cost_per_game,
                "cost_per_elo": latest.cost_per_elo,
                "efficiency_score": latest.efficiency_score,
            },
            "trends": {
                "benchmark_count": len(history),
                "elo_progression": [h.max_elo for h in history],
                "cost_progression": [h.total_cost for h in history],
                "trend": self._calculate_trend(latest),
            },
        }

        # Calculate statistics if multiple benchmarks
        if len(history) > 1:
            elos = [h.max_elo for h in history]
            analysis["statistics"] = {
                "mean_elo": mean(elos),
                "median_elo": median(elos),
                "std_dev_elo": stdev(elos) if len(elos) > 1 else 0,
                "best_performance": max(elos),
                "worst_performance": min(elos),
                "improvement_rate": (
                    (elos[-1] - elos[0]) / len(elos) if len(elos) > 1 else 0
                ),
            }

        return analysis


def create_benchmark_record(
    run_id: str,
    timestamp: datetime,
    config: Dict[str, Any],
    results: BenchmarkResult,
    budget_summary: Optional[BudgetSummary] = None,
) -> BenchmarkRecord:
    """Create a benchmark record from results."""

    models = []
    total_cost = budget_summary.total_cost if budget_summary else 0.0

    for bot_name, ladder_stats in results.bot_results.items():
        from ..llm.client import parse_bot_spec

        bot = next(
            bot for bot in parse_bot_spec(config["bots"]) if bot.name == bot_name
        )
        provider, model_name = bot.provider, bot.model

        # Calculate metrics
        total_games = ladder_stats.total_games
        elo_history = [game.elo for game in ladder_stats.games if game.completed]
        avg_elo = mean(elo_history) if elo_history else 0

        # Calculate consistency score (standard deviation of ELO)
        consistency_score = stdev(elo_history) if len(elo_history) > 1 else 0

        # Calculate improvement rate
        if len(elo_history) > 1:
            improvement_rate = (elo_history[-1] - elo_history[0]) / len(elo_history)
        else:
            improvement_rate = 0

        # Get cost info from budget summary
        bot_cost = (
            budget_summary.costs_by_bot.get(bot_name, 0.0) if budget_summary else 0.0
        )
        cost_per_game = bot_cost / total_games if total_games > 0 else 0.0
        cost_per_elo = (
            bot_cost / ladder_stats.max_elo_reached
            if ladder_stats.max_elo_reached > 0
            else 0.0
        )

        model_perf = ModelPerformance(
            model_id=f"{provider}:{model_name}",
            provider=provider,
            model_name=model_name,
            display_name=bot_name,
            max_elo=ladder_stats.max_elo_reached,
            avg_elo=avg_elo,
            total_games=total_games,
            wins=ladder_stats.wins,
            draws=ladder_stats.draws,
            losses=ladder_stats.losses,
            win_rate=ladder_stats.win_rate,
            draw_rate=ladder_stats.draw_rate,
            loss_rate=ladder_stats.loss_rate,
            consistency_score=consistency_score,
            improvement_rate=improvement_rate,
            avg_time_per_game=ladder_stats.average_game_duration,
            total_cost=bot_cost,
            cost_per_game=cost_per_game,
            cost_per_elo=cost_per_elo,
            elo_history=elo_history,
            first_seen=timestamp,
            last_seen=timestamp,
            benchmark_count=1,
        )

        models.append(model_perf)

    return BenchmarkRecord(
        run_id=run_id,
        timestamp=timestamp,
        config=config,
        models=models,
        total_games=results.total_games,
        total_cost=total_cost,
        duration=(
            budget_summary.duration
            if budget_summary and budget_summary.duration
            else timedelta(0)
        ),
        version="0.4.0",
    )


# Global results database
_results_db: Optional[ResultsDatabase] = None
_ranking_system: Optional[RankingSystem] = None


def get_results_db(db_path: Optional[Path] = None) -> ResultsDatabase:
    """Get or create the global results database."""
    global _results_db
    if _results_db is None:
        if db_path is None:
            db_path = Path("data/results-v2.db")
        _results_db = ResultsDatabase(db_path)
    return _results_db


def get_ranking_system() -> RankingSystem:
    """Get or create the global ranking system."""
    global _ranking_system
    if _ranking_system is None:
        _ranking_system = RankingSystem(get_results_db())
    return _ranking_system


def store_benchmark_results(
    run_id: str,
    timestamp: datetime,
    config: Dict[str, Any],
    results: BenchmarkResult,
    budget_summary: Optional[BudgetSummary] = None,
) -> None:
    """Store benchmark results in the database."""
    record = create_benchmark_record(run_id, timestamp, config, results, budget_summary)
    db = get_results_db()
    db.store_benchmark(record)
    from ..llm.client import parse_bot_spec

    db.store_games(results, parse_bot_spec(config["bots"]))


def show_leaderboard(limit: int = 20) -> None:
    """Display the current leaderboard."""
    ranking_system = get_ranking_system()
    table = ranking_system.create_leaderboard_table(limit)

    console = Console()
    console.print("\n")
    console.print(table)
    console.print("\n")


def show_provider_comparison() -> None:
    """Display provider comparison."""
    ranking_system = get_ranking_system()
    table = ranking_system.create_provider_comparison_table()

    console = Console()
    console.print("\n")
    console.print(table)
    console.print("\n")


def analyze_model(model_id: str) -> None:
    """Analyze and display detailed model performance."""
    ranking_system = get_ranking_system()
    analysis = ranking_system.analyze_model_performance(model_id)

    console = Console()
    console.print("\n")

    if "error" in analysis:
        console.print(f"[red]Error: {analysis['error']}[/red]")
        return

    # Model info panel
    info = analysis["model_info"]
    current = analysis["current_performance"]
    efficiency = analysis["efficiency"]

    info_text = f"""
    🤖 {info['display_name']}
    📡 Provider: {info['provider'].title()}
    Highest opponent setting: {current['max_elo']}
    📊 Win Rate: {current['win_rate']:.1%}
    🎮 Games Played: {current['total_games']}
    💰 Total Cost: ${efficiency['total_cost']:.4f}
    💵 Cost/Game: ${efficiency['cost_per_game']:.4f}
    ⭐ Rating: {current['performance_rating']}
    """

    console.print(
        Panel(
            info_text.strip(),
            title=f"Model Analysis: {info['display_name']}",
            border_style="blue",
        )
    )

    # Statistics if available
    if "statistics" in analysis:
        stats = analysis["statistics"]
        stats_text = f"""
        Mean opponent setting: {stats['mean_elo']:.0f}
        Median opponent setting: {stats['median_elo']:.0f}
        Opponent setting std dev: {stats['std_dev_elo']:.1f}
        🏅 Best: {stats['best_performance']}
        📉 Worst: {stats['worst_performance']}
        🚀 Improvement Rate: {stats['improvement_rate']:.1f}/benchmark
        """

        console.print(
            Panel(
                stats_text.strip(), title="Historical Statistics", border_style="green"
            )
        )

    console.print("\n")
