"""
Chess Puzzle Analysis and Reporting Module

This module provides comprehensive analysis and reporting capabilities for puzzle-solving
performance. It generates detailed statistics, performance comparisons, trend analysis,
and exportable reports to help understand LLM chess abilities across different domains.

The analysis includes:
- Performance breakdown by puzzle type, difficulty, and tactical motif
- Statistical analysis and confidence intervals
- Performance trends and improvement tracking
- Comparative analysis between different models
- Detailed error analysis and common failure patterns
- Export capabilities for further analysis
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import csv

from . import (
    PuzzlePosition, PuzzleAttempt, PuzzleStats, PuzzleSet,
    PuzzleType, TacticMotif, EndgameType
)


class PuzzleAnalyzer:
    """
    Comprehensive analyzer for puzzle-solving performance data.

    Provides statistical analysis, trend detection, comparative evaluation,
    and detailed reporting capabilities for LLM puzzle-solving performance.
    """

    def __init__(self):
        """Initialize the puzzle analyzer."""
        self.attempts: List[PuzzleAttempt] = []
        self.puzzles: Dict[str, PuzzlePosition] = {}
        self.sessions: List[Dict[str, Any]] = []

    def add_session_data(self, session_name: str, attempts: List[PuzzleAttempt],
                        puzzles: List[PuzzlePosition], metadata: Optional[Dict] = None) -> None:
        """
        Add data from a puzzle-solving session.

        Args:
            session_name: Name/identifier for the session
            attempts: List of puzzle attempts from the session
            puzzles: List of puzzles used in the session
            metadata: Optional session metadata (bot info, config, etc.)
        """
        session_data = {
            'name': session_name,
            'timestamp': datetime.now(),
            'attempts': attempts,
            'puzzles': {attempt.puzzle_id: puzzle for attempt, puzzle in zip(attempts, puzzles)},
            'metadata': metadata or {}
        }

        self.sessions.append(session_data)
        self.attempts.extend(attempts)

        # Update puzzle database
        for attempt, puzzle in zip(attempts, puzzles):
            self.puzzles[attempt.puzzle_id] = puzzle

    def generate_comprehensive_report(self, output_path: Path,
                                    include_charts: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.

        Args:
            output_path: Directory to save the report
            include_charts: Whether to generate chart files

        Returns:
            Dictionary containing all analysis results
        """
        if not self.attempts:
            raise ValueError("No puzzle attempt data available for analysis")

        report_data = {
            'generated_at': datetime.now().isoformat(),
            'total_attempts': len(self.attempts),
            'sessions_analyzed': len(self.sessions),
            'overview': self._generate_overview(),
            'performance_by_type': self._analyze_by_puzzle_type(),
            'performance_by_difficulty': self._analyze_by_difficulty(),
            'performance_by_motif': self._analyze_by_motif(),
            'timing_analysis': self._analyze_timing(),
            'error_analysis': self._analyze_errors(),
            'comparative_analysis': self._comparative_analysis(),
            'trends': self._analyze_trends(),
            'recommendations': self._generate_recommendations()
        }

        # Save main report
        report_file = output_path / f"puzzle_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate human-readable report
        self._generate_text_report(output_path, report_data)

        # Generate CSV exports
        self._export_csv_data(output_path)

        if include_charts:
            self._generate_charts(output_path, report_data)

        return report_data

    def _generate_overview(self) -> Dict[str, Any]:
        """Generate overall performance overview."""
        correct_attempts = [a for a in self.attempts if a.is_correct]

        return {
            'total_attempts': len(self.attempts),
            'correct_solutions': len(correct_attempts),
            'success_rate': len(correct_attempts) / len(self.attempts) if self.attempts else 0.0,
            'average_response_time': statistics.mean([a.response_time for a in self.attempts]),
            'median_response_time': statistics.median([a.response_time for a in self.attempts]),
            'total_illegal_moves': sum(a.illegal_attempts for a in self.attempts),
            'sessions_count': len(self.sessions),
            'unique_puzzles': len(set(a.puzzle_id for a in self.attempts)),
            'time_period': {
                'start': min(a.timestamp for a in self.attempts).isoformat(),
                'end': max(a.timestamp for a in self.attempts).isoformat()
            } if self.attempts else None
        }

    def _analyze_by_puzzle_type(self) -> Dict[str, Any]:
        """Analyze performance by puzzle type."""
        type_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'times': []})

        for attempt in self.attempts:
            puzzle = self.puzzles.get(attempt.puzzle_id)
            if puzzle:
                puzzle_type = puzzle.puzzle_type.value
                type_stats[puzzle_type]['total'] += 1
                type_stats[puzzle_type]['times'].append(attempt.response_time)
                if attempt.is_correct:
                    type_stats[puzzle_type]['correct'] += 1

        results = {}
        for puzzle_type, stats in type_stats.items():
            results[puzzle_type] = {
                'success_rate': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct_solutions': stats['correct'],
                'total_attempts': stats['total'],
                'average_time': statistics.mean(stats['times']) if stats['times'] else 0.0,
                'median_time': statistics.median(stats['times']) if stats['times'] else 0.0
            }

        return results

    def _analyze_by_difficulty(self) -> Dict[str, Any]:
        """Analyze performance by difficulty level."""
        difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'times': []})

        for attempt in self.attempts:
            puzzle = self.puzzles.get(attempt.puzzle_id)
            if puzzle:
                difficulty = puzzle.difficulty
                difficulty_stats[difficulty]['total'] += 1
                difficulty_stats[difficulty]['times'].append(attempt.response_time)
                if attempt.is_correct:
                    difficulty_stats[difficulty]['correct'] += 1

        results = {}
        for difficulty, stats in difficulty_stats.items():
            results[str(difficulty)] = {
                'success_rate': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct_solutions': stats['correct'],
                'total_attempts': stats['total'],
                'average_time': statistics.mean(stats['times']) if stats['times'] else 0.0
            }

        # Calculate difficulty correlation
        if len(results) > 1:
            difficulties = sorted([int(d) for d in results.keys()])
            success_rates = [results[str(d)]['success_rate'] for d in difficulties]
            correlation = self._calculate_correlation(difficulties, success_rates)
            results['difficulty_correlation'] = correlation

        return results

    def _analyze_by_motif(self) -> Dict[str, Any]:
        """Analyze performance by tactical motif."""
        motif_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'times': []})

        for attempt in self.attempts:
            puzzle = self.puzzles.get(attempt.puzzle_id)
            if puzzle and puzzle.tactic_motif:
                motif = puzzle.tactic_motif.value
                motif_stats[motif]['total'] += 1
                motif_stats[motif]['times'].append(attempt.response_time)
                if attempt.is_correct:
                    motif_stats[motif]['correct'] += 1

        results = {}
        for motif, stats in motif_stats.items():
            results[motif] = {
                'success_rate': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct_solutions': stats['correct'],
                'total_attempts': stats['total'],
                'average_time': statistics.mean(stats['times']) if stats['times'] else 0.0
            }

        return results

    def _analyze_timing(self) -> Dict[str, Any]:
        """Analyze response timing patterns."""
        times = [a.response_time for a in self.attempts]
        correct_times = [a.response_time for a in self.attempts if a.is_correct]
        incorrect_times = [a.response_time for a in self.attempts if not a.is_correct]

        analysis = {
            'overall': {
                'mean': statistics.mean(times) if times else 0,
                'median': statistics.median(times) if times else 0,
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }
        }

        if correct_times:
            analysis['correct_solutions'] = {
                'mean': statistics.mean(correct_times),
                'median': statistics.median(correct_times),
                'std_dev': statistics.stdev(correct_times) if len(correct_times) > 1 else 0
            }

        if incorrect_times:
            analysis['incorrect_solutions'] = {
                'mean': statistics.mean(incorrect_times),
                'median': statistics.median(incorrect_times),
                'std_dev': statistics.stdev(incorrect_times) if len(incorrect_times) > 1 else 0
            }

        # Time vs success correlation
        if len(times) > 1:
            success_values = [1 if a.is_correct else 0 for a in self.attempts]
            time_success_correlation = self._calculate_correlation(times, success_values)
            analysis['time_success_correlation'] = time_success_correlation

        return analysis

    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns and common failures."""
        incorrect_attempts = [a for a in self.attempts if not a.is_correct]

        error_analysis = {
            'total_errors': len(incorrect_attempts),
            'error_rate': len(incorrect_attempts) / len(self.attempts) if self.attempts else 0,
            'illegal_move_analysis': self._analyze_illegal_moves(),
            'error_by_type': defaultdict(int),
            'error_by_difficulty': defaultdict(int),
            'timeout_errors': len([a for a in incorrect_attempts if 'TIMEOUT' in a.raw_response])
        }

        # Categorize errors by puzzle type and difficulty
        for attempt in incorrect_attempts:
            puzzle = self.puzzles.get(attempt.puzzle_id)
            if puzzle:
                error_analysis['error_by_type'][puzzle.puzzle_type.value] += 1
                error_analysis['error_by_difficulty'][puzzle.difficulty] += 1

        return dict(error_analysis)

    def _analyze_illegal_moves(self) -> Dict[str, Any]:
        """Analyze illegal move patterns."""
        total_illegal = sum(a.illegal_attempts for a in self.attempts)
        attempts_with_illegal = [a for a in self.attempts if a.illegal_attempts > 0]

        return {
            'total_illegal_moves': total_illegal,
            'attempts_with_illegal': len(attempts_with_illegal),
            'illegal_move_rate': total_illegal / len(self.attempts) if self.attempts else 0,
            'average_illegal_per_attempt': total_illegal / len(self.attempts) if self.attempts else 0,
            'max_illegal_in_single_attempt': max(a.illegal_attempts for a in self.attempts) if self.attempts else 0
        }

    def _comparative_analysis(self) -> Dict[str, Any]:
        """Compare performance across different sessions/models."""
        if len(self.sessions) < 2:
            return {'message': 'Insufficient sessions for comparative analysis'}

        session_comparison = {}
        for session in self.sessions:
            session_attempts = session['attempts']
            correct = sum(1 for a in session_attempts if a.is_correct)
            total = len(session_attempts)
            avg_time = statistics.mean([a.response_time for a in session_attempts]) if session_attempts else 0

            session_comparison[session['name']] = {
                'success_rate': correct / total if total > 0 else 0,
                'total_attempts': total,
                'correct_solutions': correct,
                'average_time': avg_time,
                'timestamp': session['timestamp'].isoformat()
            }

        # Find best and worst performing sessions
        if session_comparison:
            best_session = max(session_comparison.items(), key=lambda x: x[1]['success_rate'])
            worst_session = min(session_comparison.items(), key=lambda x: x[1]['success_rate'])

            return {
                'session_comparison': session_comparison,
                'best_session': {
                    'name': best_session[0],
                    'success_rate': best_session[1]['success_rate']
                },
                'worst_session': {
                    'name': worst_session[0],
                    'success_rate': worst_session[1]['success_rate']
                },
                'performance_spread': best_session[1]['success_rate'] - worst_session[1]['success_rate']
            }

        return session_comparison

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.sessions) < 2:
            return {'message': 'Insufficient data for trend analysis'}

        # Sort sessions by timestamp
        sorted_sessions = sorted(self.sessions, key=lambda x: x['timestamp'])

        success_rates = []
        timestamps = []

        for session in sorted_sessions:
            attempts = session['attempts']
            if attempts:
                correct = sum(1 for a in attempts if a.is_correct)
                success_rate = correct / len(attempts)
                success_rates.append(success_rate)
                timestamps.append(session['timestamp'])

        if len(success_rates) < 2:
            return {'message': 'Insufficient data points for trend analysis'}

        # Calculate trend
        trend_slope = self._calculate_trend_slope(success_rates)

        return {
            'trend_direction': 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable',
            'trend_slope': trend_slope,
            'first_session_rate': success_rates[0],
            'last_session_rate': success_rates[-1],
            'improvement': success_rates[-1] - success_rates[0],
            'session_count': len(sorted_sessions)
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        if not self.attempts:
            return ["No data available for recommendations"]

        # Overall success rate recommendations
        overall_success = len([a for a in self.attempts if a.is_correct]) / len(self.attempts)
        if overall_success < 0.3:
            recommendations.append("Low overall success rate. Consider starting with easier puzzles or reviewing basic chess tactics.")
        elif overall_success > 0.8:
            recommendations.append("High success rate achieved. Consider advancing to more challenging puzzle sets.")

        # Timing recommendations
        avg_time = statistics.mean([a.response_time for a in self.attempts])
        if avg_time > 25:
            recommendations.append("Response times are quite high. Consider optimizing prompt length or model parameters.")

        # Illegal move recommendations
        illegal_rate = sum(a.illegal_attempts for a in self.attempts) / len(self.attempts)
        if illegal_rate > 0.1:
            recommendations.append("High rate of illegal moves detected. Consider improving move parsing or adding move validation training.")

        # Type-specific recommendations
        type_analysis = self._analyze_by_puzzle_type()
        for puzzle_type, stats in type_analysis.items():
            if stats['total_attempts'] >= 5 and stats['success_rate'] < 0.2:
                recommendations.append(f"Poor performance on {puzzle_type} puzzles. Consider focused training on this area.")

        # Difficulty recommendations
        difficulty_analysis = self._analyze_by_difficulty()
        if 'difficulty_correlation' in difficulty_analysis:
            correlation = difficulty_analysis['difficulty_correlation']
            if correlation > -0.3:
                recommendations.append("Difficulty scaling may not be appropriate. Success rate should decrease with higher difficulty.")

        return recommendations

    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))

        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        n = len(values)
        if n < 2:
            return 0.0

        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        return numerator / denominator if denominator != 0 else 0.0

    def _generate_text_report(self, output_path: Path, report_data: Dict[str, Any]) -> None:
        """Generate human-readable text report."""
        report_file = output_path / f"puzzle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_file, 'w') as f:
            f.write("CHESS PUZZLE PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overview
            overview = report_data['overview']
            f.write(f"OVERVIEW\n")
            f.write(f"--------\n")
            f.write(f"Total Attempts: {overview['total_attempts']}\n")
            f.write(f"Correct Solutions: {overview['correct_solutions']}\n")
            f.write(f"Success Rate: {overview['success_rate']:.1%}\n")
            f.write(f"Average Response Time: {overview['average_response_time']:.2f}s\n")
            f.write(f"Median Response Time: {overview['median_response_time']:.2f}s\n")
            f.write(f"Total Illegal Moves: {overview['total_illegal_moves']}\n\n")

            # Performance by type
            f.write("PERFORMANCE BY PUZZLE TYPE\n")
            f.write("-" * 25 + "\n")
            for puzzle_type, stats in report_data['performance_by_type'].items():
                f.write(f"{puzzle_type.title()}: {stats['success_rate']:.1%} "
                       f"({stats['correct_solutions']}/{stats['total_attempts']}) "
                       f"- Avg time: {stats['average_time']:.2f}s\n")
            f.write("\n")

            # Performance by difficulty
            f.write("PERFORMANCE BY DIFFICULTY\n")
            f.write("-" * 24 + "\n")
            difficulty_stats = report_data['performance_by_difficulty']
            for difficulty in sorted(difficulty_stats.keys()):
                if difficulty != 'difficulty_correlation':
                    stats = difficulty_stats[difficulty]
                    f.write(f"Difficulty {difficulty}: {stats['success_rate']:.1%} "
                           f"({stats['correct_solutions']}/{stats['total_attempts']})\n")
            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            for i, rec in enumerate(report_data['recommendations'], 1):
                f.write(f"{i}. {rec}\n")

    def _export_csv_data(self, output_path: Path) -> None:
        """Export detailed data to CSV files."""
        # Export attempt details
        attempts_file = output_path / f"puzzle_attempts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(attempts_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'puzzle_id', 'llm_move', 'is_correct', 'response_time',
                'evaluation_delta', 'illegal_attempts', 'timestamp',
                'puzzle_type', 'difficulty', 'best_move'
            ])

            for attempt in self.attempts:
                puzzle = self.puzzles.get(attempt.puzzle_id)
                writer.writerow([
                    attempt.puzzle_id,
                    attempt.llm_move,
                    attempt.is_correct,
                    attempt.response_time,
                    attempt.evaluation_delta,
                    attempt.illegal_attempts,
                    attempt.timestamp.isoformat(),
                    puzzle.puzzle_type.value if puzzle else '',
                    puzzle.difficulty if puzzle else '',
                    puzzle.best_move if puzzle else ''
                ])

    def _generate_charts(self, output_path: Path, report_data: Dict[str, Any]) -> None:
        """Generate simple text-based charts (placeholder for actual charting)."""
        # This is a placeholder for actual chart generation
        # In a real implementation, you might use matplotlib, plotly, or similar
        charts_file = output_path / f"puzzle_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(charts_file, 'w') as f:
            f.write("PUZZLE PERFORMANCE CHARTS\n")
            f.write("=" * 25 + "\n\n")

            # Simple text bar chart for puzzle types
            f.write("Success Rate by Puzzle Type:\n")
            for puzzle_type, stats in report_data['performance_by_type'].items():
                bar_length = int(stats['success_rate'] * 20)  # Scale to 20 chars
                bar = '█' * bar_length + '░' * (20 - bar_length)
                f.write(f"{puzzle_type:10} |{bar}| {stats['success_rate']:.1%}\n")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get quick summary statistics."""
        if not self.attempts:
            return {}

        correct = len([a for a in self.attempts if a.is_correct])

        return {
            'total_attempts': len(self.attempts),
            'success_rate': correct / len(self.attempts),
            'average_time': statistics.mean([a.response_time for a in self.attempts]),
            'illegal_moves': sum(a.illegal_attempts for a in self.attempts),
            'sessions': len(self.sessions)
        }


def compare_puzzle_performance(analyzer1: PuzzleAnalyzer, analyzer2: PuzzleAnalyzer,
                             name1: str = "Model 1", name2: str = "Model 2") -> Dict[str, Any]:
    """
    Compare puzzle performance between two analyzers.

    Args:
        analyzer1: First puzzle analyzer
        analyzer2: Second puzzle analyzer
        name1: Name for first model
        name2: Name for second model

    Returns:
        Comparison analysis dictionary
    """
    stats1 = analyzer1.get_summary_stats()
    stats2 = analyzer2.get_summary_stats()

    if not stats1 or not stats2:
        return {'error': 'Insufficient data for comparison'}

    comparison = {
        'models': {
            name1: stats1,
            name2: stats2
        },
        'comparison': {
            'success_rate_difference': stats2['success_rate'] - stats1['success_rate'],
            'time_difference': stats2['average_time'] - stats1['average_time'],
            'better_model': name1 if stats1['success_rate'] > stats2['success_rate'] else name2,
            'improvement_percentage': ((stats2['success_rate'] - stats1['success_rate']) / stats1['success_rate'] * 100) if stats1['success_rate'] > 0 else 0
        }
    }

    return comparison


def load_analysis_from_files(data_dir: Path) -> PuzzleAnalyzer:
    """
    Load puzzle analysis data from saved files.

    Args:
        data_dir: Directory containing saved puzzle data

    Returns:
        PuzzleAnalyzer with loaded data
    """
    analyzer = PuzzleAnalyzer()

    # Look for CSV files with attempt data
    for csv_file in data_dir.glob("*attempts*.csv"):
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                attempts = []
                puzzles = []

                for row in reader:
                    # Reconstruct attempt
                    attempt = PuzzleAttempt(
                        puzzle_id=row['puzzle_id'],
                        llm_move=row['llm_move'],
                        response_time=float(row['response_time']),
                        is_correct=row['is_correct'].lower() == 'true',
                        evaluation_delta=float(row['evaluation_delta']) if row['evaluation_delta'] else None,
                        illegal_attempts=int(row['illegal_attempts']),
                        timestamp=datetime.fromisoformat(row['timestamp'])
                    )
                    attempts.append(attempt)

                # Add to analyzer
                session_name = csv_file.stem
                analyzer.add_session_data(session_name, attempts, puzzles)

        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    return analyzer
