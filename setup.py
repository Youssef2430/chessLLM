#!/usr/bin/env python3
"""
Setup script for Chess LLM Benchmark.

Subscription and API chess benchmarks with explicit outcome and usage telemetry.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text(encoding="utf-8")
    if (this_directory / "README.md").exists()
    else ""
)

# Read version from package
version_file = this_directory / "chess_llm_bench" / "__init__.py"
version = "0.2.0"  # Default version
if version_file.exists():
    with open(version_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

setup(
    name="chess-llm-bench",
    version=version,
    description="Reproducible chess benchmarks and model decision analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chess LLM Bench Team",
    url="https://github.com/Youssef2430/chessLLM",
    project_urls={
        "Bug Reports": "https://github.com/Youssef2430/chessLLM/issues",
        "Source": "https://github.com/Youssef2430/chessLLM",
        "Documentation": "https://github.com/Youssef2430/chessLLM#readme",
    },
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    # Dependencies
    install_requires=[
        "python-chess>=1.999",
        "rich>=13.0.0",
        "python-dotenv>=0.19.0",
    ],
    # Optional dependencies
    extras_require={
        "charts": ["plotly>=6,<7"],
        "reports": ["matplotlib>=3.8,<4"],
        "openai": ["openai>=3.21.0,<4"],
        "anthropic": ["anthropic>=1.9.0,<2"],
        "gemini": ["google-genai>=2.25.0,<3"],
        "all": [
            "google-genai>=2.25.0,<3",
            "openai>=3.21.0,<4",
            "anthropic>=1.9.0,<2",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.0.0",
        ],
    },
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "chess-llm-bench=chess_llm_bench.cli:main",
            "chess-llm-benchmark=chess_llm_bench.cli:main",
            "chess-subscriptions=chess_llm_bench.runner:main",
            "chess-observatory=chess_llm_bench.observatory:main",
        ],
    },
    # Package data
    include_package_data=True,
    package_data={
        "chess_llm_bench": ["*.txt", "*.md", "web/*.html", "web/*.css", "web/*.js"],
    },
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Framework :: AsyncIO",
    ],
    # Keywords
    keywords=[
        "chess",
        "llm",
        "benchmark",
        "elo",
        "rating",
        "stockfish",
        "openai",
        "anthropic",
        "artificial-intelligence",
        "game-playing",
        "evaluation",
    ],
    # Minimum Python version
    zip_safe=False,
)
