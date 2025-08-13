#!/usr/bin/env python3
"""
Setup script for Chess LLM Benchmark.

A tool for testing Large Language Models with chess games and assessing their ELOs
by running them against Stockfish at various rating levels.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read version from package
version_file = this_directory / "chess_llm_bench" / "__init__.py"
version = "0.2.0"  # Default version
if version_file.exists():
    with open(version_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                version = line.split('=')[1].strip().strip('"').strip("'")
                break

setup(
    name="chess-llm-bench",
    version=version,
    description="A tool for testing LLMs with chess games and assessing their ELOs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chess LLM Bench Team",
    author_email="chess-llm-bench@example.com",
    url="https://github.com/yourusername/chess-llm-bench",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/chess-llm-bench/issues",
        "Source": "https://github.com/yourusername/chess-llm-bench",
        "Documentation": "https://github.com/yourusername/chess-llm-bench#readme",
    },

    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",

    # Dependencies
    install_requires=[
        "python-chess[engine]>=1.999",
        "rich>=13.0.0",
    ],

    # Optional dependencies
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.3.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "mypy>=1.0.0",
            "types-python-chess",
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
        ],
    },

    # Package data
    include_package_data=True,
    package_data={
        "chess_llm_bench": ["*.txt", "*.md"],
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
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

    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    ],
)
