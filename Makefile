# Chess LLM Benchmark - Makefile
# Development and build automation

.PHONY: help install install-dev test test-unit test-self lint format type-check clean demo run-quick run-standard docs build upload

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PACKAGE := chess_llm_bench
TEST_DIR := tests
MAIN_SCRIPT := main.py

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)Chess LLM Benchmark - Development Tasks$(RESET)"
	@echo "$(CYAN)=====================================$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(RESET)"
	@echo "  make install-dev  # Set up development environment"
	@echo "  make demo         # Run a quick demo"
	@echo "  make test         # Run all tests"

install: ## Install package and dependencies
	@echo "$(BLUE)Installing package and dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install development dependencies and package
	@echo "$(BLUE)Installing development environment...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Development environment ready!$(RESET)"

install-all: ## Install with all optional dependencies
	@echo "$(BLUE)Installing with all optional dependencies...$(RESET)"
	$(PIP) install -e ".[all,dev]"

test: test-unit test-self ## Run all tests

test-unit: ## Run unit tests with pytest
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR)/ -v --tb=short
	@echo "$(GREEN)Unit tests completed!$(RESET)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR)/ --cov=$(PACKAGE) --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

test-self: ## Run built-in self-tests
	@echo "$(BLUE)Running self-tests...$(RESET)"
	$(PYTHON) $(MAIN_SCRIPT) --self-test
	@echo "$(GREEN)Self-tests completed!$(RESET)"

lint: ## Run linting with flake8
	@echo "$(BLUE)Running linting...$(RESET)"
	$(PYTHON) -m flake8 $(PACKAGE)/ $(TEST_DIR)/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "$(GREEN)Linting completed!$(RESET)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(PYTHON) -m black $(PACKAGE)/ $(TEST_DIR)/ --line-length=100
	$(PYTHON) -m isort $(PACKAGE)/ $(TEST_DIR)/ --profile=black --line-length=100
	@echo "$(GREEN)Code formatting completed!$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(PYTHON) -m black --check $(PACKAGE)/ $(TEST_DIR)/ --line-length=100
	$(PYTHON) -m isort --check $(PACKAGE)/ $(TEST_DIR)/ --profile=black --line-length=100

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(RESET)"
	$(PYTHON) -m mypy $(PACKAGE)/ --ignore-missing-imports
	@echo "$(GREEN)Type checking completed!$(RESET)"

quality: lint format type-check ## Run all code quality checks

demo: ## Run a quick demo with random bots
	@echo "$(BLUE)Running demo...$(RESET)"
	$(PYTHON) $(MAIN_SCRIPT) --demo

run-quick: ## Run quick benchmark (600-800 ELO)
	@echo "$(BLUE)Running quick benchmark...$(RESET)"
	$(PYTHON) $(MAIN_SCRIPT) --bots "random::quick1,random::quick2" --start-elo 600 --max-elo 800 --elo-step 100

run-standard: ## Run standard benchmark (requires API keys)
	@echo "$(BLUE)Running standard benchmark...$(RESET)"
	@echo "$(YELLOW)Note: Requires OPENAI_API_KEY environment variable$(RESET)"
	$(PYTHON) $(MAIN_SCRIPT) --bots "openai:gpt-4o-mini:test,random::baseline" --start-elo 600 --max-elo 1200

run-comparison: ## Run model comparison (requires API keys)
	@echo "$(BLUE)Running model comparison...$(RESET)"
	@echo "$(YELLOW)Note: Requires OPENAI_API_KEY environment variable$(RESET)"
	$(PYTHON) $(MAIN_SCRIPT) --bots "openai:gpt-4o:gpt4o,openai:gpt-4o-mini:gpt4o-mini" --start-elo 800 --max-elo 1600

clean: ## Clean up generated files and caches
	@echo "$(BLUE)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf runs/
	@echo "$(GREEN)Cleanup completed!$(RESET)"

clean-runs: ## Clean up only benchmark result files
	@echo "$(BLUE)Cleaning benchmark results...$(RESET)"
	rm -rf runs/
	@echo "$(GREEN)Benchmark results cleaned!$(RESET)"

docs: ## Generate documentation (placeholder)
	@echo "$(YELLOW)Documentation generation not implemented yet$(RESET)"
	@echo "See README.md for current documentation"

build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)Packages built in dist/$(RESET)"

upload-test: build ## Upload to TestPyPI
	@echo "$(BLUE)Uploading to TestPyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)Uploaded to TestPyPI!$(RESET)"

upload: build ## Upload to PyPI
	@echo "$(BLUE)Uploading to PyPI...$(RESET)"
	@echo "$(RED)WARNING: This will upload to the real PyPI!$(RESET)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)Uploaded to PyPI!$(RESET)"

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(RESET)"
	$(PIP) list --outdated

update-deps: ## Update dependencies (be careful!)
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	@echo "$(YELLOW)This will update all packages. Continue? (Ctrl+C to cancel)$(RESET)"
	@sleep 3
	$(PIP) install --upgrade -r requirements.txt

stockfish-check: ## Check if Stockfish is available
	@echo "$(BLUE)Checking Stockfish availability...$(RESET)"
	@if command -v stockfish >/dev/null 2>&1; then \
		echo "$(GREEN)Stockfish found: $$(command -v stockfish)$(RESET)"; \
		stockfish --help | head -3; \
	else \
		echo "$(RED)Stockfish not found in PATH$(RESET)"; \
		echo "$(YELLOW)Install with:$(RESET)"; \
		echo "  macOS:    brew install stockfish"; \
		echo "  Ubuntu:   sudo apt-get install stockfish"; \
		echo "  Windows:  choco install stockfish"; \
	fi

env-check: stockfish-check ## Check environment setup
	@echo "$(BLUE)Checking environment...$(RESET)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@if [ -n "$$OPENAI_API_KEY" ]; then \
		echo "$(GREEN)OPENAI_API_KEY is set$(RESET)"; \
	else \
		echo "$(YELLOW)OPENAI_API_KEY is not set$(RESET)"; \
	fi
	@if [ -n "$$ANTHROPIC_API_KEY" ]; then \
		echo "$(GREEN)ANTHROPIC_API_KEY is set$(RESET)"; \
	else \
		echo "$(YELLOW)ANTHROPIC_API_KEY is not set$(RESET)"; \
	fi

benchmark-suite: ## Run a comprehensive benchmark suite
	@echo "$(BLUE)Running comprehensive benchmark suite...$(RESET)"
	@echo "$(YELLOW)This will take a while and requires API keys$(RESET)"
	@mkdir -p benchmark_results
	@echo "1. Running random baseline..." && $(PYTHON) $(MAIN_SCRIPT) --bots "random::baseline1,random::baseline2" --output-dir benchmark_results/baseline
	@if [ -n "$$OPENAI_API_KEY" ]; then \
		echo "2. Running OpenAI models..." && $(PYTHON) $(MAIN_SCRIPT) --bots "openai:gpt-4o-mini:gpt4o-mini" --output-dir benchmark_results/openai; \
	fi
	@if [ -n "$$ANTHROPIC_API_KEY" ]; then \
		echo "3. Running Anthropic models..." && $(PYTHON) $(MAIN_SCRIPT) --bots "anthropic:claude-3-haiku:claude" --output-dir benchmark_results/anthropic; \
	fi
	@echo "$(GREEN)Benchmark suite completed! Results in benchmark_results/$(RESET)"

# Development workflow shortcuts
dev-setup: install-dev env-check ## Complete development setup
	@echo "$(GREEN)Development environment is ready!$(RESET)"
	@echo "$(CYAN)Try: make demo$(RESET)"

ci: quality test ## Run CI pipeline (quality checks + tests)
	@echo "$(GREEN)CI pipeline completed successfully!$(RESET)"

pre-commit: format lint test-unit ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed!$(RESET)"

# Help with examples
examples: ## Show usage examples
	@echo "$(CYAN)Chess LLM Benchmark - Usage Examples$(RESET)"
	@echo "$(CYAN)================================$(RESET)"
	@echo ""
	@echo "$(GREEN)Basic Usage:$(RESET)"
	@echo "  make demo                    # Quick demo with random bots"
	@echo "  make run-quick              # Quick benchmark (600-800 ELO)"
	@echo ""
	@echo "$(GREEN)With API Keys:$(RESET)"
	@echo "  export OPENAI_API_KEY=sk-... "
	@echo "  make run-standard           # Standard benchmark with OpenAI"
	@echo "  make run-comparison         # Compare different models"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  make dev-setup              # Set up development environment"
	@echo "  make test                   # Run all tests"
	@echo "  make quality                # Code quality checks"
	@echo "  make pre-commit             # Run before committing"
	@echo ""
	@echo "$(GREEN)Advanced:$(RESET)"
	@echo "  make benchmark-suite        # Comprehensive benchmark"
	@echo "  make build                  # Build distribution packages"
