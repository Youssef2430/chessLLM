PYTHON ?= python3

.PHONY: help install test estimate smoke
help:
	@echo "install: install dependencies; test: offline tests; estimate: no-API cost scenarios; smoke: free random games"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

estimate:
	$(PYTHON) main.py --estimate-cost --preset latest --max-games 10

smoke:
	$(PYTHON) main.py --bots random::baseline --max-games 2 --max-plies 12

# Read-only dashboard; requires the charts extra.
.PHONY: observatory
observatory:
	$(PYTHON) -m chess_llm_bench.observatory
