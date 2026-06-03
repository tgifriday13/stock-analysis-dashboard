.PHONY: install install-dev test lint format clean run run-msft help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install all dependencies (prod + dev)
	pip install -r requirements.txt -r requirements-dev.txt

test:  ## Run the test suite
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=src --cov-report=term-missing

lint:  ## Lint the codebase with ruff
	ruff check src tests scripts

format:  ## Auto-format with black + isort
	black src tests scripts
	isort src tests scripts

clean:  ## Remove cache, build artifacts, and generated outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info

run:  ## Generate all reports for the configured ticker (scripts/run_all_reports.py)
	python scripts/run_all_reports.py

run-msft:  ## Generate MSFT reports (scripts/run_msft_reports.py)
	python scripts/run_msft_reports.py
