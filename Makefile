.PHONY: install dev test lint docs clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/mandol/ tests/

lint-fix:
	ruff check --fix src/mandol/ tests/

docs:
	cd docs && make html-all

docs-clean:
	cd docs && make clean-all

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
