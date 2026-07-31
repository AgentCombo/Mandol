.PHONY: install dev artifact artifact-cpu test test-unit test-integration \
	syntax lint lint-all lint-fix format docs docs-clean build clean

PYTHON_DIRS := src/mandol benchmark_locomo benchmark_longmemeval benchmark_self_host examples
TEST_DIR := examples/mandol_chat/tests
STORAGE_TEST := tests/test_rocksdb_tiered_cache.py

install:
	uv sync

dev:
	uv sync --extra dev --extra docs --group spacy-model

artifact:
	uv sync --extra dev --extra cuda --group spacy-model

artifact-cpu:
	uv sync --extra dev --group spacy-model

test:
	uv run pytest $(STORAGE_TEST) $(TEST_DIR) -v

test-unit:
	uv run pytest $(STORAGE_TEST) $(TEST_DIR)/test_benchmark_isolation.py -v

test-integration:
	uv run pytest \
		$(TEST_DIR)/test_health.py \
		$(TEST_DIR)/test_chat_api.py \
		$(TEST_DIR)/test_session_api.py -v

syntax:
	uv run python -m compileall -q $(PYTHON_DIRS)

lint:
	uv run ruff check --select E9,F63,F7,F82 src/mandol/ tests/ $(TEST_DIR)/

lint-all:
	uv run ruff check src/mandol/ tests/ $(TEST_DIR)/

lint-fix:
	uv run ruff check --fix src/mandol/ tests/ $(TEST_DIR)/

format:
	uv run ruff format src/mandol/ tests/ $(TEST_DIR)/

docs:
	uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build/

build:
	uv build

clean:
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/ htmlcov/ docs/_build/ src/*.egg-info/
	find $(PYTHON_DIRS) -type d -name __pycache__ -prune -exec rm -rf {} +
	find $(PYTHON_DIRS) -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
