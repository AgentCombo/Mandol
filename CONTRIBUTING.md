# Contributing to Mandol

Thank you for your interest in contributing to Mandol.

The `paper-repro` branch is the frozen paper artifact. Ongoing development and
general contributions should normally target `main`; changes to `paper-repro`
should be limited to corrections that preserve the published experiment setup.

## Development Setup

1. **Fork and clone the repository**:

   ```bash
   git clone https://github.com/AgentCombo/Mandol.git
   cd Mandol
   ```

2. **Install the Python 3.12 development environment with `uv`**:

   ```bash
   uv sync --extra dev --extra docs --group spacy-model
   ```

   For the CUDA environment used by the paper artifact, also add `--extra cuda`.
   The pinned flash-attention wheel is specific to Linux x86_64, Python 3.12,
   Torch 2.8, and CUDA 12.

3. **Install pre-commit hooks**:

   ```bash
   uv run pre-commit install
   ```

## Code Style

- Target Python version: 3.12.
- Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Run the baseline correctness checks with `make lint`.
- Run the complete Ruff rule set with `make lint-all`.
- Use `make lint-fix` and `make format` only after reviewing the affected files.

## Running Tests

```bash
# Current low-cost test suite
make test

# Import and strategy-isolation check
make test-unit

# Mandol chat API integration checks
make test-integration

# Syntax-check package, benchmark, and example sources
make syntax
```

The repository currently keeps its low-cost automated tests under
`examples/mandol_chat/tests`. Benchmark workflows require datasets, model
downloads, and provider credentials; run their smoke or reproduction commands
from the corresponding `REPRODUCE.md` document.

## Pull Request Process

1. Create a feature branch from `main` unless the change specifically targets
   the frozen paper artifact.
2. Make changes with clear, descriptive commit messages.
3. Run `make syntax`, `make test`, and the relevant benchmark smoke checks.
4. Run Ruff on every changed Python file and avoid unrelated formatting churn.
5. Submit a pull request with a clear description of the behavior and tests.

## Reporting Issues

- Use [GitHub Issues](https://github.com/AgentCombo/Mandol/issues) to report bugs or request features.
- Please include:
  - Python version
  - Mandol version
  - Minimal reproduction code
  - Expected vs actual behavior

## Package Areas

- `core/`: `MemoryUnit`, `MemorySpace`, `SemanticMap`, and `SemanticGraph`.
- `retrieval/`: BM25, SPLADE, cosine retrieval, fusion, and reranking.
- `triple_retrieval/`: three-tower retrieval orchestration.
- `auto_builder/`: high-level memory construction and strategy presets.
- `hierarchical/`, `entity_relation/`, and `episodic/`: tower-specific builders
  and retrievers.
- `quantification/` and `memory_router/`: sufficiency checks, query expansion,
  pruning, and benchmark routing policies.
- `llm/`, `storage/`, `cluster/`, and `utils/`: provider, persistence,
  clustering, configuration, and logging support.

Keep changes within the owning package area and preserve persistence and
benchmark compatibility unless the change explicitly documents a migration.
