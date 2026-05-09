# Contributing to Mandol

Thank you for your interest in contributing to Mandol! We welcome contributions from the community.

## Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-org/mandol.git
   cd mandol
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 100 characters
- Target Python version: 3.9+
- Run linting: `make lint`
- Auto-fix: `make lint-fix`

## Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commit messages
3. Ensure all tests pass: `make test`
4. Ensure linting passes: `make lint`
5. Submit a pull request with a clear description of the changes

## Reporting Issues

- Use [GitHub Issues](https://github.com/your-org/mandol/issues) to report bugs or request features
- Please include:
  - Python version
  - Mandol version
  - Minimal reproduction code
  - Expected vs actual behavior

## Architecture

Mandol follows a hexagonal (ports & adapters) architecture:

- `domain/` — Core data structures (MemoryUnit, MemorySpace, types)
- `ports/` — Abstract interfaces (EmbeddingProvider, VectorIndex, GraphStore, etc.)
- `application/` — Application services (MemorySystem, SemanticMap, SemanticGraph)
- `infrastructure/` — Infrastructure implementations (FAISS, Milvus, Neo4j, etc.)
- `retrieval/` — Retrieval module (BM25, sparse, RRF fusion, pipeline)

When adding new features, please follow this architecture and place code in the appropriate layer.
