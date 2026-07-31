# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Aligned root development commands, environment documentation, and release
  metadata with the current `src/mandol` package and paper artifact layout.
- Updated GitHub Actions for Python 3.12 and `uv`, and separated maintained
  documentation from archived pre-refactor material.

## [0.1.0a1] - 2026-07-06

### Added

- First public alpha release on PyPI and frozen `paper-repro` artifact.
- `MemoryUnit`, `MemorySpace`, `SemanticMap`, and `SemanticGraph` core APIs.
- Dense, BM25, SPLADE, graph-expansion, fusion, and reranking paths through
  `MultiRetriever`.
- Hierarchical, entity-relation, and episodic three-tower retrieval with router
  and semantic-quantification support.
- High-level memory construction through `auto_builder`.
- FAISS, RocksDB, tiered-storage, local-model, and OpenAI-compatible provider
  integrations used by the artifact.
- LoCoMo and LongMemEval paper-reproduction workflows, ablations, performance
  entry points, and self-host validation workflows.
- English and Chinese README guides, Sphinx documentation, and the Docusaurus
  project website.

### Requirements

- Python `>=3.12,<3.13`.
- Linux for the complete paper-performance environment.
- External datasets, model downloads, and provider credentials for full
  benchmark reproduction.

[Unreleased]: https://github.com/AgentCombo/Mandol/compare/v0.1.0-paper-repro...paper-repro
[0.1.0a1]: https://pypi.org/project/mandol/0.1.0a1/
