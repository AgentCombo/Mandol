# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-28

### Added
- Memory-native hierarchical memory system for long-conversation agents
- Layered memory model with base and high-level memories
- SemanticMap for unified key-value, vector, and graph storage
- SemanticGraph for explicit and implicit relationship management
- Three-way recall (Dense + BM25 + Sparse) → RRF fusion → BFS expansion → Cross-Encoder reranking
- Session management with LLM-driven segmentation
- Entity extraction, deduplication, and relation construction
- Event extraction, deduplication, and causal chain construction
- Cross-session entity/event merging
- Summary generation (episodic, knowledge, emotional, procedural)
- Insight extraction and global merging
- JSON-based save/load persistence
- OpenAI-compatible embedding, LLM, and reranker providers
- SentenceTransformers embedding and reranker providers
- FAISS vector index with adaptive promotion
- In-memory implementations for all ports (testing without external services)
- Milvus and Neo4j infrastructure implementations
- LoCoMo dataset adapter
- Sphinx documentation with furo theme
- README: Environment preparation section (Python version, package manager, system requirements, model download)
- README: Dual-mode quick start (Remote API mode + Local model mode)
- README: Core concepts section with plain language explanations and key terms table
- README: FAQ section covering installation, runtime errors, and performance optimization
- README: Performance section with key metrics and link to benchmarks
- Benchmarks: Complete LoCoMo reproduction guide with test environment, dataset description, key metrics, and ablation experiment details
- Sphinx: Interface status labels using Sphinx admonition directives (✅ Implemented / 🔧 Experimental / 📋 Planned)
- Sphinx: Planned interfaces documented alongside related implemented interfaces (co-locating strategy)

### Changed
- README: Performance test section now links to benchmarks/locomo/README.md instead of inline details
- Sphinx: API documentation hand-written (not auto-generated via autodoc) to preserve planned interfaces
- Terminology unified: "高阶记忆 (High-Level Memory)", "全记忆检索 (Holistic Retrieve)", "语义地图 (SemanticMap)", "语义图 (SemanticGraph)"
- Fixed incorrect type reference paths (src/memory/domain/types.py → mandol/domain/types.py)
