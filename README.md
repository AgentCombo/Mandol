[English](README.md) | [中文](README_CN.md)

# Mandol

> General-purpose in-memory hierarchical memory system for AI agents

[![CI](https://github.com/AgentCombo/Mandol/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentCombo/Mandol/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9--3.13-blue.svg)](https://www.python.org/)
[![PyPI version](https://badge.fury.io/py/mandol.svg)](https://badge.fury.io/py/mandol)
[![Docs](https://img.shields.io/badge/docs-Sphinx-green.svg)](docs/)
[![codecov](https://codecov.io/gh/AgentCombo/Mandol/branch/main/graph/badge.svg)](https://codecov.io/gh/AgentCombo/Mandol)

## What is Mandol?

Mandol is a general-purpose agent memory system built on in-memory data structures. It fuses key-value, vector, and graph indices through SemanticMap and SemanticGraph, providing unified storage and hybrid retrieval without inter-process communication. Evaluation on conversational memory benchmarks (LoCoMo: 92.21 F1, LongMemEval: 88.40 F1) is complete; assessment on code generation and image storage scenarios is underway.

## Environment Preparation

### Python Version

- Minimum requirement: Python 3.9+
- Recommended: Python 3.10 or 3.11

### Package Manager

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install mandol
```

Using conda:

```bash
conda create -n mandol python=3.10
conda activate mandol
pip install mandol
```

### System Requirements

| Component | Minimum (Remote API) | Minimum (Local Embedding) | Minimum (Local Embedding + Reranker) | Recommended |
|-----------|---------------------|--------------------------|-------------------------------------|-------------|
| CPU | 4 cores | 4 cores | 8 cores | 8+ cores |
| RAM | 8 GB | 16 GB | 32 GB | 16-64 GB |
| GPU | None (CPU OK) | None (CPU OK, GPU faster) | NVIDIA 8GB+ VRAM recommended | NVIDIA 16GB+ VRAM |
| Disk | 2 GB | 6 GB | 10 GB | 10+ GB |

### Model Download

| Model | Purpose | Size | Download Method |
|-------|---------|------|----------------|
| `Qwen/Qwen3-Embedding-4B` | Text embedding | ~4 GB | Auto-downloads to `~/.cache/huggingface/` on first run |
| `Qwen/Qwen3-Reranker-4B` | Retrieval reranking | ~4 GB | Auto-downloads to `~/.cache/huggingface/` on first run |

> **Tip**: If using remote API mode, no local model download is needed. Just configure the API endpoint.

## Installation

### Basic Install

```bash
pip install mandol
```

Or install from source:

```bash
git clone https://github.com/your-org/mandol.git
cd mandol
pip install -e .
```

### Optional Dependencies

```bash
pip install mandol[faiss]                    # FAISS vector index acceleration
pip install mandol[sentence-transformers]    # Local Embedding/Reranker models
pip install mandol[openai]                   # OpenAI API support
pip install mandol[milvus]                   # Milvus vector database
pip install mandol[neo4j]                    # Neo4j graph database
pip install mandol[all]                      # Install all optional dependencies
pip install mandol[dev]                      # Development tools (pytest, ruff, etc.)
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your API key:
# OPENAI_API_KEY=sk-your-key-here
```

### Verify Installation

```bash
python -c "from mandol import MemorySystem, MemoryUnit, Uid; print('Mandol installed successfully!')"
```

## Quick Start

### Mode 1: Remote API (Quick Start, no local models required)

```python
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")
# In config.yaml, set embedder.use_remote: true and reranker.use_remote: true,
# then configure the API endpoints and keys.

unit = MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
)
system.add(unit)

system.build_high_level(mode="auto")

hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)
for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

system.save("./memory_snapshot")
system2 = MemorySystem.load("./memory_snapshot")
```

> **About ``build_high_level()``**: The system asynchronously detects session boundaries during ``add()`` and automatically triggers high-level memory construction.
> - Retrieving raw data (BASE group): available immediately after ``add()``
> - Retrieving entities/events/summaries (ENTITY / EVENT / SUMMARY groups): wait for automatic construction to complete, or call ``build_high_level()`` manually
> - Retrieving immediately after inserting a small amount of data: call ``build_high_level()`` manually to ensure high-level memory is available

### Mode 2: Local Models (No API key required, model download needed)

```python
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem()
# Default: uses local Qwen3-Embedding-4B and Qwen3-Reranker-4B
# Models auto-download on first run (~8 GB total)

unit = MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
)
system.add(unit)

system.build_high_level(mode="auto")

hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)
for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

system.save("./memory_snapshot")
```

## Core Concepts

### What is a Memory System?

Mandol provides memory infrastructure for AI agents:

- Extracts entities, events, and summaries from text automatically
- Builds cross-session entity relation graphs and event causal chains
- Supports hybrid retrieval combining dense, sparse, and graph-based search

### Key Terms

| Term | Definition |
|------|------------|
| MemoryUnit | The smallest memory unit in the system, encapsulating text content, vector representation, and metadata |
| MemorySpace | A logical grouping container for memories, supporting organization by dimension (entity, event, summary, etc.) |
| SemanticMap | Semantic mapping table providing vector indexing and hybrid retrieval engine |
| SemanticGraph | Semantic relation graph storing associations between entities and events, supporting graph-based retrieval expansion |
| Session | A coherent interaction sequence, automatically segmented by the system based on time gaps or semantic boundaries |
| Entity | A named element (person, place, thing) extracted from text, deduplicated and linked across sessions |
| Event | An occurrence extracted from text, including causal chains and temporal relations |

> **About MemoryUnit insertion mode**: The fields in ``raw_data`` that the system automatically vectorizes are:
> - ``text_content``: plain text content → dense vector
> - ``image_path``: image file path → image vector
>
> Other fields (e.g., ``speaker``, ``source``) are stored as metadata but not automatically vectorized.

### How It Works

```
[User Input] → [Chunking + Embedding] → [Session Segmentation]
→ [Extract Entities/Events/Summaries] → [Build Relationship Graph]
→ [Retrieve: 3-way Recall → RRF Fusion → BFS Expansion → Rerank]
→ [Return Results]
```

## Core Features

### 1. Data Management

| Operation | Method | Description |
|-----------|--------|-------------|
| Add single memory | `add(unit)` | Auto-chunking, auto-embedding |
| Batch add | `add_many(units)` | More efficient batch processing |
| Save state | `save(directory)` | Export to a directory (multiple JSON files) |
| Load state | `MemorySystem.load(directory)` | Restore state from directory (class method) |

### 2. Memory Construction

After adding memories, the system automatically builds high-level memories asynchronously. For manual intervention:

| Operation | Method | Description |
|-----------|--------|-------------|
| Force rebuild | `build_high_level(mode="force")` | Clear state, reprocess all sessions |
| Async rebuild | `build_high_level_async()` | Execute construction in background |

**Automatic construction pipeline**:
- Session segmentation (LLM-driven)
- Episodic / Knowledge / Emotional / Procedural summary generation
- Insight extraction and global merging
- Entity extraction and deduplication
- Event extraction and deduplication
- Entity relation construction
- Event causal chain construction
- Cross-session entity/event merging

### 3. Retrieval

#### Holistic Retrieval (Recommended)

```python
hits = system.holistic_retrieve("query", top_k=10)
```

**Retrieval pipeline**:
1. Group recall: BASE / ENTITY / EVENT / SUMMARY four independent retrieval groups
2. Within each group: Dense + BM25 + Sparse three-way recall → RRF fusion → BFS expansion
3. Global reranking: All candidates merged and reranked by Cross-Encoder Reranker

The system also provides lower-level interfaces such as semantic retrieval and graph relation retrieval. See the [developer documentation](docs/index.rst) for details.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MANDOL_EMBEDDER_MODEL` | Embedding model | `Qwen/Qwen3-Embedding-4B` |
| `MANDOL_EMBEDDER_DEVICE` | Embedding device | `cpu` |
| `MANDOL_RERANKER_MODEL` | Reranker model | `Qwen/Qwen3-Reranker-4B` |
| `MANDOL_RERANKER_DEVICE` | Reranker device | `cpu` |
| `MANDOL_LLM_MODEL` | LLM model | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API Key | `""` |
| `MANDOL_LLM_BASE_URL` | OpenAI API Base URL | `https://api.openai.com/v1` |
| `USE_REMOTE_EMBEDDER` | Use remote Embedder | `false` |
| `USE_REMOTE_RERANKER` | Use remote Reranker | `false` |

> **Note**: These environment variables must be passed to the system via a YAML configuration file (``config.yaml``) or the ``MemorySystemConfig`` dataclass. Setting ``os.environ`` directly does not take effect. See the YAML configuration example below.

### YAML Configuration

```yaml
llm:
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."

embedder:
  model: "Qwen/Qwen3-Embedding-4B"
  device: "cuda"
  dimension: 2560
  use_remote: false
  base_url: "http://localhost:8000/v1"
  api_path: "/embeddings"
  api_key: ""
  timeout: 30

reranker:
  model: "Qwen/Qwen3-Reranker-4B"
  device: "cuda"
  use_remote: false
  base_url: ""
  api_path: "/v1/rerank"
  api_key: ""
  timeout: 30

system:
  chunk_max_tokens: 512
  session_time_gap_seconds: 1800
  session_check_interval: 20
  session_max_pending: 100
  similarity_top_k: 5
  similarity_threshold: 0.7
  similarity_recent_window: 20
  bfs_expansion_per_seed: 3
  bfs_expansion_hops: 1
  max_context_units: 20
  max_entities_per_llm: 50
  max_events_per_llm: 50
  promote_threshold: 100

storage:
  root: null
  enable_persistence: false
  auto_save_interval: 300
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Retrieval API Layer                           │
│  holistic_retrieve() → 4-group parallel → Cross-Encoder Rerank → Top-K │
├──────────────────────────────────────────────────────────────────────┤
│                         Memory Hierarchy Layer                       │
│  ┌──────────────────┐  ┌──────────────────────────────────────────┐ │
│  │  Base Memory      │  │         High-Level Memory               │ │
│  │  Raw data         │  │  Episodic │ Knowledge │ Emotional │ Proc │ │
│  │  segments         │  │  Summary  │ Entity    │ Event     │ Ins  │ │
│  └──────────────────┘  └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│                      Core Data Structure Layer                       │
│  MemoryUnit ←→ MemorySpace ←→ SemanticMap ←→ SemanticGraph          │
│  (Memory Unit)  (Logic Space)  (Semantic Map)    (Relation Graph)   │
└──────────────────────────────────────────────────────────────────────┘
```

See [Developer Documentation](docs/index.rst) for details.

## Project Structure

```
mandol/
├── domain/           # Core data structures
│   ├── memory_unit.py
│   ├── memory_space.py
│   ├── types.py
│   └── coref_graph_constants.py
├── ports/            # Abstract interfaces
│   ├── embedding_provider.py
│   ├── llm_provider.py
│   ├── reranker.py
│   ├── vector_index.py
│   ├── graph_store.py
│   ├── unit_store.py
│   ├── bm25_index.py
│   └── sparse_index.py
├── application/      # Application service layer
│   ├── memory_system.py          # Main entry point
│   ├── semantic_map.py           # Semantic map service
│   ├── semantic_graph.py         # Semantic graph service
│   ├── multidim_semantic_graph.py # Multi-dimension builder
│   ├── session_manager.py        # Session management
│   ├── chunker.py                # Chunker
│   ├── entity_dedup.py           # Entity deduplication
│   ├── event_dedup.py            # Event deduplication
│   ├── entity_relation_extract.py # Entity relation extraction
│   ├── event_causal_extract.py   # Event causal extraction
│   ├── summary_map_reducer.py    # Summary MapReduce
│   ├── insight_map_reducer.py    # Insight MapReduce
│   ├── global_insight_manager.py # Global insight management
│   ├── unified_fact_pipeline.py  # Unified fact pipeline
│   ├── cross_session_coref_manager.py # Cross-session coref manager
│   └── prompts/                  # LLM prompt templates
├── infrastructure/   # Infrastructure implementations
│   ├── in_memory_*.py
│   ├── adaptive_vector_index.py
│   ├── faiss_vector_index.py
│   ├── config.py                 # Configuration management
│   ├── json_persistence.py       # JSON persistence
│   ├── persistence_manager.py    # Persistence manager
│   ├── provider_factory.py       # Provider factory
│   ├── rank_bm25_index.py        # BM25 index
│   ├── tfidf_sparse_index.py     # TF-IDF sparse index
│   ├── openai_compatible_*.py    # OpenAI-compatible implementations
│   ├── sentence_transformers_*.py # SentenceTransformers implementations
│   ├── milvus_unit_store.py      # Milvus storage
│   ├── neo4j_graph_store.py      # Neo4j graph storage
│   └── stub_llm_provider.py      # Test stub LLM
└── retrieval/        # Retrieval module
    ├── pipeline.py   # Hybrid retriever
    ├── fusion.py     # RRF fusion
    ├── bm25.py       # BM25 retriever
    ├── sparse.py     # Sparse retriever
    ├── subgraph_hop.py # Subgraph hop retriever
    ├── text.py       # Text retrieval
    └── types.py      # SearchHit and other types
```

## Performance

Mandol is evaluated on the LoCoMo (Long Conversational Memory) and LongMemEval benchmarks. Evaluation uses LLM-as-judge to determine whether generated answers are consistent with ground truth answers.

### Key Metrics

| Metric | Description |
|--------|-------------|
| F1 Score | LLM-as-judge evaluates consistency between generated and ground truth answers |
| Response Time | End-to-end latency from query submission to result return |
| Memory Usage | Peak RSS (Resident Set Size) during system operation |
| Index Build Time | Total wall-clock time for `build_high_level()` completion |

### LoCoMo Results (GPT-4.1-mini Backbone)

| System | Avg.\,Tok. | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
|--------|------------|------------|-----------|----------|-------------|---------|
| Mem0 | 1.0k | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 |
| MemU | 4.0k | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 |
| MemOS | 2.5k | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 |
| Zep | 1.4k | 90.84 | 81.91 | 77.26 | 75.00 | 85.22 |
| EverMemOS† | 2.3k | 95.32 | 89.01 | 90.13 | 77.43 | 91.97 |
| **Mandol (Ours)** | **1.9k** | **95.36** | **92.20** | 87.85 | **79.17** | **92.21** |

### LongMemEval Results (GPT-4.1-mini Backbone)

| System | Avg.Tok. | SS-Pref | SS-Asst | Temporal | Multi-S | Know.Upd. | SS-User | Overall |
|--------|----------|---------|---------|----------|---------|-----------|---------|---------|
| EverMemOS | 2.8k | 93.33 | 85.71 | 77.44 | 73.68 | 89.74 | 97.14 | 83.00 |
| **Mandol (Ours)** | 2.3k | **96.67** | **98.21** | **87.22** | **77.44** | **89.74** | **98.57** | **88.40** |

> Overall metric excludes adversarial queries. † denotes results reproduced using the official EverMemOS implementation.

### Quick Reproduce

```bash
# LoCoMo
cd benchmarks/locomo && bash scripts/env.sh
python build_graph.py --config configs/base.yaml --output output/
python retrieve.py --config configs/base.yaml --input output/ --output output/
python generate.py --config configs/base.yaml --input output/ --output output/
python evaluate.py --input output/ --output output/

# LongMemEval
cd benchmarks/longmemeval && bash scripts/env.sh
python build_graph.py --config configs/base.yaml --output output/
python retrieve.py --config configs/base.yaml --input output/ --output output/
python generate.py --config configs/base.yaml --input output/ --output output/
python evaluate.py --input output/ --output output/
```

For complete test environment configuration, dataset details, ablation experiments, and full performance comparison tables, see [LoCoMo Benchmark](benchmarks/locomo/README.md) and [LongMemEval Benchmark](benchmarks/longmemeval/README.md).

## FAQ

### Installation Issues

**Q: `pip install mandol` reports "No matching distribution found"**
A: Ensure Python version >= 3.9 and try `pip install --upgrade pip`.

**Q: Installing `faiss-cpu` fails**
A: Try `conda install -c conda-forge faiss-cpu` or `pip install faiss-cpu --no-deps`.

### Runtime Errors

**Q: `MemorySystem()` initialization raises CUDA out of memory**
A: Set environment variables `MANDOL_EMBEDDER_DEVICE=cpu` and `MANDOL_RERANKER_DEVICE=cpu`, or use remote API mode (`USE_REMOTE_EMBEDDER=true`).

**Q: `holistic_retrieve` returns empty results**
A: Ensure `build_high_level()` has been called or wait for automatic construction to complete. Check that sufficient memory data exists (at least 5+ units recommended).

**Q: LLM API call timeout**
A: Verify `OPENAI_API_KEY` is correct and the API endpoint is reachable. You can increase the timeout with `MANDOL_LLM_TIMEOUT_S=120`.

### Performance Optimization

**Q: Retrieval is slow. How to optimize?**
A:
1. Use FAISS index acceleration: `pip install mandol[faiss]`
2. Reduce `bfs_expansion_hops` (default 1 → 0)
3. Disable reranking: `holistic_retrieve(query, use_rerank=False)`
4. Use GPU for embedding: `MANDOL_EMBEDDER_DEVICE=cuda`

**Q: Memory usage is too high. How to optimize?**
A:
1. Use remote Embedding/Reranker instead of local models
2. Reduce `similarity_recent_window`
3. Enable persistence and periodically `save`/`load`

## Documentation

- [Developer Documentation](docs/index.rst) - Architecture design, data structures, retrieval interfaces, extension guide

## License

Apache License 2.0 - See [LICENSE](LICENSE)
