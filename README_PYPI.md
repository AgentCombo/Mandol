# Mandol

Mandol is an agent memory system with hierarchical, episodic, and
entity-relation retrieval over a unified semantic graph.

[English](https://github.com/AgentCombo/Mandol/blob/v0.1.0/README.md) |
[中文](https://github.com/AgentCombo/Mandol/blob/v0.1.0/README_CN.md)

![Mandol Overview](https://raw.githubusercontent.com/AgentCombo/Mandol/v0.1.0/README.assets/Mandol-overview-v2.png)

## Installation

Mandol 0.1.0 requires Python `>=3.12,<3.13` and currently targets Linux.

```bash
python -m pip install mandol
```

Use an explicit pin for a reproducible package environment:

```bash
python -m pip install "mandol==0.1.0"
```

## Package Scope

The package provides:

- `MemoryUnit`, `MemorySpace`, `SemanticMap`, and `SemanticGraph` core APIs;
- dense, BM25, SPLADE, graph-expansion, fusion, and reranking paths;
- hierarchical, entity-relation, and episodic triple-tower retrieval;
- query routing, semantic quantification, and high-level memory builders; and
- RocksDB-backed automatic tiered paging for cold `MemoryUnit` payloads.

The package does not include benchmark datasets, generated graphs, model
caches, or provider credentials. Models may be downloaded on first use, and
complete benchmark reproduction requires external model services.

## Quick Start

The following example uses a compact sentence-transformers model and disables
realtime SPLADE generation. The embedding model may be downloaded on first use.

```python
from mandol import MemoryUnit, SemanticGraph, SemanticMap

semantic_map = SemanticMap(
    embedding_model_name="all-MiniLM-L6-v2",
    use_flash_attention=False,
)
graph = SemanticGraph(semantic_map_instance=semantic_map)

graph.add_unit(
    MemoryUnit(
        uid="msg_001",
        raw_data={"text_content": "Zhang San travelled to Beijing today."},
        metadata={"timestamp": "2026-06-21T09:00:00"},
    ),
    space_names=["demo"],
    generate_sparse_embedding=False,
)

results = graph.search_similarity_in_graph(
    query_text="Where did Zhang San go?",
    top_k=3,
    ms_names=["demo"],
    return_score=True,
)

for unit, score in results:
    print(score, unit.uid, unit.text_cached)
```

## Paper Reproduction

The published LoCoMo, LongMemEval, and performance results were produced with
the frozen [paper-repro artifact](https://github.com/AgentCombo/Mandol/tree/paper-repro),
not from the package alone. Follow the benchmark-specific guides in that branch
for datasets, model roles, configurations, and commands.

Mandol 0.1.0 is a research artifact and early public release. It is not a
production-ready service, and APIs may evolve during the `0.x` series.

## Links

- [Repository](https://github.com/AgentCombo/Mandol/tree/v0.1.0)
- [Documentation](https://agentcombo.github.io/Mandol/docs)
- [Paper](https://arxiv.org/abs/2606.29778)
- [Issues](https://github.com/AgentCombo/Mandol/issues)
- [Release notes](https://github.com/AgentCombo/Mandol/blob/v0.1.0/release-notes/v0.1.0.md)

## Citation

```bibtex
@misc{zhang2026mandol,
  title={Mandol: An Agglomerative Agent Memory System for Long-Term Conversations},
  author={Yuhan Zhang and Zhiyuan Guo and Ziheng Zeng and Wei Wang and Wentao Wu and Lijie Xu},
  year={2026},
  eprint={2606.29778},
  archivePrefix={arXiv},
  primaryClass={cs.DB},
  doi={10.48550/arXiv.2606.29778},
  url={https://arxiv.org/abs/2606.29778}
}
```

Mandol is distributed under the [Apache License 2.0](https://github.com/AgentCombo/Mandol/blob/v0.1.0/LICENSE).
