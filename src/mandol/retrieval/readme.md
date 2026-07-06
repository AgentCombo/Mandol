# Mandol Retrieval

`MultiRetriever` is the unified entry point for BM25, SPLADE, cosine retrieval,
score fusion, optional reranking, and optional graph expansion.

```python
from mandol.retrieval import MultiRetriever

retriever = MultiRetriever(semantic_map_or_graph)
```

## Basic search

```python
results = retriever.smart_search(
    query="query text",
    methods=["bm25", "splade", "cosine"],
    fusion_method="rrf",
    top_k=10,
)
```

## Space and candidate filters

`space_names` limits retrieval to the union of the selected memory spaces.
When `candidate_uids` is also supplied, the effective candidates are the
intersection of the explicit candidates and the selected spaces.

```python
results = retriever.smart_search(
    query="query text",
    methods=["bm25", "cosine", "splade"],
    top_k=10,
    space_names=["chat:messages"],
    candidate_uids=["message-1", "message-2"],
)
```

## Reranking

```python
results = retriever.smart_search(
    query="query text",
    methods=["bm25", "splade"],
    fusion_method="rrf",
    rerank_method="baai",
    rerank_params={"max_candidates": 50},
    top_k=10,
)
```

Reranking models are loaded on demand. The configured BAAI, Qwen, Jina or vLLM
backend may require additional model files or a running service.

## Graph expansion

Graph expansion is available when `MultiRetriever` is constructed with a
`SemanticGraph`:

```python
results = retriever.smart_search(
    query="query text",
    methods=["cosine", "splade"],
    enable_graph_expansion=True,
    graph_expansion_config={
        "expand_hops": 2,
        "max_neighbors": 8,
    },
    top_k=10,
)
```

## Detailed output

Set `return_detailed=True` to receive execution metadata in addition to the
final `(MemoryUnit, score)` results:

```python
detailed = retriever.smart_search(
    query="query text",
    methods=["bm25", "splade", "cosine"],
    fusion_method="rrf",
    top_k=10,
    return_detailed=True,
)
```

The older `bge_m3_vector_types` examples have been removed because that
parameter is not part of the current `MultiRetriever.smart_search` contract.
