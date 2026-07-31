Common Development Pitfalls
==============================

1. Forgetting to call build_high_level → empty retrieval results
2. Using wrong field names in raw_data (should be text_content, not content)
3. Multi-user confusion: not using spaces to isolate user data
4. Inconsistent metadata timestamp format (ISO 8601 recommended)
5. Embedding dimension mismatch: need to rebuild index after changing Embedder
6. Improper flush timing: not flushing in time with large data causes memory spike
7. LLM model name typos in YAML configuration
8. Ignoring Cross-Encoder Reranker GPU memory usage (~2-4GB)
