Advanced Troubleshooting
==========================

Retrieval Quality Issues
--------------------------

**Irrelevant retrieval results**

1. Check that query language and memory language are consistent
2. Lower ``similarity_threshold`` to 0.5
3. Confirm ``build_high_level`` has been called
4. Check that metadata timestamps are correct

**An obviously relevant memory wasn't recalled**

1. Confirm the memory has a ``text_content`` field value
2. Check raw_data field names are correct
3. Use ``retrieve_in_space`` to search within the memory's space
4. Check the memory was added to the correct space

Performance Issues
-------------------

**build_high_level timeout**

1. Reduce session size (lower session_max_pending)
2. Switch to a faster LLM model
3. Use mode="auto" for incremental instead of mode="force" for full rebuild

**Retrieval latency > 1 second**

1. Disable Rerank + BFS expansion
2. Reduce similarity_top_k
3. Confirm Embedding model is running on GPU

Build Issues
-------------

**Incorrect session segmentation**

1. The system detects topic boundaries through LLM semantic analysis; time intervals are only used as a reference
2. Manually marking session_id in metadata can override automatic segmentation
3. Use mode="force" to rebuild

**Inaccurate entity/event deduplication**

1. Increase max_entities_per_llm / max_events_per_llm
2. Use a stronger LLM model
3. Manually check deduplication logs

Data Persistence Issues
-------------------------

**Incorrect retrieval results after loading**

1. Confirm save/load uses the same Embedding model
2. Check that the save directory is complete
3. No need to re-run build_high_level after load (high-level structures are already saved)
