配置参考
========

MemorySystemConfig 全部字段。

.. code-block:: python

   from mandol.application.config import MemorySystemConfig

   config = MemorySystemConfig(
       # 分块
       chunk_max_tokens=512,

       # 会话
       session_time_gap_seconds=1800,
       session_check_interval=20,
       session_max_pending=100,

       # 相似度
       similarity_threshold=0.7,
       similarity_top_k=5,
       similarity_recent_window=20,

       # BFS
       bfs_expansion_per_seed=3,
       bfs_expansion_hops=1,

       # LLM
       max_entities_per_llm=50,
       max_events_per_llm=50,

       # 索引
       promote_threshold=100,
   )

YAML 配置映射
-------------

.. code-block:: yaml

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     session_check_interval: 20
     session_max_pending: 100
     similarity_threshold: 0.7
     similarity_top_k: 5
     similarity_recent_window: 20
     bfs_expansion_per_seed: 3
     bfs_expansion_hops: 1
     max_entities_per_llm: 50
     max_events_per_llm: 50
     promote_threshold: 100

   llm:
     provider: "openai"
     model: "gpt-4o-mini"

   embedder:
     provider: "sentence_transformers"
     model: "Qwen/Qwen3-Embedding-4B"
     device: "cpu"

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cpu"
