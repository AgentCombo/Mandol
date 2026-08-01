Frozen Paper Reproduction
=========================

This checkout is the frozen artifact used to produce the reported Mandol
results. Run reproduction commands from the repository root and keep the model
roles, dataset splits, routing policies and evaluation settings documented in
the benchmark-specific guides.

Accuracy workflows
------------------

The paper accuracy results use router + quantification over generated
hierarchical, entity-relation and episodic memory spaces:

* ``benchmark_locomo/REPRODUCE.md``
* ``benchmark_longmemeval/REPRODUCE.md``

The primary task-eval entry points are:

.. code-block:: text

   benchmark_locomo/task_eval/locomo_triple_router_quantification.py
   benchmark_longmemeval/task_eval/benchmark_triple_router_quantification.py

LoCoMo uses ``qwen-3.5-plus-thinking`` for memory extraction and LongMemEval
uses the non-thinking ``qwen-3-plus`` alias. Both workflows use
``deepseek-v3.2-dashscope`` for deduplication. The reported task-eval rows use
``gpt-4.1-mini-closeai`` and ``gpt-4o-mini-closeai`` as evaluated models, with
``gpt-4o-mini-closeai`` as judge. These names are provider aliases resolved by
the artifact configuration.

Performance workflow
--------------------

The public LoCoMo performance entry points are:

.. code-block:: text

   benchmark_locomo/task_eval/locomo_triple_input_speed.py
   benchmark_locomo/task_eval/locomo_triple_smart_search_qps.py

Build the unified per-sample graphs before the fixed-QPS search benchmark:

.. code-block:: bash

   bash benchmark_locomo/dataset_maker/build_unified.sh

The insertion benchmark times the ``SemanticGraph.add_unit`` body, including
dense embedding generation, realtime SPLADE generation and incremental index
updates. Scheduling sleep, graph initialization, warmup and report writing are
excluded. Candidate selection and eviction scheduling occur in the add path;
asynchronous RocksDB persistence is not guaranteed to finish inside the timed
add call.

The smart-search benchmark runs after graph loading and warmup. Its request
latency includes retrieval, fusion, configured reranking and Python wrapper
overhead, while fixed-QPS scheduling sleep and report writing are excluded.

Self-host workflows
-------------------

``benchmark_self_host/locomo10`` and
``benchmark_self_host/longmemeval`` validate Mandol's own high-level memory
generation path. They do not replace the router + quantification workflows used
for the paper accuracy tables.
