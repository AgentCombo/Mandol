LoCoMo Reproduction
===================

The maintained LoCoMo reproduction workflow lives under
``benchmark_locomo``. The paper accuracy numbers are produced by the
router + quantification task-eval script after the hierarchical,
entity-relation and episodic graph artifacts have been generated.

Use the repository-level guide as the source of truth:

.. code-block:: bash

   uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification --help

Before running the full benchmark, generate the offline graph artifacts and the
unified graph used by retrieval-speed tests:

.. code-block:: bash

   bash benchmark_locomo/dataset_maker/run_all_locomo_dataset_maker_workflows.sh
   bash benchmark_locomo/dataset_maker/build_unified.sh

The unified-graph wrapper calls
``benchmark_locomo/dataset_maker/build_unified_graph.py`` and writes
``benchmark_locomo/dataset/locomo/unified_per_sample_graphs``.

For a bounded real-LLM smoke test, restrict both the sample set and the number
of formal questions:

.. code-block:: bash

   uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
     --sample-ids conv-30 \
     --max-questions 1 \
     --llm-model gpt-4.1-mini-closeai \
     --llm-evaluate-model gpt-4o-mini-closeai

See ``benchmark_locomo/REPRODUCE.md`` for the full paper-aligned commands,
model roles, graph paths, and timing-boundary definitions.
