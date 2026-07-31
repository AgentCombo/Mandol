LongMemEval Reproduction
========================

The maintained LongMemEval reproduction workflow lives under
``benchmark_longmemeval``. The paper accuracy numbers are produced by the
router + quantification task-eval script after the hierarchical, episodic and
entity-relation graph artifacts have been generated from the cleaned dataset.

Use the repository-level guide as the source of truth:

.. code-block:: bash

   uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification --help

For a bounded real-LLM smoke test, restrict the QA range and the number of
formal tests:

.. code-block:: bash

   uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
     --dataset-size s \
     --start-qa 0 \
     --end-qa 0 \
     --max-tests 1 \
     --llm-model gpt-4.1-mini-closeai \
     --llm-evaluate-model gpt-4o-mini-closeai

See ``benchmark_longmemeval/REPRODUCE.md`` for the full paper-aligned commands,
model roles, graph paths, and output-file definitions.
