LLM Cost Optimization
========================

Cost Sources
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Content Per Call
   * - Session boundary detection
     - Called once per session_check_interval memories
   * - Entity extraction + deduplication
     - One extraction per session + batch deduplication
   * - Event extraction + deduplication
     - One extraction per session + batch deduplication
   * - Summary generation
     - 4 types of summaries per session
   * - Cross-session merging
     - At most once per build_high_level call

Reduction Strategies
---------------------

1. **Increase check_interval**: Reduce LLM detection frequency
   .. code-block:: yaml

      session_check_interval: 50

2. **Use lightweight models**:
   .. code-block:: yaml

      llm:
        model: "gpt-4o-mini"

3. **Reduce entity/event deduplication**:
   .. code-block:: yaml

      max_entities_per_llm: 100
      max_events_per_llm: 100

4. **Batch processing**: Accumulate enough data before building, reduce build frequency

Cost Estimate
--------------

Using gpt-4o-mini as an example, the build cost for 1000 conversations (~50 sessions) is approximately $0.50-$2.00.
