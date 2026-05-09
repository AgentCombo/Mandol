Common Troubleshooting
========================

Empty Retrieval Results
-------------------------

This is the most common issue.

**Most common cause: Not manually building high-level memories**

The system asynchronously detects topic boundaries and triggers high-level memory construction during ``add()``, but this process takes time. If you retrieve immediately after adding a small amount of data, high-level memories may not be ready yet.

.. code-block:: python

   # ❌ May return empty: retrieve immediately after add, high-level memories may not be ready
   system.add(unit)
   hits = system.holistic_retrieve("...")  # May return []

   # ✅ Correct: Manually call build_high_level to ensure high-level memories are available
   system.add(unit)
   system.build_high_level(mode="auto")
   hits = system.holistic_retrieve("...")  # Returns normally

.. note::

   When only retrieving raw conversations (BASE group), no need to wait for ``build_high_level``; retrieval works right after ``add()``. But retrieving entities/events/summaries (ENTITY/EVENT/SUMMARY groups) requires high-level memories to be built first.

**Still empty after calling build_high_level?**

1. Confirm your memories have a ``text_content`` field
2. Confirm query language matches memory language (mixed Chinese/English may affect recall)
3. Try lowering ``similarity_threshold`` (e.g., from 0.7 to 0.5)
4. Confirm sufficient number of memories added (recommend at least 5+)
5. Check logs for error messages

Installation Issues
---------------------

**pip install error "No matching distribution found"**

Please confirm Python version >= 3.9, and try ``pip install --upgrade pip``.

**faiss-cpu installation fails**

Try ``conda install -c conda-forge faiss-cpu`` or ``pip install faiss-cpu --no-deps``.

**Permission denied**

Add the ``--user`` flag or use a virtual environment (recommended).

Runtime Issues
---------------

**CUDA out of memory**

In local model mode, Embedding and Reranker models each need about 4GB VRAM. Solutions:

- Set ``embedder.device: "cpu"`` and ``reranker.device: "cpu"`` to use CPU
- Use remote API mode (``use_remote: true``), don't load models locally
- Use GPU only for Embedder, CPU or remote API for Reranker

**Remote model connection failure / API timeout**

- Check ``OPENAI_API_KEY`` is correct
- Check ``base_url`` is accessible (configure ``https_proxy`` if using a proxy)
- Increase timeout: set ``timeout: 120`` in config.yaml
- Confirm API quota hasn't been exhausted

**build_high_level errors**

- Check LLM API Key is correctly configured in ``.env`` or ``config.yaml``
- Check API base_url is accessible
- If using local model mode, confirm ``sentence-transformers`` is installed

**Irrelevant retrieval results**

- Increase ``top_k`` to see if more relevant results appear later
- Check metadata timestamps are correct
- Try more specific query phrasing
- Confirm ``build_high_level()`` has been called to build high-level memories

**High memory usage**

- Use remote Embedding/Reranker instead of local models (saves ~8GB)
- Reduce ``similarity_recent_window``
- Enable persistence and periodically ``save``/``load``
- Periodically call ``system.flush()``

**View real-time system status**

.. code-block:: python

   # One line to check memory, unit count, graph status, etc.
   print(system.monitor)

   # Or get detailed metrics
   stats = system.monitor.to_dict()
   print(f"Memory: {stats['rss_memory_mb']:.1f} MB (source: {stats['memory_source']})")
   print(f"Pending: {stats['pending_units']} units / {stats['pending_events']} events")
   print(f"Dirty flag: {stats['dirty']}")

Still Can't Resolve?
----------------------

See the more detailed advanced troubleshooting guide: :doc:`/advanced-user/troubleshooting-advanced`.
