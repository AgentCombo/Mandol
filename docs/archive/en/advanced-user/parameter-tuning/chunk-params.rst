Chunk Parameters
==================

chunk_max_tokens
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``chunk_max_tokens``
     - 512
     - tokens
     - Text exceeding this value is automatically split

**Recommended values**:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Scenario
     - Recommended
     - Description
   * - Short conversations (customer service)
     - 256
     - Finer granularity
   * - Medium conversations (assistant)
     - 512
     - Default
   * - Long documents (knowledge base)
     - 1024
     - Larger context window
