Scenario Preset Configurations
=================================

Customer Service
-----------------

.. code-block:: yaml

   system:
     chunk_max_tokens: 256
     session_time_gap_seconds: 300
     session_check_interval: 10
     similarity_top_k: 5
     similarity_threshold: 0.7
     bfs_expansion_per_seed: 3
     bfs_expansion_hops: 1

Personal Assistant
-------------------

.. code-block:: yaml

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     session_check_interval: 20
     similarity_top_k: 10
     similarity_threshold: 0.65
     bfs_expansion_per_seed: 5
     bfs_expansion_hops: 2

Knowledge Base
---------------

.. code-block:: yaml

   system:
     chunk_max_tokens: 1024
     session_time_gap_seconds: 86400
     session_check_interval: 200
     session_max_pending: 500
     similarity_top_k: 10
     similarity_threshold: 0.7
     bfs_expansion_per_seed: 3
     bfs_expansion_hops: 1
