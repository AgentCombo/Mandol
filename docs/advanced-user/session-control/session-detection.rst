会话分割机制详解
====================

Mandol 使用双重策略进行会话分割。

策略一：时间间隔分割
--------------------

两条相邻记忆的 ``timestamp`` 差值超过 ``session_time_gap_seconds`` → 强制新会话。

.. code-block:: python

   # 配置
   system = MemorySystem.from_yaml_config("config.yaml")
   # system:
   #   session_time_gap_seconds: 1800  # 30分钟

策略二：LLM 语义分割
--------------------

累积 ``session_check_interval`` 条记忆 → 调用 LLM 判断会话边界。

.. code-block:: python

   # 配置
   # session_check_interval: 20
   # session_max_pending: 100

调试分割结果
------------

.. code-block:: python

   report = system.build_high_level(mode="auto")
   print(f"处理会话数: {report.sessions_processed}")

   # 查看 SessionManager 中的会话信息
   stats = system.get_memory_stats()
   print(stats)

手动强制分割
------------

如果自动分割不满足需求，可以在 metadata 中手动标记 session_id：

.. code-block:: python

   # 同一 session_id 的记忆不会被分割
   unit.metadata["session_id"] = "手工会话-20240301"
