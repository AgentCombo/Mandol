场景会话配置
==============

客服对话
--------

.. code-block:: yaml

   system:
     session_time_gap_seconds: 300      # 5分钟
     session_check_interval: 10         # 每10条检测

稀疏对话（个人助手）
--------------------

.. code-block:: yaml

   system:
     session_time_gap_seconds: 1800     # 30分钟
     session_check_interval: 20         # 默认

密集无边界（知识库）
--------------------

.. code-block:: yaml

   system:
     session_time_gap_seconds: 86400    # 24小时，几乎不分割
     session_check_interval: 200
     session_max_pending: 1000

多用户混合
----------

.. code-block:: python

   for user_id in unique_users:
       user_units = get_user_history(user_id)
       system.add_many(user_units)
       system.build_high_level(mode="auto")

   # 每个用户一个独立空间确保隔离
   system.semantic_map.create_space(f"用户-{user_id}")

配置影响
--------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 调整
     - 效果
     - 代价
   * - 减小 gap_seconds
     - 更多会话，摘要更聚焦
     - 摘要碎片化
   * - 增大 gap_seconds
     - 更少会话，摘要更完整
     - 跨主题混淆
   * - 减小 check_interval
     - 更快检测边界
     - 更多 LLM 调用
   * - 增大 check_interval
     - 更少 LLM 调用
     - 边界检测延迟
