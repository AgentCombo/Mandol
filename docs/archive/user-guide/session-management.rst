会话管理最佳实践
================

Mandol 的会话管理是记忆系统的核心组件之一，负责将连续的对话流分割为有意义的会话单元。本指南介绍会话分割的工作原理和不同场景下的最佳实践。

会话分割机制
------------

Mandol 支持两种会话分割策略：

1. **基于时间间隔的分割**：当两条相邻记忆的时间戳间隔超过 ``session_time_gap_seconds`` 时，自动分割为新会话
2. **基于 LLM 的智能分割**：当累积的记忆单元达到 ``session_check_interval`` 时，系统调用 LLM 判断是否存在会话边界

两种策略协同工作：时间间隔作为快速预筛，LLM 作为精确判断。

分割流程
^^^^^^^^

.. mermaid::

   graph TD
       A[新记忆单元到达] --> B{时间间隔 > session_time_gap_seconds?}
       B -->|Yes| C[强制分割新会话]
       B -->|No| D{累积数量 >= session_check_interval?}
       D -->|Yes| E[调用 LLM 检测会话边界]
       D -->|No| F[继续累积]
       E --> G{LLM 检测到边界?}
       G -->|Yes| H[分割新会话]
       G -->|No| F
       C --> I[更新会话状态]
       H --> I

参数详解
--------

session_time_gap_seconds
^^^^^^^^^^^^^^^^^^^^^^^^

基于时间间隔的会话分割阈值。当两条相邻记忆的 ``timestamp`` 差值超过此值时，自动创建新会话。

- **类型**：int
- **默认值**：1800（30 分钟）
- **单位**：秒

session_check_interval
^^^^^^^^^^^^^^^^^^^^^^

每累积多少条记忆触发一次 LLM 会话边界检测。

- **类型**：int
- **默认值**：20
- **说明**：增大此值可减少 LLM 调用次数（降低成本），但会延迟分割

session_max_pending
^^^^^^^^^^^^^^^^^^^

待处理记忆最大数量。当累积的未分割记忆超过此值时，强制触发分割。

- **类型**：int
- **默认值**：100
- **说明**：这是一个安全阈值，防止极端情况下记忆无限累积

场景最佳实践
------------

客服对话
^^^^^^^^

客服对话的特点是会话短、边界清晰（用户发起咨询→问题解决→结束）。

**推荐配置**：

.. code-block:: yaml

   system:
     session_time_gap_seconds: 300
     session_check_interval: 10
     session_max_pending: 50

**理由**：

- 客服会话通常 5-10 分钟内完成，300 秒的时间间隔能准确捕捉会话边界
- 较小的 ``session_check_interval`` （10）确保快速检测到新会话
- 客服场景对话量不大，频繁 LLM 检测的成本可接受

个人助手
^^^^^^^^

个人助手的特点是会话较长、主题可能混合（工作+生活）、时间跨度大。

**推荐配置**：

.. code-block:: yaml

   system:
     session_time_gap_seconds: 1800
     session_check_interval: 20
     session_max_pending: 100

**理由**：

- 30 分钟的时间间隔适合日常对话节奏（午休、下班等自然间隔）
- 默认的 ``session_check_interval`` 平衡了检测精度和 LLM 成本

知识库导入
^^^^^^^^^^

知识库导入的特点是无自然会话边界，所有文档属于同一"会话"。

**推荐配置**：

.. code-block:: yaml

   system:
     session_time_gap_seconds: 86400
     session_check_interval: 100
     session_max_pending: 500

**理由**：

- 极大的时间间隔（24 小时）避免误分割
- 大的 ``session_check_interval`` 减少 LLM 调用（知识库不需要会话分割）
- 大的 ``session_max_pending`` 避免强制分割

多用户系统
^^^^^^^^^^

多用户系统中，不同用户的对话应属于不同会话。

**推荐做法**：

.. code-block:: python

   for user_id in ["user_001", "user_002"]:
       user_units = get_units_for_user(user_id)
       for unit in user_units:
           unit.metadata["user_id"] = user_id
           system.add(unit)
       system.build_high_level(mode="auto")

**关键点**：

- 在 ``metadata`` 中标记 ``user_id``，便于后续按用户检索
- 每个用户的对话添加完成后调用 ``build_high_level``，避免跨用户记忆混淆
- 使用 ``retrieve_in_space`` 按用户空间检索

会话分割后的影响
----------------

会话分割完成后，系统会对每个会话执行以下操作：

1. **摘要生成**：为每个会话生成情景摘要、知识摘要、情感摘要和程序摘要
2. **实体提取**：从会话中提取实体（人物、地点、概念等）
3. **事件提取**：从会话中提取事件及其因果关系
4. **跨会话合并**：将不同会话中的相同实体/事件进行合并

因此，合理的会话分割直接影响高阶记忆的质量。分割过细会导致摘要碎片化，分割过粗会导致摘要不够聚焦。

常见问题
--------

Q: 会话分割太频繁，导致摘要碎片化怎么办？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 增大 ``session_time_gap_seconds`` 和 ``session_check_interval``。如果对话主题经常切换，可以考虑在 ``metadata`` 中手动标记 ``session_id``，然后使用 ``build_high_level(mode="force")`` 重建。

Q: 不同用户的对话被合并到同一会话怎么办？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 确保在 ``metadata`` 中标记 ``user_id``，并按用户分批添加记忆。如果已经混合，可以清空后重新导入。

Q: LLM 会话检测成本太高怎么办？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 增大 ``session_check_interval`` （如 50 或 100），减少 LLM 调用频率。对于会话边界明显的场景（如客服），主要依赖时间间隔分割即可。
