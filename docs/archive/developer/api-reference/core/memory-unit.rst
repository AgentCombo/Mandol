MemoryUnit 参考
====================

MemoryUnit 是系统中最小的记忆载体。

构造函数
--------

.. code-block:: python

   MemoryUnit(
       uid: Uid,
       raw_data: dict[str, Any],
       metadata: dict[str, Any] | None = None,
       unit_type: str | None = None,
   )

字段
----

.. list-table::
   :header-rows: 1
   :widths: 18 15 67

   * - 字段
     - 类型
     - 说明
   * - ``uid``
     - Uid
     - 唯一标识
   * - ``raw_data``
     - dict
     - 原始内容，柔性容器，可存放任意键值对；仅 ``text_content`` 和 ``image_path`` 被自动向量化
   * - ``metadata``
     - dict
     - 附加标签，可存放 timestamp / speaker / source 等
   * - ``embedding``
     - Optional[numpy.ndarray]
     - 稠密向量表示（由 EmbeddingProvider 生成）
   * - ``sparse_embedding``
     - Optional[numpy.ndarray]
     - 稀疏向量表示（由 SparseIndex 生成）
   * - ``unit_type``
     - str
     - 自动推断（dialogue/knowledge/event 等）

方法
----

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 方法
     - 类型
     - 说明
   * - ``to_dict()``
     - 实例方法
     - 序列化为字典，用于持久化
   * - ``from_dict(data)``
     - 类方法
     - 从字典反序列化为 MemoryUnit
   * - ``get_user_metadata()``
     - 实例方法
     - 获取用户自定义元数据（排除 ``_system_`` 前缀的系统字段）
   * - ``touch()``
     - 实例方法
     - 更新 ``_system_updated_at`` 时间戳

raw_data 详细说明
-----------------

``raw_data`` 的类型是 ``Dict[str, Any]``，无固定 schema。系统仅对以下字段做自动向量化：

自动向量化字段
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 12 30 40

   * - 字段名
     - 类型
     - 处理方式
     - 说明
   * - ``text_content``
     - str
     - Dense Embedding + BM25 + Sparse 索引
     - 核心字段，几乎所有检索和构建流程都依赖它
   * - ``image_path``
     - str
     - ``embed_image_paths`` 生成向量
     - 接口预留，当前实现退化为路径字符串编码

文本优先原则：同时提供 ``text_content`` 和 ``image_path`` 时，仅使用 ``text_content``。

文本提取回退顺序
~~~~~~~~~~~~~~~~

::

   text_content → text → content → summary → title → message → 第一个字符串值

系统内部自动生成的字段
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - 字段名
     - 出现的单元类型
     - 说明
   * - ``text_content``
     - 实体、事件、摘要
     - 格式化的文本描述
   * - ``entity_name``
     - 实体
     - 实体名称
   * - ``entity_type``
     - 实体
     - 实体类型（Person / Organization / Location 等）
   * - ``description``
     - 实体、事件
     - 详细描述
   * - ``summary``
     - 摘要
     - 摘要文本
   * - ``type``
     - 摘要
     - 摘要类型标识
   * - ``insights``
     - 洞察
     - 洞察内容列表
   * - ``entities``
     - 基础单元
     - 旧版实体提取路径读取的字段

.. note::

   扩展 ``raw_data`` 时，避免与系统内部字段名冲突。始终使用 ``text_content`` 作为文本字段名，不要依赖回退机制。

完整字段参考见 :doc:`/shared/data-format-guide`。

使用示例
--------

.. code-block:: python

   from mandol import MemoryUnit, Uid

   unit = MemoryUnit(
       uid=Uid("unique_id"),
       raw_data={
           "text_content": "一段对话或文档内容",
           "speaker": "张三",
           "source": "微信",
       },
       metadata={"timestamp": "2024-01-15T10:00:00", "speaker": "张三"},
   )
