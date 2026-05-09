核心概念速览
============

不需要全部记住。只需要了解这三个概念，你就能开始使用 Mandol。

MemoryUnit — 记忆的最小单元
----------------------------

就像一张便签纸。每段对话、每个知识点、每个事件，都是一张便签。

.. code-block:: python

   MemoryUnit(
       uid=Uid("msg_1"),                             # 唯一编号
       raw_data={
           "text_content": "张三去北京出差",            # 文本内容（自动向量化）
           "image_path": "/path/to/photo.jpg",        # 图片路径（自动向量化，可选）
       },
       metadata={"timestamp": "2024-01-15T10:00"},   # 附加标签
   )

核心属性一览：

- ``uid`` — 唯一编号
- ``raw_data`` — 原始内容（``text_content`` 和 ``image_path`` 自动向量化，其他字段仅存储）
- ``metadata`` — 附加标签（时间戳、来源等）
- ``embedding`` — 稠密向量（系统自动生成）
- ``sparse_embedding`` — 稀疏向量（系统自动生成）

.. tip::

   ``raw_data`` 中 ``text_content`` 和 ``image_path`` 会被系统自动向量化用于检索。其他字段（如 ``speaker``、``source``）会作为元数据存储但不会自动生成向量。``metadata`` 用于存放时间戳、来源等附加信息，方便后续筛选。

.. warning::

   Mandol 目前只支持**纯文本**和**图片路径**两种内容格式。PDF、Markdown、Word 等文件不能直接插入——需先用外部工具提取为纯文本，再放入 ``text_content``。详见 :doc:`/shared/data-format-guide`。

完整的属性、方法和接口签名见 :doc:`/shared/data-structures-reference`。

MemorySpace — 便签本的分类夹
-------------------------------

就像文件夹。你可以把便签放到不同的文件夹里，文件夹还能嵌套子文件夹。

核心属性一览：

- ``name`` — 空间名称
- ``unit_uids`` — 包含的记忆单元集合
- ``child_spaces`` — 子空间集合
- ``summary_text`` — 空间摘要文本（可选）

.. code-block:: python

   # 系统自动创建的空间
   system.semantic_map.create_space("我的客户")

   # 添加到空间
   unit = MemoryUnit(
       uid=Uid("client_1"),
       raw_data={"text_content": "李总公司是做跨境电商的"},
   )
   system.add(unit, space_names=["我的客户"])

   # 之后可以只在这个空间里检索
   hits = system.retrieve_in_space("跨境电商", space_name="我的客户")

完整的属性、方法和接口签名见 :doc:`/shared/data-structures-reference`。

高阶记忆 — 系统的自动整理能力
--------------------------------

当你调用 ``build_high_level()``，系统会自动做四件事：

1. **识别话题边界**：通过 LLM 语义分析检测话题变化，将记忆按语义主题分组
2. **找关键元素**：提取出现的人物、地点、概念（实体）
3. **记录事件**：标记发生的重要事情
4. **写摘要**：给每个语义主题写一段总结

这些全自动，你不需要做任何额外操作。

.. code-block:: python

   system.build_high_level(mode="auto")
   # mode="auto"：仅处理新增内容（推荐日常使用）
   # mode="force"：清除后重新构建（用于调参后重建）

.. note::

   「会话」不等于「对话」。Mandol 中的会话 (Session) 是语义话题的边界——无论是对话、文档还是日志，系统都会按话题变化自动分组。时间间隔仅作为 LLM 判断的参考，不单独触发分割。
