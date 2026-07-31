数据格式指南
============

本文档是 MemoryUnit 数据格式的权威参考，说明 ``raw_data`` 和 ``metadata`` 中各字段的含义、系统的处理方式以及当前的限制。

raw_data 字段参考
-----------------

``raw_data`` 的类型是 ``Dict[str, Any]``，可以存放任意键值对。但系统仅对以下字段做自动向量化处理：

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
     - 自动生成 Dense Embedding + BM25 + Sparse 索引
     - **核心字段**，几乎所有检索和构建流程都依赖它
   * - ``image_path``
     - str
     - 调用 Embedder 的 ``embed_image_paths`` 生成向量
     - 接口预留，当前实现退化为路径字符串编码

.. important::

   **文本优先原则**：同时提供 ``text_content`` 和 ``image_path`` 时，系统仅使用 ``text_content`` 生成向量。一个 MemoryUnit 只能取文本或图像其一。

文本提取回退顺序
~~~~~~~~~~~~~~~~

当系统需要从 MemoryUnit 提取文本时，按以下顺序查找：

::

   text_content → text → content → summary → title → message → 第一个字符串值

这意味着 ``raw_data={"content": "一段文字"}`` 也能工作，但 ``raw_data={"body": "一段文字"}`` 不会自动被识别（除非它是字典中唯一的字符串值）。

.. tip::

   始终使用 ``text_content`` 作为文本字段名，避免依赖回退机制。回退机制是为系统内部生成的单元设计的。

系统内部自动生成的字段
~~~~~~~~~~~~~~~~~~~~~~

高阶记忆构建过程中，系统会自动创建带有以下 ``raw_data`` 字段的 MemoryUnit：

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

用户自定义字段
~~~~~~~~~~~~~~

你可以在 ``raw_data`` 中存放任意自定义字段，这些字段会被存储和序列化，但**不会自动向量化**：

.. code-block:: python

   MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={
           "text_content": "张三去北京出差",   # 自动向量化
           "speaker": "李四",                  # 仅存储，不向量化
           "source": "微信",                    # 仅存储，不向量化
           "session_id": "s_001",              # 仅存储，不向量化
       },
   )

metadata 字段参考
-----------------

``metadata`` 用于存放附加标签，不会参与向量化或文本提取，但可用于后续筛选。

常用 metadata 键名：

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - 键名
     - 类型
     - 说明
   * - ``timestamp``
     - str
     - ISO 8601 格式时间戳，系统自动填充（如未提供）
   * - ``speaker``
     - str
     - 说话人标识
   * - ``source``
     - str
     - 数据来源（微信 / 邮件 / 文档 等）
   * - ``session_id``
     - str
     - 外部会话 ID

当前不支持的内容格式
---------------------

.. warning::

   以下格式**不能**直接作为 MemoryUnit 插入系统：

   - **PDF 文件**：需先提取为纯文本，放入 ``text_content``
   - **Markdown 文件**：需先去除格式标记，放入 ``text_content``
   - **Word / Excel 文件**：需先提取为纯文本
   - **音频 / 视频文件**：需先转写为文本
   - **图片像素数据**：当前仅支持图片路径（``image_path``），且实现为接口预留

   处理方式：使用外部工具将文件内容提取为纯文本后，再放入 ``raw_data["text_content"]``。

.. code-block:: python

   # ❌ 不能直接传文件路径
   MemoryUnit(uid=Uid("doc_1"), raw_data={"file_path": "/data/report.pdf"})

   # ✅ 先提取文本
   text = extract_pdf_text("/data/report.pdf")
   MemoryUnit(uid=Uid("doc_1"), raw_data={"text_content": text})

完整示例
--------

.. code-block:: python

   from mandol import MemoryUnit, Uid

   unit = MemoryUnit(
       uid=Uid("msg_001"),
       raw_data={
           "text_content": "张三今天去北京出差了",
           "image_path": "/photos/trip.jpg",
       },
       metadata={
           "timestamp": "2024-01-15T10:00:00",
           "speaker": "系统通知",
           "source": "日程管理",
       },
   )
