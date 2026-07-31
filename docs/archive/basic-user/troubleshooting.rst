常见问题排查
==============

检索返回空结果
--------------

这是最常见的问题。

**最常见的原因：未手动构建高阶记忆**

系统在 ``add()`` 时会异步检测话题边界并自动触发高阶记忆构建，但这个过程需要时间。如果你在添加少量数据后立即检索，高阶记忆可能还未构建完成。

.. code-block:: python

   # ❌ 可能返回空：add 后立即检索，高阶记忆可能未就绪
   system.add(unit)
   hits = system.holistic_retrieve("...")  # 可能返回 []

   # ✅ 正确：手动调用 build_high_level 确保高阶记忆可用
   system.add(unit)
   system.build_high_level(mode="auto")
   hits = system.holistic_retrieve("...")  # 正常返回

.. note::

   仅检索原始对话（BASE 组）时无需等待 ``build_high_level``，``add()`` 后即可检索。但检索实体/事件/摘要（ENTITY/EVENT/SUMMARY 组）时需要高阶记忆已构建完成。

**已调用 build_high_level 仍返回空？**

1. 确认你添加的记忆中有 ``text_content`` 字段（不是 ``text``、``content``、``body`` 等）
2. 确认 ``text_content`` 的值是**纯文本**，不是 PDF 内容、Markdown 源码、HTML 标签等
3. 确认查询语言和记忆语言一致（中英文混用可能影响召回）
4. 尝试降低 ``similarity_threshold`` （如从 0.7 降到 0.5）
5. 确认添加的记忆数量足够（建议至少 5 条以上）
6. 检查 logs 是否有报错信息

.. warning::

   最常见的格式错误：

   - ❌ ``raw_data={"content": "一段文字"}`` — 字段名不对，系统不会向量化
   - ❌ ``raw_data={"text_content": "# 标题\n**加粗**"}`` — Markdown 格式，应提取为纯文本
   - ❌ ``raw_data={"file_path": "/data/report.pdf"}`` — 不支持直接传文件路径
   - ✅ ``raw_data={"text_content": "一段纯文本内容"}`` — 正确写法

   详见 :doc:`/shared/data-format-guide`。

安装问题
--------

**pip install 报错 "No matching distribution found"**

请确认 Python 版本 >= 3.9，并尝试 ``pip install --upgrade pip``。

**安装 faiss-cpu 失败**

尝试 ``conda install -c conda-forge faiss-cpu`` 或 ``pip install faiss-cpu --no-deps``。

**权限不足**

加 ``--user`` 参数或使用虚拟环境（推荐）。

运行时问题
----------

**CUDA out of memory**

本地模型模式下，Embedding 和 Reranker 模型各需约 4GB 显存。解决方案：

- 设置 ``embedder.device: "cpu"`` 和 ``reranker.device: "cpu"`` 使用 CPU 运行
- 使用远程 API 模式（``use_remote: true``），不在本地加载模型
- 仅 Embedder 用 GPU、Reranker 用 CPU 或远程 API

**远程模型连接失败 / API 超时**

- 检查 ``OPENAI_API_KEY`` 是否正确
- 检查 ``base_url`` 是否可访问（如使用代理需配置 ``https_proxy``）
- 增加超时时间：在 config.yaml 中设置 ``timeout: 120``
- 确认 API 额度是否用尽

**build_high_level 报错**

- 检查 LLM API Key 是否正确配置在 ``.env`` 或 ``config.yaml`` 中
- 检查 API base_url 是否可访问
- 如果是本地模型模式，确认已安装 ``sentence-transformers``

**检索结果不相关**

- 增加 ``top_k`` 参数看看是否有更相关的结果在靠后位置
- 检查 metadata 中的 timestamp 是否正确
- 尝试更具体的查询措辞
- 确认已调用 ``build_high_level()`` 构建高阶记忆

**内存占用过高**

- 使用远程 Embedding/Reranker 替代本地模型（节省约 8GB）
- 减小 ``similarity_recent_window``
- 启用持久化并定期 ``save``/``load``
- 定期调用 ``system.flush()``

**查看系统实时状态**

.. code-block:: python

   # 使用 monitor 属性查看内存、单元数、图状态等
   print(system.monitor)

   # 或获取详细指标
   stats = system.monitor.to_dict()
   print(f"内存: {stats['rss_memory_mb']:.1f} MB (来源: {stats['memory_source']})")
   print(f"待处理: {stats['pending_units']} 单元 / {stats['pending_events']} 事件")
   print(f"脏标记: {stats['dirty']}")

仍然无法解决？
--------------

查阅更详细的高级故障排除指南：:doc:`/advanced-user/troubleshooting-advanced`。
