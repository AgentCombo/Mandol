分块参数
========

chunk_max_tokens
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - 参数
     - 默认值
     - 单位
     - 说明
   * - ``chunk_max_tokens``
     - 512
     - tokens
     - 超过此值的文本自动拆分

**推荐值**：

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - 场景
     - 推荐值
     - 说明
   * - 短对话（客服）
     - 256
     - 粒度更细
   * - 中等对话（助手）
     - 512
     - 默认
   * - 长文档（知识库）
     - 1024
     - 大上下文窗口
