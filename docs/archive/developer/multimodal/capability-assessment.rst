多模态能力评估
================

.. note::

   本文档从 ``multimodal-capability-assessment.md`` 转换而来，内容描述了 Mandol 在多模态场景下的能力现状和限制。

主要发现
--------

- 文本处理能力：完备
- 图片处理：支持 image_path 字段，通过视觉 Embedding 模型向量化
- 多模态混合：文本和图片可共存于同一 MemoryUnit
- 已知限制：UnifiedFactPipeline 存在未实现方法的调用问题

详细内容请参阅源文件：``multimodal-capability-assessment.md``。
