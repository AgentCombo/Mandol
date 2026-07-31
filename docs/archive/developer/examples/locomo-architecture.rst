LoCoMo 架构详解
=====================

LocomoMemorySystem 封装层设计。

设计意图
--------

``LocomoMemorySystem`` 是对 ``MemorySystem`` 的业务封装，提供 LoCoMo 数据集专用的数据加载和记忆构建流程。

核心类
------

.. code-block:: python

   class LocomoMemorySystem:
       config: LocomoMemoryConfig
       system: MemorySystem

       def load_and_process_samples(self) -> dict:
           """加载 LoCoMo JSON 数据并添加为 MemoryUnit"""
           pass

       def build_high_level_memories(self, mode="auto") -> BuildReport:
           """构建高阶记忆"""
           pass

       def get_memory_stats(self) -> dict:
           """统计信息"""
           pass

       def search(self, query, top_k=5) -> list[SearchHit]:
           """使用 memory_system 检索"""
           pass

       def run_query_set(self, queries) -> list[dict]:
           """运行预设查询集"""
           pass

封装层模式
----------

.. mermaid::

   graph LR
       A[LocomoMemorySystem] -->|封装| B[MemorySystem]
       A -->|配置| C[LocomoMemoryConfig]
       B --> D[SemanticMapService]
       B --> E[SemanticGraphService]
       B --> F[SessionManager]

这种封装模式适合：
- 特定数据集有固定处理流程
- 需要预定义查询评估集
- 为特定业务场景提供简化的 API

提供者切换
----------

``LocomoMemorySystem`` 构造时可通过 config 切换提供者：

.. code-block:: python

   config = LocomoMemoryConfig(
       llm_provider="openai",
       embedder_provider="sentence_transformers",
   )
   system = LocomoMemorySystem(config=config)
