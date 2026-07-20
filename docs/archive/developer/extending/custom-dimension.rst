自定义维度构建器
==================

实现 ``DimensionBuilder`` 接口。

.. code-block:: python

   from mandol.application.multidim_semantic_graph import (
       DimensionBuilder,
       MultiDimBuildContext,
   )

   class SentimentDimension(DimensionBuilder):
       name = "sentiment_analysis"

       def build(self, ctx: MultiDimBuildContext) -> None:
           for unit in ctx.units:
               text = unit.raw_data.get("text_content", "")
               sentiment = analyze_sentiment(text)
               unit.metadata["sentiment"] = sentiment

注册维度构建器
--------------

需要在 ``MemorySystem`` 初始化时将自定义维度注入 ``MultiDimSemanticGraphBuilder``。
当前建议通过继承 ``MemorySystem`` 并覆盖构建器初始化逻辑来实现：

.. code-block:: python

   class CustomMemorySystem(MemorySystem):
       def _init_builder(self):
           from mandol.application.multidim_semantic_graph import (
               MultiDimSemanticGraphBuilder,
           )
           builder = MultiDimSemanticGraphBuilder(
               graph=self.semantic_graph,
           )
           builder.register_dimension(SentimentDimension())
           return builder
