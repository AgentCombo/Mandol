Custom Dimension Builder
============================

Implement the ``DimensionBuilder`` interface.

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

Registering Dimension Builders
---------------------------------

Custom dimensions need to be injected into ``MultiDimSemanticGraphBuilder`` during ``MemorySystem`` initialization. Currently recommended approach is to inherit ``MemorySystem`` and override the builder initialization logic:

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
