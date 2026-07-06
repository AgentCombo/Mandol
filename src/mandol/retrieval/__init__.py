"""Package exports for retrieval."""

from .retrieval_interface import (
    BaseRetriever,
    RetrievalInterface, 
    MultiRetrievalInterface,
    RetrievalMethod,
    RetrievalResult,
    parse_retrieval_methods,
    parse_weights
)

from .advance_retriever import (
    MultiRetriever,
    BM25Retriever,
    SPLADERetriever,
    CosineRetrieverAdapter,
)

from .graph_context_expander import GraphContext, PathInfo, GraphContextExpander

from .rerank_manager import RerankerManager


from .query_bundle import QueryBundle

from .score_fusion import ScoreFusion

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

__all__ = [
    'BaseRetriever',
    'RetrievalInterface',
    'MultiRetrievalInterface', 
    'RetrievalMethod',
    'RetrievalResult',
    'parse_retrieval_methods',
    'parse_weights',
    
    'MultiRetriever',
    'BM25Retriever',
    'SPLADERetriever',
    'CosineRetrieverAdapter',
    
    'GraphContext',
    'PathInfo',
    'GraphContextExpander',
    
    
    'RerankerManager',
    
    
    'QueryBundle',
    
    'ScoreFusion',

    'TRANSFORMERS_AVAILABLE'
]


__version__ = '1.0.0'

__doc__ = """
AgentMemorySystem Retrieval Module
==================================

提供多种检索方法和高级检索功能：

检索方法：
---------
- BM25: 关键词匹配检索
- SPLADE: 稀疏向量检索
- Cosine Similarity: 余弦相似度检索
- Graph Traversal: 图遍历检索

核心功能：
---------
1. MultiRetriever.smart_search(): 统一的智能检索接口
    - MultiRetriever.smart_search_async(): 异步镜像接口，支持 vLLM rerank_async
   - 支持多方法融合
   - 支持重排序 (BAAI, Qwen, Jina, MMR)
   - 支持图上下文扩展
   - 支持并行检索

2. MultiRetriever.smart_search_with_quantification(): 量化检索接口
   - 计算检索一致性分数 (稀疏/稠密方法一致性)
   - 计算语义置信度 (重排序Top-1分数)
   - 提供综合质量评分和诊断建议

使用示例：
---------
```python
from mandol.retrieval import MultiRetriever, RetrievalMethod

# 创建检索器
retriever = MultiRetriever(semantic_graph)

# 基础检索
results = retriever.smart_search(
    query="查询文本",
    methods=["bm25", "splade", "cosine"],
    top_k=10
)

# 带量化的检索
quantified = retriever.smart_search_with_quantification(
    query="查询文本",
    methods=["bm25", "splade", "cosine"],
    top_k=10,
    rerank_method="baai"
)

# 访问结果
results = quantified["results"]  # [(unit, score), ...]
metrics = quantified["quantification"]  # 量化指标
print(f"一致性: {metrics['consistency_score']}")
print(f"置信度: {metrics['confidence_score']}")
print(f"诊断: {metrics['diagnosis']}")"""
