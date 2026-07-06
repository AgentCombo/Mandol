from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING, Set

from ..core.memory_unit import MemoryUnit

if TYPE_CHECKING:
    from .query_bundle import QueryBundle

from ..utils.logging_config import create_module_logger

logger = create_module_logger("retrieval_interface")

class RetrievalMethod(Enum):
    """Retrieval backends supported by Mandol."""
    BM25 = "bm25"
    COSINE_SIMILARITY = "cosine_similarity"
    SPLADE = "splade"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID = "hybrid"
    GRAPH_CONTEXT_EXPANSION = "graph_context_expansion"

RETRIEVAL_METHOD_MAPPING = {
    "bm25": RetrievalMethod.BM25,
    "cosine": RetrievalMethod.COSINE_SIMILARITY,
    "cosine_similarity": RetrievalMethod.COSINE_SIMILARITY,
    "splade": RetrievalMethod.SPLADE,
    "graph": RetrievalMethod.GRAPH_TRAVERSAL,
    "graph_traversal": RetrievalMethod.GRAPH_TRAVERSAL,
    "graph_context_expansion": RetrievalMethod.GRAPH_CONTEXT_EXPANSION,
    "hybrid": RetrievalMethod.HYBRID,
}

def parse_retrieval_methods(methods: Union[str, List[str], List[RetrievalMethod]]) -> List[RetrievalMethod]:
    """Normalize retrieval method names into RetrievalMethod values."""
    if isinstance(methods, str):
        methods = [methods]
    
    result = []
    for method in methods:
        if isinstance(method, RetrievalMethod):
            result.append(method)
        elif isinstance(method, str):
            method_lower = method.lower()
            if method_lower in RETRIEVAL_METHOD_MAPPING:
                result.append(RETRIEVAL_METHOD_MAPPING[method_lower])
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
        else:
            raise ValueError(f"Unsupported retrieval method type: {type(method)}")
    
    return result

def parse_weights(weights: Union[Dict[str, float], Dict[RetrievalMethod, float]], 
                 methods: List[RetrievalMethod]) -> Dict[RetrievalMethod, float]:
    """Normalize fusion weights by RetrievalMethod."""
    if not weights:
        weight_value = 1.0 / len(methods)
        return {method: weight_value for method in methods}
    
    result = {}
    for key, value in weights.items():
        if isinstance(key, str):
            method = RETRIEVAL_METHOD_MAPPING.get(key.lower())
            if method is None:
                raise ValueError(f"Unknown retrieval weight key: {key}")
            result[method] = float(value)
        elif isinstance(key, RetrievalMethod):
            result[key] = float(value)
        else:
            raise ValueError(f"Unsupported weight key type: {type(key)}")
    
    for method in methods:
        if method not in result:
            result[method] = 0.0
    
    return result

class RetrievalResult:
    """Single retrieved memory unit with score and backend metadata."""
    
    def __init__(self, unit: MemoryUnit, score: float, method: RetrievalMethod, 
                 method_details: Optional[Dict[str, Any]] = None):
        self.unit = unit
        self.score = score
        self.method = method
        self.method_details = method_details or {}
        
    def __repr__(self):
        return f"RetrievalResult(uid={self.unit.uid}, score={self.score:.4f}, method={self.method.value})"

class BaseRetriever(ABC):
    """Abstract interface implemented by individual retrieval backends."""
    
    @abstractmethod
    def search(self, query: Union[str, "QueryBundle"], top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """Return scored retrieval results for a query."""
        pass
    
    @abstractmethod
    def get_method_type(self) -> RetrievalMethod:
        """Return the retrieval backend identifier."""
        pass
    
    def _get_uids_from_spaces(self, space_names: List[str]) -> List[str]:
        """Resolve memory-space names to candidate unit UIDs."""
        uids: Set[str] = set()
        
        try:
            if not hasattr(self, 'retrieval_source'):
                logger.warning("Retriever does not expose retrieval_source.")
                return []

            retrieval_source = self.retrieval_source
            if hasattr(retrieval_source, 'get_memory_space'):
                for name in space_names:
                    space = retrieval_source.get_memory_space(name)
                    if not space and isinstance(name, str) and name.startswith("ms:"):
                        space = retrieval_source.get_memory_space(name[3:])
                    if space:
                        uids.update(space.get_all_unit_uids(recursive=True))
            elif hasattr(retrieval_source, 'get_units_in_memory_space'):
                try:
                    normalized = [
                        name[3:] if isinstance(name, str) and name.startswith("ms:") else name
                        for name in space_names
                    ]
                    units = retrieval_source.get_units_in_memory_space(
                        ms_names=normalized,
                        recursive=True
                    )
                    uids.update(unit.uid for unit in units)
                except Exception as e:
                    logger.debug(f"Failed to resolve space UIDs through get_units_in_memory_space: {e}")
            elif hasattr(retrieval_source, 'semantic_map'):
                semantic_map = retrieval_source.semantic_map
                if hasattr(semantic_map, 'get_memory_space'):
                    for name in space_names:
                        space = semantic_map.get_memory_space(name)
                        if not space and isinstance(name, str) and name.startswith("ms:"):
                            space = semantic_map.get_memory_space(name[3:])
                        if space:
                            uids.update(space.get_all_unit_uids(recursive=True))
                elif hasattr(semantic_map, 'get_units_in_memory_space'):
                    normalized = [
                        name[3:] if isinstance(name, str) and name.startswith("ms:") else name
                        for name in space_names
                    ]
                    units = semantic_map.get_units_in_memory_space(
                        ms_names=normalized,
                        recursive=True
                    )
                    uids.update(unit.uid for unit in units)
            else:
                logger.warning("retrieval_source does not support memory-space UID lookup.")

            logger.debug(f"Resolved {len(uids)} candidate UIDs from spaces {space_names}.")
        
        except Exception as e:
            logger.error(f"Failed to resolve UIDs from spaces: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return list(uids)

class RetrievalInterface(ABC):
    """Storage interface required by Mandol retrieval backends."""
    
    @abstractmethod
    def search_similarity_by_vector(self, query_embedding, k: int, 
                                   ms_names: Optional[List[str]] = None,
                                   candidate_uids: Optional[List[str]] = None,
                                   **kwargs) -> List[Tuple[MemoryUnit, float]]:
        """Return dense-similarity results for an embedding vector."""
        pass
    
    @abstractmethod
    def search_similarity_by_text(self, query_text: str, k: int, 
                                 ms_names: Optional[List[str]] = None,
                                 candidate_uids: Optional[List[str]] = None,
                                 **kwargs) -> List[Tuple[MemoryUnit, float]]:
        """Embed query text and return dense-similarity results."""
        pass
    
    @abstractmethod
    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        pass
    
    @abstractmethod
    def get_all_units(self) -> List[MemoryUnit]:
        pass
    
    @abstractmethod
    def get_units_in_memory_space(self, ms_names: List[str], 
                                  recursive: bool = True) -> List[MemoryUnit]:
        """Return units in memory space."""
        pass

class MultiRetrievalInterface(ABC):
    """Interface for coordinated multi-backend retrieval."""
    
    @abstractmethod
    def add_retriever(self, retriever: BaseRetriever):
        """Register a retrieval backend."""
        pass
    
    @abstractmethod
    def smart_search(self, 
                    query: str,
                    methods: Union[str, List[str], List[RetrievalMethod]] = None,
                    top_k: int = 10,
                    
                    
                    fusion_method: str = "rrf",
                    rerank_method: Optional[str] = None,
                    weights: Optional[Union[Dict[str, float], Dict[RetrievalMethod, float]]] = None,
                    
                    enable_graph_expansion: bool = False,
                    graph_expansion_config: Optional[Dict[str, Any]] = None,
                    
                    
                    rerank_params: Optional[Dict[str, Any]] = None,
                    
                    return_detailed: bool = False,
                    
                    **kwargs) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        """Run fused retrieval across one or more backends.

        Args:
            query: Query text.
            methods: Retrieval methods to use. When omitted, the implementation
                chooses its default backend set.
            top_k: Number of final results to return.
            fusion_method: Score fusion strategy.
            rerank_method: Optional reranker identifier.
            weights: Optional per-method fusion weights.
            enable_graph_expansion: Whether to expand results with graph context.
            graph_expansion_config: Optional graph expansion settings.
            rerank_params: Optional reranker-specific settings.
            return_detailed: Whether to return backend diagnostics with results.
            **kwargs: Backend-specific retrieval options.

        Returns:
            Ranked memory units or a detailed result dictionary.
        """
        pass

    async def smart_search_async(self,
                    query: str,
                    methods: Union[str, List[str], List[RetrievalMethod]] = None,
                    top_k: int = 10,

                    
                    fusion_method: str = "rrf",
                    rerank_method: Optional[str] = None,
                    weights: Optional[Union[Dict[str, float], Dict[RetrievalMethod, float]]] = None,

                    enable_graph_expansion: bool = False,
                    graph_expansion_config: Optional[Dict[str, Any]] = None,

                    
                    rerank_params: Optional[Dict[str, Any]] = None,

                    return_detailed: bool = False,

                    **kwargs) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        """Run smart search async."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement smart_search_async(); "
            "use smart_search() or provide an async implementation."
        )
    
    @abstractmethod
    def smart_search_with_quantification(self, 
                                       query: str,
                                       methods: Union[str, List[str], List[RetrievalMethod]] = None,
                                       top_k: int = 10,
                                       rerank_method: Optional[str] = "baai",
                                       rerank_params: Optional[Dict[str, Any]] = None,
                                       **kwargs) -> Dict[str, Any]:
        """Run smart search with quantification."""
        pass

    async def smart_search_with_quantification_async(self,
                                       query: str,
                                       methods: Union[str, List[str], List[RetrievalMethod]] = None,
                                       top_k: int = 10,
                                       rerank_method: Optional[str] = "baai",
                                       rerank_params: Optional[Dict[str, Any]] = None,
                                       **kwargs) -> Dict[str, Any]:
        """Run smart search with quantification async."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "smart_search_with_quantification_async(); use the sync method or "
            "provide an async implementation."
        )
