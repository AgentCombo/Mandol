"""Dense cosine retrieval adapter backed by SemanticMap search APIs.

The adapter lets MultiRetriever treat SemanticMap dense similarity as a normal
retrieval backend. QueryBundle inputs reuse cached dense embeddings, while
candidate_uids and space_names are forwarded to SemanticMap so dense filtering
semantics remain centralized.
"""

import logging
from typing import List, Optional, Union

from .retrieval_interface import BaseRetriever, RetrievalInterface, RetrievalMethod, RetrievalResult
from .query_bundle import QueryBundle
from ..utils.logging_config import create_module_logger

logger = create_module_logger("cosine_retriever")


class CosineRetrieverAdapter(BaseRetriever):
    """Expose SemanticMap dense similarity as a RetrievalResult backend."""
    
    def __init__(self, retrieval_source: RetrievalInterface):
        self.retrieval_source = retrieval_source
    
    def get_method_type(self) -> RetrievalMethod:
        return RetrievalMethod.COSINE_SIMILARITY
    
    def search(self, query: Union[str, QueryBundle], top_k: int = 10, 
           candidate_uids: Optional[List[str]] = None,
           space_names: Optional[List[str]] = None,
           **kwargs) -> List[RetrievalResult]:
        """Retrieve dense-similarity matches.

        Args:
            query: Raw text or QueryBundle. QueryBundle avoids recomputing the
                dense embedding when other stages already requested it.
            top_k: Maximum number of dense results to return.
            candidate_uids: Optional candidate UID set forwarded to SemanticMap.
            space_names: Optional MemorySpace filter forwarded as ``ms_names``.
            **kwargs: Supports the legacy ``ms_names`` alias.

        Returns:
            RetrievalResult objects with ``metadata["cosine_score"]``.
        """
        try:
            if space_names is None and "ms_names" in kwargs:
                space_names = kwargs.get("ms_names")

            search_kwargs = {}
            
            
            if candidate_uids:
                search_kwargs['candidate_uids'] = candidate_uids
            if space_names:
                search_kwargs['ms_names'] = space_names
            
            
            if isinstance(query, QueryBundle):
                dense_vector = query.get_or_compute_dense(
                    self.retrieval_source._get_text_embedding
                )
                if dense_vector is None:
                    logger.error("Cosine retrieval failed to compute QueryBundle dense vector.")
                    return []
                
                results = self.retrieval_source.search_similarity_by_vector(
                    dense_vector, top_k, **search_kwargs
                )
            else:
                results = self.retrieval_source.search_similarity_by_text(
                    query, top_k, **search_kwargs
                )
            
            retrieval_results = []
            for unit, score in results:
                retrieval_results.append(
                    RetrievalResult(unit, score, self.get_method_type(), {"cosine_score": score})
                )
            
            if space_names:
                logger.debug(f"Cosine retrieval returned {len(retrieval_results)} results from spaces {space_names}.")
            elif candidate_uids:
                logger.debug(f"Cosine retrieval returned {len(retrieval_results)} results from {len(candidate_uids)} candidates.")
            else:
                logger.debug(f"Cosine retrieval returned {len(retrieval_results)} global results.")
            
            return retrieval_results
        except Exception as e:
            logger.error(f"Cosine similarity retrieval failed: {e}")
            return []
