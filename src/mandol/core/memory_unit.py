from typing import Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
import logging
from ..utils.logging_config import create_module_logger

logger = create_module_logger("memory_unit")


class MemoryUnit:
    """Container for one Mandol memory item.

    A MemoryUnit stores the public UID, raw payload, metadata, dense embedding,
    optional sparse embedding, and a cached text view used by retrieval
    backends.
    """

    __slots__ = ("uid", "_raw_data", "metadata", "embedding", "sparse_embedding", "text_cached", "created_time")
    _TEXT_CACHE_FIELDS = ("text_content", "content", "description", "summary", "title", "message")

    def __init__(
        self,
        uid: str,
        raw_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        sparse_embedding: Optional[Union[Dict[int, float], np.ndarray]] = None
    ):
        """Initialize a memory unit.

        Args:
            uid: Stable public identifier for the memory unit.
            raw_data: Raw payload associated with the unit.
            metadata: Optional metadata dictionary.
            embedding: Optional dense vector.
            sparse_embedding: Optional sparse vector or dense sparse-model
                output.
        """
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("MemoryUnit UID must be a non-empty string.")
        if not isinstance(raw_data, dict):
            raise ValueError("MemoryUnit raw_data must be a dictionary.")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("MemoryUnit metadata must be a dictionary.")

        self.uid: str = uid
        self.raw_data = raw_data
        self.metadata = metadata or {
            "created": str(datetime.now()),
            "updated": str(datetime.now()),
        }
        self.embedding: Optional[np.ndarray] = embedding
        self.sparse_embedding: Optional[Union[Dict[int, float], np.ndarray]] = sparse_embedding
        self.created_time = None

    @property
    def raw_data(self) -> Dict[str, Any]:
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValueError("MemoryUnit raw_data must be a dictionary.")
        self._raw_data = value
        self.text_cached = self._extract_text_cached(self.uid, value)

    def refresh_text_cache(self) -> str:
        self.text_cached = self._extract_text_cached(self.uid, self._raw_data)
        return self.text_cached

    @classmethod
    def _extract_text_cached(cls, uid: str, raw_data: Dict[str, Any]) -> str:
        text_content = ""
        if raw_data:
            for field in cls._TEXT_CACHE_FIELDS:
                value = raw_data.get(field, "")
                if isinstance(value, str) and value:
                    text_content = value
                    break

            if not text_content:
                try:
                    text_parts = []
                    for key, value in raw_data.items():
                        if isinstance(value, str) and key != "embedding":
                            text_parts.append(f"{key}: {value}")
                    text_content = "; ".join(text_parts) if text_parts else f"Unit {uid}"
                except Exception:
                    text_content = f"Memory unit {uid}"

        if not text_content.strip():
            text_content = f"Memory unit {uid}"
        return text_content

    def __str__(self) -> str:
        embedding_shape = self.embedding.shape if self.embedding is not None else None
        
        
        if self.sparse_embedding is not None:
            if isinstance(self.sparse_embedding, dict):
                sparse_info = f"Dict({len(self.sparse_embedding)} nonzero)"
            elif isinstance(self.sparse_embedding, np.ndarray):
                sparse_info = f"Array{self.sparse_embedding.shape}"
            else:
                sparse_info = f"Unknown({type(self.sparse_embedding)})"
        else:
            sparse_info = None
        
        return (f"MemoryUnit(uid='{self.uid}', raw_data={self.raw_data}, "
                f"embedding_shape={embedding_shape}, sparse_embedding={sparse_info})")

    def __repr__(self) -> str:
        return f"MemoryUnit({self.uid})"

    def __eq__(self, value):
        if not isinstance(value, MemoryUnit):
            return False
        return (
            self.uid == value.uid
            and self.raw_data == value.raw_data
            and self.metadata == value.metadata
        )

    def __hash__(self):
        return hash(self.uid)
    
    
    
    def has_sparse_embedding(self) -> bool:
        """Return whether sparse embedding is available."""
        return self.sparse_embedding is not None
    
    def set_sparse_embedding(self, sparse_vector: Union[Dict[int, float], np.ndarray], threshold: float = 1e-6):
        """Set a sparse embedding from sparse weights or a dense vector."""
        if isinstance(sparse_vector, dict):
            
            self.sparse_embedding = sparse_vector
        elif isinstance(sparse_vector, np.ndarray):
            
            self.sparse_embedding = self._dense_to_sparse(sparse_vector, threshold)
        else:
            raise ValueError(f"Unsupported sparse vector type: {type(sparse_vector)}")
    
    def get_sparse_embedding(self, format: str = "sparse", vocab_size: int = 30522) -> Optional[Union[Dict[int, float], np.ndarray]]:
        """Return sparse embedding."""
        if self.sparse_embedding is not None:
            if format == "sparse":
                if isinstance(self.sparse_embedding, dict):
                    return self.sparse_embedding
                elif isinstance(self.sparse_embedding, np.ndarray):
                    return self._dense_to_sparse(self.sparse_embedding)
            elif format == "dense":
                if isinstance(self.sparse_embedding, dict):
                    return self._sparse_to_dense(self.sparse_embedding, vocab_size)
                elif isinstance(self.sparse_embedding, np.ndarray):
                    return self.sparse_embedding
        return None
    
    @staticmethod
    def _dense_to_sparse(dense_vector: np.ndarray, threshold: float = 1e-6) -> Dict[int, float]:
        """Convert a dense vector to sparse index weights."""
        if not isinstance(dense_vector, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        
        sparse_dict = {}
        for idx, value in enumerate(dense_vector):
            if abs(value) > threshold:
                sparse_dict[int(idx)] = float(value)
        return sparse_dict
    
    @staticmethod
    def _sparse_to_dense(sparse_dict: Dict[int, float], vocab_size: int = 30522) -> np.ndarray:
        """Convert sparse index weights to a dense vector."""
        if not isinstance(sparse_dict, dict):
            raise ValueError("Input must be a dictionary.")
        
        dense_vector = np.zeros(vocab_size, dtype=np.float32)
        for idx, value in sparse_dict.items():
            if 0 <= idx < vocab_size:
                dense_vector[idx] = value
        return dense_vector
