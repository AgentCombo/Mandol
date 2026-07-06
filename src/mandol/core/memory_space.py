import pickle
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import numpy as np
from .memory_unit import MemoryUnit
from ..utils.logging_config import create_module_logger

logger = create_module_logger("memory_space")


class MemorySpace:
    """Logical memory-space membership for Mandol memory units.

    MemorySpace stores unit UID membership and child-space links. Payloads and
    embeddings remain owned by SemanticMap; this class only manages membership
    and invalidates SemanticMap space-filter caches when membership changes.
    """

    def __init__(self, ms_name: str, faiss_index_type: Optional[str] = None):
        """Initialize a memory space.

        Args:
            ms_name: Stable memory-space name.
            faiss_index_type: Deprecated per-space index type retained for
                checkpoint compatibility.
        """
        if not isinstance(ms_name, str) or not ms_name.strip():
            raise ValueError("MemorySpace name must be a non-empty string.")
        self.name: str = ms_name

        self._unit_uids: Set[str] = set()  
        self._child_space_names: Set[str] = set()

        
        self._faiss_index_type: Optional[str] = faiss_index_type

        self._semantic_map_ref = None

    def __str__(self) -> str:
        total_members = len(self._unit_uids) + len(self._child_space_names)
        return f"MemorySpace(name={self.name}, units={len(self._unit_uids)}, child_spaces={len(self._child_space_names)})"

    def __repr__(self):
        return f"MemorySpace({self.name})"

    def _set_semantic_map_ref(self, semantic_map):
        """Attach the owning SemanticMap by weak reference."""
        import weakref

        self._semantic_map_ref = weakref.ref(semantic_map)

    def _get_semantic_map(self):
        """Return the owning SemanticMap."""
        if self._semantic_map_ref is None:
            raise RuntimeError(f"MemorySpace '{self.name}' is not attached to a SemanticMap.")
        semantic_map = self._semantic_map_ref()
        if semantic_map is None:
            raise RuntimeError(f"MemorySpace '{self.name}' SemanticMap reference has expired.")
        return semantic_map

    def _notify_membership_changed(self) -> None:
        """Notify SemanticMap caches when this space membership graph changes."""
        try:
            semantic_map = self._get_semantic_map()
        except RuntimeError:
            return
        invalidate = getattr(semantic_map, "_invalidate_space_filter_cache", None)
        if callable(invalidate):
            invalidate()

    
    

    def add_unit(self, unit_or_uid: Union[str, "MemoryUnit"]):
        """Add a unit UID to this memory space."""
        if isinstance(unit_or_uid, str):
            
            self._add_unit_by_uid(unit_or_uid)
        elif hasattr(unit_or_uid, "uid"):
            self._add_unit_by_uid(unit_or_uid.uid)
        else:
            raise TypeError(
                f"add_unit() expects a str UID or MemoryUnit, got {type(unit_or_uid)}"
            )

    def add_child_space(self, space_or_name: Union[str, "MemorySpace"]):
        """Add a child memory-space reference."""
        if isinstance(space_or_name, str):
            self._add_child_space_by_name(space_or_name)
        elif hasattr(space_or_name, "name"):
            self._add_child_space_by_name(space_or_name.name)
        else:
            raise TypeError(
                f"add_child_space() expects a str name or MemorySpace, got {type(space_or_name)}"
            )

    def remove_unit(self, unit_or_uid: Union[str, "MemoryUnit"]):
        """Remove a unit UID from this memory space."""
        if isinstance(unit_or_uid, str):
            
            self._remove_unit_by_uid(unit_or_uid)
        elif hasattr(unit_or_uid, "uid"):
            self._remove_unit_by_uid(unit_or_uid.uid)
        else:
            raise TypeError(
                f"remove_unit() expects a str UID or MemoryUnit, got {type(unit_or_uid)}"
            )

    def remove_child_space(self, space_or_name: Union[str, "MemorySpace"]):
        """Remove a child memory-space reference."""
        if isinstance(space_or_name, str):
            self._remove_child_space_by_name(space_or_name)
        elif hasattr(space_or_name, "name"):
            self._remove_child_space_by_name(space_or_name.name)
        else:
            raise TypeError(
                f"remove_child_space() expects a str name or MemorySpace, got {type(space_or_name)}"
            )

    def contains_unit(
        self, unit_or_uid: Union[str, "MemoryUnit"], recursive: bool = False
    ) -> bool:
        """Return whether this space contains a unit UID."""
        if isinstance(unit_or_uid, str):
            uid = unit_or_uid
        elif hasattr(unit_or_uid, "uid"):
            uid = unit_or_uid.uid
        else:
            raise TypeError(
                f"contains_unit() expects a str UID or MemoryUnit, got {type(unit_or_uid)}"
            )

        if uid in self._unit_uids:
            return True

        if recursive:
            child_uids = self.get_all_unit_uids(recursive=True)
            return uid in child_uids

        return False

    def contains_space(
        self, space_or_name: Union[str, "MemorySpace"], recursive: bool = False
    ) -> bool:
        """Return whether this space contains a child space."""
        if isinstance(space_or_name, str):
            space_name = space_or_name
        elif hasattr(space_or_name, "name"):
            space_name = space_or_name.name
        else:
            raise TypeError(
                f"contains_space() expects a str name or MemorySpace, got {type(space_or_name)}"
            )

        if space_name in self._child_space_names:
            return True

        if recursive:
            child_names = self.get_all_child_space_names(recursive=True)
            return space_name in child_names

        return False

    
    

    def _add_unit_by_uid(self, uid: str):
        """Add a unit UID without requiring the MemoryUnit object."""
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("MemoryUnit UID must be a non-empty string.")

        
        try:
            semantic_map = self._get_semantic_map()
            if uid not in semantic_map.memory_units:
                logger.warning(
                    f"MemoryUnit '{uid}' is not present in SemanticMap; keeping the membership reference."
                )
        except RuntimeError:
            pass

        before_count = len(self._unit_uids)
        self._unit_uids.add(uid)
        if len(self._unit_uids) != before_count:
            self._notify_membership_changed()
        logger.debug(f"Added MemoryUnit reference '{uid}' to MemorySpace '{self.name}'.")

    def _add_child_space_by_name(self, space_name: str):
        """Add a child-space name without requiring the MemorySpace object."""
        if not isinstance(space_name, str) or not space_name.strip():
            raise ValueError("MemorySpace name must be a non-empty string.")

        if space_name == self.name:
            raise ValueError("A MemorySpace cannot be its own child.")

        try:
            semantic_map = self._get_semantic_map()
            if space_name not in semantic_map.memory_spaces:
                logger.warning(
                    f"MemorySpace '{space_name}' is not present in SemanticMap; keeping the child reference."
                )
        except RuntimeError:
            pass

        before_count = len(self._child_space_names)
        self._child_space_names.add(space_name)
        if len(self._child_space_names) != before_count:
            self._notify_membership_changed()
        logger.debug(
            f"Added MemorySpace reference '{space_name}' to MemorySpace '{self.name}'."
        )

    def _remove_unit_by_uid(self, uid: str):
        """Remove a unit UID membership."""
        if uid in self._unit_uids:
            self._unit_uids.remove(uid)
            self._notify_membership_changed()
            logger.debug(f"Removed MemoryUnit reference '{uid}' from MemorySpace '{self.name}'.")
        else:
            logger.warning(
                f"MemoryUnit reference '{uid}' does not exist in MemorySpace '{self.name}'."
            )

    def _remove_child_space_by_name(self, space_name: str):
        """Remove a child-space membership."""
        if space_name in self._child_space_names:
            self._child_space_names.remove(space_name)
            self._notify_membership_changed()
            logger.debug(
                f"Removed MemorySpace reference '{space_name}' from MemorySpace '{self.name}'."
            )
        else:
            logger.warning(
                f"MemorySpace reference '{space_name}' does not exist in MemorySpace '{self.name}'."
            )

    
    

    def get_unit_uids(self) -> Set[str]:
        """Return unit uids."""
        return self._unit_uids.copy()

    def get_child_space_names(self) -> Set[str]:
        """Return child space names."""
        return self._child_space_names.copy()

    def get_all_unit_uids(self, recursive: bool = True) -> Set[str]:
        """Return all unit uids."""
        result = self._unit_uids.copy()

        if recursive:
            try:
                semantic_map = self._get_semantic_map()
                for child_space_name in self._child_space_names:
                    child_space = semantic_map.memory_spaces.get(child_space_name)
                    if child_space:
                        result.update(child_space.get_all_unit_uids(recursive=True))
                    else:
                        logger.warning(
                            f"Child space '{child_space_name}' is not present in SemanticMap."
                        )
            except RuntimeError:
                logger.warning("SemanticMap is unavailable; skipping recursive lookup.")

        return result

    def get_all_child_space_names(self, recursive: bool = True) -> Set[str]:
        """Return all child space names."""
        result = self._child_space_names.copy()

        if recursive:
            try:
                semantic_map = self._get_semantic_map()
                for child_space_name in self._child_space_names:
                    child_space = semantic_map.memory_spaces.get(child_space_name)
                    if child_space:
                        result.update(
                            child_space.get_all_child_space_names(recursive=True)
                        )
                    else:
                        logger.warning(
                            f"Child space '{child_space_name}' is not present in SemanticMap."
                        )
            except RuntimeError:
                logger.warning("SemanticMap is unavailable; skipping recursive lookup.")

        return result

    def get_all_units(self) -> List["MemoryUnit"]:
        """Return all units."""
        try:
            semantic_map = self._get_semantic_map()
            unit_uids = self.get_all_unit_uids(recursive=True)

            units = []
            for uid in unit_uids:
                unit = semantic_map.memory_units.get(uid)
                if unit:
                    units.append(unit)
                else:
                    logger.warning(f"MemoryUnit '{uid}' is not present in SemanticMap.")

            return units
        except RuntimeError:
            logger.error("SemanticMap is unavailable; cannot resolve MemoryUnit objects.")
            return []

    def cluster_nodes(self, method: str = "dbscan", **kwargs) -> Dict[int, List[str]]:
        """Run cluster nodes."""
        from ..cluster import cluster_nodes

        return cluster_nodes(self, method=method, **kwargs)

    def get_all_spaces(self) -> List["MemorySpace"]:
        """Return all spaces."""
        try:
            semantic_map = self._get_semantic_map()
            space_names = self.get_all_child_space_names(recursive=True)

            spaces = []
            for space_name in space_names:
                space = semantic_map.memory_spaces.get(space_name)
                if space:
                    spaces.append(space)
                else:
                    logger.warning(f"MemorySpace '{space_name}' is not present in SemanticMap.")

            return spaces
        except RuntimeError:
            logger.error("SemanticMap is unavailable; cannot resolve MemorySpace objects.")
            return []

    
    
    

    def build_index(self, embedding_dim: int = 512, min_unit_threshold: int = 100):
        """Compatibility hook for legacy per-space index builds."""
        if hasattr(self, "_emb_index"):
            delattr(self, "_emb_index")
        if hasattr(self, "_index_to_uid"):
            delattr(self, "_index_to_uid")
        logger.debug(
            f"MemorySpace '{self.name}' skips local index construction; SemanticMap global index filtering is used."
        )

    def _convert_to_cosine_index_type(self, index_type: str) -> str:
        """Convert to cosine index type."""
        
        return index_type
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector."""
        try:
            vector = np.asarray(vector, dtype=np.float32)
            flattened = vector.reshape(-1)
            norm = np.linalg.norm(flattened)
            if norm < 1e-6:
                logger.debug("Zero or near-zero vector detected; normalization skipped.")
                return np.zeros(flattened.shape[0], dtype=np.float32)
            return (flattened / norm).astype(np.float32)
        except Exception as e:
            logger.warning(f"Vector normalization failed: {e}")
            return np.asarray(vector, dtype=np.float32).reshape(-1)

    # Compatibility helper; similarity scoring is delegated to SemanticMap.
    def search_similarity_units_by_vector(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple["MemoryUnit", float]]:
        """Return top-k units in this space by vector similarity."""
        semantic_map = self._get_semantic_map()
        try:
            return semantic_map.search_similarity_by_vector(
                query_vector,
                k=top_k,
                ms_names=[self.name],
            )
        except Exception as e:
            logger.warning(f"MemorySpace '{self.name}' global-index search failed; falling back to NumPy search: {e}")

        units = self.get_all_units()
        if not units:
            logger.debug(f"MemorySpace '{self.name}' has no units to search.")
            return []

        try:
            normalized_query = self._normalize_vector(query_vector)
            valid_units = [u for u in units if u.embedding is not None]
            if not valid_units:
                logger.warning(f"MemorySpace '{self.name}' has no valid embeddings for search.")
                return []
            matrix = np.ascontiguousarray(
                [self._normalize_vector(u.embedding) for u in valid_units],
                dtype=np.float32,
            )
            scores = matrix @ normalized_query.astype(np.float32, copy=False)
            real_k = min(top_k, len(valid_units))
            if real_k <= 0:
                return []
            if len(valid_units) > real_k:
                top_indices = np.argpartition(scores, -real_k)[-real_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                top_indices = np.argsort(scores)[::-1]
            return [
                (
                    valid_units[int(idx)],
                    max(0.0, min(1.0, (float(scores[int(idx)]) + 1.0) / 2.0)),
                )
                for idx in top_indices
            ]
        except Exception as e:
            logger.error(f"MemorySpace '{self.name}' NumPy search failed: {e}")
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        try:
            a = vec1.astype(np.float32)
            b = vec2.astype(np.float32)
            
            dot_product = float(np.dot(a, b))
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            cosine_sim = dot_product / (norm_a * norm_b + 1e-8)
            
            return (cosine_sim + 1.0) / 2.0
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0

    def _convert_l2_to_cosine_similarity(self, query_vec: np.ndarray, target_vec: np.ndarray, l2_distance: float) -> float:
        """Convert L2 to cosine similarity."""
        try:
            return self._calculate_cosine_similarity(query_vec, target_vec)
        except Exception as e:
            logger.warning(f"Failed to convert L2 distance to cosine similarity: {e}")
            return max(0.0, 1.0 / (1.0 + l2_distance))

    def _convert_normalized_l2_to_cosine_similarity(self, l2_distance: float) -> float:
        """Convert normalized L2 to cosine similarity."""
        try:
            l2_distance = max(0.0, l2_distance)
            
            cosine_sim = 1.0 - (l2_distance * l2_distance) / 2.0
            
            normalized_similarity = (cosine_sim + 1.0) / 2.0
            
            return max(0.0, min(1.0, normalized_similarity))
            
        except Exception as e:
            logger.warning(f"Failed to convert L2 distance to cosine similarity: {e}")
            return 0.0

    
    

    def save(self, file_path: str):
        """Save this memory-space membership record."""
        save_data = {
            "name": self.name,
            "unit_uids": self._unit_uids,
            "child_space_names": self._child_space_names,
            
        }

        with open(file_path, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, file_path: str) -> "MemorySpace":
        """Load a memory-space membership record."""
        with open(file_path, "rb") as f:
            save_data = pickle.load(f)

        instance = cls(save_data["name"])
        instance._unit_uids = save_data.get("unit_uids", set())
        instance._child_space_names = save_data.get("child_space_names", set())

        return instance
