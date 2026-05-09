"""Semantic map service for storing, indexing, and querying memory units.

Provides a unified interface over UnitStore, VectorIndex, EmbeddingProvider,
and Reranker. Supports space-based organization, vector and text search,
reranking, and automatic memory limit enforcement.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..domain.memory_space import MemorySpace
from ..domain.memory_unit import MemoryUnit
from ..domain.types import Embedding, SpaceName, Uid
from ..ports.embedding_provider import EmbeddingProvider
from ..ports.reranker import Reranker
from ..ports.unit_store import UnitStore
from ..ports.vector_index import VectorIndex
from ..infrastructure.adaptive_vector_index import AdaptiveVectorIndex


class SemanticMapService:
    """Manages storage, indexing, and retrieval of memory units.

    Coordinates between UnitStore (persistence), VectorIndex (ANN search),
    EmbeddingProvider (embedding generation), and optional Reranker (reranking).

    Supports space-based hierarchical organization with parent-child space
    relationships and automatic memory limit enforcement that evicts oldest
    units when the store exceeds max_units_in_memory.

    Args:
        store: Backing UnitStore for unit persistence.
        index: VectorIndex for approximate nearest neighbor search.
        embedder: Optional EmbeddingProvider for generating embeddings.
        reranker: Optional Reranker for post-retrieval reranking.
        default_text_key: Default raw_data key for text content.
        default_image_path_key: Default raw_data key for image paths.
        max_units_in_memory: Soft limit before eviction (default 10000).
    """

    def __init__(
        self,
        *,
        store: UnitStore,
        index: VectorIndex,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        default_text_key: str = "text_content",
        default_image_path_key: str = "image_path",
        max_units_in_memory: int = 10000,
    ):
        self._store = store
        self._index = index
        if isinstance(index, AdaptiveVectorIndex):
            self._abi: Optional[AdaptiveVectorIndex] = index
        else:
            self._abi = None
        self._embedder = embedder
        self._reranker = reranker
        self._default_text_key = str(default_text_key)
        self._default_image_path_key = str(default_image_path_key)
        self._max_units_in_memory = int(max_units_in_memory)

        if self._index.dim() <= 0:
            raise ValueError("index.dim() must be > 0")
        if self._embedder is not None and self._embedder.embedding_dim() != self._index.dim():
            raise ValueError("embedder dim must match index dim")

        self._dirty_units: set[Uid] = set()
        self._deleted_units: set[Uid] = set()
        self._dirty_spaces: set[SpaceName] = set()

    def set_embedder(self, embedder: Optional[EmbeddingProvider]) -> None:
        """Replace the current embedding provider.

        Args:
            embedder: New EmbeddingProvider, or None to clear.

        Raises:
            ValueError: If the new embedder's dimension doesn't match the index.
        """
        if embedder is not None and embedder.embedding_dim() != self._index.dim():
            raise ValueError("embedder dim must match index dim")
        self._embedder = embedder

    def set_reranker(self, reranker: Optional[Reranker]) -> None:
        """Replace the current reranker.

        Args:
            reranker: New Reranker, or None to clear.
        """
        self._reranker = reranker

    def get_embedder(self) -> Optional[EmbeddingProvider]:
        """Return the current embedding provider, if any."""
        return self._embedder

    def get_reranker(self) -> Optional[Reranker]:
        """Return the current reranker, if any."""
        return self._reranker

    def get_store(self) -> UnitStore:
        """Return the underlying UnitStore."""
        return self._store

    @property
    def dim(self) -> int:
        return self._index.dim()

    def create_space(self, name: Union[str, SpaceName]) -> MemorySpace:
        """Create a new memory space or return an existing one.

        Args:
            name: Name for the new space.

        Returns:
            The created or existing MemorySpace.
        """
        space_name = SpaceName(str(name))
        existing = self._store.get_space(space_name)
        if existing is not None:
            return existing
        space = MemorySpace(name=space_name)
        self._store.upsert_spaces([space])
        self._dirty_spaces.add(space_name)
        return space

    def get_space(self, name: Union[str, SpaceName]) -> Optional[MemorySpace]:
        """Look up a memory space by name.

        Args:
            name: Space name to look up.

        Returns:
            The MemorySpace, or None if not found.
        """
        return self._store.get_space(SpaceName(str(name)))

    def list_spaces(self) -> List[MemorySpace]:
        """List all memory spaces.

        Returns:
            All MemorySpace objects.
        """
        return self._store.list_spaces()

    def add_unit(
        self,
        unit: MemoryUnit,
        *,
        space_names: Optional[Sequence[Union[str, SpaceName]]] = None,
        ensure_embedding: bool = True,
        rebuild_index_immediately: bool = False,
        embedding_text: Optional[str] = None,
        embedding_image_path: Optional[str] = None,
    ) -> None:
        """Insert a unit into the store and optionally index its embedding.

        If space_names are provided, the unit is assigned to those spaces and
        the spaces are created if they don't already exist.

        Args:
            unit: The MemoryUnit to add.
            space_names: Optional sequence of space names to assign the unit to.
            ensure_embedding: If True and unit has no embedding, compute one.
            rebuild_index_immediately: If True, rebuild the full vector index after insertion.
            embedding_text: Explicit text to use for embedding (overrides raw_data lookup).
            embedding_image_path: Explicit image path for embedding (overrides raw_data lookup).
        """
        uid = Uid(str(unit.uid))

        if space_names:
            names = [SpaceName(str(n)) for n in space_names]
            unit.metadata["spaces"] = [str(n) for n in names]
            for n in names:
                space = self.create_space(n)
                space.add_unit(uid)
                self._store.upsert_spaces([space])
                self._dirty_spaces.add(n)

        if ensure_embedding and unit.embedding is None:
            unit.embedding = self._embed_for_unit(
                unit,
                explicit_text=embedding_text,
                explicit_image_path=embedding_image_path,
            )

        self._store.upsert_units([unit])
        self._dirty_units.add(uid)

        if unit.embedding is not None:
            self._index.upsert([(uid, unit.embedding)])
            if self._abi is not None and space_names:
                for n in names:
                    self._abi.upsert_to_space([(uid, unit.embedding)], n)

        if rebuild_index_immediately:
            self.rebuild_index_from_store()

        self._enforce_memory_limit()

    def upsert_unit(
        self,
        unit: MemoryUnit,
        *,
        rebuild_index_immediately: bool = False,
    ) -> None:
        """Insert or update a unit in the store and index.

        Unlike add_unit, this does not auto-compute embeddings or manage
        space assignments — it uses whatever data is already on the unit.

        Args:
            unit: The MemoryUnit to upsert.
            rebuild_index_immediately: If True, rebuild the full vector index after upserting.
        """
        uid = Uid(str(unit.uid))
        self._store.upsert_units([unit])
        self._dirty_units.add(uid)

        if unit.embedding is not None:
            self._index.upsert([(uid, unit.embedding)])
            spaces = unit.metadata.get("spaces", [])
            if self._abi is not None and isinstance(spaces, list):
                for sp in spaces:
                    self._abi.upsert_to_space([(uid, unit.embedding)], SpaceName(str(sp)))

        if rebuild_index_immediately:
            self.rebuild_index_from_store()

    def delete_unit(self, uid: Union[str, Uid]) -> None:
        """Remove a unit from the store, index, and all spaces.

        Args:
            uid: The UID (or string) of the unit to delete.
        """
        u = Uid(str(uid))
        self._store.delete_units([u])
        self._index.delete([u])
        if self._abi is not None:
            self._abi.delete(u)
        self._deleted_units.add(u)
        self._dirty_units.discard(u)

        for space in self._store.list_spaces():
            if u in space.unit_uids:
                space.remove_unit(u)
                self._store.upsert_spaces([space])
                self._dirty_spaces.add(space.name)

    def get_unit(self, uid: Union[str, Uid]) -> Optional[MemoryUnit]:
        """Retrieve a unit by its UID.

        Args:
            uid: The UID (or string) to look up.

        Returns:
            The MemoryUnit, or None if not found.
        """
        return self._store.get_unit(Uid(str(uid)))

    def list_units(self) -> List[MemoryUnit]:
        """Return all units in the store.

        Returns:
            All stored MemoryUnit objects.
        """
        return self._store.list_units()

    def add_unit_to_space(self, uid: Union[str, Uid], space_name: Union[str, SpaceName]) -> None:
        """Assign an existing unit to a space.

        Args:
            uid: The unit's UID.
            space_name: The space to add the unit to.

        Raises:
            KeyError: If the unit is not found in the store.
        """
        u = Uid(str(uid))
        s = SpaceName(str(space_name))
        unit = self._store.get_unit(u)
        if unit is None:
            raise KeyError(f"unit not found: {u}")
        space = self.create_space(s)
        space.add_unit(u)
        self._store.upsert_spaces([space])
        self._dirty_spaces.add(s)

        spaces = set(unit.metadata.get("spaces", []))
        spaces.add(str(s))
        unit.metadata["spaces"] = sorted(spaces)
        unit.touch()
        self._store.upsert_units([unit])
        self._dirty_units.add(u)

    def attach_child_space(
        self,
        parent_space_name: Union[str, SpaceName],
        child_space_name: Union[str, SpaceName],
        *,
        ensure_exists: bool = True,
    ) -> None:
        """Register a child space under a parent space.

        Spaces are created automatically if ensure_exists is True and they
        don't already exist.

        Args:
            parent_space_name: The parent space.
            child_space_name: The child space to attach.
            ensure_exists: If True, create missing spaces automatically.

        Raises:
            KeyError: If a space is not found and ensure_exists is False.
        """
        parent_name = SpaceName(str(parent_space_name))
        child_name = SpaceName(str(child_space_name))

        parent = self._store.get_space(parent_name)
        child = self._store.get_space(child_name)

        if parent is None and ensure_exists:
            parent = self.create_space(parent_name)
        if child is None and ensure_exists:
            child = self.create_space(child_name)

        if parent is None:
            raise KeyError(f"parent space not found: {parent_name}")
        if child is None:
            raise KeyError(f"child space not found: {child_name}")

        if child_name in parent.child_spaces:
            return

        parent.add_child_space(child_name)
        self._store.upsert_spaces([parent])
        self._dirty_spaces.add(parent_name)

    def ensure_child_space(
        self,
        parent_space_name: Union[str, SpaceName],
        child_space_name: Union[str, SpaceName],
    ) -> MemorySpace:
        """Create a child space and attach it to a parent.

        Shortcut for create_space + attach_child_space.

        Args:
            parent_space_name: The parent space.
            child_space_name: The child space to create and attach.

        Returns:
            The newly created or existing child MemorySpace.
        """
        child = self.create_space(child_space_name)
        self.attach_child_space(parent_space_name, child.name, ensure_exists=True)
        return child

    def get_units_in_spaces(
        self,
        space_names: Sequence[Union[str, SpaceName]],
        *,
        mode: str = "union",
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """Get all units belonging to the specified spaces.

        Args:
            space_names: Sequence of space names.
            mode: \"union\" (default) or \"intersection\".
            recursive: If True, include units from child spaces recursively.

        Returns:
            List of MemoryUnits in the specified spaces.

        Raises:
            ValueError: If mode is not \"union\" or \"intersection\".
        """
        spaces = [SpaceName(str(s)) for s in space_names]
        if not spaces:
            return []

        def resolver(name: SpaceName) -> Optional[MemorySpace]:
            return self._store.get_space(name)

        uid_sets: List[set[Uid]] = []
        for s in spaces:
            sp = self._store.get_space(s)
            if sp is None:
                continue
            uid_sets.append(set(sp.get_all_unit_uids(recursive=recursive, resolver=resolver)))

        if not uid_sets:
            return []

        if mode not in {"union", "intersection"}:
            raise ValueError("mode must be 'union' or 'intersection'")

        if mode == "union":
            uids: set[Uid] = set().union(*uid_sets)
        else:
            uids = set.intersection(*uid_sets)

        return self._store.get_units(list(uids))

    def search_by_vector(
        self,
        query: Embedding,
        *,
        top_k: int = 10,
        space_names: Optional[Sequence[Union[str, SpaceName]]] = None,
        recursive: bool = True,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search for nearest neighbors to a query embedding.

        If space_names are provided, results are post-filtered to only include
        units belonging to those spaces.

        Args:
            query: The query embedding vector.
            top_k: Number of results to return.
            space_names: Optional space filter.
            recursive: If True, include child spaces in the filter.

        Returns:
            List of (MemoryUnit, similarity_score) tuples.
        """
        candidates: Optional[set[Uid]] = None
        if space_names:
            units = self.get_units_in_spaces(space_names, mode="union", recursive=recursive)
            candidates = {Uid(str(u.uid)) for u in units}

        hits = self._index.search(query, top_k=max(1, int(top_k)))
        out: List[Tuple[MemoryUnit, float]] = []
        for uid, score in hits:
            if candidates is not None and uid not in candidates:
                continue
            unit = self._store.get_unit(uid)
            if unit is None:
                continue
            out.append((unit, float(score)))
            if len(out) >= top_k:
                break
        return out

    def search_by_text(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        space_names: Optional[Sequence[Union[str, SpaceName]]] = None,
        recursive: bool = True,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search by text via embedding and vector similarity.

        Requires an embedder to be set. Text is embedded and then delegated
        to search_by_vector.

        Args:
            query_text: The text query to search for.
            top_k: Number of results to return.
            space_names: Optional space filter.
            recursive: If True, include child spaces in the filter.

        Returns:
            List of (MemoryUnit, similarity_score) tuples.

        Raises:
            RuntimeError: If no embedder is configured.
        """
        if self._embedder is None:
            raise RuntimeError("embedder is required for search_by_text")
        emb = self._embedder.embed_text([str(query_text)])[0]
        return self.search_by_vector(
            emb, top_k=top_k, space_names=space_names, recursive=recursive
        )

    def search_by_text_with_rerank(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        recall_k: Optional[int] = None,
        space_names: Optional[Sequence[Union[str, SpaceName]]] = None,
        recursive: bool = True,
        use_rerank: bool = True,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search by text with optional two-stage reranking.

        First stage: vector search with a larger recall_k (default: top_k * 5).
        Second stage: cross-encoder reranking on the recalled candidates.

        Args:
            query_text: The text query to search for.
            top_k: Number of final results to return.
            recall_k: Number of candidates to recall in the first stage.
            space_names: Optional space filter.
            recursive: If True, include child spaces in the filter.
            use_rerank: If True and a reranker is available, apply reranking.

        Returns:
            List of (MemoryUnit, similarity_score) tuples.
        """
        recall = int(recall_k) if recall_k is not None else max(int(top_k) * 5, int(top_k))
        recalled = self.search_by_text(
            query_text,
            top_k=recall,
            space_names=space_names,
            recursive=recursive,
        )
        if not use_rerank or self._reranker is None:
            return recalled[:top_k]

        units = [u for (u, _s) in recalled]
        reranked = self._reranker.rerank(query_text, units, top_k=top_k)
        return reranked

    def search_in_space(
        self,
        query_text: str,
        space_name: SpaceName,
        candidates: Optional[Set[Uid]] = None,
        *,
        top_k: int = 10,
        recall_k: Optional[int] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search within a specific space with optional candidate filtering.

        Prefers space-level index search (AdaptiveVectorIndex) for efficiency,
        falling back to full-index search with post-filtering.

        Args:
            query_text: The text query.
            space_name: The space to search within.
            candidates: Optional set of UIDs to restrict the search to.
            top_k: Number of results to return.
            recall_k: Number of candidates to recall before filtering.

        Returns:
            List of (MemoryUnit, similarity_score) tuples.

        Raises:
            RuntimeError: If no embedder is configured.
        """
        if self._embedder is None:
            raise RuntimeError("embedder is required for search_by_text")
        recall = int(recall_k) if recall_k is not None else max(int(top_k) * 5, int(top_k))
        emb = self._embedder.embed_text([str(query_text)])[0]

        if self._abi is not None:
            hits = self._abi.search_in_space(
                emb, space_name, candidates=candidates, top_k=recall
            )
        else:
            if candidates is not None:
                all_units = self._store.list_units()
                candidate_units = [u for u in all_units if Uid(str(u.uid)) in candidates]
                hits_with_score: List[Tuple[MemoryUnit, float]] = []
                for u in candidate_units:
                    if u.embedding is not None:
                        score = float(np.dot(
                            np.asarray(emb, dtype=np.float32).reshape(-1),
                            np.asarray(u.embedding, dtype=np.float32).reshape(-1),
                        ))
                        hits_with_score.append((u, score))
                hits_with_score.sort(key=lambda x: x[1], reverse=True)
                hits = [(Uid(str(u.uid)), s) for u, s in hits_with_score]
            else:
                hits = self._index.search(emb, top_k=recall)

        out: List[Tuple[MemoryUnit, float]] = []
        seen: Set[Uid] = set()
        for uid, score in hits:
            if uid in seen:
                continue
            seen.add(uid)
            unit = self._store.get_unit(uid)
            if unit is None:
                continue
            out.append((unit, float(score)))
            if len(out) >= top_k:
                break
        return out

    def rebuild_index_from_store(self) -> None:
        """Rebuild the entire vector index from all units in the store.

        Iterates over all stored units with valid embeddings and recreates
        the index from scratch. Useful after bulk operations or when the index
        becomes stale.
        """
        items: List[Tuple[Uid, Embedding]] = []
        for unit in self._store.list_units():
            if unit.embedding is None:
                continue
            uid = Uid(str(unit.uid))
            emb = np.asarray(unit.embedding, dtype=np.float32).reshape(-1)
            if emb.shape[0] != self._index.dim():
                continue
            items.append((uid, emb))
        self._index.rebuild(items)

    def flush(self) -> None:
        """Persist pending changes and clear dirty tracking."""
        self._store.flush()
        self._dirty_units.clear()
        self._dirty_spaces.clear()
        self._deleted_units.clear()

    def _embed_for_unit(
        self,
        unit: MemoryUnit,
        *,
        explicit_text: Optional[str],
        explicit_image_path: Optional[str],
    ) -> Optional[Embedding]:
        if self._embedder is None:
            return None

        text = explicit_text
        if text is None:
            raw_val = unit.raw_data.get(self._default_text_key)
            text = str(raw_val) if raw_val is not None else ""
        if isinstance(text, str) and text.strip():
            emb = self._embedder.embed_text([text])[0]
            return np.asarray(emb, dtype=np.float32).reshape(-1)

        image_path = explicit_image_path
        if image_path is None:
            raw_img = unit.raw_data.get(self._default_image_path_key)
            image_path = str(raw_img) if raw_img is not None else ""
        if isinstance(image_path, str) and image_path.strip():
            emb = self._embedder.embed_image_paths([image_path])[0]
            return np.asarray(emb, dtype=np.float32).reshape(-1)

        return None

    def _enforce_memory_limit(self) -> None:
        current = len(self._store.list_units())
        if current <= self._max_units_in_memory:
            return
        over = current - self._max_units_in_memory
        if over <= 0:
            return

        victims: List[Uid] = []
        for u in self._store.list_units():
            victims.append(Uid(str(u.uid)))
            if len(victims) >= over:
                break
        if victims:
            self._store.delete_units(victims)
            self._index.delete(victims)
