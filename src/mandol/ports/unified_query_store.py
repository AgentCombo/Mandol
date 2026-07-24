"""Backend contract for unified multimodal physical query execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..query.vector_seeded import (
    VectorSeededTraversalExecution,
    VectorSeededTraversalSpec,
)


class UnifiedQueryStore(ABC):
    """Execute backend-neutral cross-modal physical plans."""

    @abstractmethod
    def vector_seeded_graph_traversal(
        self,
        spec: VectorSeededTraversalSpec,
        *,
        profile: bool = False,
    ) -> VectorSeededTraversalExecution:
        """Run the query, optionally collecting backend operator timings."""
        raise NotImplementedError

    @abstractmethod
    def explain_vector_seeded_graph_traversal(self, spec: VectorSeededTraversalSpec) -> str:
        """Return a backend physical plan for the unified query."""
        raise NotImplementedError
