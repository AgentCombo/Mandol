"""Core domain types for the Mandol memory system.

Defines the foundational type aliases (Uid, SpaceName, Embedding) and
the SearchResult dataclass used throughout all layers of the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NewType, Optional

import numpy as np

Uid = NewType("Uid", str)
SpaceName = NewType("SpaceName", str)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single ranked result from a retrieval operation.

    Attributes:
        uid: Unique identifier of the matched unit.
        score: Relevance or similarity score (higher is better).
        metadata: Optional metadata payload attached to this result.
    """

    uid: Uid
    score: float
    metadata: Optional[Dict[str, Any]] = None


Embedding = np.ndarray
