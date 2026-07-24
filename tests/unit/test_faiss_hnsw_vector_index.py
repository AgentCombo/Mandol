from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")

from mandol.domain.types import Uid
from mandol.infrastructure.faiss_hnsw_vector_index import FaissHNSWVectorIndex


def test_hnsw_search_update_delete_and_rebuild():
    index = FaissHNSWVectorIndex(3, m=8, ef_construction=40, ef_search=32)
    index.upsert(
        [
            (Uid("x"), np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
            (Uid("y"), np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
            (Uid("z"), np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
        ]
    )

    assert index.search(np.asarray([1.0, 0.0, 0.0]), 1)[0][0] == Uid("x")

    index.upsert([(Uid("x"), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))])
    assert {uid for uid, _ in index.search(np.asarray([0.0, 1.0, 0.0]), 2)} == {
        Uid("x"),
        Uid("y"),
    }

    index.delete([Uid("y")])
    assert Uid("y") not in {uid for uid, _ in index.search(np.ones(3), 3)}

    index.rebuild([(Uid("only"), np.asarray([1.0, 1.0, 0.0]))])
    assert index.search(np.asarray([1.0, 1.0, 0.0]), 2)[0][0] == Uid("only")


def test_hnsw_candidate_filter_adaptively_widens():
    index = FaissHNSWVectorIndex(2, m=8, ef_search=32, filter_oversample=1)
    items = [
        (Uid(f"u{i}"), np.asarray([1.0, i / 100.0], dtype=np.float32))
        for i in range(20)
    ]
    index.rebuild(items)

    hits = index.search_in_space(
        np.asarray([1.0, 0.0], dtype=np.float32),
        "unused",
        candidates={Uid("u18"), Uid("u19")},
        top_k=2,
    )

    assert {uid for uid, _ in hits} == {Uid("u18"), Uid("u19")}


def test_hnsw_validates_dimension_and_search_parameter():
    index = FaissHNSWVectorIndex(2)
    with pytest.raises(ValueError, match="dim mismatch"):
        index.upsert([(Uid("bad"), np.ones(3, dtype=np.float32))])
    with pytest.raises(ValueError, match="ef_search"):
        index.ef_search = 0
