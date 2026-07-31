"""Regression tests for RocksDB-backed automatic payload paging."""

from __future__ import annotations

import inspect
from pathlib import Path
import threading
import time

import numpy as np
import orjson
import pytest

from mandol.core.memory_unit import MemoryUnit
from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.retrieval.bm25_retriever import BM25Retriever
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.retrieval.splade_retriever import SPLADERetriever
from mandol.storage import RocksDBPayloadStore
from mandol.utils.model_manager import global_model_manager


class _DummyEmbeddingModel:
    def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _DummySpladeModel:
    def encode_query(self, text):
        del text
        return {7: 1.0}


class _DummyMultiRetriever:
    retrievers = {}

    def build_all_indexes(self, **kwargs):
        del kwargs
        return {
            "total_retrievers": 0,
            "built_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "total_duration": 0.0,
            "details": {},
        }


@pytest.fixture
def semantic_map_factory(monkeypatch):
    original_loader = global_model_manager.get_or_load_model

    def load_model(**kwargs):
        if kwargs.get("model_type") == "text_embedding":
            return _DummyEmbeddingModel()
        return original_loader(**kwargs)

    monkeypatch.setattr(global_model_manager, "get_or_load_model", load_model)
    monkeypatch.setattr(
        global_model_manager,
        "get_splade_model",
        lambda *args, **kwargs: _DummySpladeModel(),
    )
    monkeypatch.setattr(
        SemanticMap,
        "_incremental_aux_retriever_add",
        lambda self, units: None,
    )

    def create() -> SemanticMap:
        return SemanticMap(
            embedding_model_name="test/dummy",
            embedding_dim=2,
            use_flash_attention=False,
        )

    return create


def _unit(uid: str, *, sparse: bool = False) -> MemoryUnit:
    unit = MemoryUnit(
        uid,
        {"text_content": f"payload for {uid}"},
        {"source": "storage-test"},
    )
    if sparse:
        unit.sparse_embedding = {7: 1.0}
    return unit


def _add_unit(semantic_map: SemanticMap, uid: str, *, sparse: bool = False) -> None:
    semantic_map.add_unit(
        _unit(uid, sparse=sparse),
        explicit_content_for_embedding=f"payload for {uid}",
        space_names=["test-space"],
        generate_sparse_embedding=False,
    )


def _close_persistent_map(semantic_map: SemanticMap) -> None:
    semantic_map._close_tiered_storage()


def _wait_for_eviction(semantic_map: SemanticMap):
    manager = semantic_map.tiered_storage_manager
    future = manager._eviction_future
    assert future is not None
    manager.wait_for_idle()
    return future.result(timeout=10)


def _evict_now(semantic_map: SemanticMap, count: int = 1):
    manager = semantic_map.tiered_storage_manager
    assert manager is not None
    result = manager._evict_once(len(semantic_map.memory_units), count=count)
    assert result.error is None
    return result


def test_rocksdb_crud_and_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "payloads.rocksdb"
    store = RocksDBPayloadStore(str(db_path))

    first = _unit("first")
    second = _unit("second")
    assert store.add_unit(first)
    assert store.add_units_batch([second]) == 1
    assert store.get_unit("first") == first
    assert [unit.uid for unit in store.get_units_batch(["second", "first"])] == [
        "second",
        "first",
    ]
    assert set(store.list_uids()) == {"first", "second"}

    store.close()
    reopened = RocksDBPayloadStore(str(db_path))
    assert reopened.get_unit("first").raw_data == first.raw_data
    assert reopened.delete_unit("first")
    assert reopened.get_unit("first") is None
    reopened.close()


def test_default_resident_operation_and_retrievers(semantic_map_factory) -> None:
    semantic_map = semantic_map_factory()
    _add_unit(semantic_map, "dense")
    semantic_map.batch_add_units(
        [_unit("lexical", sparse=True)],
        explicit_contents_for_embedding=["payload lexical evidence"],
        per_unit_space_names=[["test-space"]],
        generate_sparse_embedding=False,
    )

    assert semantic_map.tiered_storage_manager is None
    assert semantic_map._external_storage is None
    assert semantic_map.get_unit("dense").uid == "dense"
    assert semantic_map.search_similarity_by_vector(
        np.array([1.0, 0.0], dtype=np.float32),
        k=2,
    )

    bm25 = BM25Retriever(semantic_map)
    bm25.build_index()
    assert bm25.search("lexical", top_k=1)[0].unit.uid == "lexical"

    splade = SPLADERetriever(semantic_map)
    splade.build_index()
    assert splade.search("lexical", top_k=1)[0].unit.uid == "lexical"


def test_connect_to_l2_rejects_removed_keyword(tmp_path: Path, semantic_map_factory) -> None:
    graph = SemanticGraph(semantic_map_factory())
    assert graph.connect_to_l2(str(tmp_path / "active"))

    removed_keyword = "storage" + "_mode"
    assert removed_keyword not in inspect.signature(graph.connect_to_l2).parameters
    for removed_value in ("memory", "tiered_cache", "store_only"):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            graph.connect_to_l2(
                str(tmp_path / removed_value),
                **{removed_keyword: removed_value},
            )

    graph.close()


def test_below_high_watermark_keeps_payloads_resident(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    graph = SemanticGraph(semantic_map_factory())
    assert graph.connect_to_l2(
        str(tmp_path / "runtime"),
        max_capacity=10,
        high_watermark=0.8,
        low_watermark=0.5,
    )
    for uid in ("first", "second"):
        graph.add_unit(
            _unit(uid),
            explicit_content_for_embedding=f"payload for {uid}",
            generate_sparse_embedding=False,
        )

    assert set(graph.semantic_map.memory_units) == {"first", "second"}
    assert graph._payload_store.count_units() == 0
    assert graph.tiered_storage_manager._eviction_future is None
    graph.close()


def test_connect_to_l2_checks_existing_resident_payloads(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    graph = SemanticGraph(semantic_map_factory())
    for uid in ("first", "second", "third"):
        graph.add_unit(
            _unit(uid),
            explicit_content_for_embedding=f"payload for {uid}",
            generate_sparse_embedding=False,
        )

    assert graph.connect_to_l2(
        str(tmp_path / "runtime"),
        max_capacity=4,
        high_watermark=0.75,
        low_watermark=0.50,
    )
    graph.tiered_storage_manager.wait_for_idle()
    assert len(graph.semantic_map.memory_units) == 2
    assert graph._payload_store.count_units() == 1
    graph.close()


def test_automatic_eviction_reaches_low_watermark(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    semantic_map = semantic_map_factory()
    store = RocksDBPayloadStore(str(tmp_path / "payloads.rocksdb"))
    semantic_map.enable_tiered_storage(
        store,
        max_capacity=4,
        high_watermark=0.75,
        low_watermark=0.50,
    )

    for uid in ("first", "second", "third"):
        _add_unit(semantic_map, uid)

    result = _wait_for_eviction(semantic_map)
    assert result.persisted_count == 1
    assert result.removed_count == 1
    assert len(semantic_map.memory_units) == 2
    cold_uid = result.selected_uids[0]
    assert store.get_unit(cold_uid) is not None
    assert cold_uid not in semantic_map.memory_units

    _close_persistent_map(semantic_map)


def test_direct_map_save_fails_closed_when_tiered(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    semantic_map = semantic_map_factory()
    store = RocksDBPayloadStore(str(tmp_path / "payloads.rocksdb"))
    semantic_map.enable_tiered_storage(
        store,
        max_capacity=10,
        high_watermark=9,
        low_watermark=5,
    )

    with pytest.raises(RuntimeError, match="SemanticGraph.save_graph"):
        semantic_map.save_map(str(tmp_path / "incomplete"))

    _close_persistent_map(semantic_map)


def test_repeated_connection_is_idempotent_and_reconfiguration_fails(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    graph = SemanticGraph(semantic_map_factory())
    runtime_path = tmp_path / "runtime"
    assert graph.connect_to_l2(
        str(runtime_path),
        max_capacity=4,
        high_watermark=0.75,
        low_watermark=0.50,
    )
    for uid in ("first", "second", "third"):
        graph.add_unit(
            _unit(uid),
            explicit_content_for_embedding=f"payload for {uid}",
            generate_sparse_embedding=False,
        )
    graph.tiered_storage_manager.wait_for_idle()
    manager = graph.tiered_storage_manager
    cold_uids = set(graph._payload_store.list_uids())
    assert cold_uids

    assert graph.connect_to_l2(
        str(runtime_path / ".." / "runtime"),
        max_capacity=4,
        high_watermark=0.75,
        low_watermark=0.50,
    )
    assert graph.tiered_storage_manager is manager
    assert cold_uids.isdisjoint(graph.semantic_map.memory_units)

    with pytest.raises(RuntimeError, match="already connected"):
        graph.connect_to_l2(
            str(tmp_path / "other"),
            max_capacity=4,
            high_watermark=0.75,
            low_watermark=0.50,
        )
    assert graph.tiered_storage_manager is manager
    assert all(graph.get_unit(uid) is not None for uid in cold_uids)
    graph.close()


def test_failed_initial_connection_preserves_resident_graph(
    tmp_path: Path,
    semantic_map_factory,
    monkeypatch,
) -> None:
    import mandol.storage.rocksdb_payload_store as payload_store_module

    graph = SemanticGraph(semantic_map_factory())
    graph.add_unit(
        _unit("resident"),
        explicit_content_for_embedding="resident payload",
        generate_sparse_embedding=False,
    )

    class FailingPayloadStore:
        def __init__(self, db_path):
            del db_path
            raise OSError("simulated open failure")

    monkeypatch.setattr(
        payload_store_module,
        "RocksDBPayloadStore",
        FailingPayloadStore,
    )
    assert not graph.connect_to_l2(str(tmp_path / "unavailable"))
    assert graph.tiered_storage_manager is None
    assert graph.semantic_map.tiered_storage_manager is None
    assert graph._payload_store is None
    assert graph.get_unit("resident") is not None


def test_close_waits_for_eviction_and_is_idempotent(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    graph = SemanticGraph(semantic_map_factory())
    assert graph.connect_to_l2(
        str(tmp_path / "runtime"),
        max_capacity=4,
        high_watermark=0.75,
        low_watermark=0.50,
    )
    store = graph._payload_store
    original_swap_out = store.swap_out
    started = threading.Event()
    release = threading.Event()

    def delayed_swap_out(uids, l1_data):
        started.set()
        assert release.wait(timeout=10)
        return original_swap_out(uids, l1_data)

    store.swap_out = delayed_swap_out
    for uid in ("first", "second", "third"):
        graph.add_unit(
            _unit(uid),
            explicit_content_for_embedding=f"payload for {uid}",
            generate_sparse_embedding=False,
        )
    assert started.wait(timeout=10)

    releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
    releaser.start()
    graph.close()
    releaser.join(timeout=10)

    assert not store.is_connected
    assert graph.tiered_storage_manager is None
    assert graph.semantic_map.tiered_storage_manager is None
    graph.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.get_unit("first")


def test_tiered_cache_keeps_indexes_and_pages_in_ghost_node(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    semantic_map = semantic_map_factory()
    graph = SemanticGraph(semantic_map)
    store = RocksDBPayloadStore(str(tmp_path / "payloads.rocksdb"))
    graph.enable_tiered_storage(
        store,
        max_capacity=10,
        high_watermark=9,
        low_watermark=5,
    )

    graph.add_unit(
        _unit("cold", sparse=True),
        explicit_content_for_embedding="cold payload",
        space_names=["test-space"],
        generate_sparse_embedding=False,
    )
    bm25 = BM25Retriever(semantic_map)
    bm25.build_index()
    splade = SPLADERetriever(semantic_map)
    splade.build_index()
    bm25_postings = dict(bm25.doc_lengths)
    splade_postings = {
        token: dict(postings) for token, postings in splade.inverted_index.items()
    }

    def unexpected_index_mutation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Payload paging must not mutate resident indexes.")

    semantic_map._remove_aux_retriever_uids = unexpected_index_mutation
    semantic_map._incremental_aux_retriever_add = unexpected_index_mutation
    semantic_map._incremental_faiss_add_many = unexpected_index_mutation
    int_id = semantic_map._uid_to_int_id["cold"]
    index_size = semantic_map.faiss_index.ntotal
    assert _evict_now(semantic_map).removed_count == 1
    assert "cold" not in semantic_map.memory_units
    assert semantic_map.faiss_index.ntotal == index_size
    assert semantic_map._uid_to_int_id["cold"] == int_id
    assert semantic_map.memory_spaces["test-space"].contains_unit("cold")
    assert graph.get_node_data("cold")["ghost"] is True
    assert bm25.doc_lengths == bm25_postings
    assert splade.inverted_index == splade_postings

    results = semantic_map.search_similarity_by_vector(
        np.array([1.0, 0.0], dtype=np.float32),
        k=1,
    )
    assert results[0][0].uid == "cold"
    assert "cold" in semantic_map.memory_units
    assert graph.get_node_data("cold")["ghost"] is False
    assert semantic_map.faiss_index.ntotal == index_size
    assert bm25.doc_lengths == bm25_postings
    assert splade.inverted_index == splade_postings

    graph.close()


def test_cold_payload_update_and_delete(
    tmp_path: Path,
    semantic_map_factory,
) -> None:
    semantic_map = semantic_map_factory()
    graph = SemanticGraph(semantic_map)
    store = RocksDBPayloadStore(str(tmp_path / "payloads.rocksdb"))
    graph.enable_tiered_storage(
        store,
        max_capacity=10,
        high_watermark=9,
        low_watermark=5,
    )
    graph.add_unit(
        _unit("mutable"),
        explicit_content_for_embedding="original payload",
        space_names=["test-space"],
        generate_sparse_embedding=False,
    )
    assert _evict_now(semantic_map).removed_count == 1
    assert "mutable" not in semantic_map.memory_units

    updated = MemoryUnit(
        "mutable",
        {"text_content": "updated payload"},
        {"source": "storage-test"},
    )
    graph.add_unit(
        updated,
        explicit_content_for_embedding="updated payload",
        space_names=["test-space"],
        generate_sparse_embedding=False,
    )
    assert semantic_map.get_unit("mutable").raw_data["text_content"] == "updated payload"
    assert graph.get_node_data("mutable").get("ghost") is not True

    assert _evict_now(semantic_map).removed_count == 1
    assert store.get_unit("mutable").raw_data["text_content"] == "updated payload"
    graph.delete_unit("mutable")
    assert store.get_unit("mutable") is None
    assert semantic_map.get_unit("mutable") is None
    assert not semantic_map.memory_spaces["test-space"].contains_unit("mutable")
    assert graph.get_node_data("mutable") is None

    graph.close()


@pytest.mark.parametrize("enable_paging", [False, True])
def test_search_result_format_is_stable(
    tmp_path: Path,
    semantic_map_factory,
    enable_paging: bool,
) -> None:
    semantic_map = semantic_map_factory()
    if enable_paging:
        store = RocksDBPayloadStore(str(tmp_path / "payloads.rocksdb"))
        semantic_map.enable_tiered_storage(
            store,
            max_capacity=10,
            high_watermark=9,
            low_watermark=5,
        )

    _add_unit(semantic_map, "result")
    if enable_paging:
        assert _evict_now(semantic_map).selected_uids == ["result"]

    response = semantic_map.get_multi_retriever().smart_search(
        query="payload",
        methods=[RetrievalMethod.COSINE_SIMILARITY],
        top_k=1,
        rerank_method=None,
        return_detailed=True,
    )
    assert set(response) == {
        "results",
        "execution_plan",
        "statistics",
        "method_stats",
        "config",
    }
    assert response["statistics"]["final_results_count"] == 1
    unit, score = response["results"][0]
    assert isinstance(unit, MemoryUnit)
    assert unit.uid == "result"
    assert isinstance(score, float)
    assert "result" in semantic_map.memory_units

    _close_persistent_map(semantic_map)


def test_resident_graph_checkpoint_round_trip(
    tmp_path: Path,
    semantic_map_factory,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SemanticGraph,
        "get_multi_retriever",
        lambda self: _DummyMultiRetriever(),
    )
    graph = SemanticGraph(semantic_map_factory())
    graph.add_unit(
        _unit("resident"),
        explicit_content_for_embedding="resident payload",
        generate_sparse_embedding=False,
    )

    checkpoint_dir = tmp_path / "resident-checkpoint"
    graph.save_graph(str(checkpoint_dir), build_sparse_vectors=False)
    restored = SemanticGraph.load_graph(str(checkpoint_dir))

    assert restored.semantic_map.tiered_storage_manager is None
    assert restored.get_unit("resident").raw_data["text_content"] == "payload for resident"


def test_graph_snapshot_waits_for_submitted_eviction(
    tmp_path: Path,
    semantic_map_factory,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SemanticGraph,
        "get_multi_retriever",
        lambda self: _DummyMultiRetriever(),
    )
    graph = SemanticGraph(semantic_map_factory())
    assert graph.connect_to_l2(
        str(tmp_path / "runtime"),
        max_capacity=4,
        high_watermark=0.75,
        low_watermark=0.50,
    )
    store = graph._payload_store
    original_swap_out = store.swap_out
    started = threading.Event()
    release = threading.Event()

    def delayed_swap_out(uids, l1_data):
        started.set()
        assert release.wait(timeout=10)
        return original_swap_out(uids, l1_data)

    store.swap_out = delayed_swap_out
    known_uids = {"first", "second", "third"}
    for uid in known_uids:
        graph.add_unit(
            _unit(uid),
            explicit_content_for_embedding=f"payload for {uid}",
            generate_sparse_embedding=False,
        )
    assert started.wait(timeout=10)

    def release_eviction() -> None:
        time.sleep(0.05)
        release.set()

    releaser = threading.Thread(target=release_eviction)
    releaser.start()
    checkpoint_dir = tmp_path / "tiered-checkpoint"
    graph.save_graph(str(checkpoint_dir), build_sparse_vectors=False)
    releaser.join(timeout=10)

    restored = SemanticGraph.load_graph(str(checkpoint_dir))
    assert all(restored.get_unit(uid) is not None for uid in known_uids)
    assert restored.semantic_map._all_known_uids() == known_uids
    restored.close()
    graph.close()


def test_tiered_checkpoint_restores_paging_and_graph_state(
    tmp_path: Path,
    semantic_map_factory,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SemanticGraph,
        "get_multi_retriever",
        lambda self: _DummyMultiRetriever(),
    )
    semantic_map = semantic_map_factory()
    graph = SemanticGraph(semantic_map)
    assert graph.connect_to_l2(
        str(tmp_path / "runtime"),
        max_capacity=10,
        high_watermark=9,
        low_watermark=5,
    )
    for uid in ("checkpoint", "checkpoint-target"):
        graph.add_unit(
            _unit(uid),
            explicit_content_for_embedding=f"{uid} payload",
            space_names=["test-space"],
            generate_sparse_embedding=False,
        )
    assert graph.add_relationship("checkpoint", "checkpoint-target", "RELATED_TO")
    assert _evict_now(semantic_map).removed_count == 1
    cold_uid = next(uid for uid in ("checkpoint", "checkpoint-target") if uid not in semantic_map.memory_units)

    checkpoint_dir = tmp_path / "tiered-checkpoint"
    graph.save_graph(str(checkpoint_dir), build_sparse_vectors=False)
    checkpoint_state = orjson.loads((checkpoint_dir / "graph_state.json").read_bytes())
    l2_state = checkpoint_state["l2_storage"]
    assert l2_state["backend"] == "rocksdb"
    assert l2_state["rocksdb_relative_path"] == "l2_database/payloads.rocksdb"
    assert l2_state["max_capacity"] == 10
    assert l2_state["high_watermark"] == 9
    assert l2_state["low_watermark"] == 5
    assert "mode" not in l2_state
    restored = SemanticGraph.load_graph(str(checkpoint_dir))

    assert restored.semantic_map.tiered_storage_manager is not None
    assert restored.semantic_map.tiered_storage_manager.max_capacity == 10
    assert restored.semantic_map.tiered_storage_manager.high_watermark == 9
    assert restored.semantic_map.tiered_storage_manager.low_watermark == 5
    assert restored.semantic_map.faiss_index.ntotal == 2
    assert restored.semantic_map.memory_spaces["test-space"].contains_unit(cold_uid)
    assert restored.get_node_data(cold_uid) is not None
    loaded = restored.get_unit(cold_uid)
    assert loaded is not None
    assert loaded.uid == cold_uid
    assert cold_uid in restored.semantic_map.memory_units
    assert restored.get_node_data(cold_uid)["ghost"] is False
    assert restored.get_relationship(
        "checkpoint",
        "checkpoint-target",
        "RELATED_TO",
    ) == {}

    restored.close()
    graph.close()


def test_legacy_tiered_checkpoint_metadata_remains_loadable(
    tmp_path: Path,
    semantic_map_factory,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SemanticGraph,
        "get_multi_retriever",
        lambda self: _DummyMultiRetriever(),
    )
    graph = SemanticGraph(semantic_map_factory())
    assert graph.connect_to_l2(
        str(tmp_path / "runtime"),
        max_capacity=10,
        high_watermark=9,
        low_watermark=5,
    )
    graph.add_unit(
        _unit("legacy-tiered"),
        explicit_content_for_embedding="legacy tiered payload",
        generate_sparse_embedding=False,
    )
    assert _evict_now(graph.semantic_map).removed_count == 1
    checkpoint_dir = tmp_path / "legacy-tiered-checkpoint"
    graph.save_graph(str(checkpoint_dir), build_sparse_vectors=False)

    state_path = checkpoint_dir / "graph_state.json"
    state = orjson.loads(state_path.read_bytes())
    state["l2_storage"].pop("enabled", None)
    state["l2_storage"].pop("backend", None)
    state["l2_storage"]["mode"] = "tiered_cache"
    state_path.write_bytes(orjson.dumps(state, option=orjson.OPT_INDENT_2))

    restored = SemanticGraph.load_graph(str(checkpoint_dir))
    assert restored.tiered_storage_manager is not None
    assert restored.get_unit("legacy-tiered") is not None
    restored.close()
    graph.close()


def test_removed_placement_checkpoint_fails_closed(
    tmp_path: Path,
    semantic_map_factory,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SemanticGraph,
        "get_multi_retriever",
        lambda self: _DummyMultiRetriever(),
    )
    graph = SemanticGraph(semantic_map_factory())
    graph.add_unit(
        _unit("legacy"),
        explicit_content_for_embedding="legacy payload",
        generate_sparse_embedding=False,
    )
    checkpoint_dir = tmp_path / "legacy-checkpoint"
    graph.save_graph(str(checkpoint_dir), build_sparse_vectors=False)

    state_path = checkpoint_dir / "graph_state.json"
    state = orjson.loads(state_path.read_bytes())
    state["l2_storage"] = {"mode": "store_only"}
    state_path.write_bytes(orjson.dumps(state, option=orjson.OPT_INDENT_2))

    with pytest.raises(ValueError, match="removed store_only"):
        SemanticGraph.load_graph(str(checkpoint_dir))
