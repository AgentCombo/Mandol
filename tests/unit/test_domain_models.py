"""Unit tests for domain model serialization, MemorySpace recursion, and DocumentChunker."""

from __future__ import annotations

import numpy as np
import pytest

from mandol.domain.memory_space import MemorySpace
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.application.chunker import DocumentChunker, estimate_tokens, split_into_sentences


# ── MemoryUnit round-trip ───────────────────────────────────────────────

class TestMemoryUnitRoundTrip:
    def test_to_dict_and_from_dict_basic(self):
        unit = MemoryUnit(
            uid=Uid("test-1"),
            raw_data={"text_content": "Hello world"},
            metadata={"author": "alice"},
        )
        d = unit.to_dict()
        restored = MemoryUnit.from_dict(d)
        assert str(restored.uid) == "test-1"
        assert restored.raw_data == {"text_content": "Hello world"}
        assert restored.metadata.get("author") == "alice"

    def test_to_dict_and_from_dict_with_embedding(self):
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        unit = MemoryUnit(
            uid=Uid("test-2"),
            raw_data={"text_content": "With embedding"},
            embedding=emb,
        )
        d = unit.to_dict()
        restored = MemoryUnit.from_dict(d)
        assert restored.embedding is not None
        assert np.allclose(restored.embedding, emb)

    def test_to_dict_and_from_dict_with_sparse_embedding(self):
        sparse = np.array([0.5, 0.0, 0.8], dtype=np.float32)
        unit = MemoryUnit(
            uid=Uid("test-3"),
            raw_data={"text_content": "Sparse"},
            sparse_embedding=sparse,
        )
        d = unit.to_dict()
        restored = MemoryUnit.from_dict(d)
        assert restored.sparse_embedding is not None
        assert np.allclose(restored.sparse_embedding, sparse)

    def test_from_dict_preserves_system_timestamps(self):
        unit = MemoryUnit(
            uid=Uid("test-4"),
            raw_data={"text_content": "Timestamps"},
        )
        created = unit.metadata["_system_created_at"]
        updated = unit.metadata["_system_updated_at"]

        d = unit.to_dict()
        restored = MemoryUnit.from_dict(d)
        assert restored.metadata["_system_created_at"] == created
        assert restored.metadata["_system_updated_at"] == updated

    def test_from_dict_no_embeddings(self):
        d = {"uid": "bare", "raw_data": {"text_content": "minimal"}, "metadata": {}}
        restored = MemoryUnit.from_dict(d)
        assert str(restored.uid) == "bare"
        assert restored.embedding is None
        assert restored.sparse_embedding is None

    def test_get_user_metadata_excludes_system_keys(self):
        unit = MemoryUnit(
            uid=Uid("test-5"),
            raw_data={"text_content": "Meta"},
            metadata={"custom_field": 42, "session_id": "s1"},
        )
        user_meta = unit.get_user_metadata()
        assert "custom_field" in user_meta
        assert "session_id" in user_meta
        assert "_system_created_at" not in user_meta
        assert "_system_updated_at" not in user_meta

    def test_touch_updates_timestamp(self):
        unit = MemoryUnit(
            uid=Uid("test-6"),
            raw_data={"text_content": "Touch test"},
        )
        old_updated = unit.metadata["_system_updated_at"]
        import time
        time.sleep(0.01)
        unit.touch()
        new_updated = unit.metadata["_system_updated_at"]
        assert new_updated != old_updated

    def test_eq_same_unit(self):
        u1 = MemoryUnit(uid=Uid("a"), raw_data={"text_content": "same"})
        d = u1.to_dict()
        u2 = MemoryUnit.from_dict(d)
        assert u1 == u2

    def test_eq_different_uid(self):
        u1 = MemoryUnit(uid=Uid("a"), raw_data={"text_content": "same"})
        u2 = MemoryUnit(uid=Uid("b"), raw_data={"text_content": "same"})
        assert u1 != u2

    def test_uid_must_be_non_empty(self):
        with pytest.raises(ValueError):
            MemoryUnit(uid=Uid(""), raw_data={})

    def test_raw_data_must_be_dict(self):
        with pytest.raises(ValueError):
            MemoryUnit(uid=Uid("x"), raw_data=None)  # type: ignore


# ── MemorySpace recursion ───────────────────────────────────────────────

class TestMemorySpaceRecursion:
    def test_get_all_unit_uids_flat(self):
        space = MemorySpace(name=SpaceName("root"))
        space.add_unit("u1")
        space.add_unit("u2")
        uids = space.get_all_unit_uids(recursive=False)
        assert uids == {Uid("u1"), Uid("u2")}

    def test_get_all_unit_uids_recursive(self):
        child = MemorySpace(name=SpaceName("child"))
        child.add_unit("u_child")

        parent = MemorySpace(name=SpaceName("parent"))
        parent.add_unit("u_parent")
        parent.add_child_space("child")

        def resolver(name):
            if name == SpaceName("child"):
                return child
            return None

        uids = parent.get_all_unit_uids(recursive=True, resolver=resolver)
        assert Uid("u_parent") in uids
        assert Uid("u_child") in uids
        assert len(uids) == 2

    def test_get_all_unit_uids_recursive_requires_resolver(self):
        space = MemorySpace(name=SpaceName("s"))
        space.add_child_space("child")
        with pytest.raises(ValueError, match="resolver is required"):
            space.get_all_unit_uids(recursive=True)

    def test_get_all_child_space_names_recursive(self):
        grandchild = MemorySpace(name=SpaceName("grandchild"))
        child = MemorySpace(name=SpaceName("child"))
        child.add_child_space("grandchild")
        parent = MemorySpace(name=SpaceName("parent"))
        parent.add_child_space("child")

        def resolver(name):
            if name == SpaceName("child"):
                return child
            if name == SpaceName("grandchild"):
                return grandchild
            return None

        names = parent.get_all_child_space_names(recursive=True, resolver=resolver)
        assert SpaceName("child") in names
        assert SpaceName("grandchild") in names

    def test_remove_unit_noop_if_not_present(self):
        space = MemorySpace(name=SpaceName("s"))
        space.remove_unit("nonexistent")  # should not raise
        assert len(space.unit_uids) == 0

    def test_remove_child_space_noop_if_not_present(self):
        space = MemorySpace(name=SpaceName("s"))
        space.remove_child_space("nonexistent")  # should not raise
        assert len(space.child_spaces) == 0

    def test_to_dict_and_from_dict(self):
        space = MemorySpace(name=SpaceName("s1"))
        space.add_unit("u1")
        space.add_child_space("child")
        space.set_summary("summary text")

        d = space.to_dict()
        restored = MemorySpace.from_dict(d)
        assert restored.name == SpaceName("s1")
        assert Uid("u1") in restored.unit_uids
        assert SpaceName("child") in restored.child_spaces
        assert restored.summary_text == "summary text"

    def test_name_must_be_non_empty(self):
        with pytest.raises(ValueError):
            MemorySpace(name=SpaceName(""))


# ── DocumentChunker ─────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_english_text(self):
        # "hello world" → 10 alpha chars × 0.3 ≈ 3, + 1 space × 0.4 ≈ 3
        assert estimate_tokens("hello world") > 0

    def test_chinese_text(self):
        # 4 Chinese chars × 0.6 ≈ 2
        assert estimate_tokens("你好世界") > 0


class TestSplitIntoSentences:
    def test_single_sentence(self):
        result = split_into_sentences("Hello world.")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_multiple_sentences(self):
        result = split_into_sentences("Hello. How are you? I'm fine!")
        assert len(result) == 3
        assert result[0] == "Hello."
        assert result[1] == "How are you?"
        assert result[2] == "I'm fine!"

    def test_chinese_punctuation(self):
        result = split_into_sentences("你好！最近怎么样？")
        assert len(result) == 2
        assert result[0] == "你好！"
        assert result[1] == "最近怎么样？"

    def test_empty_string(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []

    def test_no_punctuation(self):
        result = split_into_sentences("no punctuation here")
        assert len(result) == 1


class TestDocumentChunker:
    def test_should_chunk_false_for_short_text(self):
        chunker = DocumentChunker(max_tokens=512)
        unit = MemoryUnit(
            uid=Uid("u1"),
            raw_data={"text_content": "A short sentence."},
        )
        assert not chunker.should_chunk(unit)

    def test_should_chunk_true_for_long_text(self):
        chunker = DocumentChunker(max_tokens=5)
        unit = MemoryUnit(
            uid=Uid("u1"),
            raw_data={"text_content": "This is a very long sentence that should be chunked."},
        )
        assert chunker.should_chunk(unit)

    def test_should_chunk_false_for_empty_text(self):
        chunker = DocumentChunker(max_tokens=5)
        unit = MemoryUnit(uid=Uid("u1"), raw_data={"text_content": ""})
        assert not chunker.should_chunk(unit)

    def test_chunk_unit_returns_chunks(self):
        chunker = DocumentChunker(max_tokens=20)
        unit = MemoryUnit(
            uid=Uid("u1"),
            raw_data={"text_content": "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."},
        )
        result = chunker.chunk_unit(unit)
        assert len(result.chunks) >= 1
        # Each chunk should be a MemoryUnit with chunk type metadata
        for chunk in result.chunks:
            assert chunk.metadata.get("type") == "chunk"
            assert chunk.metadata.get("parent_uid") == "u1"

    def test_chunk_unit_empty_text(self):
        chunker = DocumentChunker(max_tokens=512)
        unit = MemoryUnit(uid=Uid("u1"), raw_data={})
        result = chunker.chunk_unit(unit)
        assert result.chunks == []

    def test_chunk_unit_single_sentence_within_limit(self):
        chunker = DocumentChunker(max_tokens=512)
        unit = MemoryUnit(
            uid=Uid("u1"),
            raw_data={"text_content": "Just one sentence."},
        )
        result = chunker.chunk_unit(unit)
        assert len(result.chunks) == 1
        assert result.chunks[0].raw_data.get("text_content") == "Just one sentence."

    def test_get_text_fallback_to_any_string_value(self):
        chunker = DocumentChunker(max_tokens=512, text_key="text_content")
        unit = MemoryUnit(
            uid=Uid("u1"),
            raw_data={"unusual_key": "fallback text"},
        )
        assert chunker.should_chunk(unit) is False  # short text, but shouldn't crash
        result = chunker.chunk_unit(unit)
        assert len(result.chunks) == 1
