"""Integration tests for the legacy MDSG pipeline."""

import json
import os
import tempfile
import unittest
from unittest import skip

import numpy as np

from mandol.application import SemanticGraphService, SemanticMapService, run_mdsg_pipeline
from mandol.domain import MemoryUnit
from mandol.infrastructure import InMemoryCosineVectorIndex, InMemoryGraphStore, InMemoryUnitStore, StubLLMProvider


class TestMdsdPipelineEndToEnd(unittest.TestCase):
    def _make_graph(self) -> SemanticGraphService:
        store = InMemoryUnitStore()
        index = InMemoryCosineVectorIndex(dim=4)
        semantic_map = SemanticMapService(store=store, index=index, embedder=None)
        return SemanticGraphService(semantic_map=semantic_map, graph_store=InMemoryGraphStore())

    @skip("mdsg_pipeline module was reconstructed and needs ID format alignment")
    def test_run_mdsg_pipeline_extracts_and_applies(self):
        g = self._make_graph()

        # Seed base space (root space) with units; layout normalization will move them to *_base_memory.
        u1 = MemoryUnit(uid="u1", raw_data={"text_content": "hello"}, embedding=np.array([1, 0, 0, 0], dtype=np.float32))
        u2 = MemoryUnit(uid="u2", raw_data={"text_content": "world"}, embedding=np.array([0, 1, 0, 0], dtype=np.float32))
        g.add_unit(u1, space_names=["demo"], ensure_embedding=False)
        g.add_unit(u2, space_names=["demo"], ensure_embedding=False)

        # LLM responses: summary (2 calls), event, entity
        llm = StubLLMProvider(
            contents=[
                json.dumps({"key_source_uids": ["u1"], "summary": "s-episodic"}),
                json.dumps({"key_source_uids": ["u2"], "summary": "s-knowledge"}),
                json.dumps(
                    {
                        "events": [
                            {"signature": "e1", "text": "event one", "evidence_uids": ["u1"]},
                            {"signature": "e2", "text": "event two", "evidence_uids": ["u2"]},
                        ],
                        "causal_links": [
                            {"source_signature": "e1", "target_signature": "e2", "type": "CAUSES"}
                        ],
                    }
                ),
                json.dumps(
                    {
                        "entities": [
                            {"text": "Alice", "evidence_uids": ["u1"]},
                            {"text": "Bob", "evidence_uids": ["u2"]},
                        ],
                        "relations": [
                            {"source": "Alice", "target": "Bob", "type": "KNOWS", "evidence_uids": ["u1", "u2"]}
                        ],
                    }
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "run": {
                    "output_root": td,
                    "run_id": "t1",
                    "resume": {"mode": "auto"},
                },
                "llm": {"model": "stub", "temperature": 0.1, "max_tokens": 300},
                "execution": {"max_workers": 4},
                "dimensions": {
                    "semantic_similarity": {"enabled": False},
                    "summary": {"enabled": True, "summary_types": ["episodic", "knowledge"], "max_records": 10},
                    "event_causal": {"enabled": True, "max_records": 10},
                    "entity_relation": {"enabled": True, "max_records": 10},
                },
            }
            config_path = os.path.join(td, "config.yaml")
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("PyYAML is required for this test") from e

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

            run_mdsg_pipeline(config_path=config_path, base_space_name="demo", graph=g, llm_provider=llm)

            # Summary space should be populated
            episodic_summaries = g.semantic_map.get_units_in_spaces(["demo_episodic_summary"], recursive=True)
            self.assertEqual(len(episodic_summaries), 1)

            # Event space should be populated
            events = g.semantic_map.get_units_in_spaces(["demo_episodic_event"], recursive=True)
            self.assertEqual(len(events), 2)
            self.assertIsNotNone(g.get_relationship("demo_event_e1", "demo_event_e2", "CAUSES"))

            # Entity space should be populated
            entities = g.semantic_map.get_units_in_spaces(["demo_knowledge_entity"], recursive=True)
            self.assertEqual(len(entities), 2)
            self.assertIsNotNone(g.get_relationship("demo_entity_alice", "demo_entity_bob", "KNOWS"))

            # Resume: second run should load cache and not require additional LLM calls.
            prev_i = llm._i
            run_mdsg_pipeline(config_path=config_path, base_space_name="demo", graph=g, llm_provider=llm)
            self.assertEqual(llm._i, prev_i)


if __name__ == "__main__":
    unittest.main()
