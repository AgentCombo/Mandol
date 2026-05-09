"""Unit tests for UnifiedFactPipeline and PipelineResult dataclass."""

from __future__ import annotations

import unittest

from mandol.application.pipeline.unified_fact_pipeline import (
    PipelineResult,
)


class TestPipelineResult(unittest.TestCase):
    def test_all_nine_fields_present(self):
        PipelineResult(
            entities=[],
            events=[],
            entity_relations=[],
            causal_relations=[],
            coref_edges=[],
            evidenced_by_edges=[],
            involves_edges=[],
            related_to_edges=[],
            causes_edges=[],
        )
