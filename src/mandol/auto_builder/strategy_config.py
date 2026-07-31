"""Pipeline strategy presets for auto_builder orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


HIGH_THROUGHPUT_LLM_WORKERS = 60
SAFE_DEDUP_LLM_WORKERS = 30


@dataclass(frozen=True)
class PipelineStrategy:
    """Controls high-level auto_builder workflow routing for one extraction style."""

    l0_mode: str
    build_l1: bool
    build_l2: bool
    extraction_llm_name: str
    dedup_llm_name: str
    contextual_workers: int = HIGH_THROUGHPUT_LLM_WORKERS
    hierarchical_session_workers: int = HIGH_THROUGHPUT_LLM_WORKERS
    episodic_session_workers: int = HIGH_THROUGHPUT_LLM_WORKERS
    episodic_dedup_workers: int = SAFE_DEDUP_LLM_WORKERS
    entity_session_workers: int = HIGH_THROUGHPUT_LLM_WORKERS
    entity_dedup_workers: int = SAFE_DEDUP_LLM_WORKERS
    relation_session_workers: int = HIGH_THROUGHPUT_LLM_WORKERS
    episodic_dedup_method: str = "dbscan_llm"
    episodic_dbscan_eps_range: Tuple[float, float] = (0.15, 0.5)
    episodic_dbscan_min_samples_range: Tuple[int, int] = (2, 5)
    episodic_default_eps: float = 0.25
    episodic_default_min_samples: int = 2
    episodic_large_cluster_threshold: int = 15
    entity_dbscan_eps_range: Tuple[float, float] = (0.15, 0.6)
    entity_dbscan_min_samples_range: Tuple[int, int] = (2, 6)
    entity_default_eps: float = 0.3
    entity_default_min_samples: int = 2
    entity_large_cluster_threshold: int = 12
    chunk_size: int = 512
    chunk_overlap: int = 50
    episodic_sessions_per_group: int = 1
    entity_sessions_per_group: int = 1
    build_splade: bool = True
    splade_batch_size: int = 32
    create_entity_hubs: bool = False

    def __post_init__(self) -> None:
        if self.l0_mode not in {"cr", "chunk"}:
            raise ValueError("l0_mode must be either 'cr' or 'chunk'")


STYLE_STRATEGIES: Dict[str, PipelineStrategy] = {
    "locomo10": PipelineStrategy(
        l0_mode="cr",
        build_l1=True,
        build_l2=True,
        extraction_llm_name="qwen-3.5-plus-thinking",
        dedup_llm_name="deepseek-v3.2-dashscope",
        contextual_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        hierarchical_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_dedup_workers=SAFE_DEDUP_LLM_WORKERS,
        entity_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        entity_dedup_workers=SAFE_DEDUP_LLM_WORKERS,
        relation_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_dedup_method="dbscan_llm",
        episodic_dbscan_eps_range=(0.15, 0.5),
        episodic_dbscan_min_samples_range=(2, 5),
        episodic_default_eps=0.25,
        episodic_default_min_samples=2,
        episodic_large_cluster_threshold=15,
        entity_dbscan_eps_range=(0.15, 0.6),
        entity_dbscan_min_samples_range=(2, 6),
        entity_default_eps=0.3,
        entity_default_min_samples=2,
        entity_large_cluster_threshold=12,
        chunk_size=512,
        chunk_overlap=50,
        episodic_sessions_per_group=1,
        entity_sessions_per_group=1,
        build_splade=True,
        splade_batch_size=32,
        create_entity_hubs=False,
    ),
    "longmemeval": PipelineStrategy(
        l0_mode="chunk",
        build_l1=False,
        build_l2=False,
        extraction_llm_name="qwen-3-plus-latest",
        # extraction_llm_name="qwen-3-plus",
        # extraction_llm_name="qwen-3.5-plus-thinking",
        dedup_llm_name="deepseek-v3.2-dashscope",
        contextual_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        hierarchical_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_dedup_workers=SAFE_DEDUP_LLM_WORKERS,
        entity_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        entity_dedup_workers=SAFE_DEDUP_LLM_WORKERS,
        relation_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_dedup_method="dbscan_llm",
        episodic_dbscan_eps_range=(0.1, 0.6),
        episodic_dbscan_min_samples_range=(1, 5),
        episodic_default_eps=0.15,
        episodic_default_min_samples=1,
        episodic_large_cluster_threshold=15,
        entity_dbscan_eps_range=(0.15, 0.6),
        entity_dbscan_min_samples_range=(2, 6),
        entity_default_eps=0.3,
        entity_default_min_samples=2,
        entity_large_cluster_threshold=12,
        chunk_size=512,
        chunk_overlap=50,
        episodic_sessions_per_group=1,
        entity_sessions_per_group=1,
        build_splade=True,
        splade_batch_size=32,
        create_entity_hubs=False,
    ),
    "default": PipelineStrategy(
        l0_mode="chunk",
        build_l1=True,
        build_l2=True,
        extraction_llm_name="qwen-3.5-plus-thinking",
        dedup_llm_name="deepseek-v3.2-dashscope",
        contextual_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        hierarchical_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_dedup_workers=SAFE_DEDUP_LLM_WORKERS,
        entity_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        entity_dedup_workers=SAFE_DEDUP_LLM_WORKERS,
        relation_session_workers=HIGH_THROUGHPUT_LLM_WORKERS,
        episodic_dedup_method="dbscan_llm",
        episodic_dbscan_eps_range=(0.15, 0.5),
        episodic_dbscan_min_samples_range=(2, 5),
        episodic_default_eps=0.25,
        episodic_default_min_samples=2,
        episodic_large_cluster_threshold=15,
        entity_dbscan_eps_range=(0.15, 0.6),
        entity_dbscan_min_samples_range=(2, 6),
        entity_default_eps=0.3,
        entity_default_min_samples=2,
        entity_large_cluster_threshold=12,
        chunk_size=512,
        chunk_overlap=50,
        episodic_sessions_per_group=1,
        entity_sessions_per_group=10,
        build_splade=True,
        splade_batch_size=32,
        create_entity_hubs=False,
    ),
}

STYLE_ALIASES: Dict[str, str] = {
    "locomo": "locomo10",
}
