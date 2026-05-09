"""LoCoMo end-to-end evaluation for multi-hop reasoning.

Tests the UnifiedFactPipeline with LoCoMo-style queries, verifying
that SubgraphHopRetriever surfaces entities via graph edges (e.g.
LOCATED_IN) rather than relying solely on vector similarity.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mandol.ports.embedding_provider import EmbeddingProvider
    from mandol.ports.llm_provider import LLMProvider
    from mandol.ports.reranker import Reranker

from mandol.application.memory_system import MemorySystem, MemorySystemConfig
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid
from mandol.infrastructure.openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleEmbeddingProvider,
)
from mandol.infrastructure.openai_compatible_llm_provider import (
    OpenAICompatibleLLMConfig,
    OpenAICompatibleLLMProvider,
)
from mandol.infrastructure.openai_compatible_reranker import (
    OpenAICompatibleRerankConfig,
    OpenAICompatibleReranker,
)
from mandol.retrieval.subgraph_hop import SubgraphHopRetriever, SubgraphHopConfig, SubgraphHopHit
from mandol.retrieval.pipeline import HybridRetriever, HybridRetrieverConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    conversation_id: str
    query: str
    reference: str
    evidence: List[str]
    category: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    total_queries: int = 0
    correct_predictions: int = 0
    evidence_recall: float = 0.0
    evidence_precision: float = 0.0
    avg_reasoning_path_length: float = 0.0
    avg_retrieval_time: float = 0.0
    category_breakdown: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def parse_session_number(key: str) -> Optional[int]:
    m = re.match(r"^session_(\d+)$", str(key))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_dialogue_index(dia_id: str, fallback: int) -> int:
    try:
        if ":" in str(dia_id):
            return int(str(dia_id).split(":")[-1])
    except Exception:
        pass
    return int(fallback)


class LocomoE2EEvaluator:
    def __init__(
        self,
        dataset_path: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.config = config or {}
        self._memory_system: Optional[MemorySystem] = None
        self._subgraph_retriever: Optional[SubgraphHopRetriever] = None
        self._processed_sample_ids: Set[str] = set()
        self._qa_examples: List[QAExample] = []
        self._sample_id_to_root: Dict[str, str] = {}

    def _setup_providers(self) -> Tuple["EmbeddingProvider", "Reranker", "LLMProvider"]:
        cfg = self.config.get("providers", {})
        
        embedder_model = cfg.get("embedder_model", "Qwen/Qwen3-Embedding-4B")
        embedder_dim = cfg.get("embedder_dim", 2560)
        embedder_device = cfg.get("embedder_device", "cuda")
        
        reranker_model = cfg.get("reranker_model", "Qwen/Qwen3-Reranker-4B")
        reranker_device = cfg.get("reranker_device", "cuda")
        
        llm_model = cfg.get("llm_model", "gpt-4o-mini")
        
        remote = cfg.get("remote", {})
        
        if remote.get("use_remote_embedder", False):
            emb_config = OpenAICompatibleEmbeddingConfig(
                base_url=remote.get("embedding_base_url", ""),
                api_path=remote.get("embedding_api_path", "/v1/embeddings"),
                token_env=remote.get("embedding_token_env", ""),
                timeout_s=remote.get("embedding_timeout_s", 60),
            )
            embedder = OpenAICompatibleEmbeddingProvider(
                model=embedder_model,
                dim=embedder_dim,
                config=emb_config,
            )
        else:
            from mandol.infrastructure.sentence_transformers_embedding_provider import (
                SentenceTransformersEmbeddingProvider,
            )
            embedder = SentenceTransformersEmbeddingProvider(
                model=embedder_model,
                device=embedder_device,
            )
        
        if remote.get("use_remote_reranker", False):
            rerank_config = OpenAICompatibleRerankConfig(
                base_url=remote.get("reranker_base_url", ""),
                api_path=remote.get("reranker_api_path", "/v1/rerank"),
                token_env=remote.get("reranker_token_env", ""),
                timeout_s=remote.get("reranker_timeout_s", 60),
            )
            reranker = OpenAICompatibleReranker(
                model=reranker_model,
                config=rerank_config,
            )
        else:
            from mandol.infrastructure.sentence_transformers_reranker import (
                SentenceTransformersCrossEncoderReranker,
            )
            reranker = SentenceTransformersCrossEncoderReranker(
                model=reranker_model,
                device=reranker_device,
            )
        
        if remote.get("use_remote_llm", False):
            llm_config = OpenAICompatibleLLMConfig(
                base_url=remote.get("llm_base_url", ""),
                api_key_env=remote.get("llm_api_key_env", "OPENAI_API_KEY"),
                timeout_s=remote.get("llm_timeout_s", 120),
            )
            llm_provider = OpenAICompatibleLLMProvider(
                model=llm_model,
                config=llm_config,
            )
        else:
            from mandol.infrastructure.stub_llm_provider import StubLLMProvider
            llm_provider = StubLLMProvider()
        
        return embedder, reranker, llm_provider

    def initialize(self) -> None:
        embedder, reranker, llm_provider = self._setup_providers()
        
        mem_cfg = self.config.get("memory_system", {})
        memory_config = MemorySystemConfig(
            embedder_model=mem_cfg.get("embedder_model", "Qwen/Qwen3-Embedding-4B"),
            embedder_device=mem_cfg.get("embedder_device", "cuda"),
            reranker_model=mem_cfg.get("reranker_model", "Qwen/Qwen3-Reranker-4B"),
            reranker_device=mem_cfg.get("reranker_device", "cuda"),
            llm_model=mem_cfg.get("llm_model", "gpt-4o-mini"),
            embedder_dim=mem_cfg.get("embedder_dim", 2560),
            chunk_max_tokens=mem_cfg.get("chunk_max_tokens", 512),
            session_time_gap_seconds=mem_cfg.get("session_time_gap_seconds", 1800),
            similarity_top_k=mem_cfg.get("similarity_top_k", 5),
            similarity_threshold=mem_cfg.get("similarity_threshold", 0.7),
            use_unified_pipeline=mem_cfg.get("use_unified_pipeline", True),
        )
        
        self._memory_system = MemorySystem(
            config=memory_config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
        )
        
        hybrid_config = HybridRetrieverConfig(
            per_method_k=mem_cfg.get("per_method_k", 60),
            rrf_k=mem_cfg.get("rrf_k", 60),
            bfs_per_seed=mem_cfg.get("bfs_per_seed", 3),
            bfs_hops=mem_cfg.get("bfs_hops", 1),
        )
        
        hybrid_retriever = HybridRetriever(
            graph=self._memory_system.graph,
            reranker=reranker,
            config=hybrid_config,
        )
        
        subgraph_config = SubgraphHopConfig(
            base=hybrid_config,
            max_hops=mem_cfg.get("subgraph_max_hops", 2),
            hop_decay=mem_cfg.get("hop_decay", 0.85),
            seed_top_k=mem_cfg.get("seed_top_k", 5),
            graph_branch_weight=mem_cfg.get("graph_branch_weight", 0.45),
        )
        
        self._subgraph_retriever = SubgraphHopRetriever(
            hybrid=hybrid_retriever,
            config=subgraph_config,
        )
        
        logger.info("LocomoE2EEvaluator initialized successfully")

    def load_dataset(
        self,
        conversation_ids: Optional[List[str]] = None,
        categories: Optional[List[int]] = None,
    ) -> List[QAExample]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Dataset root must be a list")
        
        self._qa_examples = []
        
        for item in data:
            sample_id = item.get("sample_id", "")
            if not sample_id:
                continue
            
            if conversation_ids and sample_id not in conversation_ids:
                continue
            
            self._sample_id_to_root[sample_id] = sample_id
            
            qa_pairs = item.get("qa", [])
            for qa in qa_pairs:
                answer = qa.get("answer", "")
                if not answer:
                    answer = qa.get("adversarial_answer", qa.get("reference", ""))
                
                question = qa.get("question", "")
                if not question:
                    continue
                
                category = qa.get("category", 0)
                if categories and category not in categories:
                    continue
                
                evidence_ids = qa.get("evidence", [])
                evidence_contents = [str(e) for e in evidence_ids if e is not None]
                
                example = QAExample(
                    conversation_id=sample_id,
                    query=question,
                    reference=str(answer),
                    evidence=evidence_contents,
                    category=category,
                    metadata={
                        "question_id": f"{sample_id}-q{len(self._qa_examples)+1}",
                        "evidence": evidence_contents,
                        "category": category,
                    },
                )
                self._qa_examples.append(example)
        
        logger.info(f"Loaded {len(self._qa_examples)} QA examples from {len(data)} samples")
        return self._qa_examples

    def process_conversations(
        self,
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self._memory_system is None:
            self.initialize()
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples_to_process = data
        if sample_ids:
            samples_to_process = [s for s in data if s.get("sample_id") in set(sample_ids)]
        
        stats = {
            "samples_processed": 0,
            "dialogues_added": 0,
            "sessions_created": 0,
        }
        
        for sample in samples_to_process:
            sample_id = sample.get("sample_id", "")
            if not sample_id:
                continue
            
            if sample_id in self._processed_sample_ids:
                continue
            
            dialogues_added = self._process_sample(sample)
            self._processed_sample_ids.add(sample_id)
            stats["samples_processed"] += 1
            stats["dialogues_added"] += dialogues_added
        
        return stats

    def _process_sample(self, sample: Dict[str, Any]) -> int:
        if self._memory_system is None:
            raise RuntimeError("MemorySystem not initialized")
        
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id:
            return 0
        
        base_root = sample_id
        
        conv = sample.get("conversation") or {}
        if not isinstance(conv, dict):
            return 0
        
        sessions: List[Tuple[int, str, List[Dict[str, Any]]]] = []
        for k, v in conv.items():
            n = parse_session_number(k)
            if n is None:
                continue
            if not isinstance(v, list):
                continue
            dt = conv.get(f"session_{n}_date_time", "")
            sessions.append((n, str(dt or ""), [x for x in v if isinstance(x, dict)]))
        sessions.sort(key=lambda x: x[0])
        
        dialogues_added = 0
        
        for sess_n, sess_dt, dialogues in sessions:
            ordered: List[Tuple[int, str]] = []
            
            for i, d in enumerate(dialogues):
                dia_id = str(d.get("dia_id") or "").strip()
                speaker = str(d.get("speaker") or "").strip()
                text = d.get("text")
                if text is None or str(text).strip() == "":
                    text = d.get("text_content") or d.get("content") or ""
                text = str(text)
                if not text.strip():
                    continue
                
                didx = parse_dialogue_index(dia_id, i)
                if not dia_id:
                    dia_id = f"D{sess_n}:{didx}"
                
                unit_uid = f"{base_root}_dialogue_{dia_id}"
                
                existing_unit = self._memory_system.semantic_map.get_unit(unit_uid)
                if existing_unit is not None:
                    ordered.append((didx, unit_uid))
                    continue
                
                unit = MemoryUnit(
                    uid=Uid(unit_uid),
                    raw_data={
                        "type": "dialogue",
                        "dia_id": dia_id,
                        "speaker": speaker,
                        "text": text,
                        "dialogue_index": didx,
                        "session_datetime": sess_dt,
                        "session_number": sess_n,
                        "text_content": f"Dialogue {dia_id} [Time {sess_dt}]: {speaker} said: {text}",
                    },
                    metadata={
                        "unit_type": "dialogue",
                        "session_number": sess_n,
                        "sample_id": sample_id,
                    },
                    embedding=None,
                )
                
                self._memory_system.add(unit)
                ordered.append((didx, unit_uid))
                dialogues_added += 1
            
            ordered.sort(key=lambda x: x[0])
            for (_, a), (_, b) in zip(ordered, ordered[1:]):
                self._memory_system.graph.add_relationship(
                    source_uid=a,
                    target_uid=b,
                    relationship_name="PRECEDES",
                    score=1.0,
                )
                self._memory_system.graph.add_relationship(
                    source_uid=b,
                    target_uid=a,
                    relationship_name="FOLLOWS",
                    score=1.0,
                )
        
        return dialogues_added

    def build_high_level_memories(self, mode: str = "auto") -> Dict[str, Any]:
        if self._memory_system is None:
            raise RuntimeError("MemorySystem not initialized")
        
        return self._memory_system.build_high_level(mode=mode)

    def retrieve_with_reasoning(
        self,
        query: str,
        sample_id: Optional[str] = None,
        top_k: int = 10,
    ) -> Tuple[List[SubgraphHopHit], float]:
        if self._subgraph_retriever is None:
            raise RuntimeError("SubgraphHopRetriever not initialized")
        
        space_names = None
        if sample_id:
            root = self._sample_id_to_root.get(sample_id, sample_id)
            base_space = f"{root}_base_memory"
            space_names = [base_space]
        
        start_time = time.time()
        hits = self._subgraph_retriever.search(
            query,
            top_k=top_k,
            space_names=space_names,
            recursive=True,
            use_rerank=True,
        )
        elapsed = time.time() - start_time
        
        return hits, elapsed

    def evaluate_evidence_recall(
        self,
        hits: List[SubgraphHopHit],
        evidence_ids: List[str],
        sample_id: str,
    ) -> Dict[str, Any]:
        if not evidence_ids:
            return {
                "recall": 1.0,
                "precision": 1.0,
                "found_evidence": [],
                "missing_evidence": [],
            }
        
        root = self._sample_id_to_root.get(sample_id, sample_id)
        
        expected_uids: Set[str] = set()
        for ev in evidence_ids:
            expected_uids.add(f"{root}_dialogue_{ev}")
        
        retrieved_uids: Set[str] = set()
        for hit in hits:
            retrieved_uids.add(str(hit.unit.uid))
        
        found = expected_uids & retrieved_uids
        missing = expected_uids - retrieved_uids
        
        recall = len(found) / len(expected_uids) if expected_uids else 1.0
        precision = len(found) / len(retrieved_uids) if retrieved_uids else 0.0
        
        return {
            "recall": recall,
            "precision": precision,
            "found_evidence": list(found),
            "missing_evidence": list(missing),
        }

    def extract_reasoning_paths(
        self,
        hits: List[SubgraphHopHit],
    ) -> List[List[Dict[str, Any]]]:
        paths = []
        for hit in hits:
            if hit.reasoning_path:
                path_info = []
                for step in hit.reasoning_path:
                    path_info.append({
                        "source_uid": str(step.source_uid),
                        "target_uid": str(step.target_uid),
                        "rel_type": step.rel_type,
                        "rel_weight": step.rel_weight,
                        "direction": step.direction,
                    })
                paths.append(path_info)
        return paths

    def run_evaluation(
        self,
        conversation_ids: Optional[List[str]] = None,
        categories: Optional[List[int]] = None,
        top_k: int = 10,
        build_high_level: bool = True,
    ) -> Dict[str, Any]:
        logger.info("Starting LoCoMo E2E evaluation...")
        
        qa_examples = self.load_dataset(
            conversation_ids=conversation_ids,
            categories=categories,
        )
        
        sample_ids = list(set(ex.conversation_id for ex in qa_examples))
        logger.info(f"Processing {len(sample_ids)} conversations...")
        
        process_stats = self.process_conversations(sample_ids=sample_ids)
        logger.info(f"Processed {process_stats['samples_processed']} samples")
        
        if build_high_level:
            logger.info("Building high-level memories...")
            build_result = self.build_high_level_memories(mode="auto")
            logger.info(f"High-level memory build result: {build_result}")
        
        metrics = EvaluationMetrics()
        metrics.total_queries = len(qa_examples)
        
        all_results = []
        category_results: Dict[int, List[Dict[str, Any]]] = {}
        
        for example in qa_examples:
            hits, retrieval_time = self.retrieve_with_reasoning(
                query=example.query,
                sample_id=example.conversation_id,
                top_k=top_k,
            )
            
            evidence_result = self.evaluate_evidence_recall(
                hits=hits,
                evidence_ids=example.evidence,
                sample_id=example.conversation_id,
            )
            
            reasoning_paths = self.extract_reasoning_paths(hits)
            
            result = {
                "query": example.query,
                "conversation_id": example.conversation_id,
                "category": example.category,
                "reference": example.reference,
                "evidence": example.evidence,
                "retrieval_time": retrieval_time,
                "evidence_recall": evidence_result["recall"],
                "evidence_precision": evidence_result["precision"],
                "found_evidence": evidence_result["found_evidence"],
                "missing_evidence": evidence_result["missing_evidence"],
                "num_hits": len(hits),
                "reasoning_paths": reasoning_paths,
                "top_hit_scores": [h.final_score for h in hits[:5]],
            }
            
            all_results.append(result)
            
            cat = example.category
            if cat not in category_results:
                category_results[cat] = []
            category_results[cat].append(result)
            
            metrics.evidence_recall += evidence_result["recall"]
            metrics.evidence_precision += evidence_result["precision"]
            metrics.avg_retrieval_time += retrieval_time
            
            if reasoning_paths:
                metrics.avg_reasoning_path_length += sum(len(p) for p in reasoning_paths) / len(reasoning_paths)
        
        if metrics.total_queries > 0:
            metrics.evidence_recall /= metrics.total_queries
            metrics.evidence_precision /= metrics.total_queries
            metrics.avg_retrieval_time /= metrics.total_queries
            metrics.avg_reasoning_path_length /= metrics.total_queries
        
        for cat, results in category_results.items():
            cat_recall = sum(r["evidence_recall"] for r in results) / len(results)
            cat_precision = sum(r["evidence_precision"] for r in results) / len(results)
            cat_avg_time = sum(r["retrieval_time"] for r in results) / len(results)
            
            metrics.category_breakdown[cat] = {
                "count": len(results),
                "avg_evidence_recall": cat_recall,
                "avg_evidence_precision": cat_precision,
                "avg_retrieval_time": cat_avg_time,
            }
        
        return {
            "overall_metrics": {
                "total_queries": metrics.total_queries,
                "avg_evidence_recall": metrics.evidence_recall,
                "avg_evidence_precision": metrics.evidence_precision,
                "avg_retrieval_time": metrics.avg_retrieval_time,
                "avg_reasoning_path_length": metrics.avg_reasoning_path_length,
            },
            "category_breakdown": metrics.category_breakdown,
            "detailed_results": all_results,
            "processing_stats": process_stats,
        }

    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print("LoCoMo E2E Evaluation Report")
        print("=" * 80)
        
        overall = results["overall_metrics"]
        print("\n📊 Overall Metrics:")
        print(f"  Total Queries: {overall['total_queries']}")
        print(f"  Avg Evidence Recall: {overall['avg_evidence_recall']:.4f}")
        print(f"  Avg Evidence Precision: {overall['avg_evidence_precision']:.4f}")
        print(f"  Avg Retrieval Time: {overall['avg_retrieval_time']:.4f}s")
        print(f"  Avg Reasoning Path Length: {overall['avg_reasoning_path_length']:.2f}")
        
        print("\n📈 Category Breakdown:")
        for cat, metrics in sorted(results["category_breakdown"].items()):
            print(f"\n  Category {cat}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Avg Evidence Recall: {metrics['avg_evidence_recall']:.4f}")
            print(f"    Avg Evidence Precision: {metrics['avg_evidence_precision']:.4f}")
            print(f"    Avg Retrieval Time: {metrics['avg_retrieval_time']:.4f}s")
        
        print("\n" + "=" * 80)

    def get_memory_stats(self) -> Dict[str, Any]:
        if self._memory_system is None:
            return {"status": "not_initialized"}
        
        spaces = self._memory_system.semantic_map.list_spaces()
        units = self._memory_system.semantic_map.list_units()
        
        space_info = {}
        for sp in spaces:
            space_info[str(sp.name)] = {
                "unit_count": len(sp.unit_uids),
                "child_spaces": [str(c) for c in sp.child_spaces],
            }
        
        return {
            "status": "ok",
            "total_spaces": len(spaces),
            "total_units": len(units),
            "processed_sample_ids": sorted(self._processed_sample_ids),
            "space_details": space_info,
        }


def run_test(
    dataset_path: str = "datasets/locomo10.json",
    conversation_ids: Optional[List[str]] = None,
    categories: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    build_high_level: bool = True,
) -> Dict[str, Any]:
    evaluator = LocomoE2EEvaluator(
        dataset_path=dataset_path,
        config=config,
    )
    
    results = evaluator.run_evaluation(
        conversation_ids=conversation_ids,
        categories=categories,
        top_k=top_k,
        build_high_level=build_high_level,
    )
    
    evaluator.print_evaluation_report(results)
    
    return results


class TestLocomoE2EEvaluator:
    """Tests for LocomoE2EEvaluator class."""
    
    def test_init(self, tmp_path: Path) -> None:
        """Test evaluator initialization."""
        dataset_path = tmp_path / "test_dataset.json"
        dataset_path.write_text("[]", encoding="utf-8")
        
        evaluator = LocomoE2EEvaluator(dataset_path=str(dataset_path))
        assert evaluator.dataset_path == dataset_path
        assert evaluator._memory_system is None
        assert evaluator._qa_examples == []
    
    def test_load_dataset_empty(self, tmp_path: Path) -> None:
        """Test loading empty dataset."""
        dataset_path = tmp_path / "empty_dataset.json"
        dataset_path.write_text("[]", encoding="utf-8")
        
        evaluator = LocomoE2EEvaluator(dataset_path=str(dataset_path))
        examples = evaluator.load_dataset()
        assert examples == []
    
    def test_load_dataset_with_samples(self, tmp_path: Path) -> None:
        """Test loading dataset with samples."""
        dataset_content = [
            {
                "sample_id": "conv-1",
                "qa": [
                    {
                        "question": "What is X?",
                        "answer": "X is Y",
                        "category": 1,
                        "evidence": ["D1:1", "D1:2"],
                    }
                ],
                "conversation": {
                    "session_1": [
                        {"dia_id": "D1:1", "speaker": "A", "text": "Hello"},
                        {"dia_id": "D1:2", "speaker": "B", "text": "Hi"},
                    ],
                    "session_1_date_time": "2024-01-01",
                },
            }
        ]
        
        dataset_path = tmp_path / "test_dataset.json"
        dataset_path.write_text(json.dumps(dataset_content), encoding="utf-8")
        
        evaluator = LocomoE2EEvaluator(dataset_path=str(dataset_path))
        examples = evaluator.load_dataset()
        
        assert len(examples) == 1
        assert examples[0].conversation_id == "conv-1"
        assert examples[0].category == 1
        assert examples[0].evidence == ["D1:1", "D1:2"]
    
    def test_load_dataset_filter_by_category(self, tmp_path: Path) -> None:
        """Test filtering dataset by category."""
        dataset_content = [
            {
                "sample_id": "conv-1",
                "qa": [
                    {"question": "Q1", "answer": "A1", "category": 1, "evidence": []},
                    {"question": "Q2", "answer": "A2", "category": 2, "evidence": []},
                ],
                "conversation": {},
            }
        ]
        
        dataset_path = tmp_path / "test_dataset.json"
        dataset_path.write_text(json.dumps(dataset_content), encoding="utf-8")
        
        evaluator = LocomoE2EEvaluator(dataset_path=str(dataset_path))
        examples = evaluator.load_dataset(categories=[1])
        
        assert len(examples) == 1
        assert examples[0].category == 1
    
    def test_evaluate_evidence_recall_empty_evidence(self, tmp_path: Path) -> None:
        """Test evidence recall with empty evidence list."""
        dataset_path = tmp_path / "test_dataset.json"
        dataset_path.write_text("[]", encoding="utf-8")
        
        evaluator = LocomoE2EEvaluator(dataset_path=str(dataset_path))
        evaluator._sample_id_to_root["test"] = "test"
        
        result = evaluator.evaluate_evidence_recall([], [], "test")
        
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
    
    def test_evaluate_evidence_recall_with_hits(self, tmp_path: Path) -> None:
        """Test evidence recall with matching hits."""
        dataset_path = tmp_path / "test_dataset.json"
        dataset_path.write_text("[]", encoding="utf-8")
        
        evaluator = LocomoE2EEvaluator(dataset_path=str(dataset_path))
        evaluator._sample_id_to_root["test"] = "test"
        
        from unittest.mock import MagicMock
        
        mock_hit = MagicMock()
        mock_hit.unit.uid = "test_dialogue_D1:1"
        
        result = evaluator.evaluate_evidence_recall(
            hits=[mock_hit],
            evidence_ids=["D1:1"],
            sample_id="test",
        )
        
        assert result["recall"] == 1.0
        assert "D1:1" in result["found_evidence"][0]
    
    def test_parse_session_number(self) -> None:
        """Test session number parsing."""
        assert parse_session_number("session_1") == 1
        assert parse_session_number("session_10") == 10
        assert parse_session_number("invalid") is None
        assert parse_session_number("") is None
    
    def test_parse_dialogue_index(self) -> None:
        """Test dialogue index parsing."""
        assert parse_dialogue_index("D1:5", 0) == 5
        assert parse_dialogue_index("D2:10", 0) == 10
        assert parse_dialogue_index("invalid", 3) == 3


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoCoMo E2E Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/locomo10.json",
        help="Path to LoCoMo dataset",
    )
    parser.add_argument(
        "--conversation-ids",
        type=str,
        nargs="+",
        default=None,
        help="List of conversation IDs to evaluate",
    )
    parser.add_argument(
        "--categories",
        type=int,
        nargs="+",
        default=None,
        help="List of categories to evaluate (e.g., 1 for multi-hop reasoning)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--no-build-high-level",
        action="store_true",
        help="Skip building high-level memories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )
    
    args = parser.parse_args()
    
    results = run_test(
        dataset_path=args.dataset,
        conversation_ids=args.conversation_ids,
        categories=args.categories,
        top_k=args.top_k,
        build_high_level=not args.no_build_high_level,
    )
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
