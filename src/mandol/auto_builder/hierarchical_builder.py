# mandol/auto_builder/hierarchical_builder.py
"""Utilities for hierarchical builder."""
import json
import logging
from ..utils.logging_config import create_module_logger
import re
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .session_tracker import SessionTracker, SessionInfo
from .hierarchical_prompts import (
    HierarchicalPromptManager, 
    ExtractionStyle, 
    L1SummaryType
)
from .l0_views import build_l0_inference_context
from .graph_write_queue import GraphWriteQueue, GraphWriteRequest, dispatch_graph_write_requests
from ..core.memory_space_registry import TowerSpace

if TYPE_CHECKING:
    from ..core.semantic_map import SemanticMap
    from ..core.semantic_graph import SemanticGraph
    from ..core.memory_unit import MemoryUnit
    from ..llm.llm_client import LLMClient

logger = create_module_logger("auto_builder.hierarchical_builder")





@dataclass
class HierarchicalBuilderConfig:
    
    extraction_style: str = "default"  # default, locomo, longmemeval
    
    enable_contextual_retrieval: bool = False
    contextual_parallel_workers: int = 60
    
    l1_summary_types: List[str] = field(default_factory=lambda: ["episodic", "knowledge"])
    l1_max_tokens: int = 500
    l1_temperature: float = 0.3
    
    l2_max_tokens: int = 1000
    l2_temperature: float = 0.3
    
    enable_chunking: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    enable_deduplication: bool = False
    dedup_similarity_threshold: float = 0.85
    
    l0_space_name: str = TowerSpace.HIERARCHICAL_L0.value
    l1_space_name: str = TowerSpace.HIERARCHICAL_L1.value
    l2_space_name: str = TowerSpace.HIERARCHICAL_L2.value
    
    parallel_workers: int = 30
    
    llm_max_retries: int = 3
    llm_retry_delay: float = 1.0


@dataclass
class L1ExtractionResult:
    session_id: str
    summary_type: str
    content: str
    unit_uid: str
    source_unit_uids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_structured: bool = False


@dataclass
class L2AggregationResult:
    sample_id: str
    content: str
    unit_uid: str
    source_l1_uids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_structured: bool = False





class HierarchicalAutoBuilder:
    
    def __init__(self,
                 semantic_system: Optional[Union["SemanticMap", "SemanticGraph"]] = None,
                 llm_client: Optional["LLMClient"] = None,
                 config: Optional[HierarchicalBuilderConfig] = None):
        self.semantic_system = semantic_system
        self.llm_client = llm_client
        self.config = config or HierarchicalBuilderConfig()
        
        self.session_tracker = SessionTracker()
        
        self.prompt_manager = HierarchicalPromptManager
        
        self._text_splitter = None
        
        self._build_stats = {
            "l0_units_processed": 0,
            "l0_units_enhanced": 0,
            "l1_summaries_generated": 0,
            "l2_insights_generated": 0,
            "sessions_processed": 0,
            "chunks_created": 0,
            "dedup_merged": 0
        }
        
        logger.info("HierarchicalAutoBuilder initialized")
        logger.info(f"   - extraction_style: {self.config.extraction_style}")
        logger.info(f"   - Contextual Retrieval: {'enabled' if self.config.enable_contextual_retrieval else 'disabled'}")
        logger.info(f"   - chunking: {'enabled' if self.config.enable_chunking else 'disabled'}")
    
    
    
    
    def _get_text_splitter(self):
        """Get text splitter."""
        if self._text_splitter is None and self.config.enable_chunking:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                logger.warning("LangChain is not installed; chunking is unavailable")
                return None
            
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
                length_function=len
            )
            logger.debug(f"LangChain splitter initialized: chunk_size={self.config.chunk_size}")
        
        return self._text_splitter
    
    def _ensure_llm_client(self):
        """Ensure LLM client."""
        if self.llm_client is None:
            raise RuntimeError("LLM client is not initialized; pass llm_client to the constructor")
    
    def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Run safe parse JSON."""
        if not text:
            return None
        
        text = text.strip()
        
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _call_llm_with_retry(self, 
                             prompt: str, 
                             temperature: float = 0.3,
                             max_tokens: int = 500,
                             context_id: str = "") -> Optional[str]:
        """Run call LLM with retry."""
        import time
        
        self._ensure_llm_client()
        
        for attempt in range(self.config.llm_max_retries):
            try:
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response
            except Exception as e:
                logger.warning(f"LLM call failed ({context_id}), attempt {attempt + 1}: {e}")
                if attempt < self.config.llm_max_retries - 1:
                    time.sleep(self.config.llm_retry_delay)
        
        logger.error(f"LLM call failed ({context_id}): all retries exhausted")
        return None
    
    
    
    
    def enhance_l0_with_context(self,
                                l0_units: List["MemoryUnit"],
                                session_date: Optional[str] = None,
                                participants: Optional[List[str]] = None,
                                custom_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run enhance L0 with context."""
        if not self.config.enable_contextual_retrieval:
            logger.debug("Contextual Retrieval is disabled; skipping enhancement")
            return []
        
        self._ensure_llm_client()
        
        logger.info(f"Starting Contextual Retrieval enhancement: {len(l0_units)} units")
        
        full_transcript = self._build_full_transcript(l0_units)
        
        session_date = session_date or datetime.now().strftime("%Y-%m-%d")
        participants = participants or ["Speaker_A", "Speaker_B"]
        
        enhanced_units = []
        
        def enhance_single_unit(unit: "MemoryUnit") -> Optional[Dict[str, Any]]:
            """Run enhance single unit."""
            original_content = unit.raw_data.get("text_content", "")
            speaker = unit.raw_data.get("speaker", unit.metadata.get("speaker", "Unknown"))
            
            if not original_content or len(original_content.strip()) < 10:
                return None
            
            prompt = self.prompt_manager.get_contextual_retrieval_prompt(
                session_date=session_date,
                participants=participants,
                full_session_transcript=full_transcript,
                speaker=speaker,
                message_text=original_content,
                custom_prompt=custom_prompt
            )
            
            enhanced_content = self._call_llm_with_retry(
                prompt=prompt,
                temperature=0.1,
                max_tokens=300,
                context_id=f"enhance_{unit.uid}"
            )
            
            if enhanced_content:
                return {
                    "uid": unit.uid,
                    "original_content": original_content,
                    "enhanced_content": enhanced_content.strip(),
                    "speaker": speaker
                }
            return None
        
        with ThreadPoolExecutor(max_workers=self.config.contextual_parallel_workers) as executor:
            futures = {executor.submit(enhance_single_unit, unit): unit for unit in l0_units}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    enhanced_units.append(result)
                    self._build_stats["l0_units_enhanced"] += 1
        
        logger.info(f"Contextual Retrieval enhancement completed: {len(enhanced_units)} units")
        return enhanced_units
    
    def _build_full_transcript(self, l0_units: List["MemoryUnit"]) -> str:
        """Build full transcript."""
        lines = []
        for i, unit in enumerate(l0_units, 1):
            speaker = unit.raw_data.get("speaker", unit.metadata.get("speaker", "Unknown"))
            content = unit.raw_data.get("text_content", "")
            if content:
                lines.append(f"[{i}] {speaker}: {content}")
        return "\n".join(lines)
    
    
    
    
    def extract_l1_from_l0_units(self,
                                  l0_units: List["MemoryUnit"],
                                  session_id: str = "session_default",
                                  session_date: Optional[str] = None,
                                  participants: Optional[List[str]] = None,
                                  summary_types: Optional[List[str]] = None,
                                  custom_prompts: Optional[Dict[str, str]] = None) -> List[L1ExtractionResult]:
        """Args: session_id: Session ID Returns:."""
        self._ensure_llm_client()
        
        logger.info(f"Starting L1 extraction: session={session_id}, units={len(l0_units)}")
        
        if summary_types is None:
            if self.config.extraction_style == ExtractionStyle.LOCOMO.value:
                summary_types = L1SummaryType.locomo_types()
            elif self.config.extraction_style == ExtractionStyle.LONGMEMEVAL.value:
                summary_types = L1SummaryType.longmemeval_types()
            else:
                summary_types = self.config.l1_summary_types
        
        l0_content = self._merge_l0_content(l0_units)
        
        if self.config.enable_chunking and len(l0_content) > self.config.chunk_size * 2:
            return self._extract_l1_with_chunking(
                l0_content=l0_content,
                l0_units=l0_units,
                session_id=session_id,
                session_date=session_date,
                participants=participants,
                summary_types=summary_types,
                custom_prompts=custom_prompts
            )
        
        results = []
        session_date = session_date or datetime.now().strftime("%Y-%m-%d")
        participants = participants or ["Speaker_A", "Speaker_B"]
        
        for summary_type in summary_types:
            try:
                prompt = self.prompt_manager.get_l1_prompt(
                    summary_type=summary_type,
                    content=l0_content,
                    extraction_style=self.config.extraction_style,
                    session_id=session_id,
                    session_date=session_date,
                    participants=participants,
                    custom_prompt=custom_prompts.get(f"l1_{summary_type}") if custom_prompts else None
                )
                
                response = self._call_llm_with_retry(
                    prompt=prompt,
                    temperature=self.config.l1_temperature,
                    max_tokens=self.config.l1_max_tokens,
                    context_id=f"l1_{session_id}_{summary_type}"
                )
                
                if not response:
                    continue
                
                is_structured = (summary_type == L1SummaryType.STRUCTURED.value or 
                                self.config.extraction_style == ExtractionStyle.LONGMEMEVAL.value)
                
                uid = f"{session_id}_L1_{summary_type}_{int(datetime.now().timestamp())}"
                result = L1ExtractionResult(
                    session_id=session_id,
                    summary_type=summary_type,
                    content=response.strip(),
                    unit_uid=uid,
                    source_unit_uids=[u.uid for u in l0_units],
                    metadata={
                        "extraction_style": self.config.extraction_style,
                        "session_date": session_date,
                        "participants": participants,
                        "created_at": datetime.now().isoformat()
                    },
                    is_structured=is_structured
                )
                
                results.append(result)
                self._build_stats["l1_summaries_generated"] += 1
                
                logger.debug(f"    L1 {summary_type}: {uid}")
                
            except Exception as e:
                    logger.error(f"    L1 {summary_type} extraction failed: {e}")
        
        self._build_stats["l0_units_processed"] += len(l0_units)
        logger.info(f"L1 extraction completed: {len(results)} summaries")
        return results
    
    def _extract_l1_with_chunking(self,
                                   l0_content: str,
                                   l0_units: List["MemoryUnit"],
                                   session_id: str,
                                   session_date: Optional[str],
                                   participants: Optional[List[str]],
                                   summary_types: List[str],
                                   custom_prompts: Optional[Dict[str, str]]) -> List[L1ExtractionResult]:
        """Extract L1 with chunking."""
        text_splitter = self._get_text_splitter()
        if text_splitter is None:
            logger.warning("Chunker is unavailable; falling back to regular extraction")
            chunks = [l0_content[:self.config.chunk_size * 3]]
        else:
            chunks = text_splitter.split_text(l0_content)
        
        logger.info(f"Long text chunked into {len(chunks)} chunks")
        self._build_stats["chunks_created"] += len(chunks)
        
        results = []
        session_date = session_date or datetime.now().strftime("%Y-%m-%d")
        participants = participants or ["Speaker_A", "Speaker_B"]
        
        for chunk_idx, chunk_content in enumerate(chunks):
            context_id = f"chunk_{chunk_idx + 1}"
            
            for summary_type in summary_types:
                try:
                    prompt = self.prompt_manager.get_l1_prompt(
                        summary_type=summary_type,
                        content=chunk_content,
                        extraction_style=self.config.extraction_style,
                        session_id=session_id,
                        session_date=session_date,
                        participants=participants,
                        context_id=context_id,
                        session_info=f"Chunk {chunk_idx + 1}/{len(chunks)}",
                        custom_prompt=custom_prompts.get(f"l1_{summary_type}") if custom_prompts else None
                    )
                    
                    response = self._call_llm_with_retry(
                        prompt=prompt,
                        temperature=self.config.l1_temperature,
                        max_tokens=self.config.l1_max_tokens,
                        context_id=f"l1_{session_id}_{summary_type}_{context_id}"
                    )
                    
                    if not response:
                        continue
                    
                    uid = f"{session_id}_L1_{summary_type}_{context_id}_{int(datetime.now().timestamp())}"
                    result = L1ExtractionResult(
                        session_id=session_id,
                        summary_type=summary_type,
                        content=response.strip(),
                        unit_uid=uid,
                        source_unit_uids=[u.uid for u in l0_units],
                        metadata={
                            "extraction_style": self.config.extraction_style,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "created_at": datetime.now().isoformat()
                        },
                        is_structured=(summary_type == L1SummaryType.STRUCTURED.value)
                    )
                    
                    results.append(result)
                    self._build_stats["l1_summaries_generated"] += 1
                    
                except Exception as e:
                    logger.error(f"    L1 {summary_type} chunk {chunk_idx} failed: {e}")
        
        self._build_stats["l0_units_processed"] += len(l0_units)
        return results
    
    
    # Dataset-specific handling used by the reproduction workflow.
    
    
    def aggregate_l2_from_l1(self,
                             l1_results: List[L1ExtractionResult],
                             sample_id: str = "sample_default",
                             participants: Optional[List[str]] = None,
                             custom_prompt: Optional[str] = None) -> Optional[L2AggregationResult]:
        """Args: sample_id: Sample ID Returns:."""
        if not l1_results:
            logger.warning("No L1 results; skipping L2 aggregation")
            return None
        
        self._ensure_llm_client()
        
        logger.info(f"Starting L2 aggregation: sample={sample_id}, L1_count={len(l1_results)}")
        
        participants = participants or ["Speaker_A", "Speaker_B"]
        
        if self.config.extraction_style == ExtractionStyle.LOCOMO.value:
            return self._aggregate_l2_locomo_style(l1_results, sample_id, participants, custom_prompt)
        elif self.config.extraction_style == ExtractionStyle.LONGMEMEVAL.value:
            return self._aggregate_l2_longmemeval_style(l1_results, sample_id, custom_prompt)
        else:
            return self._aggregate_l2_default_style(l1_results, sample_id, custom_prompt)
    
    def _aggregate_l2_default_style(self,
                                     l1_results: List[L1ExtractionResult],
                                     sample_id: str,
                                     custom_prompt: Optional[str]) -> Optional[L2AggregationResult]:
        """Run aggregate L2 default style."""
        summaries_text = "\n\n".join([
            f"[{r.summary_type} - {r.session_id}]: {r.content}"
            for r in l1_results
        ])
        
        prompt = self.prompt_manager.get_l2_prompt(
            extraction_style="default",
            summaries=summaries_text,
            custom_prompt=custom_prompt
        )
        
        response = self._call_llm_with_retry(
            prompt=prompt,
            temperature=self.config.l2_temperature,
            max_tokens=self.config.l2_max_tokens,
            context_id=f"l2_{sample_id}"
        )
        
        if not response:
            return None
        
        uid = f"{sample_id}_L2_insight_{int(datetime.now().timestamp())}"
        result = L2AggregationResult(
            sample_id=sample_id,
            content=response.strip(),
            unit_uid=uid,
            source_l1_uids=[r.unit_uid for r in l1_results],
            metadata={
                "extraction_style": "default",
                "created_at": datetime.now().isoformat()
            },
            is_structured=False
        )
        
        self._build_stats["l2_insights_generated"] += 1
        logger.info(f"L2 aggregation completed: {uid}")
        return result
    
    def _aggregate_l2_locomo_style(self,
                                    l1_results: List[L1ExtractionResult],
                                    sample_id: str,
                                    participants: List[str],
                                    custom_prompt: Optional[str]) -> Optional[L2AggregationResult]:
        """Run aggregate L2 locomo style."""
        session_data_lines = []
        dates = []
        
        for r in l1_results:
            parsed = self._safe_parse_json(r.content)
            if parsed:
                session_date = parsed.get("session_date", "unknown")
                if session_date and session_date != "unknown":
                    dates.append(session_date)
                
                session_data_lines.append(f"\n--- {r.session_id} ({session_date}) ---")
                session_data_lines.append(f"Topic: {parsed.get('session_topic', 'General')}")
                
                events = parsed.get("structured_events", [])
                if events:
                    session_data_lines.append("Events:")
                    for e in events[:10]:
                        session_data_lines.append(
                            f"  - {e.get('event_name', 'Unknown')} "
                            f"[{e.get('event_type', 'Unknown')}] "
                            f"on {e.get('date', 'unknown')}"
                        )
                
                state_updates = parsed.get("state_updates", [])
                if state_updates:
                    session_data_lines.append("State Changes:")
                    for s in state_updates[:5]:
                        session_data_lines.append(
                            f"  - {s.get('entity', 'Unknown')}'s {s.get('attribute', 'status')}: "
                            f"{s.get('old_value', '?')} -> {s.get('new_value', '?')}"
                        )

                countables = parsed.get("countable_items", [])
                if countables:
                    session_data_lines.append("Countable Items:")
                    for c in countables[:8]:
                        session_data_lines.append(
                            f"  - {c.get('by_whom', 'Someone')} {c.get('action', 'did')} "
                            f"{c.get('item_name', 'something')} ({c.get('category', 'Other')})"
                        )

                key_facts = parsed.get("key_facts", [])
                if key_facts:
                    session_data_lines.append("Key Facts:")
                    for fact in key_facts[:5]:
                        session_data_lines.append(
                            f"  - [{fact.get('fact_type', 'Fact')}] {fact.get('subject', 'Unknown')}: "
                            f"{fact.get('fact', '')}"
                        )
            else:
                session_data_lines.append(f"\n--- {r.session_id} ---")
                session_data_lines.append(r.content[:500])
        
        first_date = min(dates) if dates else "unknown"
        last_date = max(dates) if dates else "unknown"
        time_range = f"{first_date} to {last_date}"
        
        prompt = self.prompt_manager.get_l2_prompt(
            extraction_style="locomo",
            sample_id=sample_id,
            total_sessions=len(l1_results),
            participants=participants,
            time_range=time_range,
            session_data="\n".join(session_data_lines),
            first_session_date=first_date,
            last_session_date=last_date,
            aggregation_time=datetime.now().isoformat(),
            custom_prompt=custom_prompt
        )
        
        response = self._call_llm_with_retry(
            prompt=prompt,
            temperature=self.config.l2_temperature,
            max_tokens=self.config.l2_max_tokens * 2,
            context_id=f"l2_locomo_{sample_id}"
        )
        
        if not response:
            return None

        if not self._safe_parse_json(response):
            raise ValueError(
                f"LoCoMo L2 aggregation returned invalid JSON for {sample_id}; "
                f"response_length={len(response)}"
            )
        
        uid = f"{sample_id}_L2_aggregation_{int(datetime.now().timestamp())}"
        result = L2AggregationResult(
            sample_id=sample_id,
            content=response.strip(),
            unit_uid=uid,
            source_l1_uids=[r.unit_uid for r in l1_results],
            metadata={
                "extraction_style": "locomo",
                "time_range": time_range,
                "total_sessions": len(l1_results),
                "created_at": datetime.now().isoformat()
            },
            is_structured=True
        )
        
        self._build_stats["l2_insights_generated"] += 1
        logger.info(f"L2 LoCoMo aggregation completed: {uid}")
        return result
    
    def _aggregate_l2_longmemeval_style(self,
                                         l1_results: List[L1ExtractionResult],
                                         sample_id: str,
                                         custom_prompt: Optional[str]) -> Optional[L2AggregationResult]:
        """Run aggregate L2 longmemeval style."""
        chunk_summaries = []
        for r in l1_results:
            parsed = self._safe_parse_json(r.content)
            if parsed:
                summary = parsed.get("summary", r.content[:200])
                chunk_summaries.append(f"Chunk {r.metadata.get('chunk_index', '?')}: {summary}")
            else:
                chunk_summaries.append(f"Chunk: {r.content[:200]}")
        
        prompt = self.prompt_manager.get_l2_prompt(
            extraction_style="longmemeval",
            total_chunks=len(l1_results),
            chunk_summaries="\n".join(chunk_summaries),
            custom_prompt=custom_prompt
        )
        
        response = self._call_llm_with_retry(
            prompt=prompt,
            temperature=self.config.l2_temperature,
            max_tokens=self.config.l2_max_tokens,
            context_id=f"l2_longmemeval_{sample_id}"
        )
        
        if not response:
            return None
        
        uid = f"{sample_id}_L2_synthesis_{int(datetime.now().timestamp())}"
        result = L2AggregationResult(
            sample_id=sample_id,
            content=response.strip(),
            unit_uid=uid,
            source_l1_uids=[r.unit_uid for r in l1_results],
            metadata={
                "extraction_style": "longmemeval",
                "total_chunks": len(l1_results),
                "created_at": datetime.now().isoformat()
            },
            is_structured=True
        )
        
        self._build_stats["l2_insights_generated"] += 1
        logger.info(f"L2 LongMemEval aggregation completed: {uid}")
        return result
    
    
    
    
    def deduplicate_l1(self,
                       l1_results: List[L1ExtractionResult],
                       custom_prompt: Optional[str] = None) -> List[L1ExtractionResult]:
        """Deduplicate L1."""
        if not self.config.enable_deduplication or len(l1_results) <= 1:
            return l1_results
        
        self._ensure_llm_client()
        
        logger.info(f"Starting L1 deduplication: {len(l1_results)} items")
        
        by_type: Dict[str, List[L1ExtractionResult]] = {}
        for r in l1_results:
            if r.summary_type not in by_type:
                by_type[r.summary_type] = []
            by_type[r.summary_type].append(r)
        
        deduplicated = []
        
        for summary_type, results in by_type.items():
            if len(results) <= 1:
                deduplicated.extend(results)
                continue
            
            summaries_json = json.dumps([
                {"id": r.unit_uid, "content": r.content[:500]}
                for r in results
            ], ensure_ascii=False, indent=2)
            
            prompt = self.prompt_manager.get_deduplication_prompt(
                summaries=summaries_json,
                custom_prompt=custom_prompt
            )
            
            response = self._call_llm_with_retry(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000,
                context_id=f"dedup_{summary_type}"
            )
            
            if not response:
                deduplicated.extend(results)
                continue
            
            parsed = self._safe_parse_json(response)
            if not parsed:
                deduplicated.extend(results)
                continue
            
            kept_ids = set()
            
            for item in parsed.get("unique_items", []):
                if isinstance(item, str):
                    kept_ids.add(item)
            
            for merged in parsed.get("merged_items", []):
                merged_from = merged.get("merged_from", [])
                if merged_from:
                    kept_ids.add(merged_from[0])
                    self._build_stats["dedup_merged"] += len(merged_from) - 1
            
            if not kept_ids:
                deduplicated.extend(results)
            else:
                for r in results:
                    if r.unit_uid in kept_ids:
                        deduplicated.append(r)
        
        logger.info(f"L1 deduplication completed: {len(l1_results)} -> {len(deduplicated)}")
        return deduplicated
    
    
    
    
    def add_to_semantic_system(self,
                               l1_results: Optional[List[L1ExtractionResult]] = None,
                               l2_result: Optional[L2AggregationResult] = None,
                               rebuild_index: bool = True,
                               graph_writer: Optional[GraphWriteQueue] = None,
                               wait_for_completion: bool = True) -> Dict[str, int]:
        """Add to semantic system."""
        if self.semantic_system is None:
            raise RuntimeError("Semantic system is not initialized")
        
        from ..core.memory_unit import MemoryUnit
        
        stats = {"l1_added": 0, "l2_added": 0}
        write_requests: List[GraphWriteRequest] = []
        
        if l1_results:
            for r in l1_results:
                try:
                    unit = MemoryUnit(
                        uid=r.unit_uid,
                        raw_data={
                            "text_content": r.content,
                            "summary_type": r.summary_type,
                            "session_id": r.session_id,
                            "is_structured": r.is_structured
                        },
                        metadata={
                            "layer": "L1",
                            **r.metadata,
                            "source_unit_uids": r.source_unit_uids
                        }
                    )
                    
                    if r.is_structured:
                        parsed = self._safe_parse_json(r.content)
                        if parsed:
                            retrieval_text = self._build_retrieval_text_from_structured(parsed, r.summary_type)
                        else:
                            retrieval_text = r.content
                    else:
                        retrieval_text = r.content
                    
                    space_name = self.config.l1_space_name
                    
                    write_requests.append(GraphWriteRequest(
                        unit=unit,
                        explicit_content_for_embedding=retrieval_text,
                        content_type_for_embedding="text",
                        space_names=[space_name],
                        index_update_mode="none",
                        generate_sparse_embedding=False,
                        source="hierarchical_l1",
                        metadata={"unit_uid": r.unit_uid, "summary_type": r.summary_type},
                    ))
                    
                    stats["l1_added"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to add L1 unit {r.unit_uid}: {e}")
        
        if l2_result:
            try:
                unit = MemoryUnit(
                    uid=l2_result.unit_uid,
                    raw_data={
                        "text_content": l2_result.content,
                        "sample_id": l2_result.sample_id,
                        "is_structured": l2_result.is_structured
                    },
                    metadata={
                        "layer": "L2",
                        **l2_result.metadata,
                        "source_l1_uids": l2_result.source_l1_uids
                    }
                )
                
                if l2_result.is_structured:
                    parsed = self._safe_parse_json(l2_result.content)
                    if parsed:
                        retrieval_text = self._build_retrieval_text_from_l2(parsed)
                    else:
                        retrieval_text = l2_result.content
                else:
                    retrieval_text = l2_result.content
                
                write_requests.append(GraphWriteRequest(
                    unit=unit,
                    explicit_content_for_embedding=retrieval_text,
                    content_type_for_embedding="text",
                    space_names=[self.config.l2_space_name],
                    index_update_mode="none",
                    generate_sparse_embedding=False,
                    source="hierarchical_l2",
                    metadata={"unit_uid": l2_result.unit_uid, "sample_id": l2_result.sample_id},
                ))
                
                stats["l2_added"] += 1
                
            except Exception as e:
                logger.error(f"Failed to add L2 unit {l2_result.unit_uid}: {e}")
        
        if write_requests:
            dispatch_graph_write_requests(
                semantic_system=self.semantic_system,
                requests=write_requests,
                graph_writer=graph_writer,
                wait_for_completion=wait_for_completion or rebuild_index,
            )

        
        if rebuild_index:
            if hasattr(self.semantic_system, 'build_faiss_index'):
                self.semantic_system.build_faiss_index()
            elif hasattr(self.semantic_system, 'build_semantic_map_index'):
                self.semantic_system.build_semantic_map_index()
        
        logger.info(f"Added to semantic system: L1={stats['l1_added']}, L2={stats['l2_added']}")
        return stats
    
    def _build_retrieval_text_from_structured(self, parsed: Dict, summary_type: str) -> str:
        """Build retrieval text from structured."""
        parts = []
        
        if summary_type == L1SummaryType.STRUCTURED.value:
            parts.append(parsed.get("session_topic", ""))
            
            for event in parsed.get("structured_events", [])[:5]:
                parts.append(event.get("event_name", ""))
            
            for fact in parsed.get("key_facts", [])[:5]:
                parts.append(fact.get("fact", ""))
        else:
            parts.append(parsed.get("summary", ""))
            
            for fact in parsed.get("key_facts", [])[:5]:
                parts.append(fact.get("fact", ""))
            
            for topic in parsed.get("main_topics", [])[:5]:
                parts.append(topic)
        
        return " ".join(
            text for text in (self._stringify_retrieval_part(part) for part in parts) if text
        )
    
    def _build_retrieval_text_from_l2(self, parsed: Dict) -> str:
        """Build retrieval text from L2."""
        parts = []
        
        if "global_statistics" in parsed:
            parts.append(f"Total events: {parsed.get('global_statistics', {}).get('total_unique_events', 0)}")
            
            for snapshot in parsed.get("character_status_snapshot", [])[:3]:
                person = snapshot.get("person", "")
                status = snapshot.get("status_at_end", {})
                parts.append(f"{person}: {status.get('job', '')} at {status.get('location', '')}")
            
            for insight in parsed.get("cross_session_insights", [])[:3]:
                parts.append(insight)
        
        elif "narrative_summary" in parsed:
            parts.append(parsed.get("narrative_summary", ""))
            
            for fact in parsed.get("critical_facts", [])[:5]:
                parts.append(fact.get("fact", ""))
        
        return " ".join(
            text for text in (self._stringify_retrieval_part(part) for part in parts) if text
        )

    @staticmethod
    def _stringify_retrieval_part(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)
    
    
    
    
    def run_full_pipeline(self,
                          l0_units: List["MemoryUnit"],
                          session_id: str = "session_default",
                          sample_id: Optional[str] = None,
                          session_date: Optional[str] = None,
                          participants: Optional[List[str]] = None,
                          build_l1: bool = True,
                          build_l2: bool = True,
                          add_to_system: bool = True,
                          custom_prompts: Optional[Dict[str, str]] = None,
                          graph_writer: Optional[GraphWriteQueue] = None) -> Dict[str, Any]:
        """Args: session_id: Session ID Returns:."""
        start_time = datetime.now()
        sample_id = sample_id or session_id
        
        logger.info(f"\n{'='*60}")
        logger.info("Hierarchical memory build pipeline")
        logger.info(f"   - extraction_style: {self.config.extraction_style}")
        logger.info(f"   - Session: {session_id}")
        logger.info(f"   - L0 units: {len(l0_units)}")
        logger.info(f"{'='*60}")
        
        result = {
            "session_id": session_id,
            "sample_id": sample_id,
            "success": False,
            "l0_enhanced": [],
            "l1_results": [],
            "l2_result": None,
            "stats": {},
            "errors": [],
            "build_time": 0
        }
        
        try:
            if self.config.enable_contextual_retrieval:
                result["l0_enhanced"] = self.enhance_l0_with_context(
                    l0_units=l0_units,
                    session_date=session_date,
                    participants=participants,
                    custom_prompt=custom_prompts.get("contextual_retrieval") if custom_prompts else None
                )
            
            if build_l1:
                l1_results = self.extract_l1_from_l0_units(
                    l0_units=l0_units,
                    session_id=session_id,
                    session_date=session_date,
                    participants=participants,
                    custom_prompts=custom_prompts
                )
                
                if self.config.enable_deduplication:
                    l1_results = self.deduplicate_l1(
                        l1_results=l1_results,
                        custom_prompt=custom_prompts.get("deduplication") if custom_prompts else None
                    )
                
                result["l1_results"] = l1_results
            
            if build_l2 and result["l1_results"]:
                result["l2_result"] = self.aggregate_l2_from_l1(
                    l1_results=result["l1_results"],
                    sample_id=sample_id,
                    participants=participants,
                    custom_prompt=custom_prompts.get("l2") if custom_prompts else None
                )
            
            if add_to_system and self.semantic_system is not None:
                add_stats = self.add_to_semantic_system(
                    l1_results=result["l1_results"],
                    l2_result=result["l2_result"],
                    graph_writer=graph_writer,
                    wait_for_completion=graph_writer is None,
                )
                result["stats"]["added"] = add_stats
            
            result["success"] = True
            self._build_stats["sessions_processed"] += 1
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            result["errors"].append(str(e))
        
        result["build_time"] = (datetime.now() - start_time).total_seconds()
        result["stats"]["build"] = self._build_stats.copy()
        
        logger.info(f"\n{'='*60}")
        logger.info("Pipeline completed")
        logger.info(f"   - L1 summaries: {len(result['l1_results'])}")
        logger.info(f"   - L2 insight: {'yes' if result['l2_result'] else 'no'}")
        logger.info(f"   - elapsed: {result['build_time']:.2f}s")
        logger.info(f"{'='*60}\n")
        
        return result
    
    
    
    
    
    def add_unit_with_session(self,
                             unit: "MemoryUnit",
                             session_id: str = "session_default",
                             session_type: str = "default",
                             session_metadata: Optional[Dict] = None,
                             **add_unit_kwargs) -> bool:
        """Args: session_id: Session ID Returns:."""
        if self.semantic_system is None:
            logger.error("Semantic system is not initialized")
            return False
        
        try:
            if unit.metadata is None:
                unit.metadata = {}
            
            unit.metadata.update({
                "session_id": session_id,
                "session_type": session_type,
                "session_metadata": session_metadata or {},
                "added_at": datetime.now().isoformat()
            })
            
            l0_space_name = self.config.l0_space_name
            space_names = add_unit_kwargs.get("space_names", [])
            if l0_space_name not in space_names:
                space_names.append(l0_space_name)
            add_unit_kwargs["space_names"] = space_names
            
            dispatch_graph_write_requests(
                semantic_system=self.semantic_system,
                requests=[GraphWriteRequest(
                    unit=unit,
                    explicit_content_for_embedding=add_unit_kwargs.get("explicit_content_for_embedding"),
                    content_type_for_embedding=add_unit_kwargs.get("content_type_for_embedding"),
                    space_names=space_names,
                    index_update_mode=add_unit_kwargs.get("index_update_mode", "incremental"),
                    generate_sparse_embedding=add_unit_kwargs.get("generate_sparse_embedding", True),
                    sparse_model_name=add_unit_kwargs.get("sparse_model_name", "naver/splade-v3"),
                    source="hierarchical_session_l0",
                    metadata={"session_id": session_id},
                )],
            )
            
            self.session_tracker.track_unit(session_id, unit.uid)
            
            if not self.session_tracker.get_session(session_id):
                self.session_tracker.create_session(
                    session_id=session_id,
                    session_type=session_type,
                    metadata=session_metadata
                )
            
            self._build_stats["l0_units_processed"] += 1
            
            logger.debug(f"Unit {unit.uid} added to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add unit to session: {e}")
            return False
    
    def finalize_and_build_session(self,
                                   session_id: str,
                                   build_l1: bool = True,
                                   build_l2: bool = False,
                                   l1_summary_types: Optional[List[str]] = None,
                                   custom_prompts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Args: session_id: Session ID Returns:."""
        if self.semantic_system is None:
            return {"session_id": session_id, "success": False, "errors": ["Semantic system is not initialized"]}
        
        session = self.session_tracker.get_session(session_id)
        if not session:
            return {"session_id": session_id, "success": False, "errors": [f"Session {session_id} does not exist"]}
        
        l0_unit_uids = self.session_tracker.get_session_units(session_id)
        if not l0_unit_uids:
            return {"session_id": session_id, "success": False, "errors": [f"Session {session_id} has no units"]}
        
        l0_units = [self.semantic_system.get_unit(uid) for uid in l0_unit_uids]
        l0_units = [u for u in l0_units if u is not None]
        
        if l1_summary_types:
            old_types = self.config.l1_summary_types
            self.config.l1_summary_types = l1_summary_types
        
        result = self.run_full_pipeline(
            l0_units=l0_units,
            session_id=session_id,
            build_l1=build_l1,
            build_l2=build_l2,
            add_to_system=True,
            custom_prompts=custom_prompts
        )
        
        if l1_summary_types:
            self.config.l1_summary_types = old_types
        
        self.session_tracker.finalize_session(session_id)
        
        
        return {
            "session_id": session_id,
            "success": result["success"],
            "l0_units": [u.uid for u in l0_units],
            "l1_summaries": {r.summary_type: r.unit_uid for r in result["l1_results"]},
            "l2_insights": [result["l2_result"].unit_uid] if result["l2_result"] else [],
            "errors": result["errors"],
            "build_time": result["build_time"]
        }
    
    def _merge_l0_content(self, l0_units: List["MemoryUnit"]) -> str:
        """Run merge L0 content."""
        return build_l0_inference_context(l0_units)
    
    def get_session_stats(self, session_id: Optional[str] = None) -> Dict:
        """Return session stats."""
        if session_id:
            session = self.session_tracker.get_session(session_id)
            if not session:
                return {"error": f"Session {session_id} does not exist"}
            
            return {
                "session_id": session.session_id,
                "session_type": session.session_type,
                "unit_count": session.get_unit_count(),
                "is_finalized": session.is_finalized,
                "created_at": session.created_at.isoformat(),
                "finalized_at": session.finalized_at.isoformat() if session.finalized_at else None
            }
        else:
            return self.session_tracker.get_stats()
    
    def get_build_stats(self) -> Dict:
        """Return build stats."""
        return self._build_stats.copy()



HierarchicalPromptTemplateManager = HierarchicalPromptManager
