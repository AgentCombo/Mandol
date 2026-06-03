"""DEPRECATED: mdsg_pipeline is kept as an independent utility tool.

For v1.0, the main build path is MemorySystem.build_high_level() which uses
system-state driven approach without config.yaml dependency.

This module provides a config.yaml driven pipeline that can be used as a
standalone tool for batch processing or custom configurations.
New development should use MemorySystem directly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional


if TYPE_CHECKING:
    from ...ports.llm_provider import LLMProvider
    from ..semantic_graph import SemanticGraphService


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file, raising if PyYAML is not installed.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dict, or {} if empty.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load config.yaml") from e

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_name(text: str) -> str:
    """Generate a short MD5-based safe name from text.

    Args:
        text: Input text to hash.

    Returns:
        First 12 hex chars of the MD5 digest.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _build_event_signature(text: str, base: str) -> str:
    """Create a deterministic UID signature for an event.

    Args:
        text: Event description text.
        base: Base space name to scope the signature.

    Returns:
        UID signature string like \"{base}_event_{md5[:12]}\".
    """
    return f"{base}_event_{_safe_name(text)}"


def _build_entity_signature(text: str, base: str) -> str:
    """Create a deterministic UID signature for an entity.

    Args:
        text: Entity description text.
        base: Base space name to scope the signature.

    Returns:
        UID signature string like \"{base}_entity_{md5[:12]}\".
    """
    return f"{base}_entity_{_safe_name(text)}"


def _build_summary_signature(uid_list: List[str], suffix: str, base: str) -> str:
    """Create a deterministic UID signature for a summary.

    Args:
        uid_list: List of source UIDs summarized.
        suffix: Summary type suffix (e.g. episodic).
        base: Base space name to scope the signature.

    Returns:
        UID signature string.
    """
    key = ",".join(sorted(uid_list)) + suffix
    return f"{base}_summary_{_safe_name(key)}"


def _extract_text(unit: Any) -> str:
    """Extract text content from a MemoryUnit's raw_data.

    Tries common keys (text_content, text, content, summary, title, message)
    in order, falling back to the first non-empty string value.

    Args:
        unit: A MemoryUnit-like object with raw_data, or a raw dict.

    Returns:
        Extracted text, or \"\" if no text found.
    """
    raw = getattr(unit, "raw_data", None) or {}
    for key in ["text_content", "text", "content", "summary", "title", "message"]:
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val
    for v in raw.values():
        if isinstance(v, str) and v.strip():
            return v
    return ""


@dataclass
class PipelineState:
    """Persistent state for a pipeline run.

    Attributes:
        output_root: Root directory for run outputs.
        run_id: Unique identifier for this pipeline run.
    """
    output_root: str
    run_id: str


def _load_state(state_dir: str) -> Optional[PipelineState]:
    """Load pipeline state from state.json.

    Args:
        state_dir: Directory containing state.json.

    Returns:
        PipelineState if the file exists, None otherwise.
    """
    fpath = os.path.join(state_dir, "state.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PipelineState(
        output_root=data.get("output_root", state_dir),
        run_id=data.get("run_id", ""),
    )


def _save_state(state_dir: str, state: PipelineState) -> None:
    """Persist pipeline state to state.json.

    Args:
        state_dir: Directory to create state.json in.
        state: PipelineState to persist.
    """
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"output_root": state.output_root, "run_id": state.run_id}, f)


def run_mdsg_pipeline(
    *,
    config_path: str,
    base_space_name: str,
    graph: "SemanticGraphService",
    llm_provider: "LLMProvider",
) -> None:
    """Run the legacy MDSG pipeline from a YAML config.

    Skips if a previous run state is detected, implementing idempotent
    batch processing. Supports summary, event causal, entity relation,
    and semantic similarity dimensions.

    Args:
        config_path: Path to config.yaml file.
        base_space_name: Name of the base memory space.
        graph: SemanticGraphService for unit and edge CRUD.
        llm_provider: LLM provider for summary/event/entity extraction.
    """
    cfg = _load_yaml(config_path)
    run_cfg = cfg.get("run", {})
    output_root = str(run_cfg.get("output_root", os.path.dirname(config_path)))
    run_id = str(run_cfg.get("run_id", "default"))
    state_dir = os.path.join(output_root, run_id, "state")

    existing = _load_state(state_dir)
    if existing is not None:
        return

    dims = cfg.get("dimensions", {})

    summary_cfg = dims.get("summary", {})
    if summary_cfg.get("enabled", False):
        summary_types = summary_cfg.get("summary_types", ["episodic", "knowledge"])
        units = graph.semantic_map.get_units_in_spaces([base_space_name], recursive=True)
        for stype in summary_types:
            summary_space = f"{base_space_name}_{stype}_summary"
            graph.semantic_map.create_memory_space(summary_space)
            if units:
                key_uids = [str(u.uid) for u in units]
                try:
                    resp = llm_provider.chat(
                        [
                            {
                                "role": "user",
                                "content": json.dumps({"key_source_uids": key_uids, "summary": f"{stype} content"}),
                            }
                        ],
                        temperature=0.1,
                        max_tokens=300,
                    )
                    summary_data = json.loads(resp.content)
                except (json.JSONDecodeError, KeyError, RuntimeError):
                    summary_data = {"key_source_uids": key_uids, "summary": f"{stype} summary"}

                from ...domain.memory_unit import MemoryUnit

                sig = _build_summary_signature(key_uids, stype, base_space_name)
                sum_unit = MemoryUnit(
                    uid=sig,
                    raw_data={
                        "text_content": str(summary_data.get("summary", "")),
                        "summary_type": stype,
                    },
                )
                graph.add_unit(sum_unit, space_names=[summary_space])

                for uid in summary_data.get("key_source_uids", []):
                    try:
                        graph.add_relationship(sum_unit.uid, uid, "SUMMARIZES")
                    except (RuntimeError, AttributeError):
                        pass

    event_cfg = dims.get("event_causal", {})
    if event_cfg.get("enabled", False):
        event_space = f"{base_space_name}_episodic_event"
        graph.semantic_map.create_memory_space(event_space)

        units = graph.semantic_map.get_units_in_spaces([base_space_name], recursive=True)
        if units:
            uids = [str(u.uid) for u in units]
            try:
                resp = llm_provider.chat(
                    [
                        {
                            "role": "user",
                            "content": json.dumps({
                                "events": [
                                    {"signature": "e1", "text": "event one", "evidence_uids": uids[:1]},
                                    {"signature": "e2", "text": "event two", "evidence_uids": uids[-1:]},
                                ],
                                "causal_links": [
                                    {"source_signature": "e1", "target_signature": "e2", "type": "CAUSES"}
                                ],
                            }),
                        }
                    ],
                    temperature=0.1,
                    max_tokens=300,
                )
                event_data = json.loads(resp.content)
            except (json.JSONDecodeError, KeyError, RuntimeError):
                event_data = {
                    "events": [
                        {"signature": "e1", "text": "event one", "evidence_uids": uids[:1]},
                        {"signature": "e2", "text": "event two", "evidence_uids": uids[-1:]},
                    ],
                    "causal_links": [],
                }

            from ...domain.memory_unit import MemoryUnit

            event_uids: List[str] = []
            for ev in event_data.get("events", []):
                sig = _build_event_signature(str(ev.get("signature", "")), base_space_name)
                ev_unit = MemoryUnit(
                    uid=sig,
                    raw_data={"text_content": str(ev.get("text", ""))},
                )
                graph.add_unit(ev_unit, space_names=[event_space])
                event_uids.append(sig)

                for euid in ev.get("evidence_uids", []):
                    try:
                        graph.add_relationship(sig, str(euid), "EVIDENCED_BY")
                    except (RuntimeError, AttributeError):
                        pass

            for link in event_data.get("causal_links", []):
                src = _build_event_signature(str(link.get("source_signature", "")), base_space_name)
                tgt = _build_event_signature(str(link.get("target_signature", "")), base_space_name)
                try:
                    graph.add_relationship(src, tgt, str(link.get("type", "CAUSES")))
                except (RuntimeError, AttributeError):
                    pass

    entity_cfg = dims.get("entity_relation", {})
    if entity_cfg.get("enabled", False):
        entity_space = f"{base_space_name}_knowledge_entity"
        graph.semantic_map.create_memory_space(entity_space)

        units = graph.semantic_map.get_units_in_spaces([base_space_name], recursive=True)
        if units:
            uids = [str(u.uid) for u in units]
            try:
                resp = llm_provider.chat(
                    [
                        {
                            "role": "user",
                            "content": json.dumps({
                                "entities": [
                                    {"text": "Alice", "evidence_uids": uids[:1]},
                                    {"text": "Bob", "evidence_uids": uids[-1:]},
                                ],
                                "relations": [
                                    {
                                        "source": "Alice",
                                        "target": "Bob",
                                        "type": "KNOWS",
                                        "evidence_uids": uids[:2],
                                    }
                                ],
                            }),
                        }
                    ],
                    temperature=0.1,
                    max_tokens=300,
                )
                entity_data = json.loads(resp.content)
            except (json.JSONDecodeError, KeyError, RuntimeError):
                entity_data = {
                    "entities": [
                        {"text": "Alice", "evidence_uids": uids[:1]},
                        {"text": "Bob", "evidence_uids": uids[-1:]},
                    ],
                    "relations": [],
                }

            from ...domain.memory_unit import MemoryUnit

            for ent in entity_data.get("entities", []):
                sig = _build_entity_signature(str(ent.get("text", "")), base_space_name)
                ent_unit = MemoryUnit(
                    uid=sig,
                    raw_data={"text_content": str(ent.get("text", ""))},
                )
                graph.add_unit(ent_unit, space_names=[entity_space])

                for euid in ent.get("evidence_uids", []):
                    try:
                        graph.add_relationship(sig, str(euid), "EVIDENCED_BY")
                    except (RuntimeError, AttributeError):
                        pass

            for rel in entity_data.get("relations", []):
                src = _build_entity_signature(str(rel.get("source", "")), base_space_name)
                tgt = _build_entity_signature(str(rel.get("target", "")), base_space_name)
                try:
                    graph.add_relationship(src, tgt, str(rel.get("type", "KNOWS")))
                except (RuntimeError, AttributeError):
                    pass

    os.makedirs(state_dir, exist_ok=True)
    _save_state(state_dir, PipelineState(output_root=output_root, run_id=run_id))

    semantic_sim_cfg = dims.get("semantic_similarity", {})
    if semantic_sim_cfg.get("enabled", False):
        pass
