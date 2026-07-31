"""Core memory objects and lazy semantic-index entry points."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .memory_space import MemorySpace
from .memory_space_registry import MemorySpaceRegistry, TowerSpace
from .memory_unit import MemoryUnit

if TYPE_CHECKING:
    from .semantic_graph import SemanticGraph
    from .semantic_map import SemanticMap

    MemorySystemType = Union[SemanticMap, SemanticGraph]
else:
    MemorySystemType = Any

__version__ = "0.1.0a2"
__author__ = "Mandol Team"

DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_COLLECTION_NAME = "semantic_memory_units"
DEFAULT_TEXT_MODEL = "BAAI/bge-m3"
DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"

MODEL_PRESETS = {
    "bge-m3": {"text_model": "BAAI/bge-m3", "embedding_dim": 1024},
    "qwen-0.6b": {
        "text_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dim": 1024,
    },
    "qwen-4b": {
        "text_model": "Qwen/Qwen3-Embedding-4B",
        "embedding_dim": 2560,
    },
    "qwen-8b": {
        "text_model": "Qwen/Qwen3-Embedding-8B",
        "embedding_dim": 4096,
    },
    "mini": {
        "text_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
    },
}

_LAZY_EXPORTS = {
    "SemanticMap": (".semantic_map", "SemanticMap"),
    "SemanticGraph": (".semantic_graph", "SemanticGraph"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def get_default_core_config() -> Dict[str, Any]:
    """Return the default core configuration."""
    return {
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "collection_name": DEFAULT_COLLECTION_NAME,
        "text_model": DEFAULT_TEXT_MODEL,
        "image_model": DEFAULT_IMAGE_MODEL,
        "faiss_index_type": "IDMap,Flat",
    }


def get_model_preset(preset_name: str) -> Dict[str, Any]:
    """Return a copy of a named model preset."""
    return MODEL_PRESETS.get(preset_name, MODEL_PRESETS["bge-m3"]).copy()


def create_semantic_map(
    text_model: Optional[str] = None,
    image_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    preset: Optional[str] = None,
    **kwargs: Any,
):
    """Create a :class:`SemanticMap`, importing its heavy stack on demand."""
    if preset:
        preset_config = get_model_preset(preset)
        text_model = text_model or preset_config["text_model"]
        embedding_dim = embedding_dim or preset_config["embedding_dim"]

    text_model = text_model or DEFAULT_TEXT_MODEL
    if image_model is not None:
        from ..utils.logging_config import create_module_logger

        create_module_logger("core").warning(
            "create_semantic_map(image_model=...) is deprecated; "
            "multimodal support is now handled by the primary embedding model."
        )

    semantic_map_cls = __getattr__("SemanticMap")
    return semantic_map_cls(
        embedding_model_name=text_model,
        embedding_dim=embedding_dim,
        **kwargs,
    )


def create_semantic_graph(
    semantic_map_instance=None,
    preset: Optional[str] = None,
    **kwargs: Any,
):
    """Create a :class:`SemanticGraph`, importing its heavy stack on demand."""
    if semantic_map_instance is None:
        semantic_map_instance = create_semantic_map(preset=preset, **kwargs)
    semantic_graph_cls = __getattr__("SemanticGraph")
    return semantic_graph_cls(semantic_map_instance=semantic_map_instance)


def create_memory_unit(
    uid: str,
    content: Any,
    content_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> MemoryUnit:
    """Create a memory unit from a primary content value."""
    raw_data = kwargs.copy()
    if content_type == "text":
        raw_data["text_content"] = content
    elif content_type == "image_path":
        raw_data["image_path"] = content
    else:
        raw_data[content_type] = content
    return MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)


def create_memory_space(
    name: str,
    faiss_index_type: Optional[str] = None,
) -> MemorySpace:
    """Create a memory space."""
    return MemorySpace(ms_name=name, faiss_index_type=faiss_index_type)


def get_core_component_status() -> Dict[str, Any]:
    """Report component and dependency availability without importing them."""
    return {
        "components": {
            "memory_unit": True,
            "memory_space": True,
            "semantic_map": True,
            "semantic_graph": True,
        },
        "default_config": get_default_core_config(),
        "available_presets": list(MODEL_PRESETS),
        "dependencies": {
            "faiss": find_spec("faiss") is not None,
            "networkx": find_spec("networkx") is not None,
            "sentence_transformers": find_spec("sentence_transformers") is not None,
            "numpy": find_spec("numpy") is not None,
        },
    }


def validate_embedding_dim(embedding_dim: int) -> bool:
    """Return whether an embedding dimension is in the supported range."""
    return 0 < embedding_dim <= 8192


def get_recommended_faiss_index(num_vectors: int, embedding_dim: int) -> str:
    """Recommend an index shape using the existing vector-count thresholds."""
    del embedding_dim
    if num_vectors < 1_000:
        return "IDMap,Flat"
    if num_vectors < 100_000:
        return "IDMap,IVF100,Flat"
    return "IDMap,IVF1000,Flat"


__all__ = [
    "MemoryUnit",
    "MemorySpace",
    "MemorySpaceRegistry",
    "TowerSpace",
    "SemanticMap",
    "SemanticGraph",
    "MemorySystemType",
    "create_semantic_map",
    "create_semantic_graph",
    "create_memory_unit",
    "create_memory_space",
    "get_default_core_config",
    "get_model_preset",
    "get_core_component_status",
    "validate_embedding_dim",
    "get_recommended_faiss_index",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_TEXT_MODEL",
    "DEFAULT_IMAGE_MODEL",
    "MODEL_PRESETS",
]
