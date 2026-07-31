import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Iterable
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import orjson
import torch
from safetensors.torch import save_file as safetensors_save_file, load_file as safetensors_load_file

from .memory_unit import MemoryUnit
from .memory_space import MemorySpace
from .siliconflow_embedding_adapter import SiliconFlowEmbeddingAdapter
from ..retrieval.retrieval_interface import MultiRetrievalInterface, RetrievalInterface
from ..utils.config_manager import settings
from ..utils.model_manager import global_model_manager
from ..utils.logging_config import create_module_logger
from ..utils.optional_dependencies import is_flash_attention_available

logger = create_module_logger("semantic_map")


class SemanticMap(RetrievalInterface):
    """In-memory semantic index for Mandol memory units.

    SemanticMap stores memory units, dense embeddings, optional sparse
    embeddings, memory-space membership, and auxiliary retrieval indexes. It
    provides insertion, filtering, persistence, and similarity-based retrieval
    APIs while preserving stable UID-to-integer-ID mappings across dense and
    sparse indexes.
    """
    
    MODEL_CONFIG = {
        # Local text embedding backend with flash-attention enabled when available.
        "Qwen/Qwen3-Embedding-0.6B": {
            "dim": 1024, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "use_flash_attention": True,
            "model_kwargs": {
                "attn_implementation": "flash_attention_2", 
                "device_map": "auto",
                "torch_dtype": torch.bfloat16
            },
            "tokenizer_kwargs": {"padding_side": "left"}
        },
        "Qwen/Qwen3-Embedding-4B": {
            "dim": 2560, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "use_flash_attention": True,
            "model_kwargs": {
                "attn_implementation": "flash_attention_2", 
                "device_map": "auto",
                "torch_dtype": torch.bfloat16
            },
            "tokenizer_kwargs": {"padding_side": "left"}
        },
        "Qwen/Qwen3-Embedding-8B": {
            "dim": 4096, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "use_flash_attention": True,
            "model_kwargs": {
                "attn_implementation": "flash_attention_2", 
                "device_map": "auto",
                "torch_dtype": torch.bfloat16
            },
            "tokenizer_kwargs": {"padding_side": "left"}
        },
        
        
        "Qwen/Qwen3-Embedding-0.6B-remote": {
            "dim":1024,
            "type": "cloud",
            "modalities": ["text"],
            "provider": "siliconflow",
            "wrapper": None
        },
        "Qwen/Qwen3-Embedding-4B-remote": {
            "dim": 2560, 
            "type": "cloud", 
            "modalities": ["text"],
            "provider": "siliconflow",
            "wrapper": None
        },
        "Qwen/Qwen3-Embedding-8B-remote": {
            "dim": 4096, 
            "type": "cloud", 
            "modalities": ["text"],
            "provider": "siliconflow",
            "wrapper": None
        },
        
        "BAAI/bge-m3": {
            "dim": 1024, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "model_kwargs": {"torch_dtype": torch.bfloat16}
        },
        
        "sentence-transformers/clip-ViT-B-32-multilingual-v1": {
            "dim": 512, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "model_kwargs": {"torch_dtype": torch.bfloat16}
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dim": 384, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "model_kwargs": {"torch_dtype": torch.bfloat16}
        },
        
        "jinaai/jina-clip-v1": {
            "dim": 768,
            "wrapper": None,
            "type": "local",
            "modalities": ["text", "image"],
            "model_kwargs": {"torch_dtype": torch.bfloat16}
        },
        "jinaai/jina-clip-v2": {
            "dim": 1024,
            "wrapper": None,
            "type": "local",
            "modalities": ["text", "image"],
            "use_flash_attention": True,
            "sentence_transformer_kwargs": {"trust_remote_code": True},
            "model_kwargs": {"torch_dtype": torch.bfloat16},
            "image_encode_format": "path",
            "image_encode_kwargs": {"normalize_embeddings": True},
            "text_encode_kwargs": {"normalize_embeddings": True}
        },
        "jinaai/jina-embeddings-v4": {
            "dim": 2048,
            "wrapper": None,
            "type": "local",
            "modalities": ["text", "image"],
            "use_flash_attention": True,
            "sentence_transformer_kwargs": {"trust_remote_code": True},
            "model_kwargs": {
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": torch.bfloat16
            },
            "image_encode_format": "path",
            "image_encode_kwargs": {"task": "retrieval"},
            "text_encode_kwargs": {"task": "retrieval"}
        },
        "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct": {
            "dim": 1536,
            "wrapper": None,
            "type": "local",
            "modalities": ["text", "image"],
            "use_flash_attention": True,
            "sentence_transformer_kwargs": {"trust_remote_code": True},
            "model_kwargs": {
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": torch.float16
            },
            "image_encode_format": "dict_path"
        },
        "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct": {
            "dim": 3584,
            "wrapper": None,
            "type": "local",
            "modalities": ["text", "image"],
            "use_flash_attention": True,
            "sentence_transformer_kwargs": {"trust_remote_code": True},
            "model_kwargs": {
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": torch.float16
            },
            "image_encode_format": "dict_path"
        },
        
        "bge-m3": {
            "dim": 1024, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "model_kwargs": {"torch_dtype": torch.bfloat16}
        },
        "clip-ViT-B-32-multilingual-v1": {
            "dim": 512, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "model_kwargs": {"torch_dtype": torch.bfloat16}
        },
        "all-MiniLM-L6-v2": {
            "dim": 384, 
            "wrapper": None, 
            "type": "local",
            "modalities": ["text"],
            "model_kwargs": {"torch_dtype": torch.bfloat16}
        },
    }
    

    DEFAULT_TEXT_EMBEDDING_KEY = "text_content"
    DEFAULT_IMAGE_EMBEDDING_KEY = "image_path"
    DEFAULT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    
    def __init__(
        self,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_dim: Optional[int] = None,
        faiss_index_type: str = "IDMap,Flat",
        use_flash_attention: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize the semantic index and embedding backend.

        Args:
            embedding_model_name: Embedding model key from ``MODEL_CONFIG`` or a
                model name accepted by SentenceTransformer.
            embedding_dim: Explicit embedding dimension for models that are not
                listed in ``MODEL_CONFIG``.
            faiss_index_type: FAISS index descriptor used for dense retrieval.
            use_flash_attention: Optional override for Flash Attention usage.
                The Transformers flag is only passed when flash-attn is
                installed.
            **kwargs: Backward-compatible aliases accepted by older checkpoints,
                including ``text_embedding_model_name`` and
                ``image_embedding_model_name``.
        """
        default_embedding_model_name = self.DEFAULT_EMBEDDING_MODEL_NAME
        legacy_text_model_name = kwargs.pop("text_embedding_model_name", None)
        if legacy_text_model_name is not None:
            logger.warning(
                "text_embedding_model_name is deprecated; use embedding_model_name instead."
            )
            if embedding_model_name == default_embedding_model_name:
                embedding_model_name = legacy_text_model_name
            else:
                logger.warning(
                    "embedding_model_name was provided explicitly; ignoring deprecated text_embedding_model_name."
                )

        legacy_image_model_name = kwargs.pop("image_embedding_model_name", None)
        if legacy_image_model_name is not None:
            logger.warning(
                "image_embedding_model_name is deprecated; multimodal capability is selected by embedding_model_name."
            )

        if kwargs:
            logger.warning(f"SemanticMap.__init__ received unknown parameters and will ignore them: {sorted(kwargs.keys())}")
        
        config = self.MODEL_CONFIG.get(embedding_model_name)
        self.supported_modalities = config.get("modalities", ["text"]) if config else ["text"]
        
        if config:
            self.embedding_dim = config['dim']
        elif embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            raise ValueError(
                f"Cannot infer the embedding dimension for unknown model '{embedding_model_name}'."
                f"Pass the 'embedding_dim' argument during initialization."
            )

        
        self.faiss_index_type = faiss_index_type
        self.model = None
        self.text_model = None
        self.image_model = None
        self._embedding_model_name = embedding_model_name
        self._image_model_name = legacy_image_model_name
        self._text_encode_kwargs = dict(config.get("text_encode_kwargs", {})) if config else {}
        self._image_encode_kwargs = dict(config.get("image_encode_kwargs", {})) if config else {}
        self._image_encode_format = config.get("image_encode_format", "pil") if config else "pil"
        self._is_cloud_model = config and config.get("type") == "cloud"
        
        # Flash Attention is optional. Keep the configured preference, but only
        # pass its Transformers flag when the package is actually installed.
        if use_flash_attention is not None:
            self._flash_attention_requested = bool(use_flash_attention)
        elif config:
            self._flash_attention_requested = bool(
                config.get("use_flash_attention", False)
            )
        else:
            self._flash_attention_requested = False

        self._flash_attention_available = (
            is_flash_attention_available()
            if self._flash_attention_requested
            else False
        )
        self._use_flash_attention = (
            self._flash_attention_requested and self._flash_attention_available
        )
        if self._flash_attention_requested and not self._flash_attention_available:
            message = (
                "flash-attn is not installed; keeping the configured dtype and using the standard attention path."
            )
            if use_flash_attention is True:
                logger.warning(message)
            else:
                logger.info(message)

        
        try:
            logger.info(f"Initializing embedding model: {embedding_model_name}")
            
            if self._is_cloud_model:
                provider = config.get("provider", "siliconflow")
                if provider == "siliconflow":
                    # Cloud embedding credentials are read at construction time.
                    api_key = settings.get_api_key("SILICONFLOW_API_KEY")
                    if not api_key:
                        logger.warning("SILICONFLOW_API_KEY is not set in .env or the environment; cloud embeddings may be unavailable.")
                    
                    self.model = SiliconFlowEmbeddingAdapter(
                        model_name=embedding_model_name,
                        api_key=api_key,
                        dimensions=self.embedding_dim
                    )
                    logger.info(f"Cloud embedding adapter loaded: {embedding_model_name} (dim={self.embedding_dim})")
                    logger.info(f"   provider: {provider}")
                    logger.info("   note: each call may incur provider API cost")
                else:
                    raise ValueError(f"Unsupported cloud provider: {provider}")
            
            else:
                
                def load_text_model():
                    if config and config.get("wrapper"):
                        return config["wrapper"](embedding_model_name)
                    else:
                        sentence_transformer_kwargs = dict(
                            config.get("sentence_transformer_kwargs", {}) if config else {}
                        )
                        model_kwargs = dict(
                            config.get("model_kwargs", {}) if config else {}
                        )
                        tokenizer_kwargs = dict(
                            config.get("tokenizer_kwargs", {}) if config else {}
                        )

                        if self._use_flash_attention:
                            model_kwargs["attn_implementation"] = "flash_attention_2"
                            logger.info(f" Enabled flash_attention_2 optimization")
                            logger.info(f"   model_kwargs: {model_kwargs}")
                            logger.info(f"   tokenizer_kwargs: {tokenizer_kwargs}")
                        else:
                            model_kwargs.pop("attn_implementation", None)

                        def create_sentence_transformer():
                            return SentenceTransformer(
                                embedding_model_name,
                                model_kwargs=model_kwargs,
                                tokenizer_kwargs=tokenizer_kwargs,
                                **sentence_transformer_kwargs
                            )

                        if not self._use_flash_attention:
                            return create_sentence_transformer()

                        try:
                            return create_sentence_transformer()
                        except Exception as e:
                            logger.warning(f" flash_attention_2 loading failed: {e}")
                            logger.warning("   Falling back to standard attention while preserving dtype settings.")
                            model_kwargs.pop("attn_implementation", None)
                            return create_sentence_transformer()
                
                self.model = global_model_manager.get_or_load_model(
                    model_type="text_embedding",
                    model_name=f"{embedding_model_name}{'_flash' if self._use_flash_attention else ''}",
                    loader_func=load_text_model
                )
                
                flash_info = " (flash_attention_2)" if self._use_flash_attention else ""
                logger.info(f" Local embedding model initialized: {embedding_model_name}{flash_info}")

            self.text_model = self.model
            
        except Exception as e:
            logger.error(f" Embedding model initialization failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            raise RuntimeError(f"Failed to initialize text embedding model: {e}")
        
        
        if "image" not in self.supported_modalities:
            self.image_model = None
            logger.warning(
                "The primary embedding model does not declare image support; the separate image model path is disabled. "
                "Use a unified multimodal model whose MODEL_CONFIG modalities include 'image' for image embeddings."
            )
        else:
            self.image_model = self.model
            logger.info(
                f" Image modality reuses the primary embedding model: {embedding_model_name}"
            )

        self.memory_units: Dict[str, MemoryUnit] = {}
        self.memory_spaces: Dict[str, MemorySpace] = {}
        
        
        self._next_int_id: int = 0
        self._uid_to_int_id: Dict[str, int] = {}
        self._int_id_to_uid: Dict[int, str] = {}
        self._modified_units = set()
        self._deleted_units = set()
        self._external_storage = None
        self._storage_uids: Optional[Set[str]] = None
        self._max_memory_units = 100000
        self._access_counts = {}
        self._last_accessed = {}
        self.tiered_storage_manager = None
        self._high_level_memory_build_state: Dict[str, Any] = {}
        self._space_membership_version = 0
        self._space_filter_cache: Dict[Tuple[Tuple[str, ...], int], np.ndarray] = {}

        self._multi_retriever = None
        
        
        self._init_faiss_index()
        
        flash_status = " flash_attention_2" if self._use_flash_attention else "standard"
        logger.info(f" SemanticMap initialization complete:")
        logger.info(f"   - embedding model: {embedding_model_name} {' (cloud)' if self._is_cloud_model else f' (local, {flash_status})'} ")
        logger.info(f"   - supported modalities: {', '.join(self.supported_modalities)}")
        logger.info(f"   - image model: {'reuses primary model ' if self.image_model else ' (disabled/unavailable)'}")
        logger.info(f"   - embedding dimension: {self.embedding_dim}")
        logger.info(f"   - FAISS index: {faiss_index_type}")

    def set_high_level_memory_build_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Attach resumable high-level-memory build metadata to this map."""
        self._high_level_memory_build_state = dict(state or {})

    def get_high_level_memory_build_state(self) -> Dict[str, Any]:
        """Return the persisted high-level-memory build metadata."""
        return dict(getattr(self, "_high_level_memory_build_state", {}) or {})

    def update_high_level_memory_build_state(self, **updates: Any) -> Dict[str, Any]:
        """Update and return high-level-memory build metadata."""
        state = self.get_high_level_memory_build_state()
        state.update(updates)
        self.set_high_level_memory_build_state(state)
        return state

    
    

    def _set_uid_int_mapping(self, mapping: Optional[Dict[str, Any]]) -> None:
        """Replace the global uid -> int_id mapping from v3 metadata."""
        self._uid_to_int_id = {}
        self._int_id_to_uid = {}
        if mapping:
            for raw_uid, raw_int_id in mapping.items():
                try:
                    uid = str(raw_uid)
                    int_id = int(raw_int_id)
                except (TypeError, ValueError):
                    logger.warning(f"Skipping malformed global ID mapping entry: {raw_uid!r} -> {raw_int_id!r}")
                    continue
                self._uid_to_int_id[uid] = int_id
                self._int_id_to_uid[int_id] = uid
        self._next_int_id = (max(self._int_id_to_uid.keys()) + 1) if self._int_id_to_uid else 0
        self._invalidate_space_filter_cache()

    def _invalidate_space_filter_cache(self) -> None:
        """Invalidate cached MemorySpace -> FAISS int-id filters."""
        self._space_membership_version += 1
        self._space_filter_cache.clear()

    def _get_or_create_int_id(self, uid: str) -> int:
        """Return the stable global int_id for uid, allocating one if this is a new UID."""
        uid = str(uid)
        existing = self._uid_to_int_id.get(uid)
        if existing is not None:
            return int(existing)
        int_id = self._next_int_id
        self._next_int_id += 1
        self._uid_to_int_id[uid] = int_id
        self._int_id_to_uid[int_id] = uid
        return int_id

    def _get_or_create_int_ids(self, uids: Iterable[str]) -> List[int]:
        """Batch-register UIDs and return their global int IDs in input order."""
        return [self._get_or_create_int_id(uid) for uid in uids]

    def _uids_to_int_ids(self, uids: Iterable[str], create: bool = False) -> List[int]:
        """Translate UID strings to int IDs at API boundaries."""
        if create:
            return self._get_or_create_int_ids(uids)
        int_ids: List[int] = []
        for uid in uids:
            int_id = self._uid_to_int_id.get(str(uid))
            if int_id is not None:
                int_ids.append(int(int_id))
        return int_ids

    def _int_ids_to_uids(self, int_ids: Iterable[int]) -> List[str]:
        """Translate internal int IDs back to public UID strings."""
        uids: List[str] = []
        for int_id in int_ids:
            uid = self._int_id_to_uid.get(int(int_id))
            if uid is not None:
                uids.append(uid)
        return uids

    def _get_uid_to_int_id_map(self) -> Dict[str, int]:
        """Expose the canonical mapping to retrievers without handing out FAISS-specific names."""
        return self._uid_to_int_id

    def _get_int_id_to_uid_map(self) -> Dict[int, str]:
        """Expose the reverse canonical mapping to retrievers."""
        return self._int_id_to_uid

    
    

    # Tiered L1/L2 storage callbacks
    

    def enable_tiered_storage(
        self,
        payload_store,
        max_capacity: Optional[int] = None,
        high_watermark: float = 0.85,
        low_watermark: float = 0.70,
        callbacks: Optional[Dict[str, callable]] = None,
        l1_mutation_lock=None,
    ):
        """Attach RocksDB-backed automatic payload paging.

        Args:
            payload_store: Open RocksDB payload-store instance.
            max_capacity: Maximum resident payload count before paging.
            high_watermark: Resident count or capacity fraction that starts
                eviction.
            low_watermark: Resident count or capacity fraction reached after
                eviction.
            callbacks: Optional graph-aware payload cache callbacks.
            l1_mutation_lock: Optional lock shared with the graph callback path.
        """
        from ..storage.tiered_storage_manager import TieredStorageManager

        existing_manager = getattr(self, "tiered_storage_manager", None)
        if existing_manager is not None:
            raise RuntimeError("Tiered payload storage is already enabled.")
        self._external_storage = payload_store
        stored_uids = (
            payload_store.list_uids()
            if hasattr(payload_store, "list_uids")
            else []
        )
        self._storage_uids = set(self.memory_units).union(stored_uids)
        if max_capacity is not None:
            self._max_memory_units = int(max_capacity)
        callback_map = callbacks or self._build_tiered_storage_callbacks()
        try:
            self.tiered_storage_manager = TieredStorageManager(
                payload_store=payload_store,
                callbacks=callback_map,
                max_capacity=self._max_memory_units,
                high_watermark=high_watermark,
                low_watermark=low_watermark,
                l1_mutation_lock=l1_mutation_lock,
            )
        except Exception:
            self._external_storage = None
            self._storage_uids = None
            self.tiered_storage_manager = None
            raise
        self._trigger_tiered_eviction_if_needed()
        return self.tiered_storage_manager

    def _close_tiered_storage(self) -> None:
        """Close paging resources without materializing cold payloads."""
        manager = getattr(self, "tiered_storage_manager", None)
        if manager is not None:
            manager.shutdown(wait=True)
        store = self._external_storage
        if store is not None and hasattr(store, "close"):
            store.close()
        self.tiered_storage_manager = None
        self._external_storage = None
        self._storage_uids = None

    def _build_tiered_storage_callbacks(self) -> Dict[str, callable]:
        return {
            "get_l1_data_cb": self._get_l1_data_for_tiered_swap_out,
            "remove_from_l1_cb": self._remove_from_l1_for_tiered_swap,
            "add_to_l1_cb": self._add_to_l1_from_tiered_swap,
        }

    def _trigger_tiered_eviction_if_needed(self) -> None:
        manager = getattr(self, "tiered_storage_manager", None)
        if manager is not None:
            manager.check_and_trigger_eviction(len(self.memory_units))

    def _get_l1_data_for_tiered_swap_out(self, count: int) -> Dict[str, Any]:
        """Select cold resident payloads without touching retrieval indexes."""
        selected_uids = self._select_cold_l1_uids(count)
        units = [self.memory_units[uid] for uid in selected_uids if uid in self.memory_units]
        return {
            "units": units,
            "uid_order": [unit.uid for unit in units],
        }

    def _select_cold_l1_uids(self, count: int) -> List[str]:
        if count <= 0:
            return []
        candidates = list(self.memory_units.keys())
        candidates.sort(
            key=lambda uid: (
                self._access_counts.get(uid, 0),
                self._last_accessed.get(uid, 0),
            )
        )
        return candidates[: min(count, len(candidates))]

    def _remove_from_l1_for_tiered_swap(self, uids: List[str]) -> int:
        """Remove payloads while preserving all resident retrieval and graph state."""
        if not uids:
            return 0

        removed_count = 0
        for uid in uids:
            if uid in self.memory_units:
                del self.memory_units[uid]
                removed_count += 1
            self._modified_units.discard(uid)
            self._access_counts.pop(uid, None)
            self._last_accessed.pop(uid, None)
            if self._storage_uids is not None:
                self._storage_uids.add(uid)

        if removed_count:
            logger.info(
                "Tiered storage evicted %d payloads; indexes, UID mappings, "
                "MemorySpace membership, and graph topology remain resident.",
                removed_count,
            )
        return removed_count

    def _add_to_l1_from_tiered_swap(
        self,
        recovered_units: List[MemoryUnit],
    ) -> int:
        """Publish recovered payloads without rebuilding resident indexes."""
        if not recovered_units:
            return 0
        now = datetime.now().timestamp()

        for unit in recovered_units:
            self.memory_units[unit.uid] = unit
            self._access_counts[unit.uid] = self._access_counts.get(unit.uid, 0) + 1
            self._last_accessed[unit.uid] = now

        self._trigger_tiered_eviction_if_needed()
        return len(recovered_units)

    def _unit_exists(self, uid: str) -> bool:
        """Check resident or persistent payload membership without materializing."""
        if uid in self.memory_units:
            return True
        if self._storage_uids is not None:
            return uid in self._storage_uids
        return False

    def _all_known_uids(self) -> Set[str]:
        """Return the resident UID catalog used by retrieval filters."""
        if self._storage_uids is None:
            return set(self.memory_units)
        return set(self._storage_uids).union(self.memory_units)

    def _total_unit_count(self) -> int:
        if self._storage_uids is None:
            return len(self.memory_units)
        return len(self._storage_uids.union(self.memory_units))

    def _remove_aux_retriever_uids(self, uids: List[str]) -> None:
        multi_retriever = getattr(self, "_multi_retriever", None)
        if multi_retriever is None:
            return
        try:
            from ..retrieval.retrieval_interface import RetrievalMethod

            for method in (RetrievalMethod.BM25, RetrievalMethod.SPLADE):
                retriever = multi_retriever.retrievers.get(method)
                if retriever is not None and hasattr(retriever, "remove_uids"):
                    retriever.remove_uids(uids)
        except Exception as exc:
            logger.warning(f"BM25/SPLADE L1 cleanup failed: {exc}")

    
    def _init_faiss_index(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            
            if self.faiss_index_type.startswith("IDMap,"):
                if not hasattr(faiss, "IndexIDMap"):
                    raise ImportError("This FAISS version does not support IndexIDMap")
                base_index_type = self.faiss_index_type.split("IDMap,", 1)[1]
                base_index = faiss.index_factory(
                    self.embedding_dim, 
                    base_index_type,
                    faiss.METRIC_INNER_PRODUCT
                )
                self.faiss_index = faiss.IndexIDMap(base_index)
                logger.info(f"Created IndexIDMap index with base type: {base_index_type} (inner product over normalized vectors for cosine similarity)")
            else:
                self.faiss_index = faiss.index_factory(
                    self.embedding_dim, 
                    self.faiss_index_type,
                    faiss.METRIC_INNER_PRODUCT
                )
                logger.info(f"Created index {self.faiss_index_type} (inner product over normalized vectors for cosine similarity)")

            logger.info(
                f"FAISS index '{self.faiss_index_type}' initialized. Total vectors: {self.faiss_index.ntotal if self.faiss_index else 0}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index '{self.faiss_index_type}': {e}")
            self.faiss_index = None
            raise

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector."""
        try:
            
            vector = vector.astype(np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            
            norm = np.linalg.norm(vector)
            if norm < 1e-6:
                logger.debug("Zero or near-zero vector detected; skipping normalization")
                return np.zeros(vector.shape[1], dtype=np.float32)
            
            
            faiss.normalize_L2(vector)
            return vector.flatten()
        except Exception as e:
            logger.warning(f"Vector normalization failed: {e}")
            return vector.astype(np.float32).flatten()
        
    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding."""
        if not text or not text.strip():
            logger.warning("Input text is empty; cannot generate an embedding.")
            return None
        
        if self.model is None:
            logger.error("Text embedding model is not initialized.")
            return None
        
        try:
            emb = self.model.encode(
                text,
                show_progress_bar=False,
                **self._text_encode_kwargs
            )
            
            
            
            if emb is None:
                logger.warning(f"Model returned a None embedding: '{text[:50]}...'")
                return None
            
            if isinstance(emb, torch.Tensor):
                emb = emb.to(dtype=torch.float32).cpu().numpy()
            
            elif isinstance(emb, np.ndarray):
                if emb.dtype != np.float32:
                    emb = emb.astype(np.float32)
            
            elif isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            
            else:
                logger.warning(f"Unknown embedding type: {type(emb)}, attempting conversion...")
                try:
                    emb = np.array(emb, dtype=np.float32)
                except Exception as conv_err:
                    logger.error(f"Failed to convert embedding type {type(emb)}: {conv_err}")
                    return None
            
            
            if emb.ndim == 2:
                if emb.shape[0] == 1:
                    emb = emb[0]
                else:
                    logger.warning(f"Unexpected embedding shape (batch_size > 1): {emb.shape}, using the first row")
                    emb = emb[0]
            
            if emb.ndim != 1:
                logger.error(f"Embedding dimension error: expected 1D, got {emb.ndim}D, shape={emb.shape}")
                return None
            
            if emb.shape[0] != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, actual {emb.shape[0]}."
                    f"continuing, but this may cause index issues."
                )
            
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32)
            
            return emb
            
        except Exception as e:
            logger.error(f"Text embedding generation failed: '{text[:50]}...' - {e}")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())
            return None

    def _get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Get image embedding."""
        if not image_path or not isinstance(image_path, str):
            logger.warning("Invalid image path; cannot generate an embedding.")
            return None
        if not os.path.isfile(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        try:
            if self.image_model is None:
                logger.error("Image embedding model is not initialized")
                return None
            image_encode_format = getattr(self, "_image_encode_format", "pil")
            image_encode_kwargs = dict(getattr(self, "_image_encode_kwargs", {}))
            if image_encode_format == "path":
                emb = self.image_model.encode(image_path, **image_encode_kwargs)
            elif image_encode_format == "dict_path":
                emb = self.image_model.encode([{"image": image_path}], **image_encode_kwargs)
            else:
                img = Image.open(image_path)
                emb = self.image_model.encode(img, **image_encode_kwargs)

            emb = np.array(emb, dtype=np.float32)
            if emb.ndim == 2 and emb.shape[0] == 1:
                emb = emb[0]
            if emb.ndim != 1:
                logger.error(f"Image embedding dimension error: expected 1D, got {emb.ndim}D, shape={emb.shape}")
                return None
            return emb
        except Exception as e:
            logger.error(f"Image embedding generation failed: '{image_path}' - {e}")
            return None

    def _extract_text_content_for_embedding(
        self,
        unit: MemoryUnit,
        explicit_content: Optional[Any] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """Select text content used by the dense embedding path.

        Explicit content takes precedence when it is textual. Otherwise the
        method uses the configured text embedding key, common text-like fields,
        and finally a string representation of the raw payload.
        """
        if explicit_content is not None:
            if content_type == "text" or content_type is None:
                text = str(explicit_content)
                if text.strip():
                    return text
        
        text_val = unit.raw_data.get(self.DEFAULT_TEXT_EMBEDDING_KEY)
        if text_val and isinstance(text_val, str) and text_val.strip():
            return text_val
        
        common_text_fields = ["content", "text", "description", "summary", "caption", "message"]
        for field in common_text_fields:
            if field == self.DEFAULT_TEXT_EMBEDDING_KEY:
                continue
            val = unit.raw_data.get(field)
            if val and isinstance(val, str) and val.strip():
                return val
        
        fallback = str(unit.raw_data)
        if fallback.strip():
            return fallback
        
        return ""

    def _generate_embedding_for_unit(
        self,
        unit: MemoryUnit,
        explicit_content: Optional[Any] = None,
        content_type: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Generate embedding for unit."""
        image_requested = (
            content_type == "image_path"
            or self.DEFAULT_IMAGE_EMBEDDING_KEY in unit.raw_data
        )
        if image_requested and "image" not in self.supported_modalities:
            logger.debug(
                f"Unit '{unit.uid}' requested image embedding, but the primary model only supports {self.supported_modalities}, "
                "falling back to text embedding."
            )
            content_type = "text"
            explicit_content = None

        embedding = None
        
        if explicit_content is not None and content_type is not None:
            if content_type == "text":
                embedding = self._get_text_embedding(str(explicit_content))
            elif content_type == "image_path":
                embedding = self._get_image_embedding(str(explicit_content))
            else:
                logger.warning(f"Unknown content type '{content_type}'.using the default inference path.")
        
        if embedding is None:
            image_path = unit.raw_data.get(self.DEFAULT_IMAGE_EMBEDDING_KEY)
            if "image" in self.supported_modalities and content_type != "text" and image_path and isinstance(image_path, str):
                embedding = self._get_image_embedding(image_path)
            
            if embedding is None:
                text_content = self._extract_text_content_for_embedding(
                    unit, explicit_content, content_type
                )
                if text_content:
                    embedding = self._get_text_embedding(text_content)
                else:
                    logger.debug(
                        f"Unit '{unit.uid}' has no usable text or image content for embedding generation."
                    )

        if embedding is not None and embedding.shape[0] != self.embedding_dim:
            logger.error(
                f"Unit '{unit.uid}' generated embedding dimension ({embedding.shape[0]}) does not match expected dimension ({self.embedding_dim})."
            )
            return None
        return embedding

    
    
    

    def create_memory_space(self, space_name: str) -> MemorySpace:
        """Create or return a named MemorySpace and bind it to this map."""
        if space_name not in self.memory_spaces:
            space = MemorySpace(space_name)
            space._set_semantic_map_ref(self)
            self.memory_spaces[space_name] = space
            logger.info(f"Memory space '{space_name}' created")
        return self.memory_spaces[space_name]

    def get_memory_space(self, space_name: str) -> Optional[MemorySpace]:
        """Return a MemorySpace by name without creating a new one."""
        return self.memory_spaces.get(space_name)

    def add_unit_to_space(self, unit_or_uid: Union[str, MemoryUnit], space_name: str):
        """Record MemorySpace membership for an existing unit.

        Membership is separate from the L1 payload. It may outlive an in-memory
        unit during tiered eviction and therefore invalidates space-filter
        caches independently of FAISS vectors.
        """
        if isinstance(unit_or_uid, str):
            uid = unit_or_uid
        elif hasattr(unit_or_uid, "uid"):
            uid = unit_or_uid.uid
        else:
            raise TypeError(
                f"add_unit_to_space() expected str(UID) or MemoryUnit, got {type(unit_or_uid)}"
            )

        if not self._unit_exists(uid):
            logger.warning(f"Attempted to add missing memory unit '{uid}' to space '{space_name}'")
            return

        space = self.create_memory_space(space_name)
        space.add_unit(uid)

    def remove_unit_from_space(
        self, unit_or_uid: Union[str, MemoryUnit], space_name: str
    ):
        """Remove unit from space."""
        if isinstance(unit_or_uid, str):
            uid = unit_or_uid
        elif hasattr(unit_or_uid, "uid"):
            uid = unit_or_uid.uid
        else:
            raise TypeError(
                f"remove_unit_from_space() expected str(UID) or MemoryUnit, got {type(unit_or_uid)}"
            )

        space = self.get_memory_space(space_name)
        if space:
            space.remove_unit(uid)
        else:
            logger.warning(f"Attempted to remove unit '{uid}' from missing memory space '{space_name}'")

    def add_space_to_space(
        self, child_space_or_name: Union[str, MemorySpace], parent_space_name: str
    ):
        """Add space to space."""
        if isinstance(child_space_or_name, str):
            child_space_name = child_space_or_name
        elif hasattr(child_space_or_name, "name"):
            child_space_name = child_space_or_name.name
        else:
            raise TypeError(
                f"add_space_to_space() expected str(name) or MemorySpace, got {type(child_space_or_name)}"
            )

        if child_space_name not in self.memory_spaces:
            logger.warning(f"Child space '{child_space_name}' does not exist")
            return

        parent_space = self.create_memory_space(parent_space_name)
        parent_space.add_child_space(child_space_name)

    def get_units_in_memory_space(
        self, ms_names: Union[str, List[str]], recursive: bool = True
    ) -> List[MemoryUnit]:
        """Return units in memory space."""
        if isinstance(ms_names, str):
            ms_names = [ms_names]

        all_uids = set()

        for space_name in ms_names:
            space = self.memory_spaces.get(space_name)
            if space:
                if recursive:
                    all_uids.update(space.get_all_unit_uids(recursive=True))
                else:
                    all_uids.update(space.get_unit_uids())
            else:
                logger.warning(f"Memory space '{space_name}' does not exist")

        
        units = []
        for uid in all_uids:
            unit = self.get_unit(uid)
            if unit:
                units.append(unit)
            else:
                logger.warning(f"MemoryUnit '{uid}' is not present in payload storage")

        return units

    
    
    

    def _record_unit_access(self, uid: str):
        """Record unit access."""
        if uid in self.memory_units:
            if not hasattr(self, "_access_counts"):
                self._access_counts = {}
            if not hasattr(self, "_last_accessed"):
                self._last_accessed = {}
            self._access_counts[uid] = self._access_counts.get(uid, 0) + 1
            self._last_accessed[uid] = datetime.now().timestamp()

    def _incremental_faiss_add(self, unit: MemoryUnit):
        """Update FAISS for one unit while preserving UID/int-id mappings.

        If the current FAISS index type cannot remove the previous vector, the
        map falls back to a full rebuild to keep dense retrieval consistent.
        """
        if unit.embedding is None:
            logger.debug(f"Unit '{unit.uid}' has no embedding; skipping incremental FAISS update")
            return

        
        if self.faiss_index is None:
            logger.info("FAISS index is not initialized; falling back to build_index().")
            self.build_index()
            return

        
        vec = unit.embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            logger.debug(f"Unit '{unit.uid}' has a zero embedding; skipping incremental FAISS update")
            return
        faiss.normalize_L2(vec)

        
        if unit.uid in self._uid_to_int_id:
            old_id = self._uid_to_int_id[unit.uid]
            try:
                self.faiss_index.remove_ids(np.array([old_id], dtype=np.int64))
            except Exception as e:
                logger.warning(f"FAISS remove_ids({old_id}) failed: {e}, falling back to a full rebuild")
                self.build_index()
                return

        
        new_id = self._get_or_create_int_id(unit.uid)

        
        if self.faiss_index_type.startswith("IDMap,"):
            self.faiss_index.add_with_ids(vec, np.array([new_id], dtype=np.int64))
        else:
            self.faiss_index.add(vec)

        logger.debug(
            f"FAISS incremental update: uid='{unit.uid}', internal_id={new_id}, "
            f"ntotal={self.faiss_index.ntotal}"
        )

    
    

    @staticmethod
    def _normalize_index_update_mode(index_update_mode: str) -> str:
        """Normalize index update mode."""
        mode = (index_update_mode or "incremental").lower()
        valid_modes = {"incremental", "none", "rebuild"}
        if mode not in valid_modes:
            raise ValueError(
                f"index_update_mode must be one of {sorted(valid_modes)} ; got: {index_update_mode}"
            )
        return mode

    def _incremental_faiss_add_many(self, units: List[MemoryUnit]) -> int:
        """Append multiple unit embeddings to FAISS in one batch.

        Existing vectors are removed first through their stable integer IDs. A
        full rebuild is used when batched removal or insertion is unsupported.
        """
        if not units:
            return 0

        if self.faiss_index is None:
            self._init_faiss_index()
        if self.faiss_index is None:
            logger.warning("FAISS index is unavailable; skipping batched incremental update.")
            return 0

        valid_units: List[MemoryUnit] = []
        embeddings_to_add: List[np.ndarray] = []
        for unit in units:
            embedding = unit.embedding
            if embedding is None or embedding.shape[0] != self.embedding_dim:
                logger.debug(f"Unit '{unit.uid}' has no valid embedding; skipping batched incremental FAISS update")
                continue
            embedding_np = np.asarray(embedding, dtype=np.float32)
            if np.linalg.norm(embedding_np) < 1e-6:
                logger.debug(f"Unit '{unit.uid}' has a zero embedding; skipping batched incremental FAISS update")
                continue
            valid_units.append(unit)
            embeddings_to_add.append(embedding_np)

        if not valid_units:
            return 0

        existing_ids = [
            self._uid_to_int_id[unit.uid]
            for unit in valid_units
            if unit.uid in self._uid_to_int_id
        ]
        if existing_ids:
            try:
                if hasattr(self.faiss_index, "remove_ids"):
                    ids_to_remove = np.ascontiguousarray(np.array(existing_ids, dtype=np.int64))
                    self.faiss_index.remove_ids(ids_to_remove)
                else:
                    logger.warning("Current FAISS index does not support remove_ids; falling back to a full rebuild.")
                    self.build_index()
                    return len(valid_units)
            except Exception as exc:
                logger.warning(f"FAISS batch remove_ids failed: {exc}, falling back to a full rebuild")
                self.build_index()
                return len(valid_units)

        embeddings_np = np.ascontiguousarray(
            np.stack(embeddings_to_add, axis=0).astype(np.float32)
        )
        try:
            import faiss
            faiss.normalize_L2(embeddings_np)

            if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
                logger.info("FAISS index is not trained; training with the current incremental batch.")
                self.faiss_index.train(embeddings_np)

            ids_np = np.ascontiguousarray(
                np.asarray(self._get_or_create_int_ids(unit.uid for unit in valid_units), dtype=np.int64)
            )

            if self.faiss_index_type.startswith("IDMap,") and hasattr(self.faiss_index, "add_with_ids"):
                self.faiss_index.add_with_ids(embeddings_np, ids_np)
            else:
                self.faiss_index.add(embeddings_np)

            logger.debug(
                f"FAISS batched incremental update: added={len(valid_units)}, ntotal={self.faiss_index.ntotal}"
            )
            return len(valid_units)
        except Exception as exc:
            logger.warning(f"FAISS batched incremental updatefailed: {exc}, falling back to a full rebuild")
            self.build_index()
            return len(valid_units)

    def _incremental_aux_retriever_add(self, units: List[MemoryUnit]) -> None:
        """Propagate inserted units to auxiliary sparse retrievers."""
        if not units:
            return
        try:
            from ..retrieval.retrieval_interface import RetrievalMethod

            multi_retriever = self.get_multi_retriever()
            if multi_retriever is None:
                return

            has_sparse_units = any(unit.has_sparse_embedding() for unit in units)
            for method in (RetrievalMethod.BM25, RetrievalMethod.SPLADE):
                try:
                    if (
                        method == RetrievalMethod.SPLADE
                        and not has_sparse_units
                        and method not in multi_retriever.retrievers
                    ):
                        continue
                    if hasattr(multi_retriever, "_ensure_retriever_loaded"):
                        multi_retriever._ensure_retriever_loaded(method)
                    retriever = multi_retriever.retrievers.get(method)
                    if retriever is not None and hasattr(retriever, "add_units"):
                        logger.debug(f"Running {method.value} incremental update: units={len(units)}")
                        retriever.add_units(units)
                except Exception as exc:
                    logger.warning(f"{method.value} incremental index update failed: {exc}")
        except Exception as exc:
            logger.warning(f"BM25/SPLADE incremental index scheduling failed: {exc}")

    def rebuild_all_indexes(self) -> None:
        """Rebuild all indexes."""
        logger.info(" index_update_mode='rebuild': rebuilding FAISS/BM25/SPLADE indexes")
        self.build_index()
        try:
            from ..retrieval.retrieval_interface import RetrievalMethod

            multi_retriever = self.get_multi_retriever()
            if multi_retriever is not None:
                multi_retriever.build_all_indexes(
                    methods_to_build=[RetrievalMethod.BM25, RetrievalMethod.SPLADE],
                    force_rebuild=True,
                )
        except Exception as exc:
            logger.warning(f"BM25/SPLADE full index rebuild failed: {exc}")

    def _apply_index_update_mode(self, units: List[MemoryUnit], index_update_mode: str) -> None:
        """Apply index update mode."""
        mode = self._normalize_index_update_mode(index_update_mode)
        if mode == "none":
            logger.debug("index_update_mode='none': skipping FAISS/BM25/SPLADE index updates")
            return
        if mode == "rebuild":
            self.rebuild_all_indexes()
            return

        if len(units) == 1:
            self._incremental_faiss_add(units[0])
        else:
            self._incremental_faiss_add_many(units)
        self._incremental_aux_retriever_add(units)

    def add_unit(
        self,
        unit: MemoryUnit,
        explicit_content_for_embedding: Optional[Any] = None,
        content_type_for_embedding: Optional[str] = None,
        space_names: Optional[List[str]] = None,
        index_update_mode: str = "incremental",
        generate_sparse_embedding: bool = True,  
        sparse_model_name: str = "naver/splade-v3",  
        **legacy_kwargs,
    ):
        """Add or update a memory unit in the semantic map.

        Args:
            unit: MemoryUnit to insert or update.
            explicit_content_for_embedding: Optional content used to generate
                the dense embedding. If omitted, content is inferred from the
                unit payload.
            content_type_for_embedding: Content type for embedding generation,
                such as ``"text"`` or ``"image_path"``.
            space_names: Logical memory spaces that should contain this unit.
            index_update_mode: Index update strategy, such as ``"incremental"``,
                ``"none"``, or ``"rebuild"``.
            generate_sparse_embedding: Whether to generate a SPLADE sparse
                embedding during insertion.
            sparse_model_name: SPLADE model name used when sparse embedding is
                requested.
            **legacy_kwargs: Backward-compatible aliases accepted by older code.
        """
        if not isinstance(unit, MemoryUnit):
            logger.error("add_unit expected a MemoryUnit instance.")
            return

        if "rebuild_index_immediately" in legacy_kwargs:
            legacy_value = bool(legacy_kwargs.pop("rebuild_index_immediately"))
            index_update_mode = "incremental" if legacy_value else "none"
            logger.warning("rebuild_index_immediately is deprecated; use index_update_mode instead.")
        if legacy_kwargs:
            raise TypeError(f"add_unit() got unexpected keyword arguments: {list(legacy_kwargs.keys())}")
        index_update_mode = self._normalize_index_update_mode(index_update_mode)

        if self._unit_exists(unit.uid):
            existing = self.get_unit(unit.uid)
            if existing == unit:
                logger.info(f"Memory unit '{unit.uid}' already exists with identical content; skipping insertion.")
                if space_names:
                    self._update_unit_spaces(unit.uid, space_names)
                return

        if explicit_content_for_embedding is None:
            logger.debug(f"No explicit embedding content provided for unit '{unit.uid}'; inferring from payload.")
            if self.DEFAULT_TEXT_EMBEDDING_KEY in unit.raw_data:
                explicit_content_for_embedding = unit.raw_data[self.DEFAULT_TEXT_EMBEDDING_KEY]
                content_type_for_embedding = "text"
            elif self.DEFAULT_IMAGE_EMBEDDING_KEY in unit.raw_data:
                explicit_content_for_embedding = unit.raw_data[self.DEFAULT_IMAGE_EMBEDDING_KEY]
                content_type_for_embedding = "image_path"
            else:
                explicit_content_for_embedding = str(unit.raw_data)
                content_type_for_embedding = "text"

        new_embedding = self._generate_embedding_for_unit(
            unit, explicit_content_for_embedding, content_type_for_embedding
        )

        if new_embedding is None:
            logger.warning(
                f"Could not generate an embedding for memory unit '{unit.uid}'. "
                "The unit will be stored but unavailable for similarity search."
            )
        unit.embedding = new_embedding

        self.memory_units[unit.uid] = unit
        logger.info(f"Memory unit '{unit.uid}' has been added or updated in SemanticMap.")

        if space_names:
            self._update_unit_spaces(unit.uid, space_names)
            logger.debug(f"Unit '{unit.uid}' associated with spaces: {space_names}")

        
        if generate_sparse_embedding and not unit.has_sparse_embedding():
            self._generate_sparse_embedding_for_unit(
                unit, 
                explicit_content_for_embedding, 
                content_type_for_embedding,
                sparse_model_name
            )

        
        self._apply_index_update_mode([unit], index_update_mode)

        self._modified_units.add(unit.uid)
        self._access_counts[unit.uid] = self._access_counts.get(unit.uid, 0) + 1
        self._last_accessed[unit.uid] = datetime.now().timestamp()
        self._trigger_tiered_eviction_if_needed()

    def batch_add_units(
        self,
        units: List[MemoryUnit],
        batch_size: int = 32,
        space_names: Optional[List[str]] = None,
        content_type_for_embedding: Optional[str] = None,
        explicit_contents_for_embedding: Optional[List[Any]] = None,
        content_types_for_embedding: Optional[List[Optional[str]]] = None,
        per_unit_space_names: Optional[List[Optional[List[str]]]] = None,
        index_update_mode: str = "incremental",
        generate_sparse_embedding: bool = True,
        sparse_model_name: str = "naver/splade-v3",
        show_progress: bool = True,
        **legacy_kwargs,
    ) -> Dict[str, Any]:
        """Add multiple memory units and update retrieval indexes.

        Args:
            units: Memory units to insert or update.
            batch_size: Batch size for dense embedding generation.
            space_names: Shared memory spaces applied to every unit.
            content_type_for_embedding: Shared content type for embedding
                generation.
            explicit_contents_for_embedding: Optional per-unit embedding
                contents aligned with ``units``.
            content_types_for_embedding: Optional per-unit content types aligned
                with ``units``.
            per_unit_space_names: Optional per-unit memory-space lists aligned
                with ``units``.
            index_update_mode: Index update strategy, such as ``"incremental"``,
                ``"none"``, or ``"rebuild"``.
            generate_sparse_embedding: Whether to generate SPLADE sparse
                embeddings for inserted units.
            sparse_model_name: SPLADE model name used for sparse embeddings.
            show_progress: Whether embedding backends may display progress.
            **legacy_kwargs: Backward-compatible aliases accepted by older code.

        Returns:
            A statistics dictionary with insertion, skip, embedding, and timing
            counts.
        """
        import time
        from tqdm import tqdm

        if "rebuild_index" in legacy_kwargs:
            legacy_value = bool(legacy_kwargs.pop("rebuild_index"))
            index_update_mode = "rebuild" if legacy_value else "none"
            logger.warning("rebuild_index is deprecated; use index_update_mode instead.")
        if legacy_kwargs:
            raise TypeError(
                f"batch_add_units() got unexpected keyword arguments: {list(legacy_kwargs.keys())}"
            )
        index_update_mode = self._normalize_index_update_mode(index_update_mode)

        unit_count = len(units)
        if explicit_contents_for_embedding is not None and len(explicit_contents_for_embedding) != unit_count:
            raise ValueError("explicit_contents_for_embedding length must match units")
        if content_types_for_embedding is not None and len(content_types_for_embedding) != unit_count:
            raise ValueError("content_types_for_embedding length must match units")
        if per_unit_space_names is not None and len(per_unit_space_names) != unit_count:
            raise ValueError("per_unit_space_names length must match units")

        unit_index_by_uid = {
            unit.uid: index for index, unit in enumerate(units) if isinstance(unit, MemoryUnit)
        }

        def spaces_for_index(index: int) -> Optional[List[str]]:
            if per_unit_space_names is not None:
                item_spaces = per_unit_space_names[index]
                return list(item_spaces) if item_spaces else None
            return list(space_names) if space_names else None

        def embedding_text_for_unit(index: int, unit: MemoryUnit) -> str:
            explicit_content = None
            if explicit_contents_for_embedding is not None:
                explicit_content = explicit_contents_for_embedding[index]
            item_content_type = content_type_for_embedding
            if content_types_for_embedding is not None:
                item_content_type = content_types_for_embedding[index]

            if explicit_content is None:
                return self._extract_text_content_for_embedding(unit)
            if item_content_type in (None, "text"):
                return str(explicit_content)
            logger.debug(
                "batch_add_units: Unit '%s' content_type=%s does not support batched text encoding; using default text extraction",
                unit.uid,
                item_content_type,
            )
            return self._extract_text_content_for_embedding(unit)
        
        start_time = time.time()
        stats = {
            "total": len(units),
            "added": 0,
            "skipped": 0,
            "embedding_generated": 0,
            "sparse_generated": 0,
            "duration": 0.0
        }
        
        if not units:
            logger.warning("batch_add_units: no units to add")
            return stats
        
        logger.info(f" Starting batch insertion for {len(units)} memory units...")
        
        units_to_process: List[Tuple[MemoryUnit, str]] = []  # (unit, text)
        units_no_text: List[MemoryUnit] = []
        
        for index, unit in enumerate(units):
            if not isinstance(unit, MemoryUnit):
                logger.warning("Skipping non-MemoryUnit object.")
                stats["skipped"] += 1
                continue
            unit_space_names = spaces_for_index(index)
            
            if self._unit_exists(unit.uid):
                existing = self.get_unit(unit.uid)
                if existing == unit:
                    logger.debug(f"Unit '{unit.uid}' already exists with identical content; skipping.")
                    stats["skipped"] += 1
                    if unit_space_names:
                        self._update_unit_spaces(unit.uid, unit_space_names)
                    continue
            
            text = embedding_text_for_unit(index, unit)
            if text and text.strip():
                units_to_process.append((unit, text))
            else:
                units_no_text.append(unit)
                logger.debug(f"Unit '{unit.uid}' has no usable text content; skipping embedding generation.")
        
        logger.info(
            f"Text extraction complete: {len(units_to_process)} with text, "
            f"{len(units_no_text)} without text, {stats['skipped']} skipped"
        )
        
        
        embeddings_map: Dict[str, np.ndarray] = {}  # uid -> embedding
        
        if units_to_process:
            texts = [text for _, text in units_to_process]
            uids = [unit.uid for unit, _ in units_to_process]
            
            logger.info(f" Generating dense embeddings for {len(texts)} texts...")
            
            try:
                if self.model is not None:
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True,
                        **self._text_encode_kwargs
                    )
                    
                    embeddings = np.array(embeddings, dtype=np.float32)
                    
                    if embeddings.shape[1] != self.embedding_dim:
                        logger.error(
                            f"Embedding dimension mismatch: expected {self.embedding_dim}, actual {embeddings.shape[1]}"
                        )
                    else:
                        for i, uid in enumerate(uids):
                            embeddings_map[uid] = embeddings[i]
                        stats["embedding_generated"] = len(embeddings_map)
                        logger.info(f" Dense embedding generation complete: {len(embeddings_map)} embeddings")
                else:
                    logger.error("embedding model is not initialized; cannot generate vectors")
                    
            except Exception as e:
                logger.error(f"Batched dense embedding generation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        logger.info(" Storing memory units in batch...")
        
        current_timestamp = datetime.now().timestamp()
        for unit, _ in units_to_process:
            embedding = embeddings_map.get(unit.uid)
            if embedding is not None:
                unit.embedding = embedding
            self.memory_units[unit.uid] = unit
            self._modified_units.add(unit.uid)
            self._access_counts[unit.uid] = self._access_counts.get(unit.uid, 0) + 1
            self._last_accessed[unit.uid] = current_timestamp
            stats["added"] += 1
        
        for unit in units_no_text:
            self.memory_units[unit.uid] = unit
            self._modified_units.add(unit.uid)
            self._access_counts[unit.uid] = self._access_counts.get(unit.uid, 0) + 1
            self._last_accessed[unit.uid] = current_timestamp
            stats["added"] += 1
        
        if space_names or per_unit_space_names:
            if space_names:
                logger.info(f" Updating space membership: {space_names}")
            all_units = [unit for unit, _ in units_to_process] + units_no_text
            for unit in all_units:
                original_index = unit_index_by_uid.get(unit.uid, -1)
                unit_space_names = spaces_for_index(original_index) if original_index >= 0 else (list(space_names) if space_names else None)
                if unit_space_names:
                    self._update_unit_spaces(unit.uid, unit_space_names)
        
        
        if generate_sparse_embedding:
            logger.info(" Generating SPLADE sparse embeddings in batch...")
            all_units = [unit for unit, _ in units_to_process] + units_no_text
            try:
                sparse_result = self.build_sparse_embeddings(
                    units=all_units,
                    model_name=sparse_model_name,
                    batch_size=batch_size,
                    force_rebuild=False,
                    show_progress=show_progress
                )
                stats["sparse_generated"] = sparse_result.get("processed", 0)
                logger.info(f" SPLADE sparse embedding generation complete: {stats['sparse_generated']} embeddings")
            except Exception as exc:
                logger.warning(
                    f"Batched SPLADE sparse embedding generation failed: {exc}."
                    "Units were added, but sparse embeddings are missing."
                )
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    import traceback
                    logger.debug(traceback.format_exc())
        
        
        all_added_units = [unit for unit, _ in units_to_process] + units_no_text
        self._apply_index_update_mode(all_added_units, index_update_mode)
        self._trigger_tiered_eviction_if_needed()
        
        stats["duration"] = time.time() - start_time
        logger.info(
            f" Batch insertion complete: added {stats['added']}, skipped {stats['skipped']}, "
            f"dense {stats['embedding_generated']}, sparse {stats['sparse_generated']}, "
            f"elapsed {stats['duration']:.2f}s"
        )
        
        return stats

    
    def _generate_sparse_embedding_for_unit(
        self,
        unit: MemoryUnit,
        explicit_content: Optional[Any],
        content_type: Optional[str],
        model_name: str = "naver/splade-v3"
    ):
        """Generate sparse embedding for unit."""
        import torch
        
        try:
            text = self._extract_text_content_for_embedding(
                unit, explicit_content, content_type
            )
            
            if not text or not text.strip():
                logger.debug(f"Unit '{unit.uid}' has no text content for SPLADE embedding generation")
                return
            
            
            from ..utils.model_manager import global_model_manager
            model = global_model_manager.get_splade_model(model_name)
            
            if model is None:
                logger.warning(f"SPLADE model '{model_name}' is unavailable; skipping sparse embedding generation")
                return
            
            with torch.no_grad():
                sparse_tensors = model.encode_document([text])
                
                if not sparse_tensors:
                    logger.warning(f"Unit '{unit.uid}' returned an empty SPLADE encoding")
                    return
                
                sparse_tensor = sparse_tensors[0]
                
                
                coalesced = sparse_tensor.coalesce()
                
                
                indices = coalesced.indices().squeeze().cpu().numpy()
                values = coalesced.values().cpu().numpy()
                
                if indices.ndim == 0:
                    indices = np.array([indices.item()])
                    values = np.array([values.item()])
                else:
                    indices = indices.flatten()
                    values = values.flatten()
                
                
                sparse_dict = {
                    int(idx): float(val) 
                    for idx, val in zip(indices, values)
                    if val > 0
                }
                
                
                if sparse_dict:
                    unit.set_sparse_embedding(sparse_dict)
                    logger.debug(
                        f"Unit '{unit.uid}' SPLADE sparse embedding generated "
                        f"(nonzero entries: {len(sparse_dict)})"
                    )
                else:
                    logger.debug(f"Unit '{unit.uid}' has an empty SPLADE embedding")
                    
        except Exception as e:
            logger.warning(
                f"Failed to generate SPLADE sparse embedding for unit '{unit.uid}': {e}."
                f"Units were added, but sparse embeddings are missing."
            )
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())

    def _update_unit_spaces(self, uid: str, space_names: List[str]):
        """Attach a unit to the requested MemorySpaces, creating spaces as needed."""
        if not space_names:
            return
        
        for space_name in space_names:
            if space_name not in self.memory_spaces:
                logger.info(f"Automatically creating memory space: {space_name}")
                self.create_memory_space(space_name)
            
            self.add_unit_to_space(uid, space_name)

    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        """Return a payload, paging it from RocksDB when it is not resident."""
        unit = self.memory_units.get(uid)
        if unit:
            self._record_unit_access(uid)
            return unit
        manager = getattr(self, "tiered_storage_manager", None)
        if manager is not None:
            loaded_units = manager.handle_page_fault_batch([uid])
            if loaded_units:
                unit = self.memory_units.get(uid)
                if unit:
                    self._record_unit_access(uid)
                    return unit
        return None
    
    def get_units_by_spaces(
        self, 
        space_names: List[str], 
        mode: str = "union",
        recursive: bool = True
    ) -> List[MemoryUnit]:
        """Return units selected by MemorySpace set operations.

        Args:
            space_names: MemorySpace names, with or without the ``ms:`` prefix.
            mode: Set operation over the selected spaces: ``union``,
                ``intersection``, or ``difference``.
            recursive: Whether child MemorySpaces contribute their units.

        Returns:
            Memory units that satisfy the requested space operation.
        """
        if not space_names:
            return []
        
        normalized_names = [
            name[3:] if name.startswith("ms:") else name 
            for name in space_names
        ]
        
        if mode == "union":
            return self.get_units_in_memory_space(normalized_names, recursive=recursive)
        
        elif mode == "intersection":
            space_unit_sets = []
            for space_name in normalized_names:
                units = self.get_units_in_memory_space([space_name], recursive=recursive)
                space_unit_sets.append(set(u.uid for u in units))
            
            if not space_unit_sets:
                return []
            
            common_uids = set.intersection(*space_unit_sets)
            
            return [
                unit
                for uid in common_uids
                if (unit := self.get_unit(uid)) is not None
            ]
        
        elif mode == "difference":
            if len(normalized_names) < 2:
                return self.get_units_in_memory_space(normalized_names, recursive=recursive)
            
            first_space_units = self.get_units_in_memory_space(
                [normalized_names[0]], recursive=recursive
            )
            first_space_uids = set(u.uid for u in first_space_units)
            
            other_space_uids = set()
            for space_name in normalized_names[1:]:
                units = self.get_units_in_memory_space([space_name], recursive=recursive)
                other_space_uids.update(u.uid for u in units)
            
            diff_uids = first_space_uids - other_space_uids
            
            return [
                unit
                for uid in diff_uids
                if (unit := self.get_unit(uid)) is not None
            ]
        
        else:
            logger.warning(f"Unsupported query mode: {mode}, using the default union mode")
            return self.get_units_in_memory_space(normalized_names, recursive=recursive)

    def get_space_statistics(self) -> Dict[str, Any]:
        """Summarize MemorySpace membership and global FAISS index state."""
        stats = {
            "total_spaces": len(self.memory_spaces),
            "spaces": {}
        }
        
        for space_name, space in self.memory_spaces.items():
            global_index_vectors = self.faiss_index.ntotal if self.faiss_index else 0
            space_stats = {
                "units": len(space.get_all_unit_uids(recursive=True)),
                "direct_units": len(space.get_unit_uids()),
                "child_spaces": len(space.get_child_space_names()),
                "has_index": self.faiss_index is not None and global_index_vectors > 0,
                "index_vectors": global_index_vectors,
                "uses_global_index": True,
                "local_index_vectors": 0,
            }
            stats["spaces"][space_name] = space_stats
        
        return stats

    def delete_unit(self, uid: str, rebuild_index_immediately: bool = False):
        """Remove a payload and its retrieval membership from resident and persistent storage."""
        if not self._unit_exists(uid):
            logger.warning(f"Attempted to delete missing memory unit ID '{uid}'")
            return

        self.memory_units.pop(uid, None)
        if self._external_storage is not None:
            self._external_storage.delete_unit(uid)
        if self._storage_uids is not None:
            self._storage_uids.discard(uid)

        for space_obj in self.memory_spaces.values():
            if space_obj.contains_unit(uid):
                space_obj.remove_unit(uid)

        if self.faiss_index and uid in self._uid_to_int_id:
            internal_id_to_remove = self._uid_to_int_id[uid]
            try:
                if hasattr(self.faiss_index, "remove_ids"):
                    self.faiss_index.remove_ids(
                        np.array([internal_id_to_remove], dtype=np.int64)
                    )
                    logger.debug(f"Memory unit '{uid}' removed from the FAISS index")
            except Exception as e:
                logger.error(f"Failed to remove unit '{uid}' from the FAISS index: {e}")
        self._remove_aux_retriever_uids([uid])

        logger.info(f"Memory unit '{uid}' deleted from SemanticMap")
        if rebuild_index_immediately:
            self.build_index()

        if self._external_storage is None:
            self._deleted_units.add(uid)
        else:
            self._deleted_units.discard(uid)
        if uid in self._modified_units:
            self._modified_units.remove(uid)
        if uid in self._access_counts:
            del self._access_counts[uid]

    def get_all_units(self) -> List[MemoryUnit]:
        """Return all payloads, paging cold records into the resident cache."""
        if self.tiered_storage_manager is None:
            return list(self.memory_units.values())
        return [
            unit
            for uid in self._all_known_uids()
            if (unit := self.get_unit(uid)) is not None
        ]

    
    # Compatibility helpers retained for the public Mandol API.
    

    def get_all_memory_space_names(self) -> List[str]:
        """Return all memory space names."""
        result = set()

        def _collect_names(space: MemorySpace):
            result.add(space.name)
            for child_space_name in space.get_child_space_names():
                child_space = self.memory_spaces.get(child_space_name)
                if child_space:
                    _collect_names(child_space)

        for space in self.memory_spaces.values():
            _collect_names(space)
        return list(result)

    def get_memory_space_structures(self) -> List[dict]:
        """Serialize MemorySpace hierarchy and representative unit fields."""

        def _struct(space: MemorySpace):
            
            unit_uids = list(space.get_unit_uids())
            unit_fields = set()
            for uid in unit_uids:
                unit = self.get_unit(uid)
                if unit:
                    unit_fields.update(unit.raw_data.keys())

            child_space_names = list(space.get_child_space_names())
            children = []
            for child_name in child_space_names:
                child_space = self.memory_spaces.get(child_name)
                if child_space:
                    children.append(_struct(child_space))

            d = {
                "name": space.name,
                "unit_uids": unit_uids,
                "unit_fields": sorted(list(unit_fields)),
            }
            if children:
                d["children"] = children
            return d

        all_child_names = set()
        for space in self.memory_spaces.values():
            all_child_names.update(space.get_child_space_names())

        root_spaces = [
            space
            for space in self.memory_spaces.values()
            if space.name not in all_child_names
        ]

        return [_struct(space) for space in root_spaces]

    def deduplicate_units(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        """Deduplicate units."""
        seen = set()
        result = []
        for u in units:
            if u.uid not in seen:
                seen.add(u.uid)
                result.append(u)
        return result

    def units_union(self, *args) -> List[MemoryUnit]:
        """Merge unit collections while preserving first-seen order."""
        seen = set()
        result = []
        for arg in args:
            units = self._expand_to_units(arg)
            for u in units:
                if u.uid not in seen:
                    seen.add(u.uid)
                    result.append(u)
        return result

    def units_intersection(self, *args) -> List[MemoryUnit]:
        """Return units present in every expanded input collection."""
        if not args:
            return []
        sets = [set(u.uid for u in self._expand_to_units(arg)) for arg in args]
        common_uids = set.intersection(*sets) if sets else set()
        # Preserve object identity from the first input collection.
        first_units = {u.uid: u for u in self._expand_to_units(args[0])}
        return [first_units[uid] for uid in common_uids if uid in first_units]

    def units_difference(self, arg1, arg2) -> List[MemoryUnit]:
        """Return units from the first collection whose UIDs are absent from the second."""
        uids2 = set(u.uid for u in self._expand_to_units(arg2))
        return [u for u in self._expand_to_units(arg1) if u.uid not in uids2]

    def _expand_to_units(self, obj) -> List[MemoryUnit]:
        """Expand to units."""
        result = []
        if obj is None:
            return result
        if isinstance(obj, MemoryUnit):
            result.append(obj)
        elif isinstance(obj, str):
            space_name = obj[3:] if obj.startswith("ms:") else obj
            ms = self.memory_spaces.get(space_name)
            if ms:
                result.extend(
                    unit
                    for uid in ms.get_all_unit_uids(recursive=True)
                    if (unit := self.get_unit(uid)) is not None
                )
            else:
                u = self.get_unit(obj)
                if u:
                    result.append(u)
        elif hasattr(obj, "get_all_units"):
            result.extend(obj.get_all_units())
        elif isinstance(obj, (list, set, tuple)):
            for item in obj:
                result.extend(self._expand_to_units(item))
        return result

    
    
    

    def _clear_legacy_space_indexes(self) -> None:
        """Clear legacy space indexes."""
        for space in self.memory_spaces.values():
            if hasattr(space, "_emb_index"):
                delattr(space, "_emb_index")
            if hasattr(space, "_index_to_uid"):
                delattr(space, "_index_to_uid")

    def build_index(self):
        """Build index."""
        source_units = (
            list(self.memory_units.values())
            if self.tiered_storage_manager is None
            else self.get_all_units()
        )
        if not source_units:
            logger.info("No memory units are available for index construction.")
            if self.faiss_index:
                self.faiss_index.reset()  
            return

        valid_embeddings: List[np.ndarray] = []
        internal_faiss_ids_for_index: List[int] = []

        for unit in source_units:
            uid = unit.uid
            if (
                unit.embedding is not None
                and unit.embedding.shape[0] == self.embedding_dim
            ):
                valid_embeddings.append(unit.embedding)
                internal_faiss_ids_for_index.append(self._get_or_create_int_id(uid))
            else:
                logger.debug(
                    f"Memory unit '{uid}' has no valid embedding and will not be included in the FAISS index."
                )

        if not valid_embeddings:
            logger.info("No valid embeddings are available for FAISS index construction.")
            if self.faiss_index:
                self.faiss_index.reset()
            return

        embeddings_np = np.array(valid_embeddings).astype(np.float32)
        ids_np = np.array(internal_faiss_ids_for_index, dtype=np.int64)

        
        self._init_faiss_index()
        if not self.faiss_index:
            logger.error("FAISS index is not initialized; cannot build.")
            return

        
        if "IVF" in self.faiss_index_type and not self.faiss_index.is_trained:
            logger.info(f"Training FAISS index ('{self.faiss_index_type}')...")
            if embeddings_np.shape[0] < getattr(self.faiss_index, "nlist", 1):
                logger.warning(
                    f"Too few training vectors ({embeddings_np.shape[0]} vectors) for an IVF index. This may cause errors or poor performance."
                )
            if embeddings_np.shape[0] > 0:
                self.faiss_index.train(embeddings_np)
                logger.info("FAISS index training complete.")
            else:
                logger.error("No data is available to train the FAISS index.")
                return

        self.faiss_index.add_with_ids(embeddings_np, ids_np)
        logger.info(
            f"FAISS index built or rebuilt successfully with {self.faiss_index.ntotal} vectors."
        )
        self._clear_legacy_space_indexes()
        self._invalidate_space_filter_cache()

    def get_multi_retriever(self):
        """Return multi retriever."""
        if self._multi_retriever is not None:
            return self._multi_retriever

        from ..retrieval.advance_retriever import MultiRetriever
        from ..retrieval.retrieval_interface import RetrievalMethod

        self._multi_retriever = MultiRetriever(self)
        
        
        
        
        retrieval_indices_dir = getattr(self, '_index_loading_root', None)
        
        
        if not retrieval_indices_dir and hasattr(self, 'storage_path') and self.storage_path:
             retrieval_indices_dir = os.path.join(self.storage_path, "retrieval_indices")

        if retrieval_indices_dir and os.path.exists(retrieval_indices_dir):
            logger.info(f" [SemanticMap] Restoring retriever indexes from disk: {retrieval_indices_dir}")
            
            
            bm25_path = os.path.join(retrieval_indices_dir, "bm25")
            if os.path.exists(bm25_path):
                try:
                    self._multi_retriever._ensure_retriever_loaded(RetrievalMethod.BM25)
                    bm25_retriever = self._multi_retriever.retrievers.get(RetrievalMethod.BM25)
                    if bm25_retriever and not getattr(bm25_retriever, '_index_built', False):
                        if bm25_retriever.load_index(bm25_path):
                            logger.info(" BM25 index loaded successfully")
                        else:
                            logger.warning(" BM25 index load failed")
                except Exception as e:
                    logger.warning(f"Exception while loading BM25: {e}")

            
            splade_path = os.path.join(retrieval_indices_dir, "splade")
            if os.path.exists(splade_path):
                try:
                    self._multi_retriever._ensure_retriever_loaded(RetrievalMethod.SPLADE)
                    splade_retriever = self._multi_retriever.retrievers.get(RetrievalMethod.SPLADE)
                    if splade_retriever and not getattr(splade_retriever, '_index_built', False):
                        if splade_retriever.load_index(splade_path):
                            logger.info(" SPLADE index loaded successfully")
                        else:
                            logger.warning(" SPLADE index load failed")
                except Exception as e:
                    logger.warning(f"Exception while loading SPLADE: {e}")
            
            self._index_loading_root = None
            
        return self._multi_retriever

    def build_freeze_indexes(self) -> Dict[str, bool]:
        """Build freeze indexes."""
        multi_retriever = self.get_multi_retriever()
        if multi_retriever is None:
            logger.warning("MultiRetriever is not initialized; cannot build static acceleration indexes.")
            return {}

        return multi_retriever.build_freeze_indexes()
    
    # def get_multi_retriever(self):
    #     if self._multi_retriever_manager is None:
    #         try:
    #             from ..retrieval.advance_retriever import MultiRetriever
    #             self._multi_retriever_manager = MultiRetriever(self)
    #         except Exception as e:
    #             import traceback
    #             return None
    #     return self._multi_retriever_manager

    # def get_multi_retriever(self):
    #     if self._multi_retriever_manager is None:
    #         try:
    #             from ..retrieval.advance_retriever import MultiRetriever
    #             self._multi_retriever_manager = MultiRetriever(self)
    #         except Exception as e:
    #             return None
    #     return self._multi_retriever_manager

    def search_hybrid(self, query: str, **kwargs):
        """Run hybrid retrieval when auxiliary retrievers are available."""
        multi_retriever = self.get_multi_retriever()
        if multi_retriever:
            return multi_retriever.search_hybrid(query, **kwargs)
        else:
            # Dense retrieval remains available when the hybrid manager is absent.
            logger.warning("MultiRetriever is unavailable; falling back to cosine similarity retrieval.")
            return self.search_similarity_by_text(query, **kwargs)

    # def search_multi_method(self, query: str, methods: Optional[List[str]] = None, **kwargs):
    #     try:
    #         return self._multi_retriever_manager.search_hybrid(query, methods=methods, **kwargs)
    #     except Exception as e:
    # Dense retrieval remains available when the hybrid manager is absent.
    #         return self.search_similarity_by_text(query, **kwargs)

    def search_similarity_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        ms_names: Optional[List[str]] = None,
        candidate_uids: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Retrieve similarity by vector."""
        if query_embedding is None or query_embedding.shape[0] != self.embedding_dim:
            logger.error(f"Query embedding is invalid or has the wrong dimension (expected {self.embedding_dim}).")
            return []
        if k <= 0:
            logger.warning("Search parameter k must be greater than 0.")
            return []

        
        normalized_query = self._normalize_vector(query_embedding)
        query_embedding_np = normalized_query.reshape(1, -1).astype(np.float32)

        pure_space_filter = bool(ms_names and not candidate_uids)
        candidate_int_ids = self._get_space_filter_int_ids(ms_names) if pure_space_filter else None
        candidate_uids_set = None if pure_space_filter else self._get_candidate_uids_set(ms_names, candidate_uids)

        if pure_space_filter and candidate_int_ids is not None and candidate_int_ids.size == 0:
            logger.info("No candidate units are available for search.")
            return []

        if candidate_uids_set is not None and not candidate_uids_set:
            logger.info("No candidate units are available for search.")
            if ms_names:
                logger.warning(f"Requested spaces {ms_names} contain no candidate units")
                for space_name in ms_names:
                    if space_name in self.memory_spaces:
                        space = self.memory_spaces[space_name]
                        unit_count = len(space.get_all_unit_uids(recursive=True))
                        logger.warning(f"  space '{space_name}' exists and contains {unit_count} units from external storage")
                    else:
                        logger.warning(f"  space '{space_name}' does not exist")
                logger.warning(f"Available spaces: {list(self.memory_spaces.keys())}")
            return []

        if pure_space_filter and candidate_int_ids is not None:
            candidate_count = int(candidate_int_ids.size)
        else:
            candidate_count = (
                len(candidate_uids_set)
                if candidate_uids_set is not None
                else self._total_unit_count()
            )
        logger.debug(f"Global index search: candidate_count={candidate_count}, filtered={candidate_uids_set is not None or pure_space_filter}")

        
        index_available = self.faiss_index is not None and self.faiss_index.ntotal > 0
        
        if not index_available:
            logger.warning(
                f"Global FAISS index is not built or is empty (candidate_count: {candidate_count}), "
                f"falling back to brute-force cosine similarity search"
            )
            
            brute_force_candidates = (
                self._get_candidate_uids_set(ms_names, None) if pure_space_filter else candidate_uids_set
            )
            return self._search_with_brute_force_cosine(
                query_embedding_np, normalized_query, k, brute_force_candidates
            )

        
        try:
            return self._search_with_global_index_cosine(
                query_embedding_np, normalized_query, k, candidate_uids_set, candidate_int_ids
            )
        except Exception as e:
            
            logger.warning(f"Index search failed: {e}, falling back to brute-force search")
            try:
                return self._search_with_brute_force_cosine(
                    query_embedding_np, normalized_query, k, candidate_uids_set
                )
            except Exception as brute_force_error:
                logger.error(f"Brute-force search also failed: {brute_force_error}")
                import traceback
                logger.debug(f"Detailed error: {traceback.format_exc()}")
                return []
        
    def _search_with_global_index_cosine(
        self,
        query_embedding_np: np.ndarray,
        normalized_query: np.ndarray,
        k: int,
        candidate_uids_set: Optional[Set[str]],
        candidate_int_ids: Optional[np.ndarray] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search with global index cosine."""
        logger.debug("Using the global FAISS index for cosine similarity search")

        if self.faiss_index is None or self.faiss_index.ntotal <= 0:
            return []

        if candidate_uids_set is None and candidate_int_ids is None:
            similarities, internal_faiss_indices = self.faiss_index.search(query_embedding_np, k)
            return self._process_cosine_search_results(
                similarities, internal_faiss_indices, None
            )

        if (
            candidate_uids_set is not None
            and len(candidate_uids_set) >= self._total_unit_count()
        ):
            similarities, internal_faiss_indices = self.faiss_index.search(query_embedding_np, k)
            return self._process_cosine_search_results(
                similarities, internal_faiss_indices, candidate_uids_set
            )

        if candidate_int_ids is not None:
            final_target_internal_ids_np = np.ascontiguousarray(candidate_int_ids, dtype=np.int64)
        else:
            target_internal_faiss_ids = [
                self._uid_to_int_id[uid]
                for uid in candidate_uids_set
                if uid in self._uid_to_int_id
            ]
            final_target_internal_ids_np = np.ascontiguousarray(
                np.array(target_internal_faiss_ids, dtype=np.int64)
            )

        if final_target_internal_ids_np.size == 0:
            logger.info("No candidate units have matching internal IDs in the FAISS index.")
            return []

        import faiss
        try:
            id_selector = faiss.IDSelectorBatch(final_target_internal_ids_np)
        except TypeError:
            id_selector = faiss.IDSelectorBatch(
                len(final_target_internal_ids_np),
                faiss.swig_ptr(final_target_internal_ids_np),
            )

        search_params = (
            faiss.SearchParametersIVF()
            if "IVF" in self.faiss_index_type
            else faiss.SearchParameters()
        )
        search_params.sel = id_selector
        k = min(k, len(final_target_internal_ids_np))
        similarities, internal_faiss_indices = self.faiss_index.search(
            query_embedding_np, k, params=search_params
        )

        return self._process_cosine_search_results(
            similarities, internal_faiss_indices, candidate_uids_set
        )

    def _search_with_brute_force_normalized(
        self, query_embedding_np: np.ndarray, normalized_query: np.ndarray, k: int, candidate_uids_set: Set[str]
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search with brute force normalized."""
        logger.debug("Using brute-force cosine similarity search for multi-space or candidate-filtered search")

        candidate_units = []
        for uid in candidate_uids_set:
            unit = self.get_unit(uid)
            if unit and unit.embedding is not None:
                candidate_units.append(unit)

        if not candidate_units:
            logger.info("No valid candidate units are available for search.")
            return []

        results = []
        for unit in candidate_units:
            try:
                similarity = self._calculate_cosine_similarity(normalized_query, unit.embedding)
                results.append((unit, similarity))
            except Exception as e:
                logger.warning(f"Failed to compute cosine similarity for unit '{unit.uid}': {e}")
                continue

        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _search_with_single_space_index_cosine(
        self, query_embedding_np: np.ndarray, normalized_query: np.ndarray, k: int, space_name: str
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search with single space index cosine."""
        space = self.get_memory_space(space_name)
        if not space:
            logger.warning(f"Memory space '{space_name}' does not exist")
            return []
        candidate_uids = set(space.get_all_unit_uids(recursive=True))
        if not candidate_uids:
            logger.info(f"Memory space '{space_name}' contains no units")
            return []
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            try:
                return self._search_with_global_index_cosine(
                    query_embedding_np, normalized_query, k, candidate_uids
                )
            except Exception as e:
                logger.warning(f"Global selector search failed for space '{space_name}': {e}, falling back to NumPy brute-force search")
        return self._search_with_brute_force_cosine(
            query_embedding_np, normalized_query, k, candidate_uids
        )

    def _search_with_brute_force_cosine(
        self, query_embedding_np: np.ndarray, normalized_query: np.ndarray, k: int, candidate_uids_set: Optional[Set[str]]
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search with brute force cosine."""
        candidate_uid_iterable = (
            candidate_uids_set
            if candidate_uids_set is not None
            else (
                self.memory_units.keys()
                if self.tiered_storage_manager is None
                else self._all_known_uids()
            )
        )
        candidate_count = (
            len(candidate_uids_set)
            if candidate_uids_set is not None
            else self._total_unit_count()
        )
        logger.debug(f"Using brute-force cosine similarity search (candidate_count: {candidate_count})")

        
        candidate_units = []
        embeddings = []
        for uid in candidate_uid_iterable:
            unit = self.get_unit(uid)
            if unit and unit.embedding is not None:
                candidate_units.append(unit)
                embeddings.append(unit.embedding)

        if not candidate_units:
            logger.warning("No valid candidates are available for brute-force search; all candidates are missing embeddings.")
            return []

        n_candidates = len(candidate_units)
        logger.debug(f"Brute-force search valid candidate count: {n_candidates}")

        try:
            embedding_matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            valid_norms = norms.squeeze() >= 1e-6
            if not np.all(valid_norms):
                embedding_matrix = embedding_matrix[valid_norms]
                candidate_units = [unit for unit, keep in zip(candidate_units, valid_norms) if bool(keep)]
                n_candidates = len(candidate_units)
                if n_candidates == 0:
                    return []
                norms = norms[valid_norms]
            embedding_matrix = embedding_matrix / norms

            query_vec = normalized_query.reshape(-1).astype(np.float32, copy=False)
            scores_flat = embedding_matrix @ query_vec
            k_actual = min(k, n_candidates)
            if k_actual <= 0:
                return []
            if scores_flat.size > k_actual:
                indices_flat = np.argpartition(scores_flat, -k_actual)[-k_actual:]
                indices_flat = indices_flat[np.argsort(scores_flat[indices_flat])[::-1]]
            else:
                indices_flat = np.argsort(scores_flat)[::-1]

            results = []
            for idx_raw in indices_flat:
                idx = int(idx_raw)
                similarity = float(scores_flat[idx])
                normalized_similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
                results.append((candidate_units[idx], normalized_similarity))

            logger.debug(f"Brute-force search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Brute-force search execution failed: {e}")
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return []
        
    # def _search_with_brute_force_cosine(
    #     self, query_embedding_np: np.ndarray, normalized_query: np.ndarray, k: int, candidate_uids_set: Set[str]
    # ) -> List[Tuple[MemoryUnit, float]]:

    #     candidate_units = []
    #     for uid in candidate_uids_set:
    #         unit = self.get_unit(uid)
    #         if unit and unit.embedding is not None:
    #             candidate_units.append(unit)

    #     if not candidate_units:
    #         return []

    #     results = []
    #     for unit in candidate_units:
    #         try:
    #             similarity = self._calculate_cosine_similarity(normalized_query, unit.embedding)
    #             results.append((unit, similarity))
    #         except Exception as e:
    
    #             continue

    
    #     results.sort(key=lambda x: x[1], reverse=True)
    #     return results[:k]

    def _convert_l2_to_cosine_similarity_score(self, l2_distance: float) -> float:
        """Convert L2 to cosine similarity score."""
        return max(0.0, 1.0 / (1.0 + l2_distance))

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        try:
            a = vec1.astype(np.float32)
            b = vec2.astype(np.float32)
            
            dot_product = float(np.dot(a, b))
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            cosine_sim = dot_product / (norm_a * norm_b + 1e-8)
            
            return (cosine_sim + 1.0) / 2.0
            
        except Exception as e:
            logger.warning(f"Cosine similarity computation failed: {e}")
            return 0.0

    def _process_cosine_search_results(
        self,
        similarities: np.ndarray,
        internal_faiss_indices: np.ndarray,
        candidate_uids_set: Optional[Set[str]],
    ) -> List[Tuple[MemoryUnit, float]]:
        """Process cosine search results."""
        results: List[Tuple[MemoryUnit, float]] = []
        internal_id_to_uid_map = self._get_int_id_to_uid_map()

        for i in range(internal_faiss_indices.shape[1]):
            internal_id = internal_faiss_indices[0, i]
            if internal_id == -1:
                continue

            uid = internal_id_to_uid_map.get(internal_id)
            if uid and (candidate_uids_set is None or uid in candidate_uids_set):
                unit = self.get_unit(uid)
                if unit:
                    
                    similarity = float(similarities[0][i])
                    normalized_similarity = (similarity + 1.0) / 2.0
                    results.append((unit, max(0.0, min(1.0, normalized_similarity))))

        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _determine_search_strategy(
        self, ms_names: Optional[List[str]], candidate_uids: Optional[List[str]]
    ) -> str:
        """Determine search strategy."""
        return "global_index"

    # def add_ms_prefix(self, names):
    #     if not names:
    #         return []
    #     return [n if n.startswith("ms:") else f"ms:{n}" for n in names]

    def _get_space_filter_int_ids(self, ms_names: List[str]) -> np.ndarray:
        """Return cached FAISS int IDs for a pure MemorySpace filter."""
        normalized_names = tuple(sorted(str(name) for name in ms_names))
        cache_key = (normalized_names, self._space_membership_version)
        cached = self._space_filter_cache.get(cache_key)
        if cached is not None:
            return cached

        space_uids = self._get_candidate_uids_set(list(normalized_names), None)
        if not space_uids:
            int_ids = np.empty(0, dtype=np.int64)
        else:
            int_ids = np.ascontiguousarray(
                np.array(
                    [self._uid_to_int_id[uid] for uid in space_uids if uid in self._uid_to_int_id],
                    dtype=np.int64,
                )
            )
        self._space_filter_cache[cache_key] = int_ids
        return int_ids

    def _get_candidate_uids_set(
        self, ms_names: Optional[List[str]], candidate_uids: Optional[List[str]]
    ) -> Optional[Set[str]]:
        """Resolve retrieval filters without materializing payloads."""
        space_uids: Optional[Set[str]] = None
        if ms_names and candidate_uids:
            space_uids = set()
            for raw_name in ms_names:
                name = raw_name[3:] if raw_name.startswith("ms:") else raw_name
                space = self.memory_spaces.get(name)
                if space is not None:
                    space_uids.update(space.get_all_unit_uids(recursive=True))
            result = space_uids.intersection(str(uid) for uid in candidate_uids)
            logger.debug(
                "Intersection mode: spaces%s and candidate units, result_count=%d",
                ms_names,
                len(result),
            )
            return result
        elif ms_names:
            space_uids = set()
            for raw_name in ms_names:
                name = raw_name[3:] if raw_name.startswith("ms:") else raw_name
                space = self.memory_spaces.get(name)
                if space is not None:
                    space_uids.update(space.get_all_unit_uids(recursive=True))
            logger.debug(
                "Union mode: spaces%s, result_count=%d", ms_names, len(space_uids)
            )
            return space_uids
        elif candidate_uids:
            known_uids = self._all_known_uids()
            result = {str(uid) for uid in candidate_uids if str(uid) in known_uids}
            logger.debug(
                "Candidate-unit mode: candidates=%d, result_count=%d",
                len(candidate_uids),
                len(result),
            )
            return result
        else:
            logger.debug("Global mode: all units, result_count=%d", self._total_unit_count())
            return None

    def _search_with_global_index(
        self, query_embedding_np: np.ndarray, k: int, candidate_uids_set: Set[str]
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search with global index."""
        logger.debug("Using the global FAISS index for search")

        if self.faiss_index is None or self.faiss_index.ntotal <= 0:
            return []

        if len(candidate_uids_set) >= self._total_unit_count():
            distances, internal_faiss_indices = self.faiss_index.search(query_embedding_np, k)
            return self._process_search_results(
                distances, internal_faiss_indices, candidate_uids_set
            )

        target_internal_faiss_ids = [
            self._uid_to_int_id[uid]
            for uid in candidate_uids_set
            if uid in self._uid_to_int_id
        ]

        if not target_internal_faiss_ids:
            logger.info("No candidate units have matching internal IDs in the FAISS index.")
            return []

        import faiss

        final_target_internal_ids_np = np.ascontiguousarray(
            np.array(target_internal_faiss_ids, dtype=np.int64)
        )
        try:
            id_selector = faiss.IDSelectorBatch(final_target_internal_ids_np)
        except TypeError:
            id_selector = faiss.IDSelectorBatch(
                len(final_target_internal_ids_np),
                faiss.swig_ptr(final_target_internal_ids_np),
            )
        search_params = (
            faiss.SearchParametersIVF()
            if "IVF" in self.faiss_index_type
            else faiss.SearchParameters()
        )
        search_params.sel = id_selector
        k = min(k, len(final_target_internal_ids_np))
        distances, internal_faiss_indices = self.faiss_index.search(
            query_embedding_np, k, params=search_params
        )

        return self._process_search_results(
            distances, internal_faiss_indices, candidate_uids_set
        )

    def _search_with_single_space_index(
        self, query_embedding_np: np.ndarray, k: int, space_name: str
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search with single space index."""
        space = self.get_memory_space(space_name)
        if not space:
            logger.warning(f"Memory space '{space_name}' does not exist")
            return []
        if self.faiss_index is None or self.faiss_index.ntotal <= 0:
            return self._search_with_brute_force(query_embedding_np, k, set(space.get_all_unit_uids(recursive=True)))
        return self._search_with_global_index(
            query_embedding_np, k, set(space.get_all_unit_uids(recursive=True))
        )

    def _search_with_brute_force(
        self, query_embedding_np: np.ndarray, k: int, candidate_uids_set: Set[str]
    ) -> List[Tuple[MemoryUnit, float]]:
        """Search with brute force."""
        logger.debug("Using brute-force search for multi-space or candidate-filtered search")

        candidate_units = []
        for uid in candidate_uids_set:
            unit = self.get_unit(uid)
            if unit and unit.embedding is not None:
                candidate_units.append(unit)

        if not candidate_units:
            logger.info("No valid candidate units are available for search.")
            return []

        
        embeddings = np.array(
            [unit.embedding for unit in candidate_units], dtype=np.float32
        )

        
        try:
            import faiss

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            distances, indices = index.search(
                query_embedding_np, min(k, len(candidate_units))
            )

            results = []
            for i in range(len(indices[0])):
                idx = int(indices[0][i])
                if indices[0][i] == -1:
                    continue
                results.append((candidate_units[idx], float(distances[0][i])))

            return results

        except ImportError:
            logger.error("FAISS is unavailable; cannot run brute-force search.")
            return []
        except Exception as e:
            logger.error(f"Brute-force search failed: {e}")
            return []

    def _process_search_results(
        self,
        distances: np.ndarray,
        internal_faiss_indices: np.ndarray,
        candidate_uids_set: Set[str],
    ) -> List[MemoryUnit]:
        """Process search results."""
        results: List[MemoryUnit] = []
        internal_id_to_uid_map = self._get_int_id_to_uid_map()

        for i in range(internal_faiss_indices.shape[1]):
            internal_id = internal_faiss_indices[0, i]
            if internal_id == -1:
                continue

            uid = internal_id_to_uid_map.get(internal_id)
            if uid and (candidate_uids_set is None or uid in candidate_uids_set):
                unit = self.get_unit(uid)
                if unit:
                    results.append(unit)
                else:
                    logger.warning(
                        f"FAISS returned internal ID {internal_id}, but unit ID '{uid}' was not found in memory_units."
                    )
            else:
                logger.debug(
                    f"FAISS returned internal ID {internal_id} with no corresponding unit ID."
                )

        return results

    def search_similarity_by_text(
        self,
        query_text: str,
        k: int = 5,
        ms_names: Optional[List[str]] = None,
        candidate_uids: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Retrieve similarity by text."""
        query_embedding = self._get_text_embedding(query_text)
        if query_embedding is None:
            return []
        return self.search_similarity_by_vector(
            query_embedding, k, ms_names, candidate_uids
        )

    def search_similarity_by_image(
        self,
        image_path: str,
        k: int = 5,
        ms_names: Optional[List[str]] = None,
        candidate_uids: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Retrieve similarity by image."""
        if self.image_model is None:
            logger.warning("The current primary model only supports text; image retrieval is disabled.")
            return []
        query_embedding = self._get_image_embedding(image_path)
        if query_embedding is None:
            return []
        return self.search_similarity_by_vector(
            query_embedding, k, ms_names, candidate_uids
        )

    
    
    
    def save_map(
        self,
        directory_path: str,
        freeze_retrievers: bool = False,
        build_sparse_vectors: bool = True,
    ):
        """Persist a complete resident SemanticMap checkpoint.

        Tiered maps must be saved through ``SemanticGraph.save_graph()`` so the
        resident state and RocksDB payload catalog are captured together.
        """
        if self.tiered_storage_manager is not None:
            raise RuntimeError(
                "SemanticMap.save_map() cannot create a complete checkpoint "
                "while tiered paging is enabled; use SemanticGraph.save_graph()."
            )
        return self._save_map_impl(
            directory_path,
            freeze_retrievers=freeze_retrievers,
            build_sparse_vectors=build_sparse_vectors,
        )

    def _save_map_for_graph_snapshot(
        self,
        directory_path: str,
        freeze_retrievers: bool = False,
        build_sparse_vectors: bool = True,
    ):
        """Save resident map state inside a graph-managed tiered snapshot."""
        return self._save_map_impl(
            directory_path,
            freeze_retrievers=freeze_retrievers,
            build_sparse_vectors=build_sparse_vectors,
        )

    def _save_map_impl(
        self,
        directory_path: str,
        freeze_retrievers: bool = False,
        build_sparse_vectors: bool = True,
    ):
        """Persist the SemanticMap state and retrieval artifacts.

        Args:
            directory_path: Output directory for metadata, dense vectors, sparse
                vectors, FAISS index files, and retrieval index metadata.
            freeze_retrievers: Whether to materialize static acceleration
                matrices for initialized auxiliary retrievers.
            build_sparse_vectors: Whether missing SPLADE vectors should be
                generated before saving.
        """
        os.makedirs(directory_path, exist_ok=True)
        static_index_status: Dict[str, bool] = {}
        
        
        units_with_embeddings = [u for u in self.memory_units.values() if u.embedding is not None]
        faiss_needs_build = (
            len(units_with_embeddings) > 0 and 
            (self.faiss_index is None or self.faiss_index.ntotal == 0)
        )
        
        if faiss_needs_build:
            logger.info(
                f"Detected {len(units_with_embeddings)} units with dense vectors but no FAISS index; "
                f"calling build_index() before save."
            )
            self.build_index()
            logger.info(f"FAISS index built before save with {self.faiss_index.ntotal} vectors.")

        if freeze_retrievers:
            if self._multi_retriever is not None:
                logger.info("Building SemanticMap retriever static matrices (freeze_retrievers=True).")
                static_index_status = self.build_freeze_indexes()
                logger.info(f"SemanticMap retriever static matrix status: {static_index_status}")
            else:
                logger.info("freeze_retrievers=True but MultiRetriever is not initialized; skipping static matrices.")

        id_mapping_json = {uid: int(int_id) for uid, int_id in self._get_uid_to_int_id_map().items()}
        reverse_mapping_json = {str(int_id): uid for int_id, uid in self._get_int_id_to_uid_map().items()}
        with open(os.path.join(directory_path, "global_id_mapping.json"), 'wb') as f:
            f.write(orjson.dumps(id_mapping_json))
        with open(os.path.join(directory_path, "int_id_to_uid.json"), 'wb') as f:
            f.write(orjson.dumps(reverse_mapping_json))
        with open(os.path.join(directory_path, "id_mapping.json"), 'wb') as f:
            f.write(orjson.dumps(id_mapping_json))
        
        
        if self.faiss_index and self.faiss_index.ntotal > 0:
            index_path = os.path.join(directory_path, "faiss_index.bin")
            faiss.write_index(self.faiss_index, index_path)
            logger.info(f"FAISS index saved to: {index_path}")

        
        uids_with_emb = []
        embeddings_list = []
        for uid, unit in self.memory_units.items():
            if unit.embedding is not None:
                uids_with_emb.append(uid)
                embeddings_list.append(unit.embedding)
        
        if embeddings_list:
            emb_matrix = np.stack(embeddings_list).astype(np.float32)
            
            safetensors_save_file(
                {"dense_vectors": torch.from_numpy(emb_matrix)},
                os.path.join(directory_path, "dense_vectors.safetensors")
            )
            
            with open(os.path.join(directory_path, "dense_uids.json"), 'wb') as f:
                f.write(orjson.dumps(uids_with_emb))
            logger.info(f"Dense embeddings saved (safetensors), shape: {emb_matrix.shape}")

        
        units_without_sparse = [
            u for u in self.memory_units.values() 
            if u.sparse_embedding is None and u.embedding is not None
        ]
        if build_sparse_vectors and units_without_sparse:
            logger.info(
                f"Detected {len(units_without_sparse)} units from external storageunits missing SPLADE sparse embeddings, "
                f"calling build_sparse_embeddings() automatically..."
            )
            try:
                self.build_sparse_embeddings(force_rebuild=False, show_progress=True)
                logger.info("Automatic SPLADE sparse embedding build complete.")
            except Exception as e:
                logger.warning(f"Automatic SPLADE sparse embedding build failed: {e}, continuing to save existing data")
        elif units_without_sparse:
            logger.info(
                f"build_sparse_vectors=False, skipped {len(units_without_sparse)} units from external storageSPLADE sparse embedding auto-build"
            )
        
        
        sparse_uids = []
        sparse_dicts = []
        for uid, unit in self.memory_units.items():
            if unit.sparse_embedding is not None:
                sparse_uids.append(uid)
                if isinstance(unit.sparse_embedding, dict):
                    sparse_dicts.append(unit.sparse_embedding)
                elif isinstance(unit.sparse_embedding, np.ndarray):
                    sparse_dicts.append(MemoryUnit._dense_to_sparse(unit.sparse_embedding))
                else:
                    logger.warning(f"Unit {uid} has an unknown sparse embedding type: {type(unit.sparse_embedding)}, skipped")
                    sparse_uids.pop()
        
        if sparse_dicts:
            indptr = [0]
            indices = []
            data = []
            for doc_vec in sparse_dicts:
                for token_id, score in sorted(doc_vec.items()):
                    indices.append(token_id)
                    data.append(score)
                indptr.append(len(indices))
            
            safetensors_save_file(
                {
                    "csr_indptr": torch.tensor(indptr, dtype=torch.int64),
                    "csr_indices": torch.tensor(indices, dtype=torch.int64),
                    "csr_data": torch.tensor(data, dtype=torch.float32),
                },
                os.path.join(directory_path, "sparse_vectors.safetensors")
            )
            with open(os.path.join(directory_path, "sparse_uids.json"), 'wb') as f:
                f.write(orjson.dumps(sparse_uids))
            logger.info(f"Sparse embeddings saved (CSR safetensors) for {len(sparse_dicts)} units from external storage, nonzero entries {len(indices)}")

        
        meta_units = {}
        for uid, unit in self.memory_units.items():
            meta_units[uid] = {
                "uid": unit.uid,
                "raw_data": unit.raw_data,
                "metadata": unit.metadata,
                "created_time": getattr(unit, 'created_time', None),
                "has_embedding": unit.embedding is not None,
                "has_sparse": unit.sparse_embedding is not None
            }
        
        save_data = {
            "version": "3.0_safetensors",
            "config": {
                "embedding_dim": self.embedding_dim,
                "faiss_index_type": self.faiss_index_type,
                "embedding_model": self._embedding_model_name,
                "supported_modalities": self.supported_modalities,
                "image_model": self._embedding_model_name if "image" in self.supported_modalities else None,
                "text_model": self._embedding_model_name,
            },
            "memory_spaces": {
                name: {
                    "name": s.name, 
                    "unit_uids": list(s._unit_uids),
                    "child_space_names": list(s._child_space_names),
                    "faiss_index_type": getattr(s, '_faiss_index_type', None)
                } for name, s in self.memory_spaces.items()
            },
            "memory_units_meta": meta_units,
            "internal_state": {
                "_next_int_id": self._next_int_id,
                "_uid_to_int_id": {uid: int(int_id) for uid, int_id in self._uid_to_int_id.items()},
                "_modified_units": list(self._modified_units),
                "_deleted_units": list(self._deleted_units),
                "_access_counts": self._access_counts
            },
            "retrieval": {
                "indices_saved": {},
                "frozen_matrices_saved": static_index_status,
            },
            "high_level_memory_build": self.get_high_level_memory_build_state(),
        }
        
        with open(os.path.join(directory_path, "semantic_map_meta.json"), 'wb') as f:
            f.write(orjson.dumps(save_data, option=orjson.OPT_INDENT_2))
            
        logger.info("Optimized save complete (v3 safetensors).")

    @classmethod
    def load_map(cls, directory_path: str, **kwargs) -> "SemanticMap":
        """Load a SemanticMap saved by ``save_map``.

        Args:
            directory_path: Directory containing ``semantic_map_meta.json`` and
                associated vector/index files.
            **kwargs: Constructor overrides or legacy embedding-model aliases.

        Returns:
            A reconstructed SemanticMap instance.
        """
        meta_path = os.path.join(directory_path, "semantic_map_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"SemanticMap metadata file not found: {meta_path}")

        
        with open(meta_path, 'rb') as f:
            data = orjson.loads(f.read())
        
        config = data.get("config", {})
        embedding_model_name = (
            kwargs.pop("embedding_model_name", None)
            or kwargs.pop("text_embedding_model_name", None)
            or config.get("embedding_model")
            or config.get("text_model")
            or config.get("text_embedding_model_name")
        )
        legacy_image_model_name = kwargs.pop("image_embedding_model_name", None)
        if legacy_image_model_name is not None:
            logger.warning(
                "SemanticMap.load_map received deprecated image_embedding_model_name; "
                "multimodal capability is selected by embedding_model_name."
            )

        instance = cls(
            embedding_model_name=embedding_model_name,
            embedding_dim=config.get("embedding_dim"),
            faiss_index_type=config.get("faiss_index_type", "IDMap,Flat"),
            **kwargs,
        )
        instance.set_high_level_memory_build_state(data.get("high_level_memory_build", {}))
        state = data.get("internal_state", {})
        if state.get("_uid_to_int_id"):
            instance._set_uid_int_mapping(state.get("_uid_to_int_id"))

        for uid, u_data in data.get("memory_units_meta", {}).items():
            unit = MemoryUnit(
                uid=u_data["uid"],
                raw_data=u_data["raw_data"],
                metadata=u_data.get("metadata")
            )
            unit.created_time = u_data.get("created_time")
            instance.memory_units[uid] = unit

        
        dense_st_path = os.path.join(directory_path, "dense_vectors.safetensors")
        dense_uids_json_path = os.path.join(directory_path, "dense_uids.json")

        emb_uids = []
        try:
            if os.path.exists(dense_st_path) and os.path.exists(dense_uids_json_path):
                st_data = safetensors_load_file(dense_st_path)
                emb_matrix = st_data["dense_vectors"].numpy()
                with open(dense_uids_json_path, 'rb') as f:
                    emb_uids = orjson.loads(f.read())
                logger.info(f"[v3] Loaded {len(emb_uids)} dense embeddings (safetensors)")
            else:
                emb_matrix, emb_uids = None, []

            if emb_matrix is not None:
                for idx, uid in enumerate(emb_uids):
                    if uid in instance.memory_units:
                        instance.memory_units[uid].embedding = emb_matrix[idx]
        except Exception as e:
            logger.error(f"Failed to load dense embeddings: {e}")

        
        sparse_st_path = os.path.join(directory_path, "sparse_vectors.safetensors")
        sparse_uids_json_path = os.path.join(directory_path, "sparse_uids.json")

        try:
            if os.path.exists(sparse_st_path) and os.path.exists(sparse_uids_json_path):
                csr_data = safetensors_load_file(sparse_st_path)
                indptr = csr_data["csr_indptr"].numpy().astype(np.int64)
                indices = csr_data["csr_indices"].numpy().astype(np.int64)
                values = csr_data["csr_data"].numpy().astype(np.float32)
                with open(sparse_uids_json_path, 'rb') as f:
                    sparse_uids = orjson.loads(f.read())

                for i, uid in enumerate(sparse_uids):
                    if uid in instance.memory_units:
                        start, end = int(indptr[i]), int(indptr[i + 1])
                        sparse_vec = {
                            int(indices[j]): float(values[j])
                            for j in range(start, end)
                        }
                        instance.memory_units[uid].sparse_embedding = sparse_vec
                logger.info(f"[v3] Loaded {len(sparse_uids)} sparse embeddings (CSR safetensors)")
        except Exception as e:
            logger.error(f"Failed to load sparse embeddings: {e}")

        
        index_path = os.path.join(directory_path, "faiss_index.bin")
        global_mapping_json_path = os.path.join(directory_path, "global_id_mapping.json")
        mapping_json_path = os.path.join(directory_path, "id_mapping.json")
        
        if os.path.exists(index_path):
            try:
                instance.faiss_index = faiss.read_index(index_path)

                
                state_mapping = data.get("internal_state", {}).get("_uid_to_int_id")
                if os.path.exists(global_mapping_json_path):
                    with open(global_mapping_json_path, 'rb') as f:
                        instance._set_uid_int_mapping(orjson.loads(f.read()))
                    logger.info("[v3] global_id_mapping loaded from JSON")
                elif os.path.exists(mapping_json_path):
                    with open(mapping_json_path, 'rb') as f:
                        instance._set_uid_int_mapping(orjson.loads(f.read()))
                    logger.info("[v3] id_mapping loaded from JSON and migrated to the global int_id mapping")
                elif state_mapping:
                    instance._set_uid_int_mapping(state_mapping)
                    logger.info("[v3] global_id_mapping loaded from semantic_map_meta.json")
                else:
                    instance._get_or_create_int_ids(emb_uids or instance.memory_units.keys())
                    logger.warning("ID mapping file not found; rebuilt the global int_id mapping from dense_uids/memory_units order.")
                
                state = data.get("internal_state", {})
                instance._next_int_id = max(
                    int(state.get("_next_int_id", 0) or 0),
                    instance._next_int_id,
                )
                logger.info(f"Restored FAISS index directly with {instance.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"FAISS index restore failed; attempting rebuild: {e}")
                instance.build_index()
        else:
            logger.info("No prebuilt index found; rebuilding...")
            instance.build_index()

        for name, s_data in data.get("memory_spaces", {}).items():
            space = instance.create_memory_space(s_data["name"])
            space._unit_uids = set(s_data["unit_uids"])
            space._child_space_names = set(s_data["child_space_names"])
            if s_data.get("faiss_index_type"):
                space._faiss_index_type = s_data["faiss_index_type"]
        instance._invalidate_space_filter_cache()
        
        state = data.get("internal_state", {})
        instance._modified_units = set(state.get("_modified_units", []))
        instance._deleted_units = set(state.get("_deleted_units", []))
        instance._access_counts = state.get("_access_counts", {})
        
        
        
        
        
        retrieval_indices_dir = os.path.join(directory_path, "retrieval_indices")
        if os.path.exists(retrieval_indices_dir):
            instance._index_loading_root = retrieval_indices_dir
            logger.info(f"Detected SemanticMap retriever index directory: {retrieval_indices_dir}")
        else:
            instance._index_loading_root = None
        
        return instance

    def filter_memory_units(
        self,
        candidate_units: Optional[List[MemoryUnit]] = None,
        filter_condition: Optional[dict] = None,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """Filter memory units."""
        if candidate_units is not None:
            units = self.units_union(candidate_units)
        elif ms_names:
            units = self.units_union(*self.add_ms_prefix(ms_names))
        else:
            units = list(self.memory_units.values())

        def match(unit):
            if not filter_condition:
                return True
            for field, cond in filter_condition.items():
                val = getattr(unit, field, None)
                if val is None:
                    val = unit.raw_data.get(field)
                for op, op_val in cond.items():
                    if op == "eq" and not (val == op_val):
                        return False
                    if op == "ne" and not (val != op_val):
                        return False
                    if op == "in" and not (val in op_val):
                        return False
                    if op == "nin" and not (val not in op_val):
                        return False
                    if op == "gt" and not (val > op_val):
                        return False
                    if op == "gte" and not (val >= op_val):
                        return False
                    if op == "lt" and not (val < op_val):
                        return False
                    if op == "lte" and not (val <= op_val):
                        return False
                    if op == "contain" and not (op_val in str(val)):
                        return False
                    if op == "not_contain" and not (op_val not in str(val)):
                        return False
            return True

        return [u for u in units if match(u)]

    def add_memory_space(self, space: "MemorySpace"):
        """Add memory space."""
        if not isinstance(space, MemorySpace):
            raise TypeError("Argument must be a MemorySpace instance")
        space._set_semantic_map_ref(self)
        self.memory_spaces[space.name] = space
        self._invalidate_space_filter_cache()
        
    
    
    
    
    def build_sparse_embeddings(
        self,
        units: Optional[List[MemoryUnit]] = None,
        # text_field: str = "text_content",
        model_name: str = "naver/splade-v3",
        batch_size: int = 32,
        force_rebuild: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Build sparse embeddings."""
        import torch
        from tqdm import tqdm
        
        if units is None:
            units = self.get_all_units()
        
        if not units:
            logger.warning("No units need SPLADE embedding construction.")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}
        
        units_to_process = []
        for unit in units:
            if force_rebuild or not unit.has_sparse_embedding():
                units_to_process.append(unit)
        
        skipped_count = len(units) - len(units_to_process)
        
        if not units_to_process:
            logger.info(f"All {len(units)} units from external storagealready have SPLADE embeddings, skipping construction")
            return {
                "total": len(units),
                "processed": 0,
                "skipped": skipped_count,
                "failed": 0,
                "model": model_name
            }
        
        logger.info(f"Starting SPLADE embedding construction for {len(units_to_process)} units (skipped {skipped_count})")
        
        
        try:
            
            from ..utils.model_manager import global_model_manager
            model = global_model_manager.get_splade_model(model_name)
            logger.info(f" SPLADE model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load SPLADE model: {e}.skipping sparse embedding generation for this batch.")
            return {
                "total": len(units),
                "processed": 0,
                "skipped": skipped_count,
                "failed": len(units_to_process),
                "model": model_name
            }
        
        processed_count = 0
        failed_count = 0
        
        iterator = range(0, len(units_to_process), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Build SPLADE embeddings", unit="batch")
        
        for i in iterator:
            batch_units = units_to_process[i:i + batch_size]
            
            texts = []
            valid_units = []
            for unit in batch_units:
                text = self._extract_text_content_for_embedding(unit)
                
                if text and text.strip():
                    texts.append(text)
                    valid_units.append(unit)
            
            if not texts:
                continue
            
            
            try:
                with torch.no_grad():
                    
                    sparse_tensors = model.encode_document(texts)
                    
                    for unit, sparse_tensor in zip(valid_units, sparse_tensors):
                        try:
                            coalesced = sparse_tensor.coalesce()
                            
                            
                            indices = coalesced.indices().squeeze().cpu().numpy()
                            values = coalesced.values().cpu().numpy()
                            
                            if indices.ndim == 0:
                                indices = np.array([indices.item()])
                                values = np.array([values.item()])
                            
                            
                            sparse_dict = {
                                int(idx): float(val) 
                                for idx, val in zip(indices, values)
                            }
                            
                            unit.set_sparse_embedding(sparse_dict)
                            processed_count += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to process SPLADE embedding for unit {unit.uid}: {e}")
                            failed_count += 1
                            continue
            
            except Exception as e:
                logger.warning(f"Batched SPLADE embedding processing failed: {e}")
                failed_count += len(valid_units)
                continue
        
        result = {
            "total": len(units),
            "processed": processed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "model": model_name
        }
        
        logger.info(
            f" SPLADE embedding build complete: "
            f"total={result['total']}, "
            f"processed={result['processed']}, "
            f"skipped={result['skipped']}, "
            f"failed={result['failed']}"
        )
        
        return result
