from datetime import datetime
import functools
import requests
import httpx
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import SparseEncoder
from abc import ABC, abstractmethod
from enum import Enum
import torch
from collections import defaultdict
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union
import time
from dataclasses import dataclass
from contextlib import contextmanager
import dashscope
from ..utils.config_manager import settings
from ..utils.model_manager import global_model_manager
from ..utils.logging_config import create_module_logger
from ..utils.optional_dependencies import is_flash_attention_available

logger = create_module_logger("rerank_manager")


def _reranker_backend() -> str:
    return settings.reranker_backend


def _reranker_cache_key(reranker_type: str, model_name: str) -> str:
    return f"{_reranker_backend()}:{reranker_type}:{model_name}"


def _default_vllm_prompt(query: str, document: str) -> str:
    return (
        "Given a query and a document, judge whether the document is relevant to the query.\n"
        f"Query: {query}\n"
        f"Document: {document}\n"
        "Relevance score:"
    )


def _first_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().reshape(-1)[0].cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.reshape(-1)[0])
    if isinstance(value, dict):
        for key in (
            "score", "relevance_score", "logit", "logits", "scores", "data",
            "probs", "outputs", "results", "choices", "text", "value",
        ):
            scalar = _first_scalar(value.get(key))
            if scalar is not None:
                return scalar
    if isinstance(value, (list, tuple)):
        for item in value:
            scalar = _first_scalar(item)
            if scalar is not None:
                return scalar
    for attr in (
        "score", "relevance_score", "logit", "logits", "scores", "data",
        "probs", "outputs", "results", "choices", "text", "value",
    ):
        if hasattr(value, attr):
            scalar = _first_scalar(getattr(value, attr))
            if scalar is not None:
                return scalar
    return None


def _parse_vllm_score_outputs(outputs: Any, expected_count: int) -> List[float]:
    if expected_count <= 0:
        return []

    scores: List[Optional[float]] = []
    if isinstance(outputs, (list, tuple)) and len(outputs) == expected_count:
        scores = [_first_scalar(output) for output in outputs]
    elif isinstance(outputs, np.ndarray) and outputs.size == expected_count:
        scores = [_first_scalar(output) for output in outputs.reshape(-1)]
    elif isinstance(outputs, dict):
        for key in ("scores", "data", "outputs", "results", "choices"):
            value = outputs.get(key)
            if isinstance(value, np.ndarray) and value.size == expected_count:
                scores = [_first_scalar(output) for output in value.reshape(-1)]
                break
            if isinstance(value, (list, tuple)) and len(value) == expected_count:
                scores = [_first_scalar(output) for output in value]
                break
    else:
        expanded = getattr(outputs, "outputs", outputs)
        if isinstance(expanded, (list, tuple)) and len(expanded) == expected_count:
            scores = [_first_scalar(output) for output in expanded]
        else:
            scalar = _first_scalar(outputs)
            scores = [scalar] if expected_count == 1 else []

    parsed = [float(score) if score is not None else 0.0 for score in scores]
    if len(parsed) != expected_count:
        logger.warning("vLLM reranking returned an unexpected number of scores: expected=%d actual=%d", expected_count, len(parsed))
        parsed.extend([0.0] * (expected_count - len(parsed)))
    return parsed[:expected_count]


def _generate_vllm_fallback_scores(num_documents: int) -> List[float]:
    scores: List[float] = []
    current_score = 0.01
    step = 1e-5
    for _ in range(num_documents):
        scores.append(current_score)
        current_score = max(current_score - step, 1e-12)
    return scores


def _build_vllm_payload(
    api_url: str,
    model_name: str,
    query: str,
    documents: List[str],
    prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized_url = api_url.rstrip("/").lower()
    if normalized_url.endswith("/v1/completions") or normalized_url.endswith("/completions"):
        completion_prompts = prompts or [_default_vllm_prompt(query, document) for document in documents]
        return {
            "model": model_name,
            "prompts": completion_prompts,
            "prompt": completion_prompts,
            "max_tokens": 1,
            "temperature": 0,
        }
    if normalized_url.endswith("/v1/rerank") or normalized_url.endswith("/rerank"):
        return {
            "model": model_name,
            "query": query,
            "documents": documents,
        }
    return {
        "model": model_name,
        "text_1": [query] * len(documents),
        "text_2": documents,
    }


def _build_vllm_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = settings.get_api_key("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


_LOOPBACK_NO_PROXY_MOUNTS = (
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://localhost",
    "https://localhost",
    "http://[::1]",
    "https://[::1]",
)


def _loopback_no_proxy_mounts() -> Dict[str, None]:
    return {prefix: None for prefix in _LOOPBACK_NO_PROXY_MOUNTS}


def _create_proxy_aware_async_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        mounts=_loopback_no_proxy_mounts(),
        trust_env=True,
    )


def _create_vllm_http_components(model_name: str) -> Dict[str, Any]:
    return {
        "backend": "vllm",
        "model_name": model_name,
        "api_url": settings.vllm_api_url,
        "timeout": settings.vllm_timeout_seconds,
        "max_retries": settings.vllm_max_retries,
        "device": torch.device("cpu"),
    }


async def _score_with_vllm_engine(
    client: httpx.AsyncClient,
    api_url: str,
    model_name: str,
    query: str,
    documents: List[str],
    *,
    prompts: Optional[List[str]] = None,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> List[float]:
    expected_count = len(documents)
    payload = _build_vllm_payload(api_url, model_name, query, documents, prompts)
    headers = _build_vllm_headers()
    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.post(api_url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            scores = _parse_vllm_score_outputs(response_data, expected_count)
            if len(scores) == expected_count:
                return scores
            last_error = f"unexpected score count: expected={expected_count}, actual={len(scores)}"

        except httpx.TimeoutException as exc:
            last_error = f"vLLM request timeout: {exc}"
            logger.warning("vLLM HTTP reranking timeout (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
        except httpx.RequestError as exc:
            last_error = f"vLLM network request failed: {exc}"
            logger.warning("vLLM HTTP reranking network error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            last_error = f"vLLM HTTP unexpected status: {status_code}"
            logger.warning("vLLM HTTP reranking unexpected status (attempt %d/%d): %s", attempt + 1, max_retries + 1, last_error)
        except Exception as exc:
            last_error = f"vLLM response parsing failed: {exc}"
            logger.warning("vLLM HTTP reranking failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)

        if attempt < max_retries:
            await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))

    logger.error("vLLM HTTP reranking failed; using fallback scores: %s", last_error)
    return _generate_vllm_fallback_scores(expected_count)


class VLLMHttpRerankerMixin:

    def _init_vllm_http(self, components: Dict[str, Any]) -> None:
        self.vllm_model_name = components.get("model_name", self.model_name)
        self.vllm_api_url = components.get("api_url", settings.vllm_api_url)
        self.timeout = float(components.get("timeout", settings.vllm_timeout_seconds))
        self.max_retries = int(components.get("max_retries", settings.vllm_max_retries))
        self._async_client: Optional[httpx.AsyncClient] = None
        self.logger.info(
            " vLLM HTTP reranker initialized successfully: model=%s api_url=%s",
            self.vllm_model_name,
            self.vllm_api_url,
        )

    def _get_async_client(self) -> httpx.AsyncClient:
        if getattr(self, "_async_client", None) is None:
            self._async_client = _create_proxy_aware_async_client(self.timeout)
        return self._async_client

    async def _score_vllm_documents(
        self,
        query: str,
        documents: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        return await _score_with_vllm_engine(
            self._get_async_client(),
            self.vllm_api_url,
            self.vllm_model_name,
            query,
            documents,
            prompts=prompts,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    async def close_async(self) -> None:
        if getattr(self, "_async_client", None) is not None:
            await self._async_client.aclose()
            self._async_client = None
            self.logger.info(" vLLM HTTP client closed")

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
    BAAI_AVAILABLE = True
    QWEN_AVAILABLE = True
except ImportError:
    BAAI_AVAILABLE = False
    QWEN_AVAILABLE = False
    logger.warning("BAAI reranking dependency unavailable; install: pip install torch transformers")


@functools.lru_cache(maxsize=1)
def get_hardware_optimized_params() -> Dict[str, int]:
    """Return hardware optimized params."""
    conservative_params = {"batch_size": 16, "pad_to_multiple_of": 8}

    if not torch.cuda.is_available():
        return conservative_params

    try:
        device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_memory_gb = device_properties.total_memory / (1024 ** 3)
    except Exception:
        return conservative_params

    if total_memory_gb >= 70:
        return {"batch_size": 128, "pad_to_multiple_of": 64}
    if total_memory_gb >= 35:
        return {"batch_size": 64, "pad_to_multiple_of": 32}
    if total_memory_gb >= 20:
        return {"batch_size": 32, "pad_to_multiple_of": 8}
    return conservative_params





class RerankerManager:
    
    def __init__(self):
        self._available_types = ["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope"]
        self.logger = create_module_logger("RerankerManager")
        self.logger.info(f"Reranker backend: {_reranker_backend()}")
        self._rerankers: Dict[str, Any] = {}
        
        
        self._local_gpu_types = {"baai", "qwen", "jina"}
    
    def get_reranker(self, reranker_type: str, model_name: str = None) -> Any:
        """Return reranker."""
        
        if model_name is None:
            default_models = {
                "baai": "BAAI/bge-reranker-v2-m3",
                "qwen": "Qwen/Qwen3-Reranker-0.6B",
                "jina": "jinaai/jina-reranker-v3",
                "qwen-sili": "Qwen/Qwen3-Reranker-8B",
                "qwen-dashscope": "qwen3-rerank",
                "gte-dashscope": "gte-rerank-v2"
            }
            model_name = default_models.get(reranker_type.lower(), "BAAI/bge-reranker-v2-m3")
        
        
        
        cache_key = _reranker_cache_key(reranker_type.lower(), model_name)
        cached = self._rerankers.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            if reranker_type.lower() == "baai":
                reranker = BAAIReranker(model_name)
            elif reranker_type.lower() == "qwen":
                reranker = QwenReranker(model_name)
            elif reranker_type.lower() == "jina":
                reranker = JinaReranker(model_name)
            elif reranker_type.lower() == "qwen-sili":
                reranker = QwenRerankerSiliconflow(model_name)
            elif reranker_type.lower() == "qwen-dashscope":
                reranker = QwenRerankerDashscope(model_name)
            elif reranker_type.lower() == "gte-dashscope":
                reranker = GTEReranker(model_name)
            else:
                raise ValueError(f"Unsupported reranker type: {reranker_type}")
            self._rerankers[cache_key] = reranker
            return reranker
        except Exception as e:
            self.logger.error(f"Get reranker failed: {e}")
            raise
    
    def get_cached_rerankers(self) -> List[str]:
        """Return cached rerankers."""
        try:
            loaded_models = global_model_manager.get_loaded_models()
        except Exception as e:
            self.logger.warning(f"Failed to get the global reranker cache: {e}")
            loaded_models = {}
        wrapper_keys = {f"wrapper:{key}" for key in self._rerankers.keys()}
        model_keys = {key for key in loaded_models if key.startswith("reranker:")}
        return sorted(wrapper_keys | model_keys)

    def get_available_types(self) -> List[str]:
        """Return available types."""
        return list(self._available_types)

    def clear_cache(self, clear_global_models: bool = False) -> None:
        """Remove cache."""
        if clear_global_models:
            global_model_manager.cleanup_all()
            self.logger.info("Global model cache cleared")
        else:
            self.logger.debug("Cleaning reranker wrapper cache while retaining the global model cache")

        for key, reranker in list(self._rerankers.items()):
            close_async = getattr(reranker, "close_async", None)
            if callable(close_async):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(close_async())
                except RuntimeError:
                    try:
                        asyncio.run(close_async())
                    except Exception as exc:
                        self.logger.debug(f"Closing async reranker failed {key}: {exc}")
            elif hasattr(reranker, "cleanup"):
                try:
                    reranker.cleanup()
                except Exception as exc:
                    self.logger.debug(f"Reranker cleanup failed {key}: {exc}")
        self._rerankers.clear()


class BAAIReranker(VLLMHttpRerankerMixin):
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        if _reranker_backend() == "native" and not BAAI_AVAILABLE:
            raise ImportError("BAAI reranker dependency unavailable; install: pip install torch transformers")
        
        self.model_name = model_name
        self.logger = create_module_logger("BAAIReranker")
        
        def loader():
            if _reranker_backend() == "vllm":
                self.logger.info(f"Initializing BAAI vLLM HTTP reranker: {model_name}")
                return _create_vllm_http_components(model_name)

            self.logger.info(f"Initializing BAAI reranker: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # try:
            #     model = AutoModelForSequenceClassification.from_pretrained(
            #         model_name,
            #         torch_dtype=torch.bfloat16,
            #         device_map="auto" if torch.cuda.is_available() else None,
            #         attn_implementation="flash_attention_2",  #  Flash Attention 2
            #     )
            # except Exception as fa_error:
            #     model = AutoModelForSequenceClassification.from_pretrained(
            #         model_name,
            #         torch_dtype=torch.bfloat16,
            #         device_map="auto" if torch.cuda.is_available() else None,
            #     )
            
            self.logger.info(f"BAAI reranker does not support Flash Attention 2; using the default attention implementation")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            model.eval()
            
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, dynamic=True)
                    self.logger.info(" Enabling torch.compile optimization")
                except Exception as compile_error:
                    self.logger.warning(f" torch.compile failed: {compile_error}")
            
            self.logger.info(f" BAAI reranker initialized successfully: {model_name} on {device} (dtype=bfloat16)")
            return {'backend': 'native', 'model': model, 'tokenizer': tokenizer, 'device': device}
        
        try:
            components = global_model_manager.get_or_load_model(
                model_type="reranker",
                model_name=_reranker_cache_key("baai", model_name),
                loader_func=loader
            )
            self.backend = components.get('backend', 'native')
            self.device = components.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if self.backend == 'vllm':
                self._init_vllm_http(components)
                self.model = None
                self.tokenizer = None
            else:
                self.model = components['model']
                self.tokenizer = components['tokenizer']
        except Exception as e:
            self.logger.error(f"BAAI reranker initialization failed: {e}")
            raise

        hardware_params = get_hardware_optimized_params()
        self.default_batch_size = hardware_params["batch_size"]
        self.pad_to_multiple_of = hardware_params["pad_to_multiple_of"]
    
    def rerank(self, 
               query: str, 
               documents: List[str], 
               batch_size: int = None, 
               max_length: int = 512) -> List[float]:
        """Rerank."""
        if getattr(self, "backend", "native") == "vllm":
            raise RuntimeError("vLLM mode does not support synchronous calls; use rerank_async().")

        if not documents:
            return []

        if batch_size is None:
            batch_size = self.default_batch_size

        try:
            indexed_docs = [(i, doc, len(doc)) for i, doc in enumerate(documents)]
            sorted_docs = sorted(indexed_docs, key=lambda x: x[2], reverse=True)

            sorted_pairs = [[query, doc] for _, doc, _ in sorted_docs]
            original_indices = [i for i, _, _ in sorted_docs]

            num_pairs = len(sorted_pairs)
            batch_ranges = [
                (s, min(s + batch_size, num_pairs))
                for s in range(0, num_pairs, batch_size)
            ]
            num_batches = len(batch_ranges)

            if self.device.type != "cuda":
                return self._rerank_cpu_fallback(
                    sorted_pairs, original_indices, batch_ranges, max_length, len(documents)
                )

            compute_stream = torch.cuda.Stream(device=self.device)
            transfer_stream = torch.cuda.Stream(device=self.device)

            default_stream = torch.cuda.current_stream(self.device)
            compute_stream.wait_stream(default_stream)
            transfer_stream.wait_stream(default_stream)

            all_logits = torch.empty(num_pairs, dtype=torch.float32, device=self.device)

            cur_inputs = self._tokenize_batch(
                sorted_pairs, batch_ranges[0], max_length
            )
            with torch.cuda.stream(transfer_stream):
                cur_inputs = {
                    k: v.to(self.device, non_blocking=True) for k, v in cur_inputs.items()
                }

            with torch.no_grad():
                for b in range(num_batches):
                    start_idx, end_idx = batch_ranges[b]

                    compute_stream.wait_stream(transfer_stream)

                    for v in cur_inputs.values():
                        v.record_stream(compute_stream)

                    with torch.cuda.stream(compute_stream):
                        logits = self.model(**cur_inputs, return_dict=True).logits.view(-1).float()
                        all_logits[start_idx:end_idx] = logits

                    if b + 1 < num_batches:
                        cur_inputs = self._tokenize_batch(
                            sorted_pairs, batch_ranges[b + 1], max_length
                        )
                        with torch.cuda.stream(transfer_stream):
                            cur_inputs = {
                                k: v.to(self.device, non_blocking=True)
                                for k, v in cur_inputs.items()
                            }

            torch.cuda.current_stream(self.device).wait_stream(compute_stream)
            torch.cuda.current_stream(self.device).wait_stream(transfer_stream)

            sorted_scores = all_logits.cpu().tolist()

            
            final_scores = [0.0] * len(documents)
            for idx, score in zip(original_indices, sorted_scores):
                final_scores[idx] = score

            return final_scores

        except Exception as e:
            self.logger.error(f"BAAI reranking failed: {e}")
            self.logger.error(traceback.format_exc())
            return [0.0] * len(documents)

    async def rerank_async(
        self,
        query: str,
        documents: List[str],
        batch_size: int = None,
        max_length: int = 512,
    ) -> List[float]:
        if getattr(self, "backend", "native") == "native":
            return await asyncio.to_thread(self.rerank, query, documents, batch_size, max_length)
        if not documents:
            return []
        self.logger.debug(f"BAAI vLLM reranking: {len(documents)} documents")
        return await self._score_vllm_documents(query, documents)

    
    

    def _tokenize_batch(
        self,
        sorted_pairs: List[List[str]],
        batch_range: Tuple[int, int],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize batch."""
        start, end = batch_range
        batch_pairs = sorted_pairs[start:end]
        return self.tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def _rerank_cpu_fallback(
        self,
        sorted_pairs: List[List[str]],
        original_indices: List[int],
        batch_ranges: List[Tuple[int, int]],
        max_length: int,
        num_documents: int,
    ) -> List[float]:
        """Rerank cpu fallback."""
        sorted_scores: List[float] = []
        with torch.no_grad():
            for br in batch_ranges:
                inputs = self._tokenize_batch(sorted_pairs, br, max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                sorted_scores.extend(logits.tolist())

        final_scores = [0.0] * num_documents
        for idx, score in zip(original_indices, sorted_scores):
            final_scores[idx] = score
        return final_scores
    
    def cleanup(self):
        """Release associated resources."""
        try:
            if getattr(self, "backend", "native") == "vllm" and getattr(self, "_async_client", None) is not None:
                self.logger.info("BAAI vLLM HTTP client is still open; call await close_async() to close it.")
            self.logger.info("BAAI reranker resources cleaned up; CUDA cache retained.")
        except Exception as e:
            self.logger.warning(f"cleanupBAAI reranker resources raised an error: {e}")

class QwenReranker(VLLMHttpRerankerMixin):
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        if _reranker_backend() == "native" and not QWEN_AVAILABLE:
            raise ImportError("Qwen reranker dependency unavailable; install: pip install torch transformers")
        
        self.model_name = model_name
        self.logger = create_module_logger("QwenReranker")
        
        def loader():
            if _reranker_backend() == "vllm":
                self.logger.info(f"Initializing Qwen vLLM HTTP reranker: {model_name}")
                return _create_vllm_http_components(model_name)

            self.logger.info(f"Initializing Qwen reranker: {model_name}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            use_flash_attention = is_flash_attention_available()
            if use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs,
                ).eval()
                if use_flash_attention:
                    self.logger.info(" Enabling Flash Attention 2")
            except Exception as fa_error:
                if not use_flash_attention:
                    raise
                self.logger.warning(f" Flash Attention 2 load failed; using default attention: {fa_error}")
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs,
                ).eval()
            
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, dynamic=True)
                    self.logger.info(" Enabling torch.compile optimization")
                except Exception as compile_error:
                    self.logger.warning(f" torch.compile failed: {compile_error}")
            
            prefix_tokens = tokenizer.encode("<|im_start|>", add_special_tokens=False)
            suffix_tokens = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            target_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            
            self.logger.info(f" Qwen reranker initialized successfully: {model_name} on {device} (dtype=bfloat16)")
            
            return {
                'backend': 'native',
                'model': model, 
                'tokenizer': tokenizer, 
                'device': device,
                'prefix_tokens': prefix_tokens,
                'suffix_tokens': suffix_tokens,
                'target_token_id': target_token_id
            }
        
        try:
            components = global_model_manager.get_or_load_model(
                model_type="reranker",
                model_name=_reranker_cache_key("qwen", model_name),
                loader_func=loader
            )
            self.backend = components.get('backend', 'native')
            self.device = components.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if self.backend == 'vllm':
                self._init_vllm_http(components)
                self.model = None
                self.tokenizer = None
                self.prefix_tokens = []
                self.suffix_tokens = []
                self.target_token_id = None
            else:
                self.model = components['model']
                self.tokenizer = components['tokenizer']
                self.prefix_tokens = components['prefix_tokens']
                self.suffix_tokens = components['suffix_tokens']
                self.target_token_id = components['target_token_id']
        except Exception as e:
            self.logger.error(f"Qwen reranker initialization failed: {e}")
            raise

        hardware_params = get_hardware_optimized_params()
        self.default_batch_size = hardware_params["batch_size"]
        self.pad_to_multiple_of = hardware_params["pad_to_multiple_of"]
    
    def rerank(self, 
            query: str, 
            documents: List[str], 
            batch_size: int = None, 
            max_length: int = 512,
            instruction: str = None) -> List[float]:
        """Rerank."""
        if getattr(self, "backend", "native") == "vllm":
            raise RuntimeError("vLLM mode does not support synchronous calls; use rerank_async().")

        if not documents:
            return []

        if batch_size is None:
            batch_size = self.default_batch_size

        if instruction is None:
            instruction = "Given a query and a passage, determine if the passage is relevant to answer the query. Please answer Yes or No."

        try:
            indexed_docs = [(i, doc, len(doc)) for i, doc in enumerate(documents)]
            sorted_docs = sorted(indexed_docs, key=lambda x: x[2], reverse=True)

            sorted_pairs = [
                self._format_instruction(instruction, query, doc)
                for _, doc, _ in sorted_docs
            ]
            original_indices = [i for i, _, _ in sorted_docs]

            num_pairs = len(sorted_pairs)
            batch_ranges = [
                (s, min(s + batch_size, num_pairs))
                for s in range(0, num_pairs, batch_size)
            ]
            num_batches = len(batch_ranges)

            if self.device.type != "cuda":
                return self._rerank_cpu_fallback(
                    sorted_pairs, original_indices, batch_ranges, max_length, len(documents)
                )

            compute_stream = torch.cuda.Stream(device=self.device)
            transfer_stream = torch.cuda.Stream(device=self.device)

            default_stream = torch.cuda.current_stream(self.device)
            compute_stream.wait_stream(default_stream)
            transfer_stream.wait_stream(default_stream)

            all_scores = torch.empty(num_pairs, dtype=torch.float32, device=self.device)

            cur_inputs = self._process_inputs(
                sorted_pairs[batch_ranges[0][0]:batch_ranges[0][1]], max_length
            )
            with torch.cuda.stream(transfer_stream):
                cur_inputs = {
                    k: v.to(self.device, non_blocking=True) for k, v in cur_inputs.items()
                }

            with torch.no_grad():
                for b in range(num_batches):
                    start_idx, end_idx = batch_ranges[b]

                    compute_stream.wait_stream(transfer_stream)

                    for v in cur_inputs.values():
                        v.record_stream(compute_stream)

                    with torch.cuda.stream(compute_stream):
                        outputs = self.model(**cur_inputs, use_cache=False)
                        yes_logits = outputs.logits[:, -1, self.target_token_id].float()
                        all_scores[start_idx:end_idx] = yes_logits

                    if b + 1 < num_batches:
                        next_start, next_end = batch_ranges[b + 1]
                        cur_inputs = self._process_inputs(
                            sorted_pairs[next_start:next_end], max_length
                        )
                        with torch.cuda.stream(transfer_stream):
                            cur_inputs = {
                                k: v.to(self.device, non_blocking=True)
                                for k, v in cur_inputs.items()
                            }

            torch.cuda.current_stream(self.device).wait_stream(compute_stream)
            torch.cuda.current_stream(self.device).wait_stream(transfer_stream)

            sorted_scores = all_scores.cpu().tolist()

            
            final_scores = [0.0] * len(documents)
            for idx, score in zip(original_indices, sorted_scores):
                final_scores[idx] = score

            return final_scores

        except Exception as e:
            self.logger.error(f"Qwen reranking failed: {e}")
            self.logger.error(traceback.format_exc())
            return [0.0] * len(documents)

    async def rerank_async(
        self,
        query: str,
        documents: List[str],
        batch_size: int = None,
        max_length: int = 512,
        instruction: str = None,
    ) -> List[float]:
        if getattr(self, "backend", "native") == "native":
            return await asyncio.to_thread(self.rerank, query, documents, batch_size, max_length, instruction)
        if not documents:
            return []
        if instruction is None:
            instruction = "Given a query and a passage, determine if the passage is relevant to answer the query. Please answer Yes or No."
        prompts = [self._format_instruction(instruction, query, document) for document in documents]
        self.logger.debug(f"Qwen vLLM reranking: {len(documents)} documents")
        return await self._score_vllm_documents(query, documents, prompts=prompts)
    
    def _format_instruction(self, instruction_text: str, query_text: str, doc_text: str) -> str:
        """Format instruction."""
        return f'<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "Yes" or "No".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction_text}\n<Query>: {query_text}\n<Document>: {doc_text}<|im_end|>\n<|im_start|>assistant\n'
    
    def _process_inputs(self, pairs: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        """Process inputs."""
        return self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

    def _rerank_cpu_fallback(
        self,
        sorted_pairs: List[str],
        original_indices: List[int],
        batch_ranges: List[Tuple[int, int]],
        max_length: int,
        num_documents: int,
    ) -> List[float]:
        """Rerank cpu fallback."""
        sorted_scores: List[float] = []
        with torch.no_grad():
            for start, end in batch_ranges:
                inputs = self._process_inputs(sorted_pairs[start:end], max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, use_cache=False)
                yes_logits = outputs.logits[:, -1, self.target_token_id].float()
                sorted_scores.extend(yes_logits.tolist())

        final_scores = [0.0] * num_documents
        for idx, score in zip(original_indices, sorted_scores):
            final_scores[idx] = score
        return final_scores
    
    @torch.no_grad()
    def _compute_logits(self, inputs) -> List[float]:
        """Compute logits."""
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        
        yes_logits = logits[:, self.target_token_id].float()
        return yes_logits.cpu().tolist()
    
    def cleanup(self):
        """Release associated resources."""
        try:
            if getattr(self, "backend", "native") == "vllm" and getattr(self, "_async_client", None) is not None:
                self.logger.info("Qwen vLLM HTTP client is still open; call await close_async() to close it.")
            self.logger.info("Qwen reranker resources cleaned up; CUDA cache retained.")
        except Exception as e:
            self.logger.warning(f"cleanupQwen reranker resources raised an error: {e}")
            
class JinaReranker(VLLMHttpRerankerMixin):
    
    def __init__(self, model_name: str = "jinaai/jina-reranker-v3"):
        self.model_name = model_name
        self.logger = create_module_logger("JinaReranker")
        
        def loader():
            if _reranker_backend() == "vllm":
                self.logger.info(f"Initializing Jina vLLM HTTP reranker: {model_name}")
                return _create_vllm_http_components(model_name)

            from transformers import AutoModel
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Loading Jina reranking model: {model_name}")
            
            dtype = torch.bfloat16
            
            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            use_flash_attention = is_flash_attention_available()
            if use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    **model_kwargs,
                )
                if use_flash_attention:
                    self.logger.info(" Enabling Flash Attention 2")
            except Exception as fa_error:
                if not use_flash_attention:
                    raise
                self.logger.warning(f" Flash Attention 2 load failed: {fa_error}")
                model_kwargs.pop("attn_implementation", None)
                model = AutoModel.from_pretrained(
                    model_name,
                    **model_kwargs,
                )
            
            model.eval()
            
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, dynamic=True)
                    self.logger.info(" Enabling torch.compile optimization")
                except Exception as compile_error:
                    self.logger.warning(f" torch.compile failed: {compile_error}")
            
            self.logger.info(f" Jina reranker loaded successfully (dtype={dtype})")
            return {'backend': 'native', 'model': model, 'device': device}
        
        try:
            components = global_model_manager.get_or_load_model(
                model_type="reranker",
                model_name=_reranker_cache_key("jina", model_name),
                loader_func=loader
            )
            self.backend = components.get('backend', 'native')
            self.device = components.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if self.backend == 'vllm':
                self._init_vllm_http(components)
                self.model = None
            else:
                self.model = components['model']
        except Exception as e:
            self.logger.error(f" Jina reranker initialization failed: {e}")
            raise
    
    def rerank(self, 
               query: str, 
               documents: List[str], 
               batch_size: int = 32,
               max_length: int = 512) -> List[float]:
        
        if not documents:
            return []
        if getattr(self, "backend", "native") == "vllm":
            raise RuntimeError("vLLM mode does not support synchronous calls; use rerank_async().")
        
        try:
            
            
            results = self.model.rerank(
                query=query,
                documents=documents,
                top_n=None,
                return_embeddings=False,
            )
            
            
            
            scores = [0.0] * len(documents)
            for res in results:
                scores[res['index']] = res['relevance_score']

            return scores

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error(f" Jina full reranking ran out of GPU memory (Docs: {len(documents)})!reduce retrieval Top-K or enable batching.")
            self.logger.error(traceback.format_exc())
            raise e
            
        except Exception as e:
            self.logger.error(f" Jina reranking unknown error: {e}")
            self.logger.error(traceback.format_exc())
            return [0.0] * len(documents)

    async def rerank_async(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> List[float]:
        if getattr(self, "backend", "native") == "native":
            return await asyncio.to_thread(self.rerank, query, documents, batch_size, max_length)
        if not documents:
            return []
        self.logger.debug(f"Jina vLLM reranking: {len(documents)} documents")
        return await self._score_vllm_documents(query, documents)

    def cleanup(self):
        """Release associated resources."""
        try:
            if getattr(self, "backend", "native") == "vllm" and getattr(self, "_async_client", None) is not None:
                self.logger.info("Jina vLLM HTTP client is still open; call await close_async() to close it.")
            self.logger.info("Jina reranker resources cleaned up; CUDA cache retained.")
        except Exception as e:
            self.logger.warning(f"cleanupJina reranker resources raised an error: {e}")




class CloudRerankerFallbackMixin:

    def _generate_fallback_scores(
        self,
        num_documents: int,
        base_score: float = 0.01,
        step: float = 1e-5,
    ) -> List[float]:
        scores: List[float] = []
        current_score = max(float(base_score), float(step), 1e-12)
        decrement = max(float(step), 1e-12)

        for _ in range(num_documents):
            scores.append(current_score)
            next_score = current_score - decrement
            if next_score <= 0 or next_score >= current_score:
                next_score = current_score * 0.5
            current_score = next_score

        return scores


class QwenRerankerSiliconflow(CloudRerankerFallbackMixin):
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-8B"):
        self.model_name = model_name
        self.logger = create_module_logger("QwenRerankerSiliconflow")
        self.logger.setLevel(logging.DEBUG)
        self.api_url = settings.siliconflow_rerank_url
        self.api_key = settings.get_api_key("SILICONFLOW_API_KEY")
        
        if not self.api_key:
            self.logger.warning(" Environment variable not found SILICONFLOW_API_KEY")
            self.logger.warning("   Set it in the project-root .env file: SILICONFLOW_API_KEY=your_key")
        
        self.timeout = 30
        self.max_retries = 2
        
        
        self._async_client: Optional[httpx.AsyncClient] = None
        
        self.logger.info(f" Qwen cloud reranker initialized successfully: {model_name}")
        self.logger.info(f"   API URL: {self.api_url}")
        self.logger.info(f"   API key: {'configured' if self.api_key else 'not configured'}")
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Get async client."""
        if self._async_client is None:
            self._async_client = _create_proxy_aware_async_client(self.timeout)
        return self._async_client
    
    def rerank(self, 
           query: str, 
           documents: List[str], 
           batch_size: int = 32, 
           max_length: int = 512) -> List[float]:
        """Rerank."""
        if not documents:
            return []
        
        if not self.api_key:
            self.logger.error(" API key is not configured; cannot call cloud reranking.")
            return self._generate_fallback_scores(len(documents))
        
        try:
            self.logger.debug(f"Starting cloud reranking: {len(documents)} documents")
            
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "instruction": "Given a web search query, retrieve relevant passages that answer the query",
                "top_n": len(documents),
                "return_documents": False
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            scores = self._send_request_with_retry(payload, headers)
            
            self.logger.debug(f"Cloud reranking complete: returned {len(scores)} scores")
            return scores
            
        except Exception as e:
            self.logger.error(f"Cloud reranking failed: {e}")
            self.logger.debug(traceback.format_exc())
            return self._generate_fallback_scores(len(documents))
    
    def _send_request_with_retry(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[float]:
        """Send request with retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.json(), len(payload["documents"]))
                else:
                    error_msg = f"API returned error status code: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f", details: {error_detail}"
                    except:
                        error_msg += f", response: {response.text[:200]}"
                    
                    last_error = error_msg
                    
                    if attempt < self.max_retries:
                        self.logger.warning(f" request failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        time.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(error_msg)
            
            except requests.Timeout as e:
                last_error = f"request timeout: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
            
            except requests.RequestException as e:
                last_error = f"network request failed: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" network error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
        
        raise Exception(f"Still failed after {self.max_retries} retries: {last_error}")
    
    def _parse_response(self, response_data: Dict[str, Any], num_documents: int) -> List[float]:
        """Parse response."""
        try:
            self.logger.info(f" Full API response: {response_data}")
            
            results = response_data.get("results", [])
            
            if not results:
                self.logger.warning("API response has no results field; using order-preserving fallback scores")
                return self._generate_fallback_scores(num_documents)
            
            self.logger.info(f" API returned {len(results)} results:")
            for i, result in enumerate(results):
                self.logger.info(f"  result {i}: index={result.get('index')}, score={result.get('relevance_score'):.4f}")
            
            
            
            score_dict = {}
            for result in results:
                index = result.get("index")
                score = result.get("relevance_score", 0.0)
                
                if index is not None:
                    score_dict[index] = float(score)
            
            if len(score_dict) < num_documents:
                self.logger.debug(f"API returned {len(score_dict)}/{num_documents} document scores (possibly due to the top_n limit)")
            
            scores = []
            missing_scores = self._generate_fallback_scores(num_documents, base_score=0.001)
            missing_count = 0
            for i in range(num_documents):
                if i in score_dict:
                    scores.append(score_dict[i])
                else:
                    fallback_score = missing_scores[missing_count]
                    missing_count += 1
                    scores.append(fallback_score)
                    self.logger.debug(f"document index {i} did not return scores; using order-preserving fallback value {fallback_score:.6f}")
            
            normalized_scores = self._normalize_scores(scores)
            
            self.logger.debug(f"Parsed {len(normalized_scores)} scores")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"Failed to parse API response: {e}")
            self.logger.debug(f"response data: {response_data}")
            return self._generate_fallback_scores(num_documents)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores."""
        if not scores:
            return []
        
        if all(0 <= s <= 1 for s in scores):
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized
    
    async def rerank_async(self, 
                           query: str, 
                           documents: List[str], 
                           batch_size: int = 32, 
                           max_length: int = 512) -> List[float]:
        """Rerank async."""
        if not documents:
            return []
        
        if not self.api_key:
            self.logger.error(" API key is not configured; cannot call cloud reranking.")
            return self._generate_fallback_scores(len(documents))
        
        try:
            self.logger.debug(f"Starting async cloud reranking: {len(documents)} documents")
            
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "instruction": "Given a web search query, retrieve relevant passages that answer the query",
                "top_n": len(documents),
                "return_documents": False
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            client = self._get_async_client()
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        return self._parse_response(response.json(), len(documents))
                    else:
                        error_msg = f"API returned error status code: {response.status_code}"
                        last_error = error_msg
                        
                        if attempt < self.max_retries:
                            self.logger.warning(f" async request failed (attempt {attempt + 1}): {error_msg}")
                            await asyncio.sleep(1 * (attempt + 1))
                        else:
                            raise Exception(error_msg)
                
                except httpx.TimeoutException as e:
                    last_error = f"async request timeout: {e}"
                    if attempt < self.max_retries:
                        self.logger.warning(f" async timeout (attempt {attempt + 1})")
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(last_error)
                
                except httpx.RequestError as e:
                    last_error = f"async network error: {e}"
                    if attempt < self.max_retries:
                        self.logger.warning(f" async network error (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(last_error)
            
            raise Exception(f"Still failed after {self.max_retries} retries: {last_error}")
            
        except Exception as e:
            self.logger.error(f"asyncCloud reranking failed: {e}")
            self.logger.debug(traceback.format_exc())
            return self._generate_fallback_scores(len(documents))
    
    def cleanup(self):
        """Release associated resources."""
        if self._async_client is not None:
            self.logger.info(" Call close_async() to close the async client cleanly.")
        self.logger.info("Cloud reranker resources cleaned up")
    
    async def close_async(self):
        """Close async."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self.logger.info(" Async HTTP client closed")

class QwenRerankerDashscope(CloudRerankerFallbackMixin):
    
    def __init__(self, model_name: str = "qwen3-rerank"):
        self.model_name = model_name
        self.logger = create_module_logger("QwenRerankerDashscope")
        self.logger.setLevel(logging.DEBUG)
        
        self.api_url = settings.dashscope_rerank_url
        self.api_key = settings.get_api_key("DASHSCOPE_API_KEY")
        
        if not self.api_key:
            self.logger.warning(" Environment variable not found DASHSCOPE_API_KEY")
            self.logger.warning("   Set it in the project-root .env file: DASHSCOPE_API_KEY=your_key")
        
        self.timeout = 30
        self.max_retries = 2
        
        
        self._async_client: Optional[httpx.AsyncClient] = None
        
        self.logger.info(f" Qwen DashScope reranker initialized successfully: {model_name}")
        self.logger.info(f"   API URL: {self.api_url}")
        self.logger.info(f"   API key: {'configured' if self.api_key else 'not configured'}")
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Get async client."""
        if self._async_client is None:
            self._async_client = _create_proxy_aware_async_client(self.timeout)
        return self._async_client
    
    def rerank(self, 
               query: str, 
               documents: List[str], 
               batch_size: int = 32, 
               max_length: int = 512,
               instruction: str = None) -> List[float]:
        """Rerank."""
        if not documents:
            return []
        
        if not self.api_key:
            self.logger.error(" API key is not configured; cannot call cloud reranking.")
            return self._generate_fallback_scores(len(documents))
        
        try:
            self.logger.debug(f"Starting DashScope reranking: {len(documents)} documents")
            
            payload = {
                "model": self.model_name,
                "input": {
                    "query": query,
                    "documents": documents
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(documents)
                }
            }
            
            
            if instruction and self.model_name == "qwen3-rerank":
                payload["input"]["instruct"] = instruction
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            scores = self._send_request_with_retry(payload, headers)
            
            self.logger.debug(f"DashScope reranking complete: returned {len(scores)} scores")
            return scores
            
        except Exception as e:
            self.logger.error(f"DashScope reranking failed: {e}")
            self.logger.debug(traceback.format_exc())
            return self._generate_fallback_scores(len(documents))
    
    def _send_request_with_retry(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[float]:
        """Send request with retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.json(), len(payload["input"]["documents"]))
                else:
                    error_msg = f"API returned error status code: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f", details: {error_detail}"
                    except:
                        error_msg += f", response: {response.text[:200]}"
                    
                    last_error = error_msg
                    
                    if attempt < self.max_retries:
                        self.logger.warning(f" request failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        time.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(error_msg)
            
            except requests.Timeout as e:
                last_error = f"request timeout: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
            
            except requests.RequestException as e:
                last_error = f"network request failed: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" network error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
        
        raise Exception(f"Still failed after {self.max_retries} retries: {last_error}")
    
    def _parse_response(self, response_data: Dict[str, Any], num_documents: int) -> List[float]:
        """Parse response."""
        try:
            self.logger.debug(f" Full API response: {response_data}")
            
            output = response_data.get("output", {})
            results = output.get("results", [])
            
            if not results:
                self.logger.warning("API response has no results field; using order-preserving fallback scores")
                return self._generate_fallback_scores(num_documents)
            
            self.logger.debug(f" API returned {len(results)} results")
            
            score_dict = {}
            for result in results:
                index = result.get("index")
                score = result.get("relevance_score", 0.0)
                
                if index is not None:
                    score_dict[index] = float(score)
                    self.logger.debug(f"  documents {index}: score={score:.4f}")
            
            scores = []
            missing_scores = self._generate_fallback_scores(num_documents, base_score=0.001)
            missing_count = 0
            for i in range(num_documents):
                if i in score_dict:
                    scores.append(score_dict[i])
                else:
                    fallback_score = missing_scores[missing_count]
                    missing_count += 1
                    scores.append(fallback_score)
                    self.logger.debug(f"document index {i} did not return scores; using order-preserving fallback value {fallback_score:.6f}")
            
            normalized_scores = self._normalize_scores(scores)
            
            self.logger.debug(f"Parsed {len(normalized_scores)} scores")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"Failed to parse API response: {e}")
            self.logger.debug(f"response data: {response_data}")
            return self._generate_fallback_scores(num_documents)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores."""
        if not scores:
            return []
        
        if all(0 <= s <= 1 for s in scores):
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized
    
    async def rerank_async(self, 
                           query: str, 
                           documents: List[str], 
                           batch_size: int = 32, 
                           max_length: int = 512,
                           instruction: str = None) -> List[float]:
        """Rerank async."""
        if not documents:
            return []
        
        if not self.api_key:
            self.logger.error(" API key is not configured; cannot call cloud reranking.")
            return self._generate_fallback_scores(len(documents))
        
        try:
            self.logger.debug(f"Starting async DashScope reranking: {len(documents)} documents")
            
            payload = {
                "model": self.model_name,
                "input": {
                    "query": query,
                    "documents": documents
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(documents)
                }
            }
            
            if instruction and self.model_name == "qwen3-rerank":
                payload["input"]["instruct"] = instruction
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            client = self._get_async_client()
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        return self._parse_response(response.json(), len(documents))
                    else:
                        error_msg = f"API returned error status code: {response.status_code}"
                        last_error = error_msg
                        
                        if attempt < self.max_retries:
                            self.logger.warning(f" async request failed (attempt {attempt + 1}): {error_msg}")
                            await asyncio.sleep(1 * (attempt + 1))
                        else:
                            raise Exception(error_msg)
                
                except httpx.TimeoutException as e:
                    last_error = f"async request timeout: {e}"
                    if attempt < self.max_retries:
                        self.logger.warning(f" async timeout (attempt {attempt + 1})")
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(last_error)
                
                except httpx.RequestError as e:
                    last_error = f"async network error: {e}"
                    if attempt < self.max_retries:
                        self.logger.warning(f" async network error (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(last_error)
            
            raise Exception(f"Still failed after {self.max_retries} retries: {last_error}")
            
        except Exception as e:
            self.logger.error(f"asyncDashScope reranking failed: {e}")
            self.logger.debug(traceback.format_exc())
            return self._generate_fallback_scores(len(documents))
    
    def cleanup(self):
        """Release associated resources."""
        if self._async_client is not None:
            self.logger.info(" Call close_async() to close the async client cleanly.")
        self.logger.info("Cloud reranker resources cleaned up")
    
    async def close_async(self):
        """Close async."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self.logger.info(" Async HTTP client closed")

class GTEReranker(CloudRerankerFallbackMixin):
    
    def __init__(self, model_name: str = "gte-rerank-v2"):
        self.model_name = model_name
        self.logger = create_module_logger("GTEReranker")
        self.logger.setLevel(logging.DEBUG)
        
        self.api_url = settings.dashscope_rerank_url
        self.api_key = settings.get_api_key("DASHSCOPE_API_KEY")
        
        if not self.api_key:
            self.logger.warning(" Environment variable not found DASHSCOPE_API_KEY")
            self.logger.warning("   Set it in the project-root .env file: DASHSCOPE_API_KEY=your_key")
        
        self.timeout = 30
        self.max_retries = 2
        
        
        self._async_client: Optional[httpx.AsyncClient] = None
        
        self.logger.info(f" GTE reranker initialized successfully: {model_name}")
        self.logger.info(f"   API URL: {self.api_url}")
        self.logger.info(f"   API key: {'configured' if self.api_key else 'not configured'}")
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Get async client."""
        if self._async_client is None:
            self._async_client = _create_proxy_aware_async_client(self.timeout)
        return self._async_client
    
    def rerank(self, 
               query: str, 
               documents: List[str], 
               batch_size: int = 32, 
               max_length: int = 512) -> List[float]:
        """Rerank."""
        if not documents:
            return []
        
        if not self.api_key:
            self.logger.error(" API key is not configured; cannot call cloud reranking.")
            return self._generate_fallback_scores(len(documents))
        
        try:
            self.logger.debug(f"Starting GTE reranking: {len(documents)} documents")
            
            payload = {
                "model": self.model_name,
                "input": {
                    "query": query,
                    "documents": documents
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(documents)
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            scores = self._send_request_with_retry(payload, headers)
            
            self.logger.debug(f"GTE reranking complete: returned {len(scores)} scores")
            return scores
            
        except Exception as e:
            self.logger.error(f"GTE reranking failed: {e}")
            self.logger.debug(traceback.format_exc())
            return self._generate_fallback_scores(len(documents))
    
    def _send_request_with_retry(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[float]:
        """Send request with retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.json(), len(payload["input"]["documents"]))
                else:
                    error_msg = f"API returned error status code: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f", details: {error_detail}"
                    except:
                        error_msg += f", response: {response.text[:200]}"
                    
                    last_error = error_msg
                    
                    if attempt < self.max_retries:
                        self.logger.warning(f" request failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        time.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(error_msg)
            
            except requests.Timeout as e:
                last_error = f"request timeout: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
            
            except requests.RequestException as e:
                last_error = f"network request failed: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" network error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
        
        raise Exception(f"Still failed after {self.max_retries} retries: {last_error}")
    
    def _parse_response(self, response_data: Dict[str, Any], num_documents: int) -> List[float]:
        """Parse response."""
        try:
            self.logger.debug(f" Full API response: {response_data}")
            
            output = response_data.get("output", {})
            results = output.get("results", [])
            
            if not results:
                self.logger.warning("API response has no results field; using order-preserving fallback scores")
                return self._generate_fallback_scores(num_documents)
            
            self.logger.debug(f" API returned {len(results)} results")
            
            score_dict = {}
            for result in results:
                index = result.get("index")
                score = result.get("relevance_score", 0.0)
                
                if index is not None:
                    score_dict[index] = float(score)
                    self.logger.debug(f"  documents {index}: score={score:.4f}")
            
            scores = []
            missing_scores = self._generate_fallback_scores(num_documents, base_score=0.001)
            missing_count = 0
            for i in range(num_documents):
                if i in score_dict:
                    scores.append(score_dict[i])
                else:
                    fallback_score = missing_scores[missing_count]
                    missing_count += 1
                    scores.append(fallback_score)
                    self.logger.debug(f"document index {i} did not return scores; using order-preserving fallback value {fallback_score:.6f}")
            
            normalized_scores = self._normalize_scores(scores)
            
            self.logger.debug(f"Parsed {len(normalized_scores)} scores")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"Failed to parse API response: {e}")
            self.logger.debug(f"response data: {response_data}")
            return self._generate_fallback_scores(num_documents)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores."""
        if not scores:
            return []
        
        if all(0 <= s <= 1 for s in scores):
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized
    
    async def rerank_async(self, 
                           query: str, 
                           documents: List[str], 
                           batch_size: int = 32, 
                           max_length: int = 512) -> List[float]:
        """Rerank async."""
        if not documents:
            return []
        
        if not self.api_key:
            self.logger.error(" API key is not configured; cannot call cloud reranking.")
            return self._generate_fallback_scores(len(documents))
        
        try:
            self.logger.debug(f"Starting async GTE reranking: {len(documents)} documents")
            
            payload = {
                "model": self.model_name,
                "input": {
                    "query": query,
                    "documents": documents
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(documents)
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            client = self._get_async_client()
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        return self._parse_response(response.json(), len(documents))
                    else:
                        error_msg = f"API returned error status code: {response.status_code}"
                        last_error = error_msg
                        
                        if attempt < self.max_retries:
                            self.logger.warning(f" async request failed (attempt {attempt + 1}): {error_msg}")
                            await asyncio.sleep(1 * (attempt + 1))
                        else:
                            raise Exception(error_msg)
                
                except httpx.TimeoutException as e:
                    last_error = f"async request timeout: {e}"
                    if attempt < self.max_retries:
                        self.logger.warning(f" async timeout (attempt {attempt + 1})")
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(last_error)
                
                except httpx.RequestError as e:
                    last_error = f"async network error: {e}"
                    if attempt < self.max_retries:
                        self.logger.warning(f" async network error (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(last_error)
            
            raise Exception(f"Still failed after {self.max_retries} retries: {last_error}")
            
        except Exception as e:
            self.logger.error(f"asyncGTE reranking failed: {e}")
            self.logger.debug(traceback.format_exc())
            return self._generate_fallback_scores(len(documents))
    
    def cleanup(self):
        """Release associated resources."""
        if self._async_client is not None:
            self.logger.info(" Call close_async() to close the async client cleanly.")
        self.logger.info("Cloud reranker resources cleaned up")
    
    async def close_async(self):
        """Close async."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self.logger.info(" Async HTTP client closed")

class QwenRerankerCstcloud(CloudRerankerFallbackMixin):
    
    def __init__(self, model_name: str = "qwen3-reranker:8b"):
        self.model_name = model_name
        self.logger = create_module_logger("QwenRerankerCstcloud")
        self.logger.setLevel(logging.DEBUG)
        self.api_url = settings.cstcloud_rerank_url
        self.api_key = settings.get_api_key("CSTCLOUD_API_KEY")
        
        if not self.api_key:
            self.logger.warning(" Environment variable not found CSTCLOUD_API_KEY")
            self.logger.warning("   Set it in the project-root .env file: CSTCLOUD_API_KEY=your_key")
        
        self.timeout = 30
        self.max_retries = 2
        
        self.logger.info(f" Qwen cloud reranker initialized successfully: {model_name}")
        self.logger.info(f"   API URL: {self.api_url}")
        self.logger.info(f"   API key: {'configured' if self.api_key else 'not configured'}")
    
    def rerank(self, 
               query: str, 
               documents: List[str], 
               batch_size: int = 32, 
               max_length: int = 512) -> List[float]:
        """Rerank."""
        if not documents:
            return []
        
        if not self.api_key:
            self.logger.error(" API key is not configured; cannot call cloud reranking.")
            return self._generate_fallback_scores(len(documents))
        
        try:
            self.logger.debug(f"Starting cloud reranking: {len(documents)} documents")
            
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "instruct": "Given a web search query, retrieve relevant passages that answer the query",
                "top_n": len(documents),
                "return_documents": False
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            scores = self._send_request_with_retry(payload, headers)
            
            self.logger.debug(f"Cloud reranking complete: returned {len(scores)} scores")
            return scores
            
        except Exception as e:
            self.logger.error(f"Cloud reranking failed: {e}")
            self.logger.debug(traceback.format_exc())
            return self._generate_fallback_scores(len(documents))
    
    def _send_request_with_retry(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[float]:
        """Send request with retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.json(), len(payload["documents"]))
                else:
                    error_msg = f"API returned error status code: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f", details: {error_detail}"
                    except:
                        error_msg += f", response: {response.text[:200]}"
                    
                    last_error = error_msg
                    
                    if attempt < self.max_retries:
                        self.logger.warning(f" request failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        time.sleep(1 * (attempt + 1))
                    else:
                        raise Exception(error_msg)
            
            except requests.Timeout as e:
                last_error = f"request timeout: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
            
            except requests.RequestException as e:
                last_error = f"network request failed: {e}"
                if attempt < self.max_retries:
                    self.logger.warning(f" network error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(1 * (attempt + 1))
                else:
                    raise Exception(last_error)
        
        raise Exception(f"Still failed after {self.max_retries} retries: {last_error}")
    
    def _parse_response(self, response_data: Dict[str, Any], num_documents: int) -> List[float]:
        """Parse response."""
        try:
            self.logger.info(f" Full API response: {response_data}")
            
            results = response_data.get("results", [])
            
            if not results:
                self.logger.warning("API response has no results field; using order-preserving fallback scores")
                return self._generate_fallback_scores(num_documents)
            
            self.logger.info(f" API returned {len(results)} results:")
            for i, result in enumerate(results):
                self.logger.info(f"  result {i}: index={result.get('index')}, score={result.get('relevance_score'):.4f}")
            
            
            
            score_dict = {}
            for result in results:
                index = result.get("index")
                score = result.get("relevance_score", 0.0)
                
                if index is not None:
                    score_dict[index] = float(score)
            
            if len(score_dict) < num_documents:
                self.logger.debug(f"API returned {len(score_dict)}/{num_documents} document scores (possibly due to the top_n limit)")
            
            scores = []
            missing_scores = self._generate_fallback_scores(num_documents, base_score=0.001)
            missing_count = 0
            for i in range(num_documents):
                if i in score_dict:
                    scores.append(score_dict[i])
                else:
                    fallback_score = missing_scores[missing_count]
                    missing_count += 1
                    scores.append(fallback_score)
                    self.logger.debug(f"document index {i} did not return scores; using order-preserving fallback value {fallback_score:.6f}")
            
            normalized_scores = self._normalize_scores(scores)
            
            self.logger.debug(f"Parsed {len(normalized_scores)} scores")
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"Failed to parse API response: {e}")
            self.logger.debug(f"response data: {response_data}")
            return self._generate_fallback_scores(num_documents)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores."""
        if not scores:
            return []
        
        if all(0 <= s <= 1 for s in scores):
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized
    
    def cleanup(self):
        """Release associated resources."""
        self.logger.info("Cloud reranker does not require resource cleanup")

def test_qwen_reranker():
    try:
        reranker = QwenReranker()
        
        query = "What is the capital of China?"
        documents = [
            "The capital of China is Beijing.",
            "Shanghai is a major city in China.",
            "Gravity is a force that attracts objects."
        ]
        
        scores = reranker.rerank(query, documents)
        print(f" Qwen reranking test succeeded")
        print(f"Query: {query}")
        for i, (doc, score) in enumerate(zip(documents, scores)):
            print(f"  documents{i+1} (scores={score:.4f}): {doc[:50]}...")
        
        assert scores[0] > scores[1] > scores[2], "Scores are not sorted as expected"
        print(" Scores are sorted correctly")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen_reranker()
