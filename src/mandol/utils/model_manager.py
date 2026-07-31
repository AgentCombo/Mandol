"""Utilities for model manager."""
import logging
from .logging_config import create_module_logger
import threading
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Union, List
import torch
import numpy as np
import os
import re
from .config_manager import settings
from .optional_dependencies import is_flash_attention_available

logger = create_module_logger("utils.model_manager")




class NLTKProcessor:
    
    _WORDNET_POS_MAP = {
        'J': 'a',  # Adjective
        'V': 'v',  # Verb
        'N': 'n',  # Noun
        'R': 'r',  # Adverb
    }
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NLTKProcessor")
        self._initialized = False
        self._lemmatizer = None
        self._stopwords = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize."""
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag
            
            required_resources = [
                ('tokenizers/punkt', 'punkt'),
                ('tokenizers/punkt_tab', 'punkt_tab'),
                ('corpora/stopwords', 'stopwords'),
                ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
                ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
                ('corpora/wordnet', 'wordnet'),
            ]
            
            for path, name in required_resources:
                try:
                    nltk.data.find(path)
                except LookupError:
                    self.logger.info(f"Downloading NLTK resource: {name}")
                    nltk.download(name, quiet=True)
            
            self._lemmatizer = WordNetLemmatizer()
            self._stopwords = set(stopwords.words('english'))
            
            self._stopwords.update({
                'um', 'uh', 'oh', 'ah', 'yeah', 'yes', 'no', 'okay', 'ok',
                'well', 'like', 'just', 'actually', 'basically', 'really',
                'gonna', 'wanna', 'gotta', 'kinda', 'sorta'
            })
            
            self._initialized = True
            self.logger.info("NLTKProcessor initialized successfully")
            
        except ImportError as e:
            self.logger.warning(f"NLTK is not installed. Run: pip install nltk. Error: {e}")
            self._initialized = False
        except Exception as e:
            self.logger.error(f"NLTKProcessor initialization failed: {e}")
            self._initialized = False
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Get wordnet pos."""
        return self._WORDNET_POS_MAP.get(treebank_tag[0], 'n')
    
    def process(self, text: str) -> List[str]:
        """Process."""
        if not self._initialized:
            return self._fallback_process(text)
        
        try:
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag
            
            tokens = word_tokenize(text.lower())
            
            filtered_tokens = [
                token for token in tokens
                if token.isalpha() and token not in self._stopwords and len(token) > 1
            ]
            
            if not filtered_tokens:
                return []
            
            pos_tagged = pos_tag(filtered_tokens)
            
            lemmatized = [
                self._lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
                for word, tag in pos_tagged
            ]
            
            return lemmatized
            
        except Exception as e:
            self.logger.warning(f"NLTK processing failed; falling back to simple tokenization: {e}")
            return self._fallback_process(text)
    
    def _fallback_process(self, text: str) -> List[str]:
        """Run fallback process."""
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        
        basic_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        return [w for w in words if len(w) > 1 and w not in basic_stopwords]
    
    @property
    def is_available(self) -> bool:
        """Run is available."""
        return self._initialized

class GlobalModelManager:

    _STATUS_NOT_STARTED = "NOT_STARTED"
    _STATUS_LOADING = "LOADING"
    _STATUS_READY = "READY"
    _LOAD_WAIT_TIMEOUT_SECONDS = 120.0
    
    _instance = None
    _init_lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
                    cls._instance._model_configs = {}
                    cls._instance._access_history = {}
                    cls._instance._model_statuses = {}
                    cls._instance._model_events = {}
                    
                    
                    cls._instance._model_locks = {}
                    
                    cls._instance._locks_dict_lock = threading.RLock()
                    
                    cls._instance.logger = logging.getLogger(f"{__name__}.GlobalModelManager")
        return cls._instance
    
    def _get_model_lock(self, key: str) -> threading.Lock:
        """Get model lock."""
        with self._locks_dict_lock:
            if key not in self._model_locks:
                self._model_locks[key] = threading.RLock()
            return self._model_locks[key]

    def _claim_model_load(self, key: str):
        """Returns: (cache_hit, cached_model, event, is_worker)."""
        with self._locks_dict_lock:
            status = self._model_statuses.get(key, self._STATUS_NOT_STARTED)

            if status == self._STATUS_READY and key in self._models:
                return True, self._models[key], None, False

            if status == self._STATUS_LOADING:
                event = self._model_events.get(key)
                if event is None:
                    event = threading.Event()
                    self._model_events[key] = event
                return False, None, event, False

            event = self._model_events.get(key)
            if event is None:
                event = threading.Event()
                self._model_events[key] = event
            event.clear()
            self._model_statuses[key] = self._STATUS_LOADING
            return False, None, event, True

    def _publish_model_ready(self, key: str, model: Any, config: Dict[str, Any]) -> None:
        """Run publish model ready."""
        with self._locks_dict_lock:
            self._models[key] = model
            self._model_configs[key] = config
            self._access_history[key] = datetime.now()
            self._model_statuses[key] = self._STATUS_READY

    def _publish_model_failure(self, key: str) -> None:
        """Run publish model failure."""
        with self._locks_dict_lock:
            self._models.pop(key, None)
            self._model_configs.pop(key, None)
            self._access_history.pop(key, None)
            self._model_statuses[key] = self._STATUS_NOT_STARTED

    def _wait_for_model_ready(self, key: str, event: threading.Event, raise_on_failure: bool = True) -> Any:
        """Run wait for model ready."""
        completed = event.wait(timeout=self._LOAD_WAIT_TIMEOUT_SECONDS)

        with self._locks_dict_lock:
            status = self._model_statuses.get(key, self._STATUS_NOT_STARTED)
            if status == self._STATUS_READY and key in self._models:
                return self._models[key]

        if not completed:
            raise TimeoutError(f"Timed out waiting for model {key} to load; current status: {status}")

        if raise_on_failure:
            raise RuntimeError(f"Model {key} failed to load; status reset to {status}")

        self.logger.error(f"Model {key} failed to load; status reset to {status}")
        return None

    def get_or_load_model(self, model_type: str, model_name: str, loader_func: Callable) -> Any:
        """Return or load model."""
        key = f"{model_type}:{model_name}"
        
        if key in self._models and self._model_statuses.get(key) == self._STATUS_READY:
            return self._models[key]

        cache_hit, cached_model, event, is_worker = self._claim_model_load(key)
        if cache_hit:
            return cached_model

        if is_worker:
            try:
                self.logger.info(f"[Load] Loading new model: {key} ...")
                start_time = datetime.now()
                model = loader_func()
                elapsed = (datetime.now() - start_time).total_seconds()

                self._publish_model_ready(
                    key,
                    model,
                    {"type": model_type, "name": model_name},
                )
                self.logger.info(f"[Ready] Model {key} loaded in {elapsed:.2f}s")
                return model
            except Exception as e:
                self.logger.error(f"[Error] Model {key} failed to load: {e}")
                self._publish_model_failure(key)
                raise
            finally:
                event.set()

        return self._wait_for_model_ready(key, event, raise_on_failure=True)
    
    def preload_model(self, model_type: str, model_name: str, **kwargs):
        """Preload model."""
        cache_key = f"{model_type}:{model_name}"
        
        if cache_key in self._models and self._model_statuses.get(cache_key) == self._STATUS_READY:
            self.logger.info(f"Model {cache_key} is already cached; skipping preload")
            return self._models[cache_key]

        cache_hit, cached_model, event, is_worker = self._claim_model_load(cache_key)
        if cache_hit:
            return cached_model

        if not is_worker:
            return self._wait_for_model_ready(cache_key, event, raise_on_failure=False)

        self.logger.info(f"Starting optimized preload: {cache_key}")
        try:
            model = None
            start_time = datetime.now()
            if model_type == "text_embedding":
                model = self._load_optimized_embedding(model_name, **kwargs)
            elif model_type == "reranker":
                model = self._load_optimized_reranker(model_name, **kwargs)
            elif model_type == "splade":
                model = self._load_optimized_splade(model_name, **kwargs)
            elif model_type == "spacy":
                
                model = self._load_spacy_model(model_name, **kwargs)
            elif model_type == "nltk":
                
                model = self._load_nltk_processor(model_name, **kwargs)
            else:
                self.logger.warning(f"Unknown model type {model_type}; skipping optimized preload")
                self._publish_model_failure(cache_key)
                return None

            elapsed = (datetime.now() - start_time).total_seconds()
            if model:
                self._publish_model_ready(
                    cache_key,
                    model,
                    {"type": model_type, "name": model_name, "optimized": True},
                )
                self.logger.info(f"Model {cache_key} preloaded in {elapsed:.2f}s")
                return model

            self._publish_model_failure(cache_key)
            return None
        except Exception as e:
            self.logger.error(f"Model {cache_key} preload failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self._publish_model_failure(cache_key)
            return None
        finally:
            event.set()

    
    
    

    def _get_device_and_dtype(self):
        """Get device and dtype."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        use_flash_attn = False

        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            
            # Flash Attention requires both compatible hardware and the
            # separately installed optional package.
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8 and is_flash_attention_available():
                    use_flash_attn = True
            except (RuntimeError, AssertionError):
                pass
        
        return device, dtype, use_flash_attn

    def _load_optimized_embedding(self, model_name: str, **kwargs):
        """Load optimized embedding."""
        from sentence_transformers import SentenceTransformer
        
        device, dtype, use_flash_attn = self._get_device_and_dtype()
        self.logger.info(f"Configuring embedding model: {model_name} | device={device} | dtype={dtype} | flash_attn={use_flash_attn}")

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None
        }

        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        try:
            model = SentenceTransformer(model_name, model_kwargs=model_kwargs, trust_remote_code=True)
        except Exception as e:
            if use_flash_attn:
                self.logger.warning(f"Flash Attention loading failed; using the standard attention path: {e}")
                model_kwargs.pop("attn_implementation", None)
                model = SentenceTransformer(model_name, model_kwargs=model_kwargs, trust_remote_code=True)
            else:
                raise e
        return model

    def _load_optimized_reranker(self, model_name: str, **kwargs):
        """Load optimized reranker."""
        backend = settings.reranker_backend

        if backend == "vllm":
            self.logger.info(
                "Configuring vLLM HTTP reranker: %s | api_url=%s",
                model_name,
                settings.vllm_api_url,
            )
            return {
                "backend": "vllm",
                "model_name": model_name,
                "api_url": settings.vllm_api_url,
                "timeout": settings.vllm_timeout_seconds,
                "max_retries": settings.vllm_max_retries,
                "device": torch.device("cpu"),
            }

        from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
        
        device, dtype, use_flash_attn = self._get_device_and_dtype()
        self.logger.info(f"Configuring reranker: {model_name} | device={device} | dtype={dtype} | flash_attn={use_flash_attn}")

        is_causal = "qwen" in model_name.lower()
        
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None
        }
        
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        try:
            model_cls = AutoModelForCausalLM if is_causal else AutoModelForSequenceClassification
            model = model_cls.from_pretrained(model_name, **model_kwargs).eval()
        except Exception as e:
            if use_flash_attn:
                self.logger.warning(f"Flash Attention loading failed; using the standard attention path: {e}")
                model_kwargs.pop("attn_implementation", None)
                model = model_cls.from_pretrained(model_name, **model_kwargs).eval()
            else:
                raise e

        wrapper = {
            'backend': 'native',
            'model': model,
            'tokenizer': tokenizer,
            'device': torch.device(device),
        }
        
        if is_causal:
            wrapper['prefix_tokens'] = tokenizer.encode("<|im_start|>", add_special_tokens=False)
            wrapper['suffix_tokens'] = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
            wrapper['target_token_id'] = yes_ids[0] if yes_ids else None

        return wrapper

    def _load_optimized_splade(self, model_name: str, **kwargs):
        """Load optimized splade."""
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        
        device, dtype, _ = self._get_device_and_dtype()
        self.logger.info(f"Configuring SPLADE model: {model_name} | device={device} | dtype={dtype}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        ).eval()
        
        return self._create_splade_wrapper(tokenizer, model, device)

    def _load_nltk_processor(self, model_name: str, **kwargs) -> Optional[NLTKProcessor]:
        """Load nltk processor."""
        self.logger.info(f"Initializing NLTK processor: {model_name}")
        
        try:
            processor = NLTKProcessor()
            if processor.is_available:
                self.logger.info("NLTK processor initialized successfully")
                return processor
            else:
                self.logger.warning("NLTK processor initialization failed; using fallback tokenization")
                return processor
        except Exception as e:
            self.logger.error(f"NLTK processor loading failed: {e}")
            return None
        
    def _load_spacy_model(self, model_name: str, **kwargs):
        """Load spacy model."""
        self.logger.info(f"Configuring spaCy model: {model_name}")
        
        try:
            import spacy
            
            disable_components = kwargs.get("disable", ["ner", "parser", "textcat", "senter"])
            
            try:
                
                nlp = spacy.load(model_name, disable=disable_components)
                self.logger.info(f"spaCy model {model_name} loaded successfully (disabled={disable_components})")
                return nlp
            except OSError:
                self.logger.warning(f"spaCy model '{model_name}' was not found; attempting download...")
                from spacy.cli import download
                download(model_name)
                nlp = spacy.load(model_name, disable=disable_components)
                self.logger.info(f"spaCy model {model_name} downloaded and loaded successfully")
                return nlp
                
        except ImportError:
            self.logger.error("spaCy is not installed. Run: pip install spacy")
            return None
        except Exception as e:
            self.logger.error(f"spaCy model loading failed: {e}")
            return None

    def get_spacy_model(self, model_name: str = "en_core_web_lg", disable_components: list = None):
        """Return spacy model."""
        if disable_components is None:
            disable_components = ["ner", "parser", "textcat", "senter"]
        return self.preload_model("spacy", model_name, disable=disable_components)
    
    def get_nltk_processor(self, processor_name: str = "default") -> Optional[NLTKProcessor]:
        """Return nltk processor."""
        return self.preload_model("nltk", processor_name)

    def get_loaded_models(self) -> Dict[str, str]:
        """Return loaded models."""
        return {key: str(type(model).__name__) for key, model in self._models.items()}
    
    def cleanup_all(self):
        """Release associated resources."""
        logger.info("Cleaning up all global models...")
        
        
        
        for cache_key, model in list(self._models.items()):
            try:
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                elif isinstance(model, dict) and 'model' in model:
                     del model['model']
                
                logger.debug(f"Cleaning model object: {cache_key}")
            except Exception as e:
                logger.warning(f"Model cleanup failed for {cache_key}: {e}")
        
        self._models.clear()
        self._model_configs.clear()
        
        with self._locks_dict_lock:
            self._access_history.clear()
            self._model_statuses.clear()
            for event in self._model_events.values():
                event.set()
            self._model_events.clear()
            self._model_locks.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache released")
        
        logger.info("Global model cleanup complete")
    
    def get_splade_model(self, model_name: str = "naver/splade-v3"):
        """Return splade model."""
        return self.preload_model("splade", model_name)
    
    def _create_splade_wrapper(self, tokenizer, model, device):
        """Create splade wrapper."""
        class SPLADEWrapper:
            def __init__(self, tokenizer, model, device):
                self.tokenizer = tokenizer
                self.model = model
                self.device = device
            
            def encode_document(self, texts):
                if isinstance(texts, str): texts = [texts]
                results = []
                with torch.no_grad():
                    inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    values = torch.log1p(torch.relu(logits)).max(dim=1).values.float()
                    values_cpu = values.cpu().numpy()
                    vocab_size = values_cpu.shape[1]
                    for i in range(len(texts)):
                        nz_indices = np.nonzero(values_cpu[i])[0]
                        nz_values = values_cpu[i][nz_indices]
                        idx_tensor = torch.from_numpy(nz_indices.astype(np.int64)).unsqueeze(0)
                        val_tensor = torch.from_numpy(nz_values.astype(np.float32))
                        sparse_t = torch.sparse_coo_tensor(
                            idx_tensor, val_tensor, size=(vocab_size,), device=self.device
                        )
                        results.append(sparse_t)
                return results
            
            def encode_query(self, text):
                return self.encode_document(text)[0]
        
        return SPLADEWrapper(tokenizer, model, device)


global_model_manager = GlobalModelManager()
