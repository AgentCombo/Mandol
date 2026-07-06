import logging
from ..utils.logging_config import create_module_logger
import torch
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils.model_manager import global_model_manager

logger = create_module_logger("llm.local_llm_client")

class LocalLLMClient:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        
    def _ensure_loaded(self):
        """Ensure loaded."""
        if self.model is not None:
            return

        logger.info(f"Loading local model through the global model manager: {self.model_name}")

        def _loader():
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model.eval()
            return {"model": model, "tokenizer": tokenizer, "device": self.device}

        try:
            model_wrapper = global_model_manager.get_or_load_model(
                model_type="causal_lm",
                model_name=self.model_name,
                loader_func=_loader
            )
            self.model = model_wrapper["model"]
            self.tokenizer = model_wrapper["tokenizer"]
            logger.info(f"Retrieving model from the global manager: {self.model_name}")
        except Exception as e:
            logger.error(f"Local model loading failed: {e}")
            raise

    def generate(self, 
                 prompt: Union[str, List[Dict]], 
                 max_tokens: int = 512, 
                 temperature: float = 0.1, 
                 stop_words: List[str] = None, 
                 **kwargs) -> str:
        """Generate."""
        self._ensure_loaded()
        
        if isinstance(prompt, list):
            model_inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.device)
        else:
            model_inputs = self.tokenizer(
                prompt, 
                return_tensors="pt"
            ).to(self.device)
            
        # Dataset-specific handling used by the reproduction workflow.
        # Dataset-specific handling used by the reproduction workflow.
        explicit_do_sample = kwargs.pop("do_sample", None)
        if explicit_do_sample is not None:
             do_sample = explicit_do_sample
        else:
             do_sample = (temperature > 0)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )
            
        input_len = model_inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response

    def cleanup(self):
        """Release associated resources."""
        self.model = None
        self.tokenizer = None
