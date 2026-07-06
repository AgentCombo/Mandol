"""Utilities for siliconflow embedding adapter."""

import logging
from typing import List, Optional, Union

import numpy as np
import requests

from ..utils.config_manager import settings


class SiliconFlowEmbeddingAdapter:
    def __init__(self, model_name: str, api_key: Optional[str] = None, dimensions: int = None):
        self.api_model_name = model_name.replace("-remote", "")
        self.api_key = api_key or settings.get_api_key("SILICONFLOW_API_KEY")
        self.dimensions = dimensions
        self.url = settings.siliconflow_embeddings_url
        
        if not self.api_key:
            logging.warning("SILICONFLOW_API_KEY was not found in .env or the environment; cloud embeddings will be unavailable.")

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode."""
        is_single = isinstance(sentences, str)
        texts = [sentences] if is_single else sentences
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._call_api(batch)
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
            else:
                
                logging.error(f"Batch {i} API call failed; filling zero vectors")
                dim = self.dimensions if self.dimensions else 4096
                all_embeddings.extend([np.zeros(dim) for _ in batch])
        
        result = np.array(all_embeddings, dtype=np.float32)
        
        if is_single and result.shape[0] == 1:
            return result[0]
            
        return result

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Run call API."""
        if not self.api_key:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.api_model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        if self.dimensions and "Qwen" in self.api_model_name:
            payload["dimensions"] = self.dimensions

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                sorted_data = sorted(data['data'], key=lambda x: x['index'])
                return [item['embedding'] for item in sorted_data]
            else:
                logging.error(f"SiliconFlow API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logging.error(f"Cloud embedding request failed: {e}")
            return []
