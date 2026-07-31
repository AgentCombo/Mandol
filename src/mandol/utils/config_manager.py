"""Utilities for config manager."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _discover_project_env_file() -> str:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "mandol").exists():
            return str(parent / ".env")
    return ".env"


PROJECT_ENV_FILE = _discover_project_env_file()


class AppSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    deepseek_api_key: Optional[SecretStr] = Field(default=None, alias="DEEPSEEK_API_KEY")
    closeai_api_key: Optional[SecretStr] = Field(default=None, alias="CLOSEAI_API_KEY")
    cstcloud_api_key: Optional[SecretStr] = Field(default=None, alias="CSTCLOUD_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[SecretStr] = Field(default=None, alias="OPENROUTER_API_KEY")
    dashscope_api_key: Optional[SecretStr] = Field(default=None, alias="DASHSCOPE_API_KEY")
    siliconflow_api_key: Optional[SecretStr] = Field(default=None, alias="SILICONFLOW_API_KEY")
    vllm_api_key: Optional[SecretStr] = Field(default=None, alias="VLLM_API_KEY")

    hf_token: Optional[SecretStr] = Field(default=None, alias="HF_TOKEN")
    huggingface_token: Optional[SecretStr] = Field(default=None, alias="HUGGINGFACE_TOKEN")
    hf_endpoint: str = Field(default="https://huggingface.co", alias="HF_ENDPOINT")
    hf_home: Optional[str] = Field(default=None, alias="HF_HOME")

    # OpenAI-compatible provider base URLs。
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    cstcloud_base_url: str = Field(default="https://uni-api.cstcloud.cn/v1", alias="CSTCLOUD_BASE_URL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    closeai_base_url: str = Field(default="https://api.openai-proxy.org/v1", alias="CLOSEAI_BASE_URL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )

    
    siliconflow_embeddings_url: str = Field(
        default="https://api.siliconflow.cn/v1/embeddings",
        alias="SILICONFLOW_EMBEDDINGS_URL",
    )
    siliconflow_rerank_url: str = Field(
        default="https://api.siliconflow.cn/v1/rerank",
        alias="SILICONFLOW_RERANK_URL",
    )
    dashscope_rerank_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        alias="DASHSCOPE_RERANK_URL",
    )
    cstcloud_rerank_url: str = Field(
        default="https://uni-api.cstcloud.cn/v1/rerank",
        alias="CSTCLOUD_RERANK_URL",
    )

    
    reranker_backend: str = Field(default="native", alias="RERANKER_BACKEND")
    vllm_gpu_memory_utilization: float = Field(default=0.5, alias="VLLM_GPU_MEMORY_UTILIZATION")
    vllm_api_url: str = Field(default="http://localhost:8000/score", alias="VLLM_API_URL")
    vllm_timeout_seconds: float = Field(default=30.0, alias="VLLM_TIMEOUT_SECONDS")
    vllm_max_retries: int = Field(default=2, alias="VLLM_MAX_RETRIES")

    _API_KEY_FIELD_BY_ENV: ClassVar[dict[str, str]] = {
        "DEEPSEEK_API_KEY": "deepseek_api_key",
        "CLOSEAI_API_KEY": "closeai_api_key",
        "CSTCLOUD_API_KEY": "cstcloud_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "OPENROUTER_API_KEY": "openrouter_api_key",
        "DASHSCOPE_API_KEY": "dashscope_api_key",
        "SILICONFLOW_API_KEY": "siliconflow_api_key",
        "VLLM_API_KEY": "vllm_api_key",
        "HF_TOKEN": "hf_token",
        "HUGGINGFACE_TOKEN": "huggingface_token",
    }

    @staticmethod
    def _secret_to_plain(secret: Optional[SecretStr]) -> Optional[str]:
        """Run secret to plain."""
        if secret is None:
            return None
        value = secret.get_secret_value().strip()
        return value or None

    @field_validator("reranker_backend")
    @classmethod
    def _validate_reranker_backend(cls, value: str) -> str:
        normalized = (value or "native").strip().lower()
        if normalized not in {"native", "vllm"}:
            raise ValueError("RERANKER_BACKEND must be either 'native' or 'vllm'")
        return normalized

    @field_validator("vllm_gpu_memory_utilization")
    @classmethod
    def _validate_vllm_gpu_memory_utilization(cls, value: float) -> float:
        utilization = float(value)
        if utilization <= 0.0 or utilization > 1.0:
            raise ValueError("VLLM_GPU_MEMORY_UTILIZATION must be in the range (0, 1]")
        return utilization

    @field_validator("vllm_timeout_seconds")
    @classmethod
    def _validate_vllm_timeout_seconds(cls, value: float) -> float:
        timeout = float(value)
        if timeout <= 0.0:
            raise ValueError("VLLM_TIMEOUT_SECONDS must be positive")
        return timeout

    @field_validator("vllm_max_retries")
    @classmethod
    def _validate_vllm_max_retries(cls, value: int) -> int:
        retries = int(value)
        if retries < 0:
            raise ValueError("VLLM_MAX_RETRIES must be non-negative")
        return retries

    def get_api_key(self, env_name: str) -> Optional[str]:
        """Return api key."""
        normalized_name = env_name.upper()
        field_name = self._API_KEY_FIELD_BY_ENV.get(normalized_name)
        if field_name is None:
            return None

        value = self._secret_to_plain(getattr(self, field_name))
        if value:
            return value

        
        if normalized_name == "HF_TOKEN":
            return self._secret_to_plain(self.huggingface_token)
        if normalized_name == "HUGGINGFACE_TOKEN":
            return self._secret_to_plain(self.hf_token)
        return None

    def require_api_key(self, env_name: str) -> str:
        """Run require API key."""
        api_key = self.get_api_key(env_name)
        if not api_key:
            raise ValueError(f"未找到 API 密钥，请在项目根目录 .env 中配置 {env_name}")
        return api_key


settings = AppSettings()