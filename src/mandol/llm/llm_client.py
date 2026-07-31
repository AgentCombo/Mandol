from ..utils.logging_config import create_module_logger
import re
import asyncio
import traceback
from typing import List, Dict, Any, Optional
from ..utils.config_manager import settings

from .configs import MODEL_CONFIGS, API_PROVIDERS
from .providers.base import DEFAULT_REQUEST_MAX_RETRIES, DEFAULT_REQUEST_TIMEOUT_SECONDS
from .providers.standard import StandardProvider
from .providers.reasoning import ReasoningProvider


DEFAULT_ASYNC_MAX_CONCURRENCY = 10
QWEN_DASHSCOPE_ASYNC_MAX_CONCURRENCY = 60
DEEPSEEK_DASHSCOPE_ASYNC_MAX_CONCURRENCY = 30


def estimate_tokens(text: str) -> int:
    """Estimate tokens."""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    other_chars = len(text) - chinese_chars - english_chars
    
    return int(chinese_chars * 0.6 + english_chars * 0.3 + other_chars * 0.4)


class LLMClient:
    
    
    MODEL_CONFIGS = MODEL_CONFIGS
    API_PROVIDERS = API_PROVIDERS
    
    def __init__(self, 
             model_name: str = "deepseek-v4-pro-thinking",
             api_key: Optional[str] = None,
             base_url: Optional[str] = None,
             max_context_ratio: float = 0.85,
             request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
             request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS):
        self.logger = create_module_logger("llm.llm_client.LLMClient")
        
        model_config = MODEL_CONFIGS.get(model_name)
        if not model_config:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_CONFIGS.keys())}")
        
        provider_name = model_config["provider"]
        provider_config = API_PROVIDERS.get(provider_name)
        if not provider_config:
            raise ValueError(f"Unsupported API provider: {provider_name}")
            
        final_api_key = api_key
        if not final_api_key:
            
            primary_env = provider_config["api_key_env"]
            final_api_key = settings.get_api_key(primary_env)
            if not final_api_key and "fallback_env" in provider_config:
                fallback_env = provider_config["fallback_env"]
                final_api_key = settings.get_api_key(fallback_env)
                if final_api_key:
                    self.logger.info(f"Using fallback environment variable {fallback_env}")
        
        if not final_api_key:
            raise ValueError(f"API key not found. Set {provider_config['api_key_env']} in the project .env file")
            
        final_base_url = base_url or provider_config["base_url"]
        
        model_type = model_config.get("model_type", "standard")
        
        if model_type == "reasoning":
            self.logger.debug("Instantiating ReasoningProvider")
            self.provider = ReasoningProvider(
                model_name=model_name,
                model_config=model_config,
                provider_config=provider_config,
                api_key=final_api_key,
                base_url=final_base_url,
                max_context_ratio=max_context_ratio,
                request_max_retries=request_max_retries,
                request_timeout=request_timeout
            )
        else:
            self.logger.debug("Instantiating StandardProvider")
            self.provider = StandardProvider(
                model_name=model_name,
                model_config=model_config,
                provider_config=provider_config,
                api_key=final_api_key,
                base_url=final_base_url,
                max_context_ratio=max_context_ratio,
                request_max_retries=request_max_retries,
                request_timeout=request_timeout
            )
            
        
        self.context_length = self.provider.context_length
        self.max_output_tokens = self.provider.max_output_tokens
        self.default_output_tokens = self.provider.default_output_tokens
        self.max_context_tokens = self.provider.max_context_tokens
        self.model_name = model_name
        self.actual_model = self.provider.actual_model
        self.base_url = final_base_url
        self.supports_json_format = self.provider.supports_json_format
        self.tokenizer = self.provider.tokenizer
        self.client = self.provider.client
        self.async_client = self.provider.async_client
        self.provider_name = provider_name
        self.request_max_retries = self.provider.request_max_retries
        self.request_timeout = self.provider.request_timeout

    def count_tokens(self, text: str) -> int:
        return self.provider.count_tokens(text)

    def truncate_context(self, context_text: str, max_tokens: Optional[int] = None) -> str:
        return self.provider.truncate_context(context_text, max_tokens)

    def generate_answer(self, 
                       prompt: str, 
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.1,
                       generate_strategy: str = "default",
                       json_format: bool = False,
                       **kwargs) -> str:
        """Generate answer."""
        return self.provider.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            generate_strategy=generate_strategy,
            json_format=json_format,
            **kwargs
        )

    async def generate_answer_async(self, 
                                    prompt: str, 
                                    max_tokens: Optional[int] = None,
                                    temperature: float = 0.1,
                                    generate_strategy: str = "default",
                                    json_format: bool = False,
                                    **kwargs) -> str:
        """Generate answer async."""
        return await self.provider.generate_async(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            generate_strategy=generate_strategy,
            json_format=json_format,
            **kwargs
        )

    async def batch_generate_async(self, 
                                   prompts: List[str], 
                                   max_tokens: Optional[int] = None,
                                   temperature: float = 0.1,
                                   max_concurrency: Optional[int] = None,
                                   **kwargs) -> List[str]:
        """Run batch generate async."""
        if max_concurrency is None:
            max_concurrency = self._default_async_max_concurrency()
        max_concurrency = max(1, int(max_concurrency))
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _generate_one(prompt: str) -> str:
            async with semaphore:
                return await self.generate_answer_async(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
        
        self.logger.info(f"[Async] Batch generation started: {len(prompts)} requests, max_concurrency={max_concurrency}")
        raw_results = await asyncio.gather(
            *[_generate_one(p) for p in prompts],
            return_exceptions=True,
        )
        results = []
        for result in raw_results:
            if isinstance(result, Exception):
                self.logger.error(f"[Async] One batch-generation item failed: {result}")
                results.append(f"Generation failed: {str(result)}")
            else:
                results.append(result)
        self.logger.info("[Async] Batch generation completed")
        return list(results)

    def _default_async_max_concurrency(self) -> int:
        actual_model = str(getattr(self, "actual_model", "")).lower()
        provider_name = str(getattr(self, "provider_name", "")).lower()
        if provider_name == "dashscope" and ("qwen-plus" in actual_model or "qwen3.5-plus" in actual_model):
            return QWEN_DASHSCOPE_ASYNC_MAX_CONCURRENCY
        if provider_name == "dashscope" and "deepseek-v3.2" in actual_model:
            return DEEPSEEK_DASHSCOPE_ASYNC_MAX_CONCURRENCY
        return DEFAULT_ASYNC_MAX_CONCURRENCY

    def batch_generate(self, 
                      prompts: List[str], 
                      max_tokens: Optional[int] = None,
                      temperature: float = 0.1,
                      **kwargs) -> List[str]:
        """Run batch generate."""
        results = []
        self.logger.info(f"Batch generation started: {len(prompts)} requests")
        for i, prompt in enumerate(prompts, 1):
            self.logger.debug(f"Processing {i}/{len(prompts)}")
            answer = self.generate_answer(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            results.append(answer)
        self.logger.info("Batch generation completed")
        return results

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Run analyze text."""
        total_tokens = self.count_tokens(text)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        return {
            "total_tokens": total_tokens,
            "character_count": len(text),
            "chinese_chars": chinese_chars,
            "english_chars": english_chars,
            "tokens_per_char": total_tokens / len(text) if text else 0,
            "fits_in_context": total_tokens <= self.max_context_tokens,
            "usage_ratio": total_tokens / self.context_length,
            "can_process": total_tokens <= (self.context_length - self.default_output_tokens)
        }

    def get_context_info(self) -> Dict[str, Any]:
        """Return context info."""
        return {
            "model_name": self.model_name,
            "actual_model": self.actual_model,
            "provider": self.provider.model_config["provider"],
            "base_url": self.base_url,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "default_output_tokens": self.default_output_tokens,
            "max_context_tokens": self.max_context_tokens,
            "tokenizer_available": self.tokenizer is not None,
            "encoding": self.provider.model_config.get("encoding", "unknown"),
            "supports_json_format": self.supports_json_format,
            "request_max_retries": self.request_max_retries,
            "request_timeout": self.request_timeout
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        return self.get_context_info()

    @classmethod
    def list_available_models(cls) -> Dict[str, List[str]]:
        models_by_provider = {}
        for model_name, config in MODEL_CONFIGS.items():
            provider = config["provider"]
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model_name)
        return models_by_provider
    
    @classmethod
    def get_provider_info(cls, provider: str) -> Optional[Dict[str, Any]]:
        return API_PROVIDERS.get(provider)
    
    @classmethod
    def list_required_env_vars(cls) -> Dict[str, List[str]]:
        env_vars = {}
        for provider, config in API_PROVIDERS.items():
            vars_list = [config["api_key_env"]]
            if "fallback_env" in config:
                vars_list.append(f"{config['fallback_env']} (fallback)")
            env_vars[provider] = vars_list
        return env_vars


def create_llm_client(model: str = "deepseek-v4-pro-thinking", 
                     api_key: Optional[str] = None,
                     base_url: Optional[str] = None,
                     request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
                     request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS) -> LLMClient:
    return LLMClient(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        request_max_retries=request_max_retries,
        request_timeout=request_timeout,
    )

def create_deepseek_client(model: str = "deepseek-v4-flash", 
                          api_key: Optional[str] = None,
                          request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
                          request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS) -> LLMClient:
    return LLMClient(
        model_name=model,
        api_key=api_key,
        request_max_retries=request_max_retries,
        request_timeout=request_timeout,
    )




if __name__ == "__main__":
    import time
    
    print("=" * 80)
    print("LLM客户端测试工具 (Flat Structure Refactor)")
    print("=" * 80)
    
    print("\n 可用模型:")
    models = LLMClient.list_available_models()
    all_models = []
    model_index = 1
    for provider, model_list in models.items():
        print(f"\n{provider}:")
        for model in model_list:
            print(f"  [{model_index}] {model}")
            all_models.append(model)
            model_index += 1
    
    print("\n" + "=" * 80)
    model_choice = input(f"请选择要测试的模型 (1-{len(all_models)}, 直接回车默认使用 deepseek-v3:671b): ").strip()
    
    if model_choice == "":
        selected_model = "deepseek-v3:671b"
        print(f"使用默认模型: {selected_model}")
    elif model_choice.isdigit() and 1 <= int(model_choice) <= len(all_models):
        selected_model = all_models[int(model_choice) - 1]
        print(f"已选择模型: {selected_model}")
    else:
        print(f" 无效选择，使用默认模型: deepseek-v3:671b")
        selected_model = "deepseek-v3:671b"
    
    print("\n" + "=" * 80)
    test_prompt = input("请输入测试问题 (直接回车使用默认问题): ").strip()
    if test_prompt == "":
        test_prompt = "请简单介绍一下人工智能。"
        print(f"使用默认问题: {test_prompt}")
    
    print("\n" + "=" * 80)
    print(f"测试模型: {selected_model}")
    print("=" * 80)
    
    try:
        print("\nInitializing client...")
        init_start = time.time()
        client = LLMClient(selected_model)
        init_time = time.time() - init_start
        print(f" 客户端初始化完成 (耗时: {init_time:.2f}秒)")
        
        print("\n 分析测试文本...")
        test_text = "Hello world! 你好世界！这是一个测试文本。"
        analysis = client.analyze_text(test_text)
        print(f"   文本: {test_text}")
        print(f"   Token数: {analysis['total_tokens']}")
        print(f"   字符数: {analysis['character_count']}")
        print(f"   中文字符: {analysis['chinese_chars']}")
        print(f"   英文字符: {analysis['english_chars']}")
        
        print(f"\n 正在生成回答...")
        print(f"   问题: {test_prompt}")
        generate_start = time.time()
        answer = client.generate_answer(test_prompt, max_tokens=1000)
        generate_time = time.time() - generate_start
        
        print("\n" + "=" * 80)
        print(" 生成结果:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        print(f"\nPerformance statistics:")
        print(f"   初始化耗时: {init_time:.2f}秒")
        print(f"   生成耗时: {generate_time:.2f}秒")
        print(f"   总耗时: {init_time + generate_time:.2f}秒")
        
        answer_analysis = client.analyze_text(answer)
        print(f"\n 回答分析:")
        print(f"   字符数: {answer_analysis['character_count']}")
        print(f"   Token数: {answer_analysis['total_tokens']}")
        print(f"   中文字符: {answer_analysis['chinese_chars']}")
        print(f"   英文字符: {answer_analysis['english_chars']}")
        
        info = client.get_model_info()
        print(f"\n 模型信息:")
        print(f"   配置模型: {info['model_name']}")
        print(f"   实际模型: {info['actual_model']}")
        print(f"   提供商: {info['provider']}")
        print(f"   API地址: {info['base_url']}")
        print(f"   上下文长度: {info['context_length']:,}")
        print(f"   最大输出: {info['max_output_tokens']:,}")
        print(f"   默认输出: {info['default_output_tokens']:,}")
        print(f"   支持JSON: {info['supports_json_format']}")
        
    except Exception as e:
        print(f"\n 测试失败: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
