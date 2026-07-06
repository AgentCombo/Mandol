"""Targeted query expansion for Mandol's quantification stage.

The expander turns insufficiency signals from the quantifier into follow-up
search queries. It supports local and API-backed LLM clients, parses structured
JSON when possible, and falls back to the original query when expansion fails.
"""

import logging
from ..utils.logging_config import create_module_logger
import json
import re
from typing import List, Optional, Dict, Any

from .multi_query_prompts import OPTIMIZED_QUERY_EXPANSION_PROMPT, TARGETED_SEARCH_PROMPT
from ..llm.llm_client import LLMClient, create_llm_client
from ..llm.local_llm_client import LocalLLMClient
from ..utils.model_manager import global_model_manager

logger = create_module_logger("quantification.query_expander")


class QueryExpander:
    """Generate follow-up retrieval queries from missing-evidence descriptions.

    Args:
        model_source: ``"local"`` or ``"api"``. API-style model names passed
            with ``"local"`` are promoted to API mode for compatibility.
        model_name: Local model ID or API model name used for query expansion.
        llm_client: Optional API client. If omitted in API mode, the default
            Mandol LLM client factory is used.
        device: Optional device passed to ``LocalLLMClient``.

    Notes:
        Local clients are owned by this object and released by ``cleanup``.
        API clients may be supplied by the caller and are not recreated unless
        they are missing during initialization.
    """
    
    def __init__(self,
                 model_source: str = "local",
                 model_name: str = "Qwen/Qwen3-4B",
                 llm_client: Optional[LLMClient] = None,
                 device: Optional[str] = None):
        self.model_source = model_source
        self.model_name = model_name
        self.llm_client = llm_client
        
        is_api_model = (
            model_name in LLMClient.MODEL_CONFIGS or 
            any(model_name.startswith(prefix) for prefix in ["gpt-", "claude-", "deepseek-", "gemini-"])
        )
        
        if model_source == "local" and is_api_model:
            logger.info(
                "Detected API model name %r; switching QueryExpander to API mode.",
                model_name,
            )
            model_source = "api"
            self.model_source = "api"
        
        self.local_client = None
        self.device = device
        
        if model_source == "local":
            if device is None:
                # LocalLLMClient will handle device selection
                self.device = None
            self.local_client = LocalLLMClient(model_name, device=self.device)
            
        elif model_source == "api":
            if self.llm_client is None:
                logger.info(
                    "No API client was provided; creating one for model %s.",
                    model_name,
                )
                try:
                    self.llm_client = create_llm_client(model=model_name)
                    logger.info("LLMClient created successfully.")
                except Exception as e:
                    raise ValueError(
                        f"API mode requires llm_client, and automatic creation failed: {e}"
                    )
        else:
            raise ValueError(f"Unsupported model_source: {model_source}")
        
        logger.info("QueryExpander initialized in %s mode.", model_source)
    
    def _format_prompt(self, query: str, missing_info: str) -> str:
        """Render the structured expansion prompt used by both model backends."""
        return OPTIMIZED_QUERY_EXPANSION_PROMPT.format(
            original_query=query,
            missing_info=missing_info
        )
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse model JSON, accepting markdown-wrapped responses as fallback."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # markdown json block
                r'```\s*(\{.*?\})\s*```',       # markdown generic block
                r'(\{.*\})'                      # raw JSON
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        continue
            
            logger.warning("Failed to parse JSON response: %s...", response[:200])
            return {}
    
    def _expand_with_local_model(self, query: str, missing_info: str) -> List[str]:
        """Generate expansion queries with the local LLM client.

        The local model is loaded lazily through ``LocalLLMClient``. Any model
        or parsing failure returns the original query so retrieval can continue.
        """
        if self.local_client is None:
            self.local_client = LocalLLMClient(
                model_name=self.model_name,
                device=self.device
            )
        
        prompt = self._format_prompt(query, missing_info)
        
        messages = [
            {"role": "system", "content": "You are a search query optimization expert."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.local_client.generate(
                messages,
                max_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
            parsed = self._parse_json_response(response)
            queries = parsed.get("queries", [])
            
            if not queries:
                logger.warning("No queries were extracted; falling back to the original query.")
                return [query]
            
            logger.debug("Generated expansion queries: %s", queries)
            return queries
            
        except Exception as e:
            logger.error("Local query expansion failed: %s", e)
            return [query]
    
    def _expand_with_api(self, query: str, missing_info: str) -> List[str]:
        """Expand a query through the configured model path.

        Expansion is best-effort: invalid model output, empty query lists, and
        runtime failures all fall back to the original query without changing
        retrieval control flow.
        """
        logger.info("Starting query expansion for %r.", query)
        logger.debug("Missing evidence description: %s", missing_info)
        
        try:
            if self.model_source == "local":
                queries = self._expand_with_local_model(query, missing_info)
            else:  # api
                queries = self._expand_with_api(query, missing_info)
            
            if not queries or not isinstance(queries, list):
                logger.warning("Query expansion returned an invalid result; using the original query.")
                return [query]
            
            queries = [q.strip() for q in queries if q and q.strip()]
            
            if not queries:
                logger.warning("All expanded queries were empty; using the original query.")
                return [query]
            
            logger.info("Query expansion completed with %d queries.", len(queries))
            return queries
            
        except Exception as e:
            logger.error("Query expansion failed: %s", e)
            return [query]
    
    def generate_targeted_query(self, query: str, missing_info: str) -> str:
        """Generate one focused search query for a missing evidence item.

        Args:
            query: Original user question.
            missing_info: Specific gap identified by the quantifier.

        Returns:
            A targeted search query. If generation or parsing fails, the return
            value is ``query`` combined with ``missing_info``.
        """
        if not missing_info or missing_info.strip() == "":
            logger.warning("Missing-evidence description is empty; returning the original query.")
            return query
        
        logger.info("Generating targeted query for %r; missing=%s", query, missing_info)
        
        try:
            prompt = TARGETED_SEARCH_PROMPT.format(
                original_query=query,
                missing_info=missing_info
            )
            
            messages = [
                {"role": "system", "content": "You are a precision search query expert."},
                {"role": "user", "content": prompt}
            ]
            
            if self.model_source == "local":
                if self.local_client is None:
                    self.local_client = LocalLLMClient(
                        model_name=self.model_name,
                        device=self.device
                    )
                response = self.local_client.generate(
                    messages,
                    max_tokens=256,
                    temperature=0.1
                )
            else:  # API mode
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.1,
                    json_format=True
                )
            
            parsed = self._parse_json_response(response)
            targeted_query = parsed.get("targeted_query", "").strip()
            
            if targeted_query:
                logger.info("Targeted query generated: %r", targeted_query)
                return targeted_query
            else:
                logger.warning("Targeted-query response was empty; using fallback query.")
                fallback = f"{query} {missing_info}"
                return fallback
                
        except Exception as e:
            logger.error("Targeted-query generation failed; using fallback query: %s", e)
            fallback = f"{query} {missing_info}"
            return fallback
    
    def cleanup(self):
        """Release associated resources."""
        if self.local_client is not None:
            logger.info("Releasing local query-expansion client references.")
            self.local_client.cleanup()
            self.local_client = None
            logger.info("Local query-expansion references released.")


def create_query_expander(
    model_source: str = "local",
    model_name: str = "Qwen/Qwen3-4B",
    llm_client: Optional[LLMClient] = None
) -> QueryExpander:
    """Create a QueryExpander with Mandol's default expansion model."""
    return QueryExpander(
        model_source=model_source,
        model_name=model_name,
        llm_client=llm_client
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("QueryExpander smoke test")
    print("=" * 80)
    
    print("\n### Example 1: local model mode")
    try:
        expander_local = QueryExpander(model_source="local")
        
        queries = expander_local.expand(
            query="Alice's hobbies",
            missing_info="Specific details about Alice's free time activities"
        )
        
        print(f"\nOriginal query: Alice's hobbies")
        print("Expanded queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        expander_local.cleanup()
        
    except Exception as e:
        print(f"Local model test failed: {e}")
    
    print("\n### Example 2: API mode")
    try:
        from mandol.llm.llm_client import LLMClient
        
        llm_client = LLMClient(model_name="deepseek-v4-flash")
        expander_api = QueryExpander(
            model_source="api",
            llm_client=llm_client
        )
        
        queries = expander_api.expand(
            query="Meeting on Oct 5th",
            missing_info="Location and participants of the meeting"
        )
        
        print(f"\nOriginal query: Meeting on Oct 5th")
        print("Expanded queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        expander_api.cleanup()
        
    except Exception as e:
        print(f"API mode test failed: {e}")
    
    print("\n### Example 3: fallback with empty missing evidence")
    try:
        expander = QueryExpander(model_source="local")
        
        queries = expander.expand(
            query="What is the capital of France?",
            missing_info=""
        )
        
        print(f"Expanded queries: {queries}")
        
        expander.cleanup()
        
    except Exception as e:
        print(f"Fallback test failed: {e}")
    
    print("\n" + "=" * 80)
