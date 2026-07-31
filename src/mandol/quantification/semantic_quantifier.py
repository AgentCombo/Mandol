"""Sufficiency quantification for retrieved Mandol memory evidence.

SemanticQuantifier evaluates whether retrieved memory units contain enough
evidence to answer a query. It supports a fast local binary gate, an API-backed
full sufficiency analysis, and hybrid mode where the local gate decides whether
the slower API expert is needed.
"""

import logging
from ..utils.logging_config import create_module_logger
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.memory_unit import MemoryUnit
from ..llm.llm_client import LLMClient, create_llm_client
from ..llm.local_llm_client import LocalLLMClient
from .quantifier_prompts import QUANTIFICATION_PROMPT, FAST_BINARY_PROMPT

logger = create_module_logger("quantification.semantic_quantifier")

COMBINED_QUANTIFICATION_EXPANSION_PROMPT = """You are a Memory Relevance Expert and Precision Search Optimizer.
Your goal is to determine if the retrieved documents provide sufficient information to answer the user's query.
If they are INSUFFICIENT, you must identify what is missing AND generate ONE highly targeted search query to find that specific information.

**Input Context**:
- User Query: "{query}"
- Retrieved Documents:
{context}

**Instructions**:
1. **Analyze Sufficiency**: Do the documents contain the full answer?
   - If YES -> Status: SUFFICIENT.
   - If NO -> Status: MISSING.
2. **If MISSING**:
   - Explicitly describe the missing information (be specific about what entity, attribute, or fact is absent).
   - Generate ONE single, highly targeted search query that focuses solely on finding the missing piece.
     * Use specific entity names (no pronouns).
     * Make it standalone (understandable without context).
     * Target the gap, not the whole topic.
3. **If SUFFICIENT**:
   - Explain why the documents satisfy the query.
   - Set targeted_query to null.

**Output Format** (STRICT JSON):
{{
  "status": "SUFFICIENT" or "MISSING",
  "thought_process": "Brief analysis of the context vs query",
  "missing_info": "Description of what is missing (or empty string if sufficient)",
  "targeted_query": "A single targeted query to find the missing info (or null if sufficient)"
}}

**Examples**:

Example 1 (SUFFICIENT):
Query: "What is Alice's age?"
Documents: ["Alice is 25 years old."]
Output:
{{
  "status": "SUFFICIENT",
  "thought_process": "Document explicitly states Alice's age as 25.",
  "missing_info": "",
  "targeted_query": null
}}

Example 2 (MISSING):
Query: "When did Alice fly to Paris?"
Documents: ["Alice loves traveling to Europe.", "Paris is a beautiful city."]
Output:
{{
  "status": "MISSING",
  "thought_process": "Documents mention Alice and Paris separately but lack the specific date of her flight.",
  "missing_info": "Specific date of Alice's flight to Paris",
  "targeted_query": "What is the specific date of Alice's flight to Paris?"
}}
"""


class SemanticQuantifier:
    """Assess retrieved evidence and optionally generate targeted follow-up queries.

    Args:
        model_source: ``"local"``, ``"api"``, or ``"hybrid"``. Hybrid mode runs
            the local fast check first and calls the API expert only when needed.
        local_model_name: Local gatekeeper model used by local and hybrid modes.
        llm_client: Optional API client used by API and hybrid slow paths.
        device: Optional device for ``LocalLLMClient``.
        model_name: Backward-compatible alias for ``local_model_name``.

    Notes:
        The local model is lazily loaded. Quantification does not mutate memory
        units or retrieval indexes; it only serializes retrieved evidence into a
        prompt and returns sufficiency metadata.
    """
    
    def __init__(self,
                 model_source: str = "local",
                 local_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 llm_client: Optional[LLMClient] = None,
                 device: Optional[str] = None,
                 
                 model_name: Optional[str] = None):
        self.model_source = model_source
        
        
        self.local_model_name = model_name or local_model_name
        self.device = device
        
        
        self.local_client: Optional[LocalLLMClient] = None
        
        self.llm_client = llm_client
        
        is_api_model = (
            self.local_model_name in LLMClient.MODEL_CONFIGS or 
            any(self.local_model_name.startswith(prefix) for prefix in ["gpt-", "claude-", "deepseek-", "gemini-"])
        )
        
        if model_source == "local" and is_api_model:
            logger.info(
                "Detected API model name %r; switching SemanticQuantifier to API mode.",
                self.local_model_name,
            )
            self.model_source = "api"
        
        if self.model_source == "api" and self.llm_client is None:
            logger.info(
                "No API client was provided; creating one for model %s.",
                self.local_model_name,
            )
            try:
                self.llm_client = create_llm_client(model=self.local_model_name)
                logger.info("LLMClient created successfully.")
            except Exception as e:
                raise ValueError(
                    f"API mode requires llm_client, and automatic creation failed: {e}"
                )
        
        logger.info("SemanticQuantifier initialized in %s mode.", self.model_source)
        if self.model_source in ["local", "hybrid"]:
            logger.info("  local model: %s (lazy loading)", self.local_model_name)
        if self.model_source in ["api", "hybrid"] and self.llm_client:
            logger.info("  API model: %s", self.llm_client.model_name)
    
    def _ensure_local_client(self):
        """Load the local gatekeeper only when the first local check is needed."""
        if self.local_client is None:
            logger.info("Initializing LocalLLMClient: %s", self.local_model_name)
            self.local_client = LocalLLMClient(
                model_name=self.local_model_name,
                device=self.device
            )
            logger.info("LocalLLMClient initialized.")
    
    def _serialize_units(self, units: List[MemoryUnit]) -> str:
        """Serialize retrieved memory units into numbered evidence blocks."""
        if not units:
            return "[No documents provided]"
        
        serialized_parts = []
        for idx, unit in enumerate(units, 1):
            content = unit.raw_data.get("text_content") or \
                     unit.raw_data.get("content") or \
                     str(unit.raw_data)
            serialized_parts.append(f"[Document {idx}] {content}")
        
        return "\n".join(serialized_parts)
    
    def _fast_check_local(self, query: str, context: str) -> bool:
        """Run the local binary sufficiency gate.

        The fast path only asks for a true/false decision. Any local-model
        failure returns ``False`` so the caller can fall back to the slower API
        path without losing the retrieval result.
        """
        try:
            
            self._ensure_local_client()
            
            prompt = FAST_BINARY_PROMPT.format(query=query, context=context)
            
            messages = [{"role": "user", "content": prompt}]
            
            raw_output = self.local_client.generate(
                messages,
                max_tokens=2,
                temperature=0.01,
                do_sample=False
            )
            
            clean_output = raw_output.strip().upper()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Local gatekeeper raw output: %r", raw_output)
            
            is_sufficient = clean_output.startswith("T")
            
            logger.info(
                "Fast path result: %s",
                "sufficient" if is_sufficient else "insufficient",
            )
            
            return is_sufficient
            
        except Exception as e:
            logger.warning("Fast path failed; falling back to slow path: %s", e)
            return False
        
    def _quantify_with_hybrid_api(self, query: str, context: str) -> Dict[str, Any]:
        """Run full API sufficiency analysis and targeted-query generation."""
        prompt = COMBINED_QUANTIFICATION_EXPANSION_PROMPT.format(
            query=query, 
            context=context
        )
        
        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.0,
                json_format=True
            )
            
            result_json = self._parse_json_response(response)
            
            if result_json is None:
                return self._create_error_result("JSON Parse Error")
            
            status = result_json.get("status", "MISSING").upper()
            thought_process = result_json.get("thought_process", "")
            missing_info = result_json.get("missing_info", "")
            
            expanded_queries = self._extract_expanded_queries(result_json)
            
            is_sufficient = (status == "SUFFICIENT")
            
            reasoning = f"[{status}] {thought_process}"
            if not is_sufficient and missing_info:
                reasoning += f" Missing: {missing_info}"

            return {
                "is_sufficient": is_sufficient,
                "confidence": 1.0 if is_sufficient else 0.9,
                "reasoning": reasoning,
                "raw_response": response,
                "missing_info": missing_info,
                "expanded_queries": expanded_queries
            }
            
        except Exception as e:
            logger.error("Slow-path API call failed: %s", e)
            return self._create_error_result(f"API Error: {str(e)}")
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse strict JSON, accepting plain fenced responses as fallback."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            clean_response = response.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(clean_response)
            except json.JSONDecodeError:
                logger.error("Failed to parse quantifier JSON response: %s...", response[:100])
                return None
    
    def _extract_expanded_queries(self, result_json: Dict[str, Any]) -> List[str]:
        """Extract targeted and optional expanded queries from API JSON."""
        expanded_queries = []
        
        targeted_query = result_json.get("targeted_query")
        if targeted_query and isinstance(targeted_query, str) and targeted_query.strip():
            if targeted_query.lower() not in ["null", "none", ""]:
                expanded_queries.append(targeted_query.strip())
        
        
        extra_queries = result_json.get("expansion_queries") or result_json.get("queries")
        if extra_queries and isinstance(extra_queries, list):
            for q in extra_queries:
                if isinstance(q, str) and q.strip() and q not in expanded_queries:
                    expanded_queries.append(q.strip())
        
        return expanded_queries
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Build the conservative fallback result for quantifier failures."""
        return {
            "is_sufficient": False,
            "confidence": 0.0,
            "reasoning": error_msg,
            "missing_info": "System Error",
            "expanded_queries": []
        }

    def quantify(self, query: str, units: List[MemoryUnit]) -> Dict[str, Any]:
        """Evaluate whether retrieved units are sufficient for the query.

        Args:
            query: User query being answered.
            units: Retrieved memory units used as evidence.

        Returns:
            A dictionary containing ``is_sufficient``, ``confidence``,
            ``missing_info``, optional ``expanded_queries``, timing metadata,
            and the selected ``model_source``.

        Notes:
            In hybrid mode, a sufficient local gate result skips the API slow
            path. If no API client is available after an insufficient local
            result, this method returns a conservative insufficient result.
        """
        logger.info(
            "Starting semantic quantification (query=%r, units=%d).",
            query[:30],
            len(units),
        )
        start_time = datetime.now()
        
        context = self._serialize_units(units)
        
        result = {}
        
        
        if self.model_source in ["local", "hybrid"]:
            logger.info("Entering fast path (local gatekeeper).")
            
            is_sufficient = self._fast_check_local(query, context)
            
            if is_sufficient:
                result = {
                    "is_sufficient": True,
                    "confidence": 1.0,
                    "reasoning": "Validated by Local Gatekeeper (Fast Path)",
                    "missing_info": "",
                    "expanded_queries": []
                }
                logger.info("Fast path succeeded; skipping slow path.")
            else:
                if self.llm_client is not None:
                    logger.info("Fast path was insufficient; entering slow path (API expert).")
                    result = self._quantify_with_hybrid_api(query, context)
                else:
                    logger.warning("No API client is available; returning conservative insufficient result.")
                    result = {
                        "is_sufficient": False,
                        "confidence": 0.5,
                        "reasoning": "Local gatekeeper marked the context insufficient, but no API client is available.",
                        "missing_info": "More context is required.",
                        "expanded_queries": []
                    }
        
        elif self.model_source == "api":
            logger.info("API-only mode; entering slow path directly.")
            result = self._quantify_with_hybrid_api(query, context)
        
        else:
            raise ValueError(f"Unsupported model_source: {self.model_source}")
        
        
        duration = (datetime.now() - start_time).total_seconds()
        
        final_result = {
            "quantified_score": 1.0 if result["is_sufficient"] else 0.0,
            "is_sufficient": result["is_sufficient"],
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "missing_info": result.get("missing_info", ""),
            "expanded_queries": result.get("expanded_queries", []),
            "num_units": len(units),
            "timestamp": str(datetime.now()),
            "model_source": self.model_source,
            "duration": duration
        }
        
        logger.info(
            "Quantification complete: sufficient=%s | expanded_queries=%d | duration=%.2fs",
            result["is_sufficient"],
            len(final_result["expanded_queries"]),
            duration,
        )
        
        return final_result
    
    def cleanup(self):
        """Release associated resources."""
        if self.local_client is not None:
            logger.info("Releasing local quantifier client.")
            self.local_client.cleanup()
            self.local_client = None





def create_semantic_quantifier(
    model_source: str = "hybrid",
    local_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    llm_client: Optional[LLMClient] = None,
    
    model_name: Optional[str] = None
) -> SemanticQuantifier:
    """Create a SemanticQuantifier with Mandol's default hybrid policy."""
    return SemanticQuantifier(
        model_source=model_source,
        local_model_name=model_name or local_model_name,
        llm_client=llm_client
    )





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SemanticQuantifier - Cascade RAG Architecture")
    print("Initialize and test this component from an application context.")
