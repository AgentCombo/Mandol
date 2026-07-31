from .semantic_quantifier import SemanticQuantifier

class FastFinetunedQuantifier(SemanticQuantifier):
    def _build_binary_prompt(self, query: str, context: str) -> str:
        """Build binary prompt."""
        instruction = (
            "Evaluate if the provided Documents fully answer the Query. "
            "Output 'T' (True/Sufficient) or 'F' (False/Insufficient) immediately, "
            "followed by reasoning.\nFormat: [T/F]\\n### Reasoning:\\n..."
        )
        
        user_input_content = f"Query: {query}\nDocuments:\n{context}"
        
        return f"{instruction}\n\n{user_input_content}"

    # def _quantify_with_local_model(self, query: str, context: str):
    #     """
    #     """
    #     prompt_content = self._build_binary_prompt(query, context)
        
    #     messages = [{"role": "user", "content": prompt_content}]
        
    #     raw_output = self.local_client.generate(
    #         messages,
    # Dataset-specific handling used by the reproduction workflow.
    #         temperature=0.01
    #     )
    
    def _quantify_with_local_model(self, query: str, context: str):
        prompt_content = self._build_binary_prompt(query, context)
        
        raw_output = self.local_client.generate(
            prompt_content,
            max_tokens=1,
            do_sample=False,
            temperature=0.01
        )
        
        clean_response = raw_output.strip()
        
        is_sufficient = clean_response.startswith("T")
        
        return {
            "is_sufficient": is_sufficient,
            "confidence": 1.0 if is_sufficient else 0.5,
            "reasoning": "Model prediction (Latency Optimized)", 
            "raw_response": raw_output
        }


if __name__ == "__main__":
    quantifier = FastFinetunedQuantifier(
        model_source="local",
        model_name="/path/to/your/finetuned/model_directory" 
    )