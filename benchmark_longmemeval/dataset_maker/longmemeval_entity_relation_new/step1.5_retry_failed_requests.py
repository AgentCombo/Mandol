#!/usr/bin/env python3
"""Utilities for step1.5 retry failed requests."""
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set

from step1_entity_batch_requests import LongMemEvalBatchEntityExtractor, LongMemEvalEntityType
from mandol.core import paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetryBatchGenerator:
    def __init__(self, dataset_path: str):
        self.extractor = LongMemEvalBatchEntityExtractor(dataset_path=dataset_path)
        self.qa_data = self.extractor.load_dataset()
        self.prompt_template = self.extractor._build_unified_extraction_prompt()
    
    def parse_failed_ids(self, error_files: List[str]) -> Set[str]:
        """Parse failed ids."""
        failed_ids = set()
        files_to_process = []

        # 1. Expand directories into file lists
        for file_path in error_files:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f" Path does not exist: {path}")
                continue
            
            if path.is_dir():
                logger.info(f" Detected directory: {path}, scanning for .jsonl files...")
                # Automatically find all jsonl files in the directory
                files_to_process.extend(path.glob("*.jsonl"))
            else:
                files_to_process.append(path)

        if not files_to_process:
            logger.warning(" No .jsonl files found to process.")
            return failed_ids

        # 2. Process each file
        for path in files_to_process:
            logger.info(f" Reading error log: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            # Check for error field
                            if data.get("error"):
                                failed_ids.add(data["custom_id"])
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f" Failed to read file {path}: {e}")
                
        return failed_ids
    
    # def parse_failed_ids(self, error_files: List[str]) -> Set[str]:
    #     failed_ids = set()
    #     for file_path in error_files:
    #         path = Path(file_path)
    #         if not path.exists():
    #             continue
            
    # Avoid mutating LogRecord fields before other handlers process the record.
    #         with open(path, 'r', encoding='utf-8') as f:
    #             for line in f:
    #                 if not line.strip(): continue
    #                 try:
    #                     data = json.loads(line)
    #                     if data.get("error"):
    #                         failed_ids.add(data["custom_id"])
    #                 except json.JSONDecodeError:
    #                     continue
    #     return failed_ids

    def generate_retry_file(self, failed_ids: Set[str], output_file: str, model: str,
                            enable_thinking: bool = False, thinking_budget: int = 1024):
        """Generate retry file."""
        if not failed_ids:
            logger.info(" 没有发现失败的 ID，无需重试。")
            return

        logger.info(f" 正在为 {len(failed_ids)} 个失败请求生成重试文件...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            success_count = 0
            
            for custom_id in failed_ids:
                try:
                    parts = custom_id.split('_')
                    if len(parts) < 4:
                        logger.warning(f"无法解析 ID 格式: {custom_id}")
                        continue
                        
                    qa_index = int(parts[1])
                    start_session_idx = int(parts[3])
                    end_session_idx_inclusive = int(parts[4])
                    end_session_idx = end_session_idx_inclusive + 1

                    qa_sample = self.qa_data[qa_index]
                    haystack_sessions = qa_sample.get("haystack_sessions", [])
                    haystack_session_ids = qa_sample.get("haystack_session_ids", [])
                    haystack_dates = qa_sample.get("haystack_dates", [])

                    sessions_text = self.extractor._build_sessions_text(
                        haystack_sessions,
                        haystack_session_ids,
                        haystack_dates,
                        start_session_idx,
                        end_session_idx
                    )

                    user_prompt = self.prompt_template.substitute(
                        sessions_text=sessions_text,
                        entity_types_description=LongMemEvalEntityType.get_priority_description()
                    )

                    messages = [
                        {"role": "system", "content": "You are a professional entity extraction expert. Extract entities in JSON format only."},
                        {"role": "user", "content": user_prompt}
                    ]

                    body = {
                        "model": model,
                        "messages": messages,
                        "enable_thinking": enable_thinking
                    }
                    if enable_thinking:
                        body["thinking_budget"] = thinking_budget

                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body
                    }
                    
                    f.write(json.dumps(request, ensure_ascii=True) + '\n')
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f" 处理 {custom_id} 时出错: {e}")

        logger.info(f" 重试文件已生成: {output_file}")
        logger.info(f" 成功生成: {success_count}/{len(failed_ids)}")

def main():
    parser = argparse.ArgumentParser(description="生成重试 Batch 请求（阿里云百炼格式）")
    parser.add_argument("--error-files", nargs='+', required=True, help="包含错误的 jsonl 结果文件路径")
    parser.add_argument("--output-file", default="repair_batch_requests.jsonl", help="输出的重试请求文件名")
    # parser.add_argument("--model", default="qwen-plus-latest", 
    parser.add_argument("--model", 
                       default="qwen-plus-latest", 
                       choices=[
                           "qwen-plus-latest",
                           "qwen-max-latest",
                           "qwen-turbo-latest",
                           "qwen-long",
                           "qwen3.5-plus"
                       ],
                       help="模型名称 (默认: qwen3.5-plus)")
    
    parser.add_argument("--enable-thinking", action="store_true",
                       help="启用模型的推理/思考模式 (默认禁用，以防 Qwen 3.5 默认开启导致 token 浪费)")
    parser.add_argument("--thinking-budget", type=int, default=2048,
                       help="思考过程的 Token 预算（仅在启用 enable-thinking 时生效）")
    
    args = parser.parse_args()
    
    generator = RetryBatchGenerator(dataset_path=str(paths.LONGMEMEVAL_S_CLEANED_FILE))
    
    failed_ids = generator.parse_failed_ids(args.error_files)
    generator.generate_retry_file(failed_ids, args.output_file, args.model,
                                  enable_thinking=args.enable_thinking,
                                  thinking_budget=args.thinking_budget)

if __name__ == "__main__":
    main()