#!/usr/bin/env python3
"""
Step 2a: L1 Session-Level Structured Extraction (ETL)
- Read L0 JSON files from step1
- Extract structured data points (Events, States, Counts) for each session
- Output structured JSON for downstream aggregation
- Focus on fact extraction, NOT narrative summaries

Key Features:
- Absolute time resolution (relative dates -> YYYY-MM-DD)
- State change tracking (Old -> New)
- Event classification (Occurrence vs Reference)
- Countable action extraction

Concurrency Model (V2: Intra-Sample Session-Level Parallelism):
- Outer (Sample): Sequential processing for easier debugging
- Inner (Session): Parallel LLM calls within each sample
- max_workers controls concurrent session processing per sample
"""

import json
import logging
import os
import sys
import re
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


from mandol.llm.llm_client import LLMClient
from mandol.core import paths



#  L1 Structured Extraction Prompt Template
# Purpose: Extract factual structured data, NOT narrative summaries

L1_EXTRACTION_PROMPT_TEMPLATE = """
You are a Data Extraction Specialist. Analyze this conversation session.
Your goal is to EXTRACT structured data points (Events, States, Counts) for a Knowledge Graph.

<input_context>
Session ID: {session_id}
Session Date: {session_date} (Reference this for absolute date calculation)
Participants: {participants}
Transcript:
{transcript}
</input_context>

<extraction_rules>
1. **Absolute Time Resolution**: Convert relative times (e.g., "next Friday", "last week", "yesterday") to YYYY-MM-DD format using Session Date as reference.
2. **State Change Tracking**: Identify status changes (Job, Location, Relationship, Health, Mood). Format: Old -> New.
3. **Event Classification**:
   - "Occurrence": Happening NOW or Planned for future.
   - "Reference": Discussing past events.
4. **Countable Actions**: Extract items that can be counted (games played, trips taken, meals, recommendations made).
5. **Quote Extraction**: Include exact quotes that support state changes or key facts.
</extraction_rules>

<output_format>
Return ONLY valid JSON (no markdown, no explanation):
{{
    "session_id": "{session_id}",
    "session_date": "{session_date}",
    "session_topic": "Brief 5-10 word topic description",
    "structured_events": [
        {{
            "event_name": "Descriptive name of the event",
            "event_type": "Activity|Crisis|Milestone|Plan|Social|Health|Work",
            "date": "YYYY-MM-DD or 'unknown'",
            "date_source": "explicit|calculated|unknown",
            "is_new_occurrence": true,
            "participants": ["Name1", "Name2"],
            "location": "Location if mentioned or null",
            "supporting_quote": "Exact quote from transcript"
        }}
    ],
    "state_updates": [
        {{
            "entity": "Person name",
            "attribute": "Job|Location|Relationship|Health|Mood|Hobby|Goal",
            "old_value": "Previous state or 'unknown'",
            "new_value": "Current/new state",
            "change_date": "YYYY-MM-DD or 'during_session'",
            "trigger_quote": "Exact quote that reveals this change"
        }}
    ],
    "countable_items": [
        {{
            "category": "Game|Place|Food|Activity|Recommendation|Purchase",
            "item_name": "Specific name of the item",
            "action": "Played|Visited|Ate|Did|Recommended|Bought",
            "count": 1,
            "by_whom": "Person who performed action"
        }}
    ],
    "mentioned_dates": [
        {{
            "original_text": "The relative/absolute date as mentioned",
            "resolved_date": "YYYY-MM-DD",
            "context": "What this date refers to"
        }}
    ],
    "key_facts": [
        {{
            "fact_type": "Identity|Preference|History|Plan|Opinion",
            "subject": "Who this fact is about",
            "fact": "The factual statement",
            "supporting_quote": "Exact quote"
        }}
    ]
}}
</output_format>

IMPORTANT: 
- Return ONLY the JSON object, no additional text.
- If no items exist for a category, use empty array [].
- All text content must be in English.
"""


@dataclass
class L1ExtractionConfig:
    """L1 Extraction Configuration"""
    # Input/Output paths
    l0_graphs_dir: str = str(paths.LOCOMO_HIERARCHICAL_STEP1_DIR)
    output_dir: str = str(paths.LOCOMO_HIERARCHICAL_STEP2_DIR)
    
    # Alternative: Read directly from locomo10.json
    locomo_source_file: str = str(paths.LOCOMO_RAW_FILE)
    use_l0_graphs: bool = True  # If False, read from locomo_source_file directly
    
    #  Parallel processing V2: Intra-Sample Session-Level Parallelism
    # Samples are processed sequentially; sessions within each sample are parallel
    enable_session_parallel: bool = True  # Enable parallel session processing within each sample
    max_workers: int = 8  # Max concurrent session LLM calls per sample
    
    # LLM configuration
    llm_model: str = "qwen-3.5-plus-thinking"  # Updated model for better extraction performance
    llm_temperature: float = 0.1  # Low temperature for factual extraction
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    sample_ids: Optional[List[str]] = None  # Dataset-specific handling used by the reproduction workflow.
    
    # Debug
    debug_mode: bool = False


class L1Extractor:
    """L1 Session-Level Structured Data Extractor"""
    
    def __init__(self, config: L1ExtractionConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Statistics (thread-safe via lock when parallel)
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'total_sessions': 0,
            'total_events_extracted': 0,
            'total_state_updates': 0,
            'total_countable_items': 0,
            'failed_samples': [],
            'failed_sessions': [],
            'processing_time': 0
        }
        self._stats_lock = threading.Lock()  # Lock for thread-safe stats update
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model_name=self.config.llm_model,
            max_context_ratio=0.8
        )
        
        self.logger.info("=" * 60)
        self.logger.info(" L1 Structured Extractor Initialized")
        self.logger.info(f"   LLM Model: {self.config.llm_model}")
        self.logger.info(f"   Output Dir: {self.config.output_dir}")
        self.logger.info(f"   Concurrency: Sequential Sample, {'Parallel' if self.config.enable_session_parallel else 'Sequential'} Sessions")
        if self.config.enable_session_parallel:
            self.logger.info(f"   Max Workers per Sample: {self.config.max_workers}")
        self.logger.info("=" * 60)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f"{__name__}.L1Extractor")
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_extraction(self) -> Dict[str, Any]:
        """
        Main extraction pipeline
        
        V2 Concurrency Model:
        - Outer loop: Sequential sample processing
        - Inner loop: Parallel session processing within each sample
        """
        start_time = time.time()
        self.logger.info("\n Starting L1 Structured Extraction Pipeline")
        self.logger.info("   Mode: Sequential Samples, Parallel Sessions per Sample")
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Load data source
        if self.config.use_l0_graphs:
            samples = self._load_from_l0_graphs()
        else:
            samples = self._load_from_locomo_source()
        
        if not samples:
            raise ValueError("No samples found to process")
        
        # Dataset-specific handling used by the reproduction workflow.
        if self.config.sample_ids:
            id_set = set(self.config.sample_ids)
            samples = [s for s in samples if self._get_sample_id(s) in id_set]
            self.logger.info(f" 筛选指定样本: {self.config.sample_ids} → {len(samples)} 个匹配")
        
        self.stats['total_samples'] = len(samples)
        self.logger.info(f" Found {len(samples)} samples to process")
        
        #  V2: Always process samples sequentially
        # Parallel processing happens at session level within each sample
        self._process_samples_sequential(samples)
        
        self.stats['processing_time'] = time.time() - start_time
        self._save_extraction_stats()
        self._print_summary()
        
        return self.stats
    
    def _load_from_l0_graphs(self) -> List[Dict[str, Any]]:
        """Load data from L0 graph JSON files"""
        l0_dir = Path(self.config.l0_graphs_dir)
        if not l0_dir.exists():
            self.logger.error(f"L0 graphs directory not found: {l0_dir}")
            return []
        
        samples = []
        for l0_file in sorted(l0_dir.glob("*_l0_graph.json")):
            try:
                with open(l0_file, 'r', encoding='utf-8') as f:
                    l0_data = json.load(f)
                samples.append({
                    'source': 'l0_graph',
                    'file_path': str(l0_file),
                    'data': l0_data
                })
            except Exception as e:
                self.logger.error(f"Failed to load {l0_file}: {e}")
        
        return samples
    
    def _load_from_locomo_source(self) -> List[Dict[str, Any]]:
        """Load data directly from locomo10.json"""
        source_path = paths.PROJECT_ROOT / self.config.locomo_source_file
        if not source_path.exists():
            self.logger.error(f"Locomo source file not found: {source_path}")
            return []
        
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                locomo_data = json.load(f)
            
            samples = []
            for sample in locomo_data:
                samples.append({
                    'source': 'locomo_direct',
                    'data': sample
                })
            return samples
        except Exception as e:
            self.logger.error(f"Failed to load locomo source: {e}")
            return []
    
    def _process_samples_sequential(self, samples: List[Dict[str, Any]]):
        """
        Sequential sample processing with parallel sessions within each sample
        
        V2 Model: For each sample, sessions are processed in parallel
        """
        for i, sample in enumerate(samples, 1):
            sample_id = self._get_sample_id(sample)
            self.logger.info(f"\n[{i}/{len(samples)}] Processing Sample: {sample_id}")
            sample_start_time = time.time()
            
            try:
                result = self._extract_sample(sample)
                if result:
                    self._save_sample_result(sample_id, result)
                    self.stats['processed_samples'] += 1
                    
                    sample_time = time.time() - sample_start_time
                    session_count = result.get('total_sessions', 0)
                    self.logger.info(f" [{i}/{len(samples)}] {sample_id}: {session_count} sessions in {sample_time:.2f}s")
            except Exception as e:
                self.logger.error(f" [{i}/{len(samples)}] Failed to process {sample_id}: {e}")
                traceback.print_exc()
                self.stats['failed_samples'].append({
                    'sample_id': sample_id,
                    'error': str(e)
                })
    
    def _get_sample_id(self, sample: Dict[str, Any]) -> str:
        """Extract sample ID from sample data"""
        if sample['source'] == 'l0_graph':
            return sample['data'].get('sample_id', 'unknown')
        else:
            return sample['data'].get('sample_id', 'unknown')
    
    def _extract_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structured data from a single sample"""
        sample_id = self._get_sample_id(sample)
        
        if sample['source'] == 'l0_graph':
            return self._extract_from_l0_graph(sample['data'], sample_id)
        else:
            return self._extract_from_locomo_direct(sample['data'], sample_id)
    
    def _extract_from_l0_graph(self, l0_data: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """
        Extract from L0 graph format
        
        V2: Parallel session processing within the sample
        """
        conversations = l0_data.get('l0_conversations', [])
        speakers = l0_data.get('metadata', {}).get('speakers', ['Speaker_A', 'Speaker_B'])
        
        # Group conversations by session
        sessions = {}
        for conv in conversations:
            session_id = conv.get('raw_data', {}).get('session_id', 'unknown')
            if session_id not in sessions:
                sessions[session_id] = {
                    'session_id': session_id,
                    'session_datetime': conv.get('raw_data', {}).get('session_datetime', ''),
                    'date': conv.get('raw_data', {}).get('date', 'unknown'),
                    'messages': []
                }
            sessions[session_id]['messages'].append(conv)
        
        # Sort messages within each session by message_index
        for session_id in sessions:
            sessions[session_id]['messages'].sort(
                key=lambda x: x.get('metadata', {}).get('message_index', 0)
            )
        
        # Prepare session tasks (sorted by session_id for consistent ordering)
        session_tasks = []
        for session_id, session_data in sorted(sessions.items(), 
                                                key=lambda x: self._extract_session_number(x[0])):
            session_tasks.append({
                'session_id': session_id,
                'session_date': session_data['date'],
                'participants': speakers,
                'messages': session_data['messages'],
                'sample_id': sample_id,
                'raw_format': False  # L0 graph format
            })
        
        self.logger.info(f"    {sample_id}: {len(session_tasks)} sessions to extract")
        
        #  V2: Parallel or Sequential session processing
        if self.config.enable_session_parallel and len(session_tasks) > 1:
            session_extractions = self._process_sessions_parallel(session_tasks)
        else:
            session_extractions = self._process_sessions_sequential(session_tasks)
        
        # Sort results by session number (maintain order after parallel processing)
        session_extractions.sort(key=lambda x: self._extract_session_number(x.get('session_id', '')))
        
        return {
            'sample_id': sample_id,
            'extraction_time': datetime.now().isoformat(),
            'source': 'l0_graph',
            'total_sessions': len(session_extractions),
            'participants': speakers,
            'session_extractions': session_extractions
        }
    
    def _extract_from_locomo_direct(self, sample_data: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """
        Extract directly from locomo10.json format
        
        V2: Parallel session processing within the sample
        """
        conversation = sample_data.get('conversation', {})
        speaker_a = conversation.get('speaker_a', 'Speaker_A')
        speaker_b = conversation.get('speaker_b', 'Speaker_B')
        participants = [speaker_a, speaker_b]
        
        # Prepare session tasks
        session_tasks = []
        for key, messages in conversation.items():
            if not key.startswith('session_') or not isinstance(messages, list):
                continue
            
            # Get session datetime
            datetime_key = f"{key}_date_time"
            session_datetime = conversation.get(datetime_key, '')
            session_date = self._parse_date_from_datetime(session_datetime)
            
            session_tasks.append({
                'session_id': key,
                'session_date': session_date,
                'participants': participants,
                'messages': messages,
                'sample_id': sample_id,
                'raw_format': True  # Raw locomo format
            })
        
        # Sort tasks by session number for consistent processing order
        session_tasks.sort(key=lambda x: self._extract_session_number(x['session_id']))
        
        self.logger.info(f"    {sample_id}: {len(session_tasks)} sessions to extract")
        
        #  V2: Parallel or Sequential session processing
        if self.config.enable_session_parallel and len(session_tasks) > 1:
            session_extractions = self._process_sessions_parallel(session_tasks)
        else:
            session_extractions = self._process_sessions_sequential(session_tasks)
        
        # Sort results by session number (maintain order after parallel processing)
        session_extractions.sort(key=lambda x: self._extract_session_number(x.get('session_id', '')))
        
        return {
            'sample_id': sample_id,
            'extraction_time': datetime.now().isoformat(),
            'source': 'locomo_direct',
            'total_sessions': len(session_extractions),
            'participants': participants,
            'session_extractions': session_extractions
        }
    
    
    #  Session-Level Parallel Processing Methods (V2)
    
    
    def _process_sessions_parallel(self, session_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process sessions in parallel within a single sample
        
        Thread-safe: Stats are collected and updated after all futures complete
        """
        self.logger.info(f"    Parallel session extraction: {self.config.max_workers} workers")
        
        session_extractions = []
        local_stats = {
            'total_sessions': 0,
            'total_events_extracted': 0,
            'total_state_updates': 0,
            'total_countable_items': 0,
            'failed_sessions': []
        }
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all session extraction tasks
            future_to_task = {
                executor.submit(
                    self._extract_session,
                    task['session_id'],
                    task['session_date'],
                    task['participants'],
                    task['messages'],
                    task['sample_id'],
                    task['raw_format']
                ): task
                for task in session_tasks
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                session_id = task['session_id']
                completed += 1
                
                try:
                    extraction = future.result()
                    if extraction:
                        session_extractions.append(extraction)
                        # Collect stats locally (no lock needed here)
                        local_stats['total_sessions'] += 1
                        local_stats['total_events_extracted'] += len(extraction.get('structured_events', []))
                        local_stats['total_state_updates'] += len(extraction.get('state_updates', []))
                        local_stats['total_countable_items'] += len(extraction.get('countable_items', []))
                        self.logger.debug(f"       [{completed}/{len(session_tasks)}] {session_id}")
                except Exception as e:
                    self.logger.warning(f"       [{completed}/{len(session_tasks)}] {session_id}: {e}")
                    local_stats['failed_sessions'].append(session_id)
        
        #  Thread-safe: Update global stats after parallel processing completes
        with self._stats_lock:
            self.stats['total_sessions'] += local_stats['total_sessions']
            self.stats['total_events_extracted'] += local_stats['total_events_extracted']
            self.stats['total_state_updates'] += local_stats['total_state_updates']
            self.stats['total_countable_items'] += local_stats['total_countable_items']
            self.stats['failed_sessions'].extend(local_stats['failed_sessions'])
        
        return session_extractions
    
    def _process_sessions_sequential(self, session_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process sessions sequentially within a single sample"""
        self.logger.info(f"    Sequential session extraction")
        
        session_extractions = []
        
        for i, task in enumerate(session_tasks, 1):
            session_id = task['session_id']
            
            try:
                extraction = self._extract_session(
                    session_id=task['session_id'],
                    session_date=task['session_date'],
                    participants=task['participants'],
                    messages=task['messages'],
                    sample_id=task['sample_id'],
                    raw_format=task['raw_format']
                )
                
                if extraction:
                    session_extractions.append(extraction)
                    # Update stats directly (no lock needed for sequential)
                    self.stats['total_sessions'] += 1
                    self.stats['total_events_extracted'] += len(extraction.get('structured_events', []))
                    self.stats['total_state_updates'] += len(extraction.get('state_updates', []))
                    self.stats['total_countable_items'] += len(extraction.get('countable_items', []))
                    self.logger.debug(f"       [{i}/{len(session_tasks)}] {session_id}")
                    
            except Exception as e:
                self.logger.warning(f"       [{i}/{len(session_tasks)}] {session_id}: {e}")
                self.stats['failed_sessions'].append(session_id)
        
        return session_extractions
    
    def _extract_session(self,
                         session_id: str,
                         session_date: str,
                         participants: List[str],
                         messages: List[Any],
                         sample_id: str,
                         raw_format: bool = False) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from a single session
        
        Note: Stats are NOT updated here - they are collected and updated
        by the calling method (_process_sessions_parallel or _process_sessions_sequential)
        to ensure thread safety.
        """
        
        # Build transcript
        transcript_lines = []
        for i, msg in enumerate(messages):
            if raw_format:
                # Raw locomo format
                if isinstance(msg, dict):
                    speaker = msg.get('speaker', participants[i % 2])
                    text = msg.get('text', str(msg))
                else:
                    speaker = participants[i % 2]
                    text = str(msg)
            else:
                # L0 graph format
                raw_data = msg.get('raw_data', {})
                speaker = raw_data.get('speaker', 'Unknown')
                text = raw_data.get('message', raw_data.get('text_content', ''))
            
            dialogue_id = f"D{self._extract_session_number(session_id)}:{i+1}"
            transcript_lines.append(f"{dialogue_id} [{speaker}]: {text}")
        
        transcript = "\n".join(transcript_lines)
        
        if len(transcript.strip()) < 50:
            self.logger.debug(f"Skipping empty session: {session_id}")
            return None
        
        # Build extraction prompt
        prompt = L1_EXTRACTION_PROMPT_TEMPLATE.format(
            session_id=session_id,
            session_date=session_date,
            participants=", ".join(participants),
            transcript=transcript
        )
        
        # Call LLM with retry
        extraction = self._call_llm_with_retry(prompt, session_id)
        
        # Note: Stats update is handled by the caller for thread safety
        
        return extraction
    
    def _call_llm_with_retry(self, prompt: str, context_id: str) -> Optional[Dict[str, Any]]:
        """Call LLM with retry logic and JSON parsing"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=4000
                )
                
                # Parse JSON response
                parsed = self._parse_json_response(response)
                if parsed:
                    return parsed
                
                self.logger.warning(f"Invalid JSON response for {context_id}, attempt {attempt + 1}")
                
            except Exception as e:
                self.logger.warning(f"LLM call failed for {context_id}, attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        self.logger.error(f"All retries failed for {context_id}")
        self.stats['failed_sessions'].append(context_id)
        return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response with cleanup"""
        if not response:
            return None
        
        # Clean up response
        text = response.strip()
        
        # Remove markdown code blocks
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        text = text.strip()
        
        # Try to find JSON object
        try:
            # Direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _parse_date_from_datetime(self, datetime_str: str) -> str:
        """Parse date from locomo datetime format"""
        if not datetime_str:
            return "unknown"
        
        # Pattern: "12:19 am on 4 January, 2024"
        match = re.search(r'(\d{1,2})\s+(\w+),?\s*(\d{4})', datetime_str)
        if match:
            day, month_name, year = match.groups()
            month_map = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            month = month_map.get(month_name.lower(), '01')
            return f"{year}-{month}-{day.zfill(2)}"
        
        return "unknown"
    
    def _extract_session_number(self, session_id: str) -> int:
        """Extract numeric session number"""
        match = re.search(r'session_(\d+)', session_id)
        if match:
            return int(match.group(1))
        return 0
    
    def _save_sample_result(self, sample_id: str, result: Dict[str, Any]):
        """Save extraction result to JSON file"""
        output_file = Path(self.config.output_dir) / f"{sample_id}_l1_extracted.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved: {output_file}")
    
    def _save_extraction_stats(self):
        """Save extraction statistics"""
        stats_file = Path(self.config.output_dir) / "l1_extraction_stats.json"
        
        stats_output = {
            **self.stats,
            'config': {
                'llm_model': self.config.llm_model,
                'enable_session_parallel': self.config.enable_session_parallel,
                'max_workers_per_sample': self.config.max_workers,
                'concurrency_model': 'Sequential Samples, Parallel Sessions'
            },
            'completion_time': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f" Stats saved: {stats_file}")
    
    def _print_summary(self):
        """Print extraction summary"""
        print("\n" + "=" * 60)
        print(" L1 Extraction Summary")
        print("=" * 60)
        print(f"Total Samples:        {self.stats['total_samples']}")
        print(f"Processed:            {self.stats['processed_samples']}")
        print(f"Failed:               {len(self.stats['failed_samples'])}")
        print(f"Total Sessions:       {self.stats['total_sessions']}")
        print(f"Events Extracted:     {self.stats['total_events_extracted']}")
        print(f"State Updates:        {self.stats['total_state_updates']}")
        print(f"Countable Items:      {self.stats['total_countable_items']}")
        print(f"Processing Time:      {self.stats['processing_time']:.2f}s")
        print(f"Output Directory:     {self.config.output_dir}")
        print("=" * 60 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Step 2a: L1 Session-Level Structured Extraction (Intra-Sample Session Parallelism)"
    )
    parser.add_argument(
        "--l0-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP1_DIR),
        help="L0 graphs directory"
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP2_DIR),
        help="Output directory"
    )
    parser.add_argument(
        "--locomo-source",
        default=str(paths.LOCOMO_RAW_FILE),
        help="Direct locomo source file"
    )
    parser.add_argument(
        "--use-locomo-direct",
        action="store_true",
        help="Read directly from locomo10.json instead of L0 graphs"
    )
    parser.add_argument(
        "--extract-model",
        default="qwen-3.5-plus-thinking",
        help="抽取模型名称"
    )
    parser.add_argument(
        "--no-session-parallel",
        action="store_true",
        help="Disable parallel session processing within each sample"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=40,
        help="Max concurrent session workers per sample"
    )
    parser.add_argument(
        "--sample-ids",
        nargs='+',
        help="指定要处理的sample ID列表，如 --sample-ids conv-26 conv-30"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    config = L1ExtractionConfig(
        l0_graphs_dir=args.l0_dir,
        output_dir=args.output_dir,
        locomo_source_file=args.locomo_source,
        use_l0_graphs=not args.use_locomo_direct,
        llm_model=args.extract_model,
        enable_session_parallel=not args.no_session_parallel,
        max_workers=args.max_workers,
        sample_ids=args.sample_ids,
        debug_mode=args.debug
    )
    
    extractor = L1Extractor(config)
    
    try:
        stats = extractor.run_extraction()
        print(" L1 Extraction Complete!")
        return 0
    except Exception as e:
        print(f" L1 Extraction Failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
