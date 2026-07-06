#!/usr/bin/env python3
"""
Step 2b: L2 Sample-Level Global Aggregation
- Read L1 extracted JSON files from step2_l1_extracted/
- Aggregate across all sessions to produce global statistics
- Output comprehensive sample-level summary for retrieval
- Focus on answering cross-session analytical questions

Key Features:
- Global statistics computation (totals, counts)
- Character status snapshots (final states)
- Master timeline construction
- Cross-session pattern detection
"""

import json
import logging
import os
import sys
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


from mandol.llm.llm_client import LLMClient
from mandol.core import paths



#  L2 Global Aggregation Prompt Template
# Purpose: Aggregate L1 extraction results into global summary

L2_AGGREGATION_PROMPT_TEMPLATE = """
You are a Data Aggregation Specialist. Given the session-level facts extracted from multiple sessions, create a GLOBAL summary.

<input_context>
Sample ID: {sample_id}
Total Sessions: {total_sessions}
Participants: {participants}
Time Range: {time_range}

Session Extractions (L1 Data):
{session_data}
</input_context>

<aggregation_rules>
1. **Global Statistics**: Compute totals across ALL sessions (total games played, total trips taken, etc.)
2. **Character Status Snapshot**: For each person, provide their LATEST status as of the final session.
3. **Master Timeline**: Combine all events into a single chronological timeline.
4. **Cross-Session Patterns**: Identify recurring topics, relationships, activities.
5. **Count Deduplication**: Do NOT double-count the same event mentioned in multiple sessions.
6. **Temporal Logic**: For any recurring events (e.g., Doctor Visits, Meetings), you MUST CALCULATE the exact time gap between them (in days or months) and include it in the output.
</aggregation_rules>

<output_format>
Return ONLY valid JSON (no markdown, no explanation):
{{
    "sample_id": "{sample_id}",
    "aggregation_time": "{aggregation_time}",
    "time_range": {{
        "first_session": "{first_session_date}",
        "last_session": "{last_session_date}",
        "total_sessions": {total_sessions}
    }},
    "global_statistics": {{
        "total_conversations": {total_sessions},
        "total_unique_events": 0,
        "total_state_changes": 0,
        "activity_counts": {{
            "games_played": {{
                "total": 0,
                "items": ["game1", "game2"]
            }},
            "places_visited": {{
                "total": 0,
                "items": ["place1", "place2"]
            }},
            "foods_mentioned": {{
                "total": 0,
                "items": ["food1"]
            }},
            "recommendations_made": {{
                "total": 0,
                "items": ["rec1"]
            }}
        }},
        "topic_frequency": [
            {{"topic": "Gaming", "count": 5}},
            {{"topic": "Work", "count": 3}}
        ]
    }},
    "character_status_snapshot": [
        {{
            "person": "Person Name",
            "status_at_end": {{
                "job": "Current job or 'unknown'",
                "location": "Current city/location or 'unknown'",
                "relationship_status": "Status or 'unknown'",
                "current_mood": "Mood or 'unknown'",
                "active_hobbies": ["hobby1", "hobby2"],
                "ongoing_goals": ["goal1"],
                "health_status": "Status or 'unknown'"
            }},
            "key_changes_during_timeline": [
                {{
                    "attribute": "job",
                    "from": "Engineer",
                    "to": "Manager",
                    "change_date": "2024-02-15"
                }}
            ]
        }}
    ],
    "master_timeline": [
        {{
            "date": "YYYY-MM-DD",
            "events": [
                {{
                    "event": "Description of event",
                    "participants": ["Person1"],
                    "source_session": "session_1"
                }}
            ]
        }}
    ],
    "relationship_graph": {{
        "edges": [
            {{
                "from": "Person A",
                "to": "Person B",
                "relationship": "friends|colleagues|family",
                "interaction_count": 10
            }}
        ]
    }},
    "recurring_topics": [
        {{
            "topic": "Topic name",
            "occurrences": 5,
            "sessions": ["session_1", "session_3", "session_5"]
        }}
    ],
    "cross_session_insights": [
        "Insight 1: Summary of a pattern observed across sessions",
        "Insight 2: Another cross-session observation"
    ],
    "temporal_analysis": [
        {{
            "event_pair": "Doctor Visit 1 -> Doctor Visit 2",
            "date_1": "2023-08-15",
            "date_2": "2023-11-15",
            "calculated_gap": "3 months" 
        }}
    ]
}}
</output_format>

IMPORTANT: 
- Return ONLY the JSON object, no additional text.
- Compute actual totals from the input data.
- Deduplicate events that appear in multiple sessions.
- All text content must be in English.
"""


@dataclass
class L2AggregationConfig:
    """L2 Aggregation Configuration"""
    # Input/Output paths
    l1_extracted_dir: str = str(paths.LOCOMO_HIERARCHICAL_STEP2_DIR)
    output_dir: str = str(paths.LOCOMO_HIERARCHICAL_STEP2_L2_DIR)
    
    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = 4  # Lower than L1 since each aggregation is heavier
    
    # LLM configuration
    llm_model: str = "qwen-3.5-plus-thinking"  # Updated model for better aggregation performance
    llm_temperature: float = 0.1
    
    # Chunking for large samples
    max_sessions_per_chunk: int = 10  # If sample has many sessions, process in chunks
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    sample_ids: Optional[List[str]] = None  # Dataset-specific handling used by the reproduction workflow.
    
    # Debug
    debug_mode: bool = False


class L2Aggregator:
    """L2 Sample-Level Global Aggregator"""
    
    def __init__(self, config: L2AggregationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'total_sessions_aggregated': 0,
            'total_events_in_timeline': 0,
            'total_activity_counts': 0,
            'failed_samples': [],
            'processing_time': 0
        }
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model_name=self.config.llm_model,
            max_context_ratio=0.8
        )
        
        self.logger.info("=" * 60)
        self.logger.info(" L2 Global Aggregator Initialized")
        self.logger.info(f"   LLM Model: {self.config.llm_model}")
        self.logger.info(f"   Input Dir: {self.config.l1_extracted_dir}")
        self.logger.info(f"   Output Dir: {self.config.output_dir}")
        self.logger.info("=" * 60)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f"{__name__}.L2Aggregator")
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_aggregation(self) -> Dict[str, Any]:
        """Main aggregation pipeline"""
        start_time = time.time()
        self.logger.info("\n Starting L2 Global Aggregation Pipeline")
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Load L1 extracted files
        l1_files = self._load_l1_files()
        
        if not l1_files:
            raise ValueError("No L1 extracted files found to process")
        
        # Dataset-specific handling used by the reproduction workflow.
        if self.config.sample_ids:
            id_set = set(self.config.sample_ids)
            l1_files = [f for f in l1_files if f['data'].get('sample_id', 'unknown') in id_set]
            self.logger.info(f" 筛选指定样本: {self.config.sample_ids} → {len(l1_files)} 个匹配")
        
        self.stats['total_samples'] = len(l1_files)
        self.logger.info(f" Found {len(l1_files)} L1 files to aggregate")
        
        # Process samples
        if self.config.enable_parallel:
            self._process_parallel(l1_files)
        else:
            self._process_sequential(l1_files)
        
        self.stats['processing_time'] = time.time() - start_time
        self._save_aggregation_stats()
        self._print_summary()
        
        return self.stats
    
    def _load_l1_files(self) -> List[Dict[str, Any]]:
        """Load L1 extracted JSON files"""
        l1_dir = Path(self.config.l1_extracted_dir)
        if not l1_dir.exists():
            self.logger.error(f"L1 extracted directory not found: {l1_dir}")
            return []
        
        l1_files = []
        for l1_file in sorted(l1_dir.glob("*_l1_extracted.json")):
            # Skip stats file
            if 'stats' in l1_file.name:
                continue
            
            try:
                with open(l1_file, 'r', encoding='utf-8') as f:
                    l1_data = json.load(f)
                l1_files.append({
                    'file_path': str(l1_file),
                    'data': l1_data
                })
            except Exception as e:
                self.logger.error(f"Failed to load {l1_file}: {e}")
        
        return l1_files
    
    def _process_sequential(self, l1_files: List[Dict[str, Any]]):
        """Sequential processing"""
        for i, l1_file in enumerate(l1_files, 1):
            sample_id = l1_file['data'].get('sample_id', 'unknown')
            self.logger.info(f"\n[{i}/{len(l1_files)}] Aggregating: {sample_id}")
            
            try:
                result = self._aggregate_sample(l1_file['data'])
                if result:
                    self._save_sample_result(sample_id, result)
                    self.stats['processed_samples'] += 1
            except Exception as e:
                self.logger.error(f"Failed to aggregate {sample_id}: {e}")
                self.stats['failed_samples'].append({
                    'sample_id': sample_id,
                    'error': str(e)
                })
    
    def _process_parallel(self, l1_files: List[Dict[str, Any]]):
        """Parallel processing"""
        self.logger.info(f" Parallel mode: {self.config.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._aggregate_sample, l1_file['data']): l1_file
                for l1_file in l1_files
            }
            
            completed = 0
            for future in as_completed(futures):
                l1_file = futures[future]
                sample_id = l1_file['data'].get('sample_id', 'unknown')
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self._save_sample_result(sample_id, result)
                        self.stats['processed_samples'] += 1
                        self.logger.info(f" [{completed}/{len(l1_files)}] {sample_id}")
                except Exception as e:
                    self.logger.error(f" [{completed}/{len(l1_files)}] {sample_id}: {e}")
                    self.stats['failed_samples'].append({
                        'sample_id': sample_id,
                        'error': str(e)
                    })
    
    def _aggregate_sample(self, l1_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Aggregate L1 data for a single sample"""
        sample_id = l1_data.get('sample_id', 'unknown')
        session_extractions = l1_data.get('session_extractions', [])
        participants = l1_data.get('participants', [])
        
        if not session_extractions:
            self.logger.warning(f"No session extractions for {sample_id}")
            return None
        
        total_sessions = len(session_extractions)
        self.stats['total_sessions_aggregated'] += total_sessions
        
        # Pre-aggregate some stats locally before LLM call
        pre_aggregated = self._pre_aggregate_stats(session_extractions)
        
        # Build session data summary for prompt
        session_data_str = self._format_sessions_for_prompt(session_extractions)
        
        # Determine time range
        dates = [s.get('session_date', '') for s in session_extractions if s.get('session_date')]
        first_date = min(dates) if dates else 'unknown'
        last_date = max(dates) if dates else 'unknown'
        
        # Build aggregation prompt
        prompt = L2_AGGREGATION_PROMPT_TEMPLATE.format(
            sample_id=sample_id,
            total_sessions=total_sessions,
            participants=", ".join(participants),
            time_range=f"{first_date} to {last_date}",
            session_data=session_data_str,
            aggregation_time=datetime.now().isoformat(),
            first_session_date=first_date,
            last_session_date=last_date
        )
        
        # Call LLM for aggregation
        aggregation = self._call_llm_with_retry(prompt, sample_id)
        
        if aggregation:
            # Merge pre-aggregated stats with LLM output
            aggregation = self._merge_pre_aggregated(aggregation, pre_aggregated)
            
            # Update stats
            timeline = aggregation.get('master_timeline', [])
            self.stats['total_events_in_timeline'] += sum(
                len(entry.get('events', [])) for entry in timeline
            )
        
        return aggregation
    
    def _pre_aggregate_stats(self, session_extractions: List[Dict]) -> Dict[str, Any]:
        """Pre-aggregate statistics before LLM call"""
        pre_agg = {
            'total_events': 0,
            'total_state_updates': 0,
            'countable_by_category': defaultdict(lambda: {'total': 0, 'items': set()}),
            'all_dates_mentioned': [],
            'all_facts': []
        }
        
        for session in session_extractions:
            # Count events
            events = session.get('structured_events', [])
            pre_agg['total_events'] += len(events)
            
            # Count state updates
            state_updates = session.get('state_updates', [])
            pre_agg['total_state_updates'] += len(state_updates)
            
            # Aggregate countable items
            countables = session.get('countable_items', [])
            for item in countables:
                category = item.get('category', 'Other')
                item_name = item.get('item_name', '')
                count = item.get('count', 1)
                
                pre_agg['countable_by_category'][category]['total'] += count
                if item_name:
                    pre_agg['countable_by_category'][category]['items'].add(item_name)
            
            # Collect dates
            mentioned_dates = session.get('mentioned_dates', [])
            pre_agg['all_dates_mentioned'].extend(mentioned_dates)
            
            # Collect facts
            key_facts = session.get('key_facts', [])
            pre_agg['all_facts'].extend(key_facts)
        
        # Convert sets to lists for JSON serialization
        for cat in pre_agg['countable_by_category']:
            pre_agg['countable_by_category'][cat]['items'] = list(
                pre_agg['countable_by_category'][cat]['items']
            )
        
        return pre_agg
    
    def _format_sessions_for_prompt(self, session_extractions: List[Dict]) -> str:
        """Format session extractions for LLM prompt"""
        formatted_lines = []
        
        for session in session_extractions:
            session_id = session.get('session_id', 'unknown')
            session_date = session.get('session_date', 'unknown')
            topic = session.get('session_topic', 'General conversation')
            
            formatted_lines.append(f"\n--- {session_id} ({session_date}) ---")
            formatted_lines.append(f"Topic: {topic}")
            
            # Events
            events = session.get('structured_events', [])
            if events:
                formatted_lines.append("Events:")
                for e in events[:10]:  # Limit to 10 per session
                    formatted_lines.append(
                        f"  - {e.get('event_name', 'Unknown')} "
                        f"[{e.get('event_type', 'Unknown')}] "
                        f"on {e.get('date', 'unknown')}"
                    )
            
            # State updates
            state_updates = session.get('state_updates', [])
            if state_updates:
                formatted_lines.append("State Changes:")
                for s in state_updates[:5]:  # Limit to 5 per session
                    formatted_lines.append(
                        f"  - {s.get('entity', 'Unknown')}'s {s.get('attribute', 'status')}: "
                        f"{s.get('old_value', '?')} -> {s.get('new_value', '?')}"
                    )
            
            # Countables
            countables = session.get('countable_items', [])
            if countables:
                formatted_lines.append("Countable Items:")
                for c in countables[:8]:  # Limit to 8 per session
                    formatted_lines.append(
                        f"  - {c.get('by_whom', 'Someone')} {c.get('action', 'did')} "
                        f"{c.get('item_name', 'something')} ({c.get('category', 'Other')})"
                    )
            
            # Key facts
            facts = session.get('key_facts', [])
            if facts:
                formatted_lines.append("Key Facts:")
                for f in facts[:5]:  # Limit to 5 per session
                    formatted_lines.append(
                        f"  - [{f.get('fact_type', 'Fact')}] {f.get('subject', 'Unknown')}: "
                        f"{f.get('fact', '')}"
                    )
        
        return "\n".join(formatted_lines)
    
    def _merge_pre_aggregated(self, llm_output: Dict, pre_aggregated: Dict) -> Dict:
        """Merge pre-aggregated statistics with LLM output"""
        # Add pre-aggregated stats to output
        if 'global_statistics' in llm_output:
            llm_output['global_statistics']['pre_aggregated_totals'] = {
                'total_events_from_sessions': pre_aggregated['total_events'],
                'total_state_updates': pre_aggregated['total_state_updates']
            }
            
            # Add detailed countable breakdown
            llm_output['global_statistics']['detailed_countables'] = dict(
                pre_aggregated['countable_by_category']
            )
        
        # Add all mentioned dates for reference
        llm_output['all_mentioned_dates'] = pre_aggregated['all_dates_mentioned']
        
        # Add fact collection
        llm_output['all_extracted_facts'] = pre_aggregated['all_facts']
        
        return llm_output
    
    def _call_llm_with_retry(self, prompt: str, context_id: str) -> Optional[Dict[str, Any]]:
        """Call LLM with retry logic and JSON parsing"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=8000  # Larger for aggregation
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
        return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response with cleanup"""
        if not response:
            return None
        
        text = response.strip()
        
        # Remove markdown code blocks
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        text = text.strip()
        
        try:
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
    
    def _save_sample_result(self, sample_id: str, result: Dict[str, Any]):
        """Save aggregation result to JSON file"""
        output_file = Path(self.config.output_dir) / f"{sample_id}_l2_aggregated.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved: {output_file}")
    
    def _save_aggregation_stats(self):
        """Save aggregation statistics"""
        stats_file = Path(self.config.output_dir) / "l2_aggregation_stats.json"
        
        stats_output = {
            **self.stats,
            'config': {
                'llm_model': self.config.llm_model,
                'enable_parallel': self.config.enable_parallel,
                'max_workers': self.config.max_workers
            },
            'completion_time': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f" Stats saved: {stats_file}")
    
    def _print_summary(self):
        """Print aggregation summary"""
        print("\n" + "=" * 60)
        print(" L2 Aggregation Summary")
        print("=" * 60)
        print(f"Total Samples:           {self.stats['total_samples']}")
        print(f"Processed:               {self.stats['processed_samples']}")
        print(f"Failed:                  {len(self.stats['failed_samples'])}")
        print(f"Sessions Aggregated:     {self.stats['total_sessions_aggregated']}")
        print(f"Events in Timelines:     {self.stats['total_events_in_timeline']}")
        print(f"Processing Time:         {self.stats['processing_time']:.2f}s")
        print(f"Output Directory:        {self.config.output_dir}")
        print("=" * 60 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Step 2b: L2 Sample-Level Global Aggregation"
    )
    parser.add_argument(
        "--l1-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP2_DIR),
        help="L1 extracted directory"
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP3_DIR),
        help="Output directory"
    )
    parser.add_argument(
        "--extract-model",
        default="qwen-3.5-plus-thinking",
        help="抽取模型名称"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Max parallel workers"
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
    
    config = L2AggregationConfig(
        l1_extracted_dir=args.l1_dir,
        output_dir=args.output_dir,
        llm_model=args.extract_model,
        enable_parallel=not args.no_parallel,
        max_workers=args.max_workers,
        sample_ids=args.sample_ids,
        debug_mode=args.debug
    )
    
    aggregator = L2Aggregator(config)
    
    try:
        stats = aggregator.run_aggregation()
        print(" L2 Aggregation Complete!")
        return 0
    except Exception as e:
        print(f" L2 Aggregation Failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
