"""Extract episodic facts from LoCoMo conversations."""

import json
import logging
import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Lock


from mandol.llm.llm_client import LLMClient
from mandol.core import paths


@dataclass
class EpisodicFactConfig:
    locomo_file_path: str = str(paths.LOCOMO_RAW_FILE)
    output_dir: str = str(paths.LOCOMO_EPISODIC_STEP1_DIR)
    
    llm_model: str = "qwen-3.5-plus-thinking"
    llm_temperature: float = 0.1
    
    max_workers: int = 6
    
    sample_ids: List[str] = None  # Dataset-specific handling used by the reproduction workflow.
    enable_detail_extraction: bool = True
    
    debug_mode: bool = False



class FactType:
    
    EVENT = "EVENT"
    STATE_CHANGE = "STATE_CHANGE"
    ACTIVITY = "ACTIVITY"
    PLAN = "PLAN"
    ACHIEVEMENT = "ACHIEVEMENT"
    
    RECOMMENDATION = "RECOMMENDATION"
    OPINION = "OPINION"
    PREFERENCE = "PREFERENCE"
    
    RELATIONSHIP = "RELATIONSHIP"
    INTERACTION = "INTERACTION"
    
    POSSESSION = "POSSESSION"
    ATTRIBUTE = "ATTRIBUTE"
    NUMERICAL = "NUMERICAL"



@dataclass
class TimeInfo:
    original_text: str = ""
    absolute_start: str = ""
    absolute_end: str = ""
    reference_date: str = ""
    is_exact: bool = False
    is_future: bool = False


@dataclass  
class EpisodicFact:
    fact_id: str
    content: str
    fact_type: str
    
    participants: List[str] = field(default_factory=list)
    
    time: TimeInfo = field(default_factory=TimeInfo)
    
    location: str = ""
    
    details: Dict[str, Any] = field(default_factory=dict)
    
    source_session_id: str = ""
    source_turns: List[str] = field(default_factory=list)
    
    retrieval_keys: List[str] = field(default_factory=list)
    
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Run to dict."""
        result = asdict(self)
        if isinstance(self.time, TimeInfo):
            result['time'] = asdict(self.time)
        return result



EPISODIC_EXTRACTION_PROMPT = """
You are an expert fact extractor for a Question-Answering system. Your task is to extract ALL answerable facts from this conversation session.

## Context
- **Session Date (Reference)**: {session_date}
- **Speakers**: {speakers}
- **Session ID**: {session_id}

## Extraction Goals
Extract facts that can answer these question types:
1. **When** questions: "When did X happen?" → Need precise time
2. **What** questions: "What did X do/say/recommend?" → Need specific details
3. **How many** questions: "How many times did X?" → Need countable events
4. **Who** questions: "Who did X meet/help/talk to?" → Need participants
5. **Where** questions: "Where did X go?" → Need locations
6. **What kind/type** questions: "What kind of food?" → Need specific names

## Time Resolution Rules (CRITICAL)
- Reference Date: {session_date}
- "last Friday" + Reference 2023-11-17 → 2023-11-10
- "Thursday before December 17" → 2023-12-14
- "towards the end of summer" → 2023-08-15 to 2023-09-01
- "next weekend" → calculate from reference date
- If time is unclear, use the session date as default

## Dialogue to Process
{dialogue_text}

## Output Format (JSON)
{{
    "facts": [
        {{
            "content": "Complete, self-contained description of the fact. Include WHO, WHAT, WHEN context. Example: 'Sam fell in love with a Canadian woman towards the end of summer 2023.'",
            "fact_type": "EVENT|STATE_CHANGE|ACTIVITY|PLAN|ACHIEVEMENT|RECOMMENDATION|OPINION|PREFERENCE|RELATIONSHIP|INTERACTION|POSSESSION|ATTRIBUTE|NUMERICAL",
            "participants": ["Person1", "Person2"],
            "time": {{
                "original_text": "towards the end of summer",
                "absolute_start": "2023-08-15",
                "absolute_end": "2023-09-01",
                "is_exact": false,
                "is_future": false
            }},
            "location": "Canada (or empty string if unknown)",
            "details": {{
                "what": "fell in love",
                "with_whom": "Canadian woman",
                "specific_items": [],  // For food, games, books, etc.
                "numerical_value": null,  // For counts, durations, etc.
                "advice_content": null  // For recommendations
            }},
            "source_turns": ["D5:1"],
            "retrieval_keys": ["Sam love", "Canadian woman", "summer 2023", "Sam relationship"]
        }}
    ]
}}

## IMPORTANT Rules
1. Extract EVERY fact that could be asked about, even small details
2. For recommendations/suggestions, include the EXACT content of what was recommended
3. For food/games/books, include SPECIFIC names, not generic descriptions
4. For numerical facts (how many, how long), extract the exact number
5. Each fact should be self-contained and understandable without context
6. Generate multiple retrieval_keys for each fact (synonyms, related terms)
7. If a speaker shares something (photo, recipe), describe WHAT they shared specifically
"""



class EpisodicFactExtractor:
    
    def __init__(self, config: EpisodicFactConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        self.llm = LLMClient(model_name=config.llm_model)
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_lock = Lock()
        self.stats = {
            'samples_processed': 0,
            'sessions_processed': 0,
            'facts_extracted': 0,
            'failed_samples': [],
            'sample_details': {}
        }
        
        self.month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4, 
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        
        self.logger.info(f" 情景事实抽取器初始化完成")
        self.logger.info(f" 输出目录: {self.output_dir}")
        self.logger.info(f" LLM模型: {config.llm_model}")
    
    def _setup_logging(self) -> logging.Logger:
        """Run setup logging."""
        logger = logging.getLogger(f"{__name__}.EpisodicFactExtractor")
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _parse_session_date(self, date_str: str) -> str:
        """Parse session date."""
        if not date_str or date_str == "Unknown":
            return datetime.now().strftime("%Y-%m-%d")
        
        try:
            parts = date_str.replace(",", "").split()
            if len(parts) >= 3:
                day = int(parts[0])
                month = self.month_map.get(parts[1].lower(), 1)
                year = int(parts[2])
                return f"{year}-{month:02d}-{day:02d}"
        except Exception:
            pass
        
        return date_str
    
    def _extract_session_from_conversation(self, conversation: Dict, session_key: str) -> Dict:
        """Extract session from conversation."""
        session_data = conversation.get(session_key, [])
        date_key = f"{session_key}_date_time"
        session_date = conversation.get(date_key, "Unknown")
        
        return {
            'session_id': session_key,
            'date': session_date,
            'dialogues': session_data
        }
    
    def _build_dialogue_text(self, dialogues: List, speakers: Dict) -> Tuple[str, List[str]]:
        """Build dialogue text."""
        lines = []
        turn_ids = []
        
        for i, turn in enumerate(dialogues, 1):
            if isinstance(turn, dict):
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                
                if 'share' in turn:
                    share_content = turn['share']
                    if isinstance(share_content, dict):
                        share_type = share_content.get('type', 'content')
                        if share_type == 'image':
                            caption = share_content.get('caption', 'an image')
                            text = f"{text} [Shared image: {caption}]"
                        else:
                            shared_text = share_content.get('text', '')
                            text = f"{text} [Shared: {shared_text}]"
                
                if text:
                    speaker_name = speakers.get(f'speaker_{speaker.lower()}', speaker)
                    if speaker_name == speaker and 'speaker_a' in speakers:
                        if speaker.upper() == 'A':
                            speaker_name = speakers.get('speaker_a', speaker)
                        elif speaker.upper() == 'B':
                            speaker_name = speakers.get('speaker_b', speaker)
                    
                    lines.append(f"[Turn {i}] {speaker_name}: {text}")
                    turn_ids.append(f"D{i}")
        
        return "\n".join(lines), turn_ids
    
    def _safe_parse_json(self, content: str) -> Dict:
        """Run safe parse JSON."""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                import re
                content = re.sub(r',\s*}', '}', content)
                content = re.sub(r',\s*]', ']', content)
                return json.loads(content)
            except:
                pass
            
            try:
                from json_repair import repair_json
                repaired = repair_json(content)
                if isinstance(repaired, str):
                    return json.loads(repaired)
                return repaired
            except:
                pass
        
        return {"facts": []}
    
    def extract_facts_from_session(
        self, 
        session_data: Dict, 
        sample_id: str,
        speakers: Dict
    ) -> List[EpisodicFact]:
        """Extract facts from session."""
        session_id = session_data['session_id']
        session_date = self._parse_session_date(session_data.get('date', 'Unknown'))
        dialogues = session_data.get('dialogues', [])
        
        if not dialogues:
            return []
        
        dialogue_text, turn_ids = self._build_dialogue_text(dialogues, speakers)
        
        if not dialogue_text.strip():
            return []
        
        speaker_str = f"{speakers.get('speaker_a', 'Speaker A')} and {speakers.get('speaker_b', 'Speaker B')}"
        
        prompt = EPISODIC_EXTRACTION_PROMPT.format(
            session_date=session_date,
            speakers=speaker_str,
            session_id=session_id,
            dialogue_text=dialogue_text
        )
        
        try:
            response = self.llm.generate_answer(
                prompt=prompt,
                temperature=self.config.llm_temperature,
                json_format=True
            )
            
            parsed = self._safe_parse_json(response)
            raw_facts = parsed.get('facts', [])
            
            if not isinstance(raw_facts, list):
                raw_facts = []
            
            facts = []
            for idx, raw in enumerate(raw_facts):
                if not isinstance(raw, dict):
                    continue
                    
                content = raw.get('content', '')
                if not content:
                    continue
                
                time_data = raw.get('time', {})
                if not isinstance(time_data, dict):
                    time_data = {}
                    
                time_info = TimeInfo(
                    original_text=time_data.get('original_text', ''),
                    absolute_start=time_data.get('absolute_start', session_date),
                    absolute_end=time_data.get('absolute_end', time_data.get('absolute_start', session_date)),
                    reference_date=session_date,
                    is_exact=time_data.get('is_exact', False),
                    is_future=time_data.get('is_future', False)
                )
                
                details = raw.get('details', {})
                if not isinstance(details, dict):
                    details = {}
                
                retrieval_keys = raw.get('retrieval_keys', [])
                if not isinstance(retrieval_keys, list):
                    retrieval_keys = []
                
                source_turns = raw.get('source_turns', [])
                if not isinstance(source_turns, list):
                    source_turns = []
                
                participants = raw.get('participants', [])
                if not isinstance(participants, list):
                    participants = []
                
                fact = EpisodicFact(
                    fact_id=f"{sample_id}_{session_id}_f{idx}",
                    content=content,
                    fact_type=raw.get('fact_type', 'EVENT'),
                    participants=participants,
                    time=time_info,
                    location=raw.get('location', '') or '',
                    details=details,
                    source_session_id=session_id,
                    source_turns=source_turns,
                    retrieval_keys=retrieval_keys,
                    confidence=1.0
                )
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            self.logger.error(f"抽取Session {session_id} 失败: {e}")
            if self.config.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return []
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process sample."""
        sample_id = sample.get('sample_id', 'unknown')
        conversation = sample.get('conversation', {})
        speakers = sample.get('speakers', {})
        
        self.logger.info(f"处理样本: {sample_id}")
        
        all_facts = []
        sessions_processed = 0
        
        session_keys = sorted([
            k for k in conversation.keys() 
            if re.match(r'^session_\d+$', k)
        ], key=lambda x: int(x.split('_')[1]))
        
        for session_key in session_keys:
            session_data = self._extract_session_from_conversation(conversation, session_key)
            
            facts = self.extract_facts_from_session(session_data, sample_id, speakers)
            all_facts.extend(facts)
            sessions_processed += 1
            
            self.logger.debug(f"  Session {session_key}: {len(facts)} facts")
        
        
        temporal_index = self._build_temporal_index(all_facts)
        participant_index = self._build_participant_index(all_facts)
        fact_type_index = self._build_fact_type_index(all_facts)
        
        output_data = {
            "sample_id": sample_id,
            "metadata": {
                "speakers": speakers,
                "total_sessions": sessions_processed,
                "total_facts": len(all_facts),
                "extraction_time": datetime.now().isoformat(),
                "llm_model": self.config.llm_model
            },
            "episodic_facts": [f.to_dict() for f in all_facts],
            "indices": {
                "temporal": temporal_index,
                "participants": participant_index,
                "fact_types": fact_type_index
            }
        }
        
        
        output_file = self.output_dir / f"{sample_id}_episodic_facts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        with self.stats_lock:
            self.stats['samples_processed'] += 1
            self.stats['sessions_processed'] += sessions_processed
            self.stats['facts_extracted'] += len(all_facts)
            self.stats['sample_details'][sample_id] = {
                'sessions': sessions_processed,
                'facts': len(all_facts)
            }
        
        self.logger.info(f" 样本 {sample_id} 完成: {sessions_processed} sessions, {len(all_facts)} facts")
        
        return {
            'sample_id': sample_id,
            'sessions': sessions_processed,
            'facts': len(all_facts)
        }
    
    def _build_temporal_index(self, facts: List[EpisodicFact]) -> Dict:
        """Build temporal index."""
        index = defaultdict(list)
        
        for fact in facts:
            if fact.time and fact.time.absolute_start:
                
                try:
                    date_parts = fact.time.absolute_start.split('-')
                    if len(date_parts) >= 2:
                        year_month = f"{date_parts[0]}-{date_parts[1]}"
                        index[year_month].append(fact.fact_id)
                except:
                    pass
        
        return dict(index)
    
    def _build_participant_index(self, facts: List[EpisodicFact]) -> Dict:
        """Build participant index."""
        index = defaultdict(list)
        
        for fact in facts:
            for participant in fact.participants:
                index[participant.lower()].append(fact.fact_id)
        
        return dict(index)
    
    def _build_fact_type_index(self, facts: List[EpisodicFact]) -> Dict:
        """Build fact type index."""
        index = defaultdict(list)
        
        for fact in facts:
            index[fact.fact_type].append(fact.fact_id)
        
        return dict(index)
    
    def run(self) -> Dict:
        """Run."""
        self.logger.info("=" * 80)
        self.logger.info(" 开始情景事实抽取 (Episodic Fact Extraction)")
        self.logger.info("=" * 80)
        
        
        if not os.path.exists(self.config.locomo_file_path):
            self.logger.error(f" 数据文件未找到: {self.config.locomo_file_path}")
            return self.stats
        
        with open(self.config.locomo_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if self.config.sample_ids:
            samples = [s for s in data if s.get('sample_id') in self.config.sample_ids]
        else:
            samples = data
        
        self.logger.info(f" 待处理样本数: {len(samples)}")
        
        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.process_sample, s): s for s in samples}
                
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        sample_id = sample.get('sample_id', 'unknown')
                        self.logger.error(f" 样本 {sample_id} 处理失败: {e}")
                        self.stats['failed_samples'].append(sample_id)
        else:
            for sample in samples:
                try:
                    self.process_sample(sample)
                except Exception as e:
                    sample_id = sample.get('sample_id', 'unknown')
                    self.logger.error(f" 样本 {sample_id} 处理失败: {e}")
                    self.stats['failed_samples'].append(sample_id)
        
        
        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Run print summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(" 情景事实抽取完成")
        self.logger.info("=" * 80)
        self.logger.info(f" 处理样本数: {self.stats['samples_processed']}")
        self.logger.info(f" 处理Session数: {self.stats['sessions_processed']}")
        self.logger.info(f" 抽取事实数: {self.stats['facts_extracted']}")
        
        if self.stats['failed_samples']:
            self.logger.warning(f" 失败样本: {self.stats['failed_samples']}")
        
        self.logger.info(f" 输出目录: {self.output_dir}")



def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Step 1: 情景事实抽取 (Episodic Fact Extraction)"
    )
    
    parser.add_argument(
        "--input-file",
        default=str(paths.LOCOMO_RAW_FILE),
        help="输入的locomo数据文件"
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.LOCOMO_EPISODIC_STEP1_DIR),
        help="输出目录"
    )
    parser.add_argument(
        "--extract-model",
        default="qwen-3.5-plus-thinking",
        help="抽取模型名称"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="最大并行工作线程数"
    )
    parser.add_argument(
        "--sample-ids",
        nargs='+',
        help="指定处理的sample ID列表"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    config = EpisodicFactConfig(
        locomo_file_path=args.input_file,
        output_dir=args.output_dir,
        llm_model=args.extract_model,
        max_workers=args.max_workers,
        sample_ids=args.sample_ids,
        debug_mode=args.debug
    )
    
    extractor = EpisodicFactExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()
