"""Utilities for pipline."""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from mandol.core import paths


class LoCoMoKnowledgeGraphPipeline:
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 base_dir: str = "benchmark/dataset/locomo",
                 log_level: str = "INFO"):
        self.base_dir = Path(base_dir)
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging(log_level)
        
        self.stages = {
            "1_entity_extraction": {
                "name": "实体抽取与去重",
                "input_dir": self.base_dir / "locomo10.json",
                "output_dir": self.base_dir / "step1_entities",
                "script": "benchmark/dataset_maker/locomo_graph_maker/locomo_entity_extractor.py",
                "required": True,
                "description": "从对话中抽取实体，使用DBSCAN聚类去重"
            },
            "2_relation_generation": {
                "name": "关系生成与构建",
                "input_dir": self.base_dir / "step1_entities",
                "output_dir": self.base_dir / "step2_relations", 
                "script": "benchmark/dataset_maker/locomo_graph_maker/locomo_relation_generator.py",
                "required": True,
                "description": "基于实体生成关系，构建知识图谱"
            },
            "3_semantic_graph": {
                "name": "语义图谱构建",
                "input_dir": self.base_dir / "step2_relations",
                "output_dir": self.base_dir / "step3_semantic_graph",
                "script": "benchmark/dataset_maker/locomo_graph_maker/locomo_entity_relation_semantic_graph.py",
                "required": True,
                "description": "将实体关系转换为SemanticGraph结构"
            },
            "4_benchmark_test": {
                "name": "知识图谱Benchmark测试",
                "input_dir": self.base_dir / "step3_semantic_graph",
                "output_dir": self.base_dir / "knowledge_graph_benchmark_results",
                "script": "benchmark/task_eval/locomo_benchmark_entity_relation.py",
                "required": False,
                "description": "执行知识图谱检索性能测试"
            }
        }
        
        self.execution_stats = {
            "pipeline_start_time": None,
            "pipeline_end_time": None,
            "total_execution_time": 0,
            "stages_executed": [],
            "stages_skipped": [],
            "stages_failed": [],
            "overall_success": False,
            "samples_processed": 0,
            "entities_extracted": 0,
            "relations_generated": 0,
            "semantic_graphs_created": 0
        }
        
        self.stats_lock = threading.Lock()
        
        self.logger.info("LoCoMo知识图谱生成流水线初始化完成")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load config."""
        default_config = {
            "entity_extraction": {
                "llm_model": "deepseek-reasoner",
                "enable_cross_session_dedup": True,
                "parallel_workers": 10,
                "similarity_threshold": 0.85,
                "entity_embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "relation_generation": {
                "llm_model": "deepseek-reasoner", 
                "parallel_workers": 10,
                "confidence_threshold": 0.7,
                "enable_cross_session_relations": True
            },
            "semantic_graph": {
                "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "create_multi_layer_spaces": True,
                "enable_qa_optimization": True
            },
            "benchmark_test": {
                "enable": True,
                "use_entity_relation": True,
                "max_workers": 1,
                "llm_model": "deepseek-v3.2-dashscope"
            },
            "global": {
                "skip_existing": True,
                "backup_existing": True,
                "cleanup_temp": True,
                "max_parallel_samples": 3,
                "enable_sample_parallelization": True
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self._merge_config(default_config, user_config)
                self.logger.info(f"已加载配置文件: {config_file}")
            except Exception as e:
                self.logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    def _merge_config(self, default: Dict, user: Dict):
        """Run merge config."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Run setup logging."""
        logger = logging.getLogger("LoCoMoKGPipeline")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            log_file = self.base_dir / "kg_pipeline_logs" / f"kg_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            self.log_file = log_file
        
        return logger
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.execution_stats:
                    if isinstance(self.execution_stats[key], (int, float)):
                        self.execution_stats[key] += value
                    else:
                        self.execution_stats[key] = value
    
    def run_pipeline(self, 
                    stages: Optional[List[str]] = None,
                    sample_ids: Optional[List[str]] = None,
                    force_rebuild: bool = False,
                    dry_run: bool = False) -> bool:
        """Run pipeline."""
        self.execution_stats["pipeline_start_time"] = datetime.now()
        self.logger.info(" 开始执行LoCoMo知识图谱生成流水线")
        
        if dry_run:
            self.logger.info(" 干运行模式：只检查不执行")
        
        if sample_ids:
            self.logger.info(f" 指定样本处理: {sample_ids}")
        
        try:
            stages_to_run = stages or list(self.stages.keys())
            self.logger.info(f" 计划执行阶段: {stages_to_run}")
            
            if not self._pre_check(stages_to_run, sample_ids, dry_run):
                return False
            
            for stage_key in stages_to_run:
                if stage_key not in self.stages:
                    self.logger.error(f" 未知阶段: {stage_key}")
                    continue
                
                stage_info = self.stages[stage_key]
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f" 执行阶段: {stage_info['name']} ({stage_key})")
                self.logger.info(f" 描述: {stage_info['description']}")
                self.logger.info(f"{'='*80}")
                
                if dry_run:
                    self.logger.info(f" [干运行] 将执行: {stage_info['script']}")
                    with self.stats_lock:
                        self.execution_stats["stages_executed"].append(stage_key)
                    continue
                
                if not force_rebuild and self._should_skip_stage(stage_key, sample_ids):
                    self.logger.info(f"  跳过阶段 {stage_key}: 输出已存在")
                    with self.stats_lock:
                        self.execution_stats["stages_skipped"].append(stage_key)
                    continue
                
                success = self._execute_stage(stage_key, sample_ids)
                
                if success:
                    self.logger.info(f" 阶段 {stage_key} 执行成功")
                    with self.stats_lock:
                        self.execution_stats["stages_executed"].append(stage_key)
                else:
                    self.logger.error(f" 阶段 {stage_key} 执行失败")
                    with self.stats_lock:
                        self.execution_stats["stages_failed"].append(stage_key)
                    
                    if stage_info["required"]:
                        self.logger.error(f" 关键阶段失败，终止流水线")
                        return False
                    else:
                        self.logger.warning(f"  可选阶段失败，继续执行")
            
            self._collect_final_statistics()
            
            self._generate_pipeline_report()
            
            success_count = len(self.execution_stats["stages_executed"])
            total_count = len(stages_to_run)
            
            if success_count == total_count:
                with self.stats_lock:
                    self.execution_stats["overall_success"] = True
                self.logger.info(f" 知识图谱流水线执行完成！成功执行 {success_count}/{total_count} 个阶段")
                return True
            else:
                self.logger.warning(f"  知识图谱流水线部分完成：{success_count}/{total_count} 个阶段成功")
                return False
        
        except Exception as e:
            self.logger.error(f" 知识图谱流水线执行异常: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
        
        finally:
            self.execution_stats["pipeline_end_time"] = datetime.now()
            if self.execution_stats["pipeline_start_time"]:
                self.execution_stats["total_execution_time"] = (
                    self.execution_stats["pipeline_end_time"] - 
                    self.execution_stats["pipeline_start_time"]
                ).total_seconds()
    
    def _pre_check(self, stages_to_run: List[str], sample_ids: Optional[List[str]], dry_run: bool) -> bool:
        """Run pre check."""
        self.logger.info(" 执行预检查...")
        
        locomo10_file = self.base_dir / "locomo10.json"
        if not locomo10_file.exists():
            self.logger.error(f" 基础数据文件不存在: {locomo10_file}")
            return False
        
        if sample_ids:
            try:
                with open(locomo10_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                available_samples = {item.get("sample_id") for item in data if item.get("sample_id")}
                invalid_samples = set(sample_ids) - available_samples
                
                if invalid_samples:
                    self.logger.error(f" 无效的样本ID: {invalid_samples}")
                    self.logger.info(f" 可用样本: {sorted(available_samples)}")
                    return False
                
                self.logger.info(f" 样本ID验证通过: {len(sample_ids)} 个样本")
                
            except Exception as e:
                self.logger.error(f" 验证样本ID时出错: {e}")
                return False
        
        for stage_key in stages_to_run:
            stage_info = self.stages[stage_key]
            script_path = Path(stage_info["script"])
            
            if not script_path.exists():
                self.logger.error(f" 阶段 {stage_key} 脚本不存在: {script_path}")
                return False
        
        if not dry_run:
            try:
                test_dir = self.base_dir / "temp_permission_test"
                test_dir.mkdir(exist_ok=True)
                test_dir.rmdir()
            except Exception as e:
                self.logger.error(f" 基础目录无写入权限: {e}")
                return False
        
        self.logger.info(" 预检查通过")
        return True
    
    def _should_skip_stage(self, stage_key: str, sample_ids: Optional[List[str]]) -> bool:
        """Run should skip stage."""
        if not self.config["global"]["skip_existing"]:
            return False
        
        stage_info = self.stages[stage_key]
        output_dir = stage_info["output_dir"]
        
        if stage_key == "1_entity_extraction":
            if sample_ids:
                for sample_id in sample_ids:
                    entity_file = output_dir / f"{sample_id}_entities.json"
                    if not entity_file.exists():
                        return False
                return True
            else:
                entity_files = list(output_dir.glob("*_entities.json"))
                return len(entity_files) > 0
        
        elif stage_key == "2_relation_generation":
            if sample_ids:
                for sample_id in sample_ids:
                    sample_dir = output_dir / sample_id
                    relation_file = sample_dir / f"{sample_id}_complete_entity_relation.json"
                    if not relation_file.exists():
                        return False
                return True
            else:
                relation_files = list(output_dir.glob("*//*_complete_entity_relation.json"))
                return len(relation_files) > 0
        
        elif stage_key == "3_semantic_graph":
            if sample_ids:
                for sample_id in sample_ids:
                    sample_dir = output_dir / sample_id
                    semantic_graph_file = sample_dir / "semantic_graph.json"
                    if not semantic_graph_file.exists():
                        return False
                return True
            else:
                semantic_files = list(output_dir.glob("*//semantic_graph.json"))
                return len(semantic_files) > 0
        
        elif stage_key == "4_benchmark_test":
            return (output_dir.exists() and 
                   (output_dir / "locomo_graph_benchmark_report.json").exists())
        
        return False
    
    def _execute_stage(self, stage_key: str, sample_ids: Optional[List[str]]) -> bool:
        """Execute stage."""
        stage_info = self.stages[stage_key]
        script_path = stage_info["script"]
        
        try:
            if self.config["global"]["backup_existing"] and stage_info["output_dir"].exists():
                self._backup_output_dir(stage_info["output_dir"], stage_key)
            
            stage_info["output_dir"].parent.mkdir(parents=True, exist_ok=True)
            
            cmd = self._build_stage_command(stage_key, sample_ids)
            
            self.logger.info(f" 执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=paths.PROJECT_ROOT,
                timeout=144000
            )
            
            if result.returncode == 0:
                self.logger.info(f" 阶段 {stage_key} 执行成功")
                if result.stdout:
                    self.logger.debug(f"输出: {result.stdout}")
                return True
            else:
                self.logger.error(f" 阶段 {stage_key} 执行失败 (返回码: {result.returncode})")
                if result.stderr:
                    self.logger.error(f"错误输出: {result.stderr}")
                if result.stdout:
                    self.logger.error(f"标准输出: {result.stdout}")
                return False
        
        except subprocess.TimeoutExpired:
            self.logger.error(f" 阶段 {stage_key} 执行超时")
            return False
        except Exception as e:
            self.logger.error(f" 阶段 {stage_key} 执行异常: {e}")
            return False
    
    def _build_stage_command(self, stage_key: str, sample_ids: Optional[List[str]]) -> List[str]:
        """Build stage command."""
        stage_info = self.stages[stage_key]
        cmd = [sys.executable, stage_info["script"]]
        
        if stage_key == "1_entity_extraction":
            config = self.config["entity_extraction"]
            
            if sample_ids and len(sample_ids) == 1:
                cmd.extend(["--sample-id", sample_ids[0]])
            elif sample_ids and len(sample_ids) > 1:
                pass
            
            cmd.extend([
                "--llm-model", config["llm_model"],
                "--parallel-workers", str(config["parallel_workers"])
            ])
            
            if not config["enable_cross_session_dedup"]:
                cmd.append("--no-cross-session-dedup")
        
        elif stage_key == "2_relation_generation":
            config = self.config["relation_generation"]
            
            cmd.extend([
                "--llm-model", config["llm_model"],
                "--parallel-workers", str(config["parallel_workers"])
            ])
            
            if sample_ids:
                if len(sample_ids) == 1:
                    cmd.extend(["--sample-id", sample_ids[0]])
                else:
                    cmd.extend(["--sample-ids"] + sample_ids)
            else:
                cmd.append("--process-all")
        
        elif stage_key == "3_semantic_graph":
            config = self.config["semantic_graph"]
            
            cmd.extend([
                "--text-embedding-model", config["text_embedding_model"],
                "--batch-mode"
            ])
            
            if sample_ids:
                cmd.extend(["--sample-ids"] + sample_ids)
        
        elif stage_key == "4_benchmark_test":
            config = self.config["benchmark_test"]
            
            if not config["use_entity_relation"]:
                cmd.append("--no-entity-relation")
            
            if config["max_workers"] > 1:
                cmd.extend(["--max-workers", str(config["max_workers"])])
            
            if sample_ids:
                cmd.extend(["--sample-ids"] + sample_ids)
        
        return cmd
    
    def _backup_output_dir(self, output_dir: Path, stage_key: str):
        """Run backup output dir."""
        if not output_dir.exists():
            return
        
        backup_dir = output_dir.parent / f"{output_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            shutil.copytree(output_dir, backup_dir)
            self.logger.info(f" 备份输出目录: {output_dir} -> {backup_dir}")
        except Exception as e:
            self.logger.warning(f"  备份失败: {e}")
    
    def _collect_final_statistics(self):
        """Run collect final statistics."""
        try:
            entity_dir = self.base_dir / "step1_entities"
            if entity_dir.exists():
                entity_files = list(entity_dir.glob("*_entities.json"))
                entity_count = 0
                for entity_file in entity_files:
                    try:
                        with open(entity_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        entity_count += len(data.get("entities", []))
                    except Exception:
                        pass
                
                self._update_stats(entities_extracted=entity_count)
            
            relation_dir = self.base_dir / "step2_relations"
            if relation_dir.exists():
                relation_files = list(relation_dir.glob("*//*_complete_entity_relation.json"))
                relation_count = 0
                for relation_file in relation_files:
                    try:
                        with open(relation_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        relation_count += len(data.get("relations", []))
                    except Exception:
                        pass
                
                self._update_stats(relations_generated=relation_count)
            
            semantic_dir = self.base_dir / "step3_semantic_graph"
            if semantic_dir.exists():
                semantic_files = list(semantic_dir.glob("*//semantic_graph.json"))
                self._update_stats(semantic_graphs_created=len(semantic_files))
            
            locomo_file = self.base_dir / "locomo10.json"
            if locomo_file.exists():
                try:
                    with open(locomo_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._update_stats(samples_processed=len(data))
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.warning(f"  收集统计信息时出错: {e}")
    
    def _generate_pipeline_report(self):
        """Generate pipeline report."""
        report = {
            "pipeline_info": {
                "pipeline_type": "knowledge_graph_generation",
                "execution_date": datetime.now().isoformat(),
                "total_execution_time_seconds": self.execution_stats["total_execution_time"],
                "pipeline_start_time": self.execution_stats["pipeline_start_time"].isoformat() if self.execution_stats["pipeline_start_time"] else None,
                "pipeline_end_time": self.execution_stats["pipeline_end_time"].isoformat() if self.execution_stats["pipeline_end_time"] else None,
                "overall_success": self.execution_stats["overall_success"]
            },
            "stage_results": {
                "stages_executed": self.execution_stats["stages_executed"],
                "stages_skipped": self.execution_stats["stages_skipped"],
                "stages_failed": self.execution_stats["stages_failed"],
                "execution_summary": {
                    "total_stages": len(self.stages),
                    "executed_count": len(self.execution_stats["stages_executed"]),
                    "skipped_count": len(self.execution_stats["stages_skipped"]),
                    "failed_count": len(self.execution_stats["stages_failed"])
                }
            },
            "knowledge_graph_statistics": {
                "samples_processed": self.execution_stats["samples_processed"],
                "entities_extracted": self.execution_stats["entities_extracted"],
                "relations_generated": self.execution_stats["relations_generated"],
                "semantic_graphs_created": self.execution_stats["semantic_graphs_created"],
                "avg_entities_per_sample": self.execution_stats["entities_extracted"] / max(1, self.execution_stats["samples_processed"]),
                "avg_relations_per_sample": self.execution_stats["relations_generated"] / max(1, self.execution_stats["samples_processed"])
            },
            "configuration": self.config,
            "output_locations": {
                stage_key: str(stage_info["output_dir"])
                for stage_key, stage_info in self.stages.items()
            },
            "complementary_pipelines": {
                "hierarchical_pipeline": "benchmark/dataset_maker/locomo_hierarchical_maker/locomo_pipeline.py",
                "relationship": "双塔召回系统数据准备：知识图谱塔（本流水线）+ 分层检索塔（分层流水线）"
            }
        }
        
        
        report_file = self.base_dir / "kg_pipeline_reports" / f"kg_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f" 知识图谱流水线报告已生成: {report_file}")
    
    def list_stages(self):
        """Run list stages."""
        print(" 可用的知识图谱流水线阶段:")
        print("=" * 100)
        
        for stage_key, stage_info in self.stages.items():
            required_mark = " 必需" if stage_info["required"] else " 可选"
            print(f"{stage_key}: {stage_info['name']} [{required_mark}]")
            print(f"     描述: {stage_info['description']}")
            print(f"     输入: {stage_info['input_dir']}")
            print(f"     输出: {stage_info['output_dir']}")
            print(f"     脚本: {stage_info['script']}")
            print()
    
    def clean_outputs(self, stages: Optional[List[str]] = None, confirm: bool = False):
        """Run clean outputs."""
        if not confirm:
            self.logger.warning("  清理操作需要确认，请设置 --confirm 参数")
            return
        
        stages_to_clean = stages or list(self.stages.keys())
        
        for stage_key in stages_to_clean:
            if stage_key not in self.stages:
                continue
            
            output_dir = self.stages[stage_key]["output_dir"]
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                    self.logger.info(f" 已清理: {output_dir}")
                except Exception as e:
                    self.logger.error(f" 清理失败 {output_dir}: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Return pipeline status."""
        status = {}
        
        for stage_key, stage_info in self.stages.items():
            output_dir = stage_info["output_dir"]
            
            if stage_key == "1_entity_extraction":
                entity_files = list(output_dir.glob("*_entities.json")) if output_dir.exists() else []
                status[stage_key] = {
                    "completed": len(entity_files) > 0,
                    "output_count": len(entity_files),
                    "output_files": [f.name for f in entity_files[:5]]
                }
            
            elif stage_key == "2_relation_generation":
                relation_files = list(output_dir.glob("*//*_complete_entity_relation.json")) if output_dir.exists() else []
                status[stage_key] = {
                    "completed": len(relation_files) > 0,
                    "output_count": len(relation_files),
                    "output_samples": [f.parent.name for f in relation_files[:5]]
                }
            
            elif stage_key == "3_semantic_graph":
                semantic_files = list(output_dir.glob("*//semantic_graph.json")) if output_dir.exists() else []
                status[stage_key] = {
                    "completed": len(semantic_files) > 0,
                    "output_count": len(semantic_files),
                    "output_samples": [f.parent.name for f in semantic_files[:5]]
                }
            
            elif stage_key == "4_benchmark_test":
                benchmark_file = output_dir / "locomo_graph_benchmark_report.json"
                status[stage_key] = {
                    "completed": benchmark_file.exists(),
                    "output_exists": benchmark_file.exists()
                }
        
        return status


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="LoCoMo知识图谱生成流水线")
    
    parser.add_argument("--config", type=str,
                       help="配置文件路径")
    parser.add_argument("--base-dir", type=str,
                       default="benchmark/dataset/locomo",
                       help="基础数据目录")
    parser.add_argument("--log-level", type=str,
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    parser.add_argument("--stages", nargs='+',
                       choices=["1_entity_extraction", "2_relation_generation", 
                               "3_semantic_graph", "4_benchmark_test"],
                       help="指定要执行的阶段")
    parser.add_argument("--sample-ids", nargs='+',
                       help="指定要处理的样本ID")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="强制重新构建，即使输出已存在")
    parser.add_argument("--dry-run", action="store_true",
                       help="干运行，只检查不执行")
    
    parser.add_argument("--list-stages", action="store_true",
                       help="列出所有可用阶段")
    parser.add_argument("--status", action="store_true",
                       help="查看流水线状态")
    parser.add_argument("--clean", nargs='*',
                       help="清理指定阶段的输出")
    parser.add_argument("--confirm", action="store_true",
                       help="确认清理操作")
    
    args = parser.parse_args()
    
    pipeline = LoCoMoKnowledgeGraphPipeline(
        config_file=args.config,
        base_dir=args.base_dir,
        log_level=args.log_level
    )
    
    try:
        if args.list_stages:
            pipeline.list_stages()
            return 0
        
        if args.status:
            print(" 知识图谱流水线状态:")
            print("=" * 80)
            status = pipeline.get_pipeline_status()
            for stage_key, stage_status in status.items():
                stage_name = pipeline.stages[stage_key]["name"]
                completed = " 完成" if stage_status["completed"] else " 未完成"
                print(f"{stage_key}: {stage_name} - {completed}")
                if "output_count" in stage_status:
                    print(f"     输出数量: {stage_status['output_count']}")
            return 0
        
        if args.clean is not None:
            clean_stages = args.clean if args.clean else None
            pipeline.clean_outputs(stages=clean_stages, confirm=args.confirm)
            return 0
        
        print(" LoCoMo知识图谱生成流水线")
        print("=" * 80)
        print(" 流水线组成:")
        print("   1  实体抽取与去重 (DBSCAN聚类)")
        print("   2  关系生成与构建 (跨会话关系)")
        print("   3  语义图谱构建 (多层空间优化)")
        print("   4  知识图谱Benchmark测试")
        print()
        print(" 与分层检索流水线互补，共同构成双塔召回系统数据准备")
        print("=" * 80)
        
        success = pipeline.run_pipeline(
            stages=args.stages,
            sample_ids=args.sample_ids,
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run
        )
        
        if success:
            print("\n 知识图谱流水线执行成功！")
            print(f" 详细报告: {pipeline.base_dir}/kg_pipeline_reports/")
            print(f" 日志文件: {pipeline.log_file}")
            
            stats = pipeline.execution_stats
            if stats["samples_processed"] > 0:
                print(f" 处理统计:")
                print(f"   - 样本处理: {stats['samples_processed']} 个")
                print(f"   - 实体抽取: {stats['entities_extracted']} 个")
                print(f"   - 关系生成: {stats['relations_generated']} 个")
                print(f"   - 语义图谱: {stats['semantic_graphs_created']} 个")
            
            return 0
        else:
            print("\n 知识图谱流水线执行失败！")
            print(f" 查看日志: {pipeline.log_file}")
            return 1
    
    except KeyboardInterrupt:
        print("\n  用户中断操作")
        return 1
    except Exception as e:
        print(f"\n 知识图谱流水线异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main())