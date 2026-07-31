#!/usr/bin/env python3
"""Utilities for generate comparison table."""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import argparse
from tabulate import tabulate
from mandol.core import paths

BASELINE_SYSTEMS = {
    "MIRIX": {
        "tokens": "-",
        "single_hop": 68.22,
        "multi_hop": 54.26,
        "temporal_reasoning": 68.54,
        "open_domain": 46.88,
        "adversarial": 0,
        "overall": 64.33,
        "overall_f1": 28.10
    },
    "Mem0": {
        "tokens": "1172",
        "single_hop": 73.33,
        "multi_hop": 58.75,
        "temporal_reasoning": 52.34,
        "open_domain": 45.83,
        "adversarial": 0,
        "overall": 64.57,
        "overall_f1": 43.46
    },
    "Zep": {
        "tokens": "2701",
        "single_hop": 66.23,
        "multi_hop": 52.12,
        "temporal_reasoning": 54.82,
        "open_domain": 33.33,
        "adversarial": 0,
        "overall": 59.22,
        "overall_f1": 41.23
    },
    "Memobase": {
        "tokens": "2102",
        "single_hop": 73.12,
        "multi_hop": 64.65,
        "temporal_reasoning": 81.20,
        "open_domain": 53.12,
        "adversarial": 0,
        "overall": 72.01,
        "overall_f1": 50.18
    },
    "MemU": {
        "tokens": "617",
        "single_hop": 66.34,
        "multi_hop": 63.12,
        "temporal_reasoning": 27.10,
        "open_domain": 50.01,
        "adversarial": 0,
        "overall": 56.55,
        "overall_f1": 35.15
    },
    "Supermemory": {
        "tokens": "500",
        "single_hop": 67.30,
        "multi_hop": 51.12,
        "temporal_reasoning": 31.77,
        "open_domain": 42.67,
        "adversarial": 0,
        "overall": 55.34,
        "overall_f1": 34.87
    },
    "MemOS-1031": {
        "tokens": "1589",
        "single_hop": 81.09,
        "multi_hop": 67.49,
        "temporal_reasoning": 75.18,
        "open_domain": 55.90,
        "adversarial": 0,
        "overall": 75.80,
        "overall_f1": 45.27
    }
}

CATEGORY_MAP = {
    "1": "single_hop",
    "2": "multi_hop",
    "3": "temporal_reasoning",
    "4": "open_domain",
    "5": "adversarial"
}

CATEGORY_NAMES = {
    "single_hop": "Single-hop",
    "multi_hop": "Multi-hop",
    "temporal_reasoning": "Temporal Reasoning",
    "open_domain": "Open-domain",
    "adversarial": "Adversarial"
}


class LoCoMoResultsAnalyzer:
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.sample_results: List[Dict] = []
        self.aggregated_stats: Dict = {}
        
    def load_results(self) -> int:
        """Load results."""
        pattern = str(self.results_dir / "sample_conv-*_*.json")
        files = glob.glob(pattern)
        
        json_files = [f for f in files if "readable" not in f]
        
        print(f" 找到 {len(json_files)} 个结果文件")
        
        for file_path in sorted(json_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sample_results.append(data)
                    print(f"   加载: {Path(file_path).name}")
            except Exception as e:
                print(f"   加载失败: {file_path} - {e}")
        
        return len(self.sample_results)
    
    def analyze(self) -> Dict:
        """Run analyze."""
        if not self.sample_results:
            print(" 没有加载任何结果文件")
            return {}
        
        category_stats = defaultdict(lambda: {
            "test_count": 0,
            "total_llm_accuracy": 0.0,
            "total_f1_score": 0.0,
            "samples": []
        })
        
        total_test_count = 0
        total_llm_accuracy_sum = 0.0
        total_f1_sum = 0.0
        
        non_adv_test_count = 0
        non_adv_llm_accuracy_sum = 0.0
        non_adv_f1_sum = 0.0
        
        for sample in self.sample_results:
            sample_id = sample.get("sample_info", {}).get("sample_id", "unknown")
            cat_perf = sample.get("category_performance", {})
            
            for cat_id, cat_data in cat_perf.items():
                cat_key = CATEGORY_MAP.get(cat_id, f"category_{cat_id}")
                test_count = cat_data.get("test_count", 0)
                llm_acc = cat_data.get("avg_llm_accuracy", 0)
                f1_score = cat_data.get("avg_f1_score", 0)
                
                category_stats[cat_key]["test_count"] += test_count
                category_stats[cat_key]["total_llm_accuracy"] += llm_acc * test_count
                category_stats[cat_key]["total_f1_score"] += f1_score * test_count
                category_stats[cat_key]["samples"].append({
                    "sample_id": sample_id,
                    "test_count": test_count,
                    "llm_accuracy": llm_acc,
                    "f1_score": f1_score
                })
                
                total_test_count += test_count
                total_llm_accuracy_sum += llm_acc * test_count
                total_f1_sum += f1_score * test_count
                
                if cat_key != "adversarial":
                    non_adv_test_count += test_count
                    non_adv_llm_accuracy_sum += llm_acc * test_count
                    non_adv_f1_sum += f1_score * test_count
        
        self.aggregated_stats = {}
        for cat_key, stats in category_stats.items():
            if stats["test_count"] > 0:
                self.aggregated_stats[cat_key] = {
                    "test_count": stats["test_count"],
                    "avg_llm_accuracy": stats["total_llm_accuracy"] / stats["test_count"],
                    "avg_f1_score": stats["total_f1_score"] / stats["test_count"],
                    "samples": stats["samples"]
                }
        
        if total_test_count > 0:
            self.aggregated_stats["overall_with_adv"] = {
                "test_count": total_test_count,
                "avg_llm_accuracy": total_llm_accuracy_sum / total_test_count,
                "avg_f1_score": total_f1_sum / total_test_count
            }
        
        if non_adv_test_count > 0:
            self.aggregated_stats["overall_no_adv"] = {
                "test_count": non_adv_test_count,
                "avg_llm_accuracy": non_adv_llm_accuracy_sum / non_adv_test_count,
                "avg_f1_score": non_adv_f1_sum / non_adv_test_count
            }
        
        return self.aggregated_stats
    
    def get_our_system_data(self) -> Dict:
        """Return our system data."""
        if not self.aggregated_stats:
            self.analyze()
        
        our_data = {
            "tokens": "~1500",
            "single_hop": self.aggregated_stats.get("single_hop", {}).get("avg_llm_accuracy", 0) * 100,
            "multi_hop": self.aggregated_stats.get("multi_hop", {}).get("avg_llm_accuracy", 0) * 100,
            "temporal_reasoning": self.aggregated_stats.get("temporal_reasoning", {}).get("avg_llm_accuracy", 0) * 100,
            "open_domain": self.aggregated_stats.get("open_domain", {}).get("avg_llm_accuracy", 0) * 100,
            "adversarial": self.aggregated_stats.get("adversarial", {}).get("avg_llm_accuracy", 0) * 100,
            "overall": self.aggregated_stats.get("overall_no_adv", {}).get("avg_llm_accuracy", 0) * 100,
            "overall_with_adv": self.aggregated_stats.get("overall_with_adv", {}).get("avg_llm_accuracy", 0) * 100,
            "overall_f1": self.aggregated_stats.get("overall_with_adv", {}).get("avg_f1_score", 0) * 100
        }
        
        return our_data
    
    def print_detailed_stats(self):
        """Run print detailed stats."""
        if not self.aggregated_stats:
            self.analyze()
        
        print("\n" + "=" * 80)
        print(" LoCoMo Benchmark 详细统计结果")
        print("=" * 80)
        
        print("\n### 各维度统计 ###\n")
        for cat_key in ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]:
            if cat_key in self.aggregated_stats:
                stats = self.aggregated_stats[cat_key]
                cat_name = CATEGORY_NAMES.get(cat_key, cat_key)
                print(f" {cat_name}:")
                print(f"   测试数量: {stats['test_count']}")
                print(f"   平均 LLM Accuracy: {stats['avg_llm_accuracy'] * 100:.2f}%")
                print(f"   平均 F1 Score: {stats['avg_f1_score'] * 100:.2f}%")
                print()
        
        print("### Overall 统计 ###\n")
        if "overall_no_adv" in self.aggregated_stats:
            stats = self.aggregated_stats["overall_no_adv"]
            print(f" Overall (不含 Adversarial):")
            print(f"   测试数量: {stats['test_count']}")
            print(f"   平均 LLM Accuracy: {stats['avg_llm_accuracy'] * 100:.2f}%")
            print(f"   平均 F1 Score: {stats['avg_f1_score'] * 100:.2f}%")
            print()
        
        if "overall_with_adv" in self.aggregated_stats:
            stats = self.aggregated_stats["overall_with_adv"]
            print(f" Overall (含 Adversarial):")
            print(f"   测试数量: {stats['test_count']}")
            print(f"   平均 LLM Accuracy: {stats['avg_llm_accuracy'] * 100:.2f}%")
            print(f"   平均 F1 Score: {stats['avg_f1_score'] * 100:.2f}%")
            print()


def generate_comparison_table(our_data: Dict, output_format: str = "markdown") -> str:
    """Generate comparison table."""
    
    headers = [
        "Method", "Tokens", 
        "Single-hop ↑", "Multi-hop ↑", "Temporal ↑", "Open-domain ↑", 
        "Overall ↑\n(no Adv)", "Overall F1 ↑",
        "Adversarial ↑", "Overall ↑\n(with Adv)"
    ]
    
    rows = []
    
    for system_name, data in BASELINE_SYSTEMS.items():
        row = [
            system_name,
            data["tokens"],
            f"{data['single_hop']:.2f}",
            f"{data['multi_hop']:.2f}",
            f"{data['temporal_reasoning']:.2f}",
            f"{data['open_domain']:.2f}",
            f"{data['overall']:.2f}",
            f"{data['overall_f1']:.2f}",
            f"{data['adversarial']:.2f}" if data['adversarial'] > 0 else "-",
            "-"
        ]
        rows.append(row)
    
    our_row = [
        "**Ours (Three-Tower)**",
        our_data["tokens"],
        f"**{our_data['single_hop']:.2f}**" if our_data['single_hop'] > 81.09 else f"{our_data['single_hop']:.2f}",
        f"**{our_data['multi_hop']:.2f}**" if our_data['multi_hop'] > 67.49 else f"{our_data['multi_hop']:.2f}",
        f"**{our_data['temporal_reasoning']:.2f}**" if our_data['temporal_reasoning'] > 81.20 else f"{our_data['temporal_reasoning']:.2f}",
        f"**{our_data['open_domain']:.2f}**" if our_data['open_domain'] > 55.90 else f"{our_data['open_domain']:.2f}",
        f"**{our_data['overall']:.2f}**" if our_data['overall'] > 75.80 else f"{our_data['overall']:.2f}",
        f"{our_data['overall_f1']:.2f}",
        f"**{our_data['adversarial']:.2f}**",
        f"**{our_data['overall_with_adv']:.2f}**"
    ]
    rows.append(our_row)
    
    if output_format == "markdown":
        return tabulate(rows, headers=headers, tablefmt="pipe")
    elif output_format == "latex":
        return tabulate(rows, headers=headers, tablefmt="latex_booktabs")
    else:
        return tabulate(rows, headers=headers, tablefmt="grid")


def generate_summary_table(our_data: Dict) -> str:
    """Generate summary table."""
    
    headers = ["Method", "Single-hop", "Multi-hop", "Temporal", "Open-domain", "Overall*", "F1"]
    
    rows = []
    for system_name, data in BASELINE_SYSTEMS.items():
        rows.append([
            system_name,
            f"{data['single_hop']:.2f}",
            f"{data['multi_hop']:.2f}",
            f"{data['temporal_reasoning']:.2f}",
            f"{data['open_domain']:.2f}",
            f"{data['overall']:.2f}",
            f"{data['overall_f1']:.2f}"
        ])
    
    rows.append([
        "**Ours**",
        f"{our_data['single_hop']:.2f}",
        f"{our_data['multi_hop']:.2f}",
        f"{our_data['temporal_reasoning']:.2f}",
        f"{our_data['open_domain']:.2f}",
        f"{our_data['overall']:.2f}",
        f"{our_data['overall_f1']:.2f}"
    ])
    
    return tabulate(rows, headers=headers, tablefmt="pipe")


def generate_latex_table(our_data: Dict) -> str:
    """Generate latex table."""
    
    latex = r"""
\begin{table*}[htbp]
\centering
\caption{Performance comparison on LoCoMo benchmark. All methods use GPT-4o-mini as the foundation LLM. 
↑ indicates higher is better. * indicates Overall without Adversarial questions.
\textbf{Bold} values indicate the best performance in each column.}
\label{tab:locomo_comparison}
\begin{tabular}{lccccccccc}
\toprule
\textbf{Method} & \textbf{Tokens} & \textbf{Single-hop} & \textbf{Multi-hop} & \textbf{Temporal} & \textbf{Open-domain} & \textbf{Adversarial} & \textbf{Overall*} & \textbf{Overall} & \textbf{F1} \\
\midrule
"""
    
    for system_name, data in BASELINE_SYSTEMS.items():
        latex += f"{system_name} & {data['tokens']} & {data['single_hop']:.2f} & {data['multi_hop']:.2f} & "
        latex += f"{data['temporal_reasoning']:.2f} & {data['open_domain']:.2f} & - & {data['overall']:.2f} & - & {data['overall_f1']:.2f} \\\\\n"
    
    latex += r"\midrule" + "\n"
    
    best_single = our_data['single_hop'] > 81.09
    best_multi = our_data['multi_hop'] > 67.49
    best_temporal = our_data['temporal_reasoning'] > 81.20
    best_open = our_data['open_domain'] > 55.90
    best_overall = our_data['overall'] > 75.80
    
    latex += r"\textbf{Ours (Three-Tower)} & " + f"~{our_data['tokens']} & "
    latex += (r"\textbf{" + f"{our_data['single_hop']:.2f}" + r"}" if best_single else f"{our_data['single_hop']:.2f}") + " & "
    latex += (r"\textbf{" + f"{our_data['multi_hop']:.2f}" + r"}" if best_multi else f"{our_data['multi_hop']:.2f}") + " & "
    latex += (r"\textbf{" + f"{our_data['temporal_reasoning']:.2f}" + r"}" if best_temporal else f"{our_data['temporal_reasoning']:.2f}") + " & "
    latex += (r"\textbf{" + f"{our_data['open_domain']:.2f}" + r"}" if best_open else f"{our_data['open_domain']:.2f}") + " & "
    latex += r"\textbf{" + f"{our_data['adversarial']:.2f}" + r"} & "
    latex += (r"\textbf{" + f"{our_data['overall']:.2f}" + r"}" if best_overall else f"{our_data['overall']:.2f}") + " & "
    latex += r"\textbf{" + f"{our_data['overall_with_adv']:.2f}" + r"} & "
    latex += f"{our_data['overall_f1']:.2f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark 结果统计与对比")
    parser.add_argument(
        "--results-dir",
        default=str(paths.LOCOMO_TASK_EVAL_RESULTS_DIR / "locomo_dual_tower_benchmark_new_dataset/all_conv_sota_11_13"),
        help="结果文件目录"
    )
    parser.add_argument(
        "--output",
        default=str(paths.LOCOMO_TASK_EVAL_DIR / "comparison_table.md"),
        help="输出文件路径"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "grid", "all"],
        default="all",
        help="输出格式"
    )
    
    args = parser.parse_args()
    
    analyzer = LoCoMoResultsAnalyzer(args.results_dir)
    
    
    count = analyzer.load_results()
    if count == 0:
        print(" 没有找到结果文件")
        return
    
    analyzer.analyze()
    analyzer.print_detailed_stats()
    
    our_data = analyzer.get_our_system_data()
    
    print("\n" + "=" * 80)
    print(" 我们系统的评分 (百分制)")
    print("=" * 80)
    for key, value in our_data.items():
        if key != "tokens":
            print(f"  {key}: {value:.2f}%")
    
    output_content = ""
    
    print("\n" + "=" * 80)
    print(" 对比表格")
    print("=" * 80)
    
    if args.format in ["markdown", "all"]:
        md_table = generate_comparison_table(our_data, "markdown")
        print("\n### Markdown 格式 ###\n")
        print(md_table)
        output_content += "# LoCoMo Benchmark 对比结果\n\n"
        output_content += "## 完整对比表格\n\n"
        output_content += md_table + "\n\n"
        output_content += "**注释:**\n"
        output_content += "- ↑ 表示越高越好\n"
        output_content += "- Overall (no Adv): 不含对抗性问题的总体评分\n"
        output_content += "- Overall (with Adv): 含对抗性问题的总体评分\n"
        output_content += "- Adversarial: 对抗性问题评分（其他系统未测试）\n\n"
        
        summary_table = generate_summary_table(our_data)
        output_content += "## 简化对比表格\n\n"
        output_content += summary_table + "\n\n"
    
    if args.format in ["latex", "all"]:
        latex_table = generate_latex_table(our_data)
        print("\n### LaTeX 格式 ###\n")
        print(latex_table)
        output_content += "## LaTeX 格式 (用于论文)\n\n```latex\n"
        output_content += latex_table + "\n```\n\n"
    
    if args.format in ["grid", "all"]:
        grid_table = generate_comparison_table(our_data, "grid")
        print("\n### Grid 格式 ###\n")
        print(grid_table)
    
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"\n 结果已保存到: {output_path}")
    
    stats_output = {
        "analysis_info": {
            "results_dir": str(args.results_dir),
            "sample_count": count
        },
        "our_system": our_data,
        "baseline_systems": BASELINE_SYSTEMS,
        "category_details": {
            k: {
                "test_count": v["test_count"],
                "avg_llm_accuracy": v["avg_llm_accuracy"],
                "avg_f1_score": v["avg_f1_score"]
            }
            for k, v in analyzer.aggregated_stats.items()
            if k not in ["overall_with_adv", "overall_no_adv"]
        }
    }
    
    stats_path = Path(args.output).with_suffix('.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f" 详细统计已保存到: {stats_path}")


if __name__ == "__main__":
    main()
