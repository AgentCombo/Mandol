# LongMemEval Script Index

These scripts are convenience launchers for LongMemEval task-eval reproduction.
Run them from anywhere inside the repository; each script resolves the
repository root before launching Python modules.

| Script | Purpose |
| --- | --- |
| `run_reproduction_suite.sh` | Expanded aggregate launcher for router+quantification+cascade, router-only, ablation, and GPT-5 sections. |
| `run_router_quantification_cascade_closeai.sh` | Paper-style router + quantification + cascade benchmark template using CloseAI models. |
| `run_router_quantification_cascade_expanded_closeai.sh` | Copy-friendly expanded router + quantification + cascade commands. |
| `run_router_only_closeai.sh` | Router-only baseline runs through CloseAI models. |
| `run_router_only_openrouter.sh` | Router-only baseline runs through OpenRouter models. |
| `run_triple_tower_cli.sh` | Configurable wrapper for one LongMemEval triple-tower run. |
| `run_triple_tower_ablation_closeai.sh` | Triple-tower baseline and tower-removal ablations through CloseAI models. |
| `run_triple_tower_ablation_openrouter.sh` | Triple-tower baseline and tower-removal ablations through OpenRouter models. |
| `run_quantification_ablation_expanded.sh` | Full quantification ablation matrix with all parameters expanded. |
| `run_quantification_ablation_gpt41_mini.sh` | Four quantification ablations for GPT-4.1-mini generation. |
| `run_quantification_ablation_gpt4o_mini.sh` | Four quantification ablations for GPT-4o-mini generation. |
| `run_gpt5_triple_tower_ablation.sh` | GPT-5 triple-tower and tower-removal ablations. |
| `rerun_gpt5_empty_generation_outputs.sh` | Rerun GPT-5 cases with empty generated answers into a separate output directory. |
| `reeval_mem0_judge_alignment_gpt4o_mini.sh` | Re-evaluate individual reports with the Mem0-style GPT-4o-mini judge prompt. |
| `run_speed_benchmarks.sh` | Notes that LongMemEval speed helpers are private; public latency/QPS runs use the LoCoMo speed script. |
