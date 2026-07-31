# LoCoMo Script Index

These scripts are convenience launchers for LoCoMo task-eval reproduction. Run
them from anywhere inside the repository; each script resolves the repository
root before launching Python modules.

| Script | Purpose |
| --- | --- |
| `check_task_eval_entrypoints.sh` | Lightweight import/`--help` check for LoCoMo task-eval modules. |
| `run_reproduction_suite.sh` | Expanded aggregate launcher for ablations, router-only runs, router+quantification+cascade runs, and speed sections. |
| `run_router_quantification_cascade_closeai.sh` | Paper-style router + quantification + cascade benchmark template using CloseAI models. |
| `run_router_quantification_cascade_expanded_closeai.sh` | Copy-friendly expanded router + quantification + cascade commands. |
| `run_router_only_closeai.sh` | Router-only baseline runs through CloseAI models. |
| `run_router_only_openrouter.sh` | Router-only baseline runs through OpenRouter models. |
| `run_quantification_ablation_expanded.sh` | Full quantification ablation matrix with all parameters expanded. |
| `run_quantification_ablation_gpt41_mini.sh` | Four quantification ablations for GPT-4.1-mini generation. |
| `run_quantification_ablation_gpt4o_mini.sh` | Four quantification ablations for GPT-4o-mini generation. |
| `run_triple_tower_ablation_closeai.sh` | Triple-tower baseline and tower-removal ablations through CloseAI models. |
| `run_triple_tower_ablation_no_baseline_closeai.sh` | Tower-removal ablations without relaunching the full baseline. |
| `run_gpt5_triple_tower_ablation.sh` | GPT-5 triple-tower generation ablations. |
| `reeval_mem0_judge_alignment_gpt4o_mini.sh` | Re-evaluate individual reports with the Mem0-style GPT-4o-mini judge prompt. |
| `run_speed_benchmarks.sh` | Serial insertion and smart-search QPS speed benchmarks. |

`env.sh` is a local legacy environment helper and is not a main reproduction
entrypoint.
