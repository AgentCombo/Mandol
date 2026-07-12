# Experimental Self-host Evaluation Workflows

> [!NOTE]
> These workflows validate the refactored `main` branch and support smoke
> testing, integration testing, debugging, and continued self-host workflow
> development.
>
> They are not the frozen benchmark pipelines used to produce the results
> reported in the Mandol paper. For exact paper reproduction, use the
> [`paper-repro`](https://github.com/AgentCombo/Mandol/tree/paper-repro)
> branch.

## Available Workflows

| Workflow | Description | Development dataset | Source |
|----------|-------------|---------------------|--------|
| [LoCoMo](locomo/) | Long-conversation memory evaluation with single-hop, multi-hop, temporal, and open-domain questions | Bundled development/evaluation copy of `locomo10.json` | [GitHub](https://github.com/snap-research/locomo) |
| [LongMemEval](longmemeval/) | Long-term memory evaluation across session, preference, temporal, multi-session, and knowledge-update questions | `longmemeval_s_cleaned.json` downloaded separately | [Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |

## Pipeline

Both workflows follow the same four-stage development pipeline:

```text
build_graph -> retrieve -> generate -> evaluate
```

| Step | Description |
|------|-------------|
| `build_graph.py` | Load the development dataset, construct the semantic graph, and build high-level memories |
| `retrieve.py` | Execute retrieval queries against the built graph |
| `generate.py` | Generate answers from the retrieved context |
| `evaluate.py` | Score generated answers against the reference answers with an LLM judge |

Stages exchange JSON files on disk so they can be resumed, inspected, and run
independently. `run.py` provides the end-to-end entry point.

## Smoke Tests

Run these commands from the repository root:

```bash
python experimental/self_host_benchmarks/locomo/run.py \
  --smoke \
  --config experimental/self_host_benchmarks/locomo/configs/base.yaml

python experimental/self_host_benchmarks/longmemeval/run.py \
  --smoke \
  --config experimental/self_host_benchmarks/longmemeval/configs/base.yaml
```

Each smoke test uses a small data subset and performs build, retrieval,
generation, evaluation, and a persistence round trip without spawning the
stage scripts. Provider credentials and model access are still required.

## Full Development Evaluation

Run from the selected workflow directory so config, dataset, and output paths
remain local to that workflow:

```bash
cd experimental/self_host_benchmarks/locomo
python run.py --config configs/base.yaml --output output/
```

Use `experimental/self_host_benchmarks/longmemeval` for LongMemEval. To rebuild
all stages or run only selected stages:

```bash
python run.py --config configs/base.yaml --output output/ --force
python run.py --config configs/base.yaml --stages build,retrieve
```

### Step-by-step execution

```bash
python build_graph.py --config configs/base.yaml --output output/
python retrieve.py --config configs/base.yaml --output output/
python generate.py --config configs/base.yaml --output output/
python evaluate.py --config configs/base.yaml --output output/
```

Add `--force` to a stage to replace an existing stage result.

## Configuration

The workflows use two configuration layers:

1. Repository-level environment variables in [`.env.example`](../../.env.example)
   define provider connectivity, API keys, model endpoints, timeouts, and retries.
2. Workflow-local YAML files under `locomo/configs/` and
   `longmemeval/configs/` define sample selection, retrieval settings,
   generation parameters, storage behavior, and output locations.

Environment variables take precedence over corresponding YAML provider values.
Relative dataset and output paths in a YAML file are interpreted from the
selected workflow directory by the end-to-end runner.

## Output Files

A complete development run writes the following structure:

```text
output/<config_name>/
├── build_stats.json
├── evaluation_summary.json
├── evaluation_report.txt
└── <sample_id-or-question_id>/
    ├── build.json
    ├── retrieval.json
    ├── generation.json
    ├── evaluation.json
    └── graph/
```

The per-sample JSON files preserve stage outputs for resume and debugging. The
graph directory contains the persisted `MemorySystem` state used by later
stages.

## Current Validation Environment

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon Platinum 8458P |
| RAM | 120 GB |
| GPU | NVIDIA H800 80 GB |
| Python | 3.10.12 |
| OS | Ubuntu 22.04 LTS |

Results can vary across hardware, dependency versions, and model-provider
versions. This is the environment currently used to validate these development
workflows; it is not a claim that the frozen paper results are reproduced here.

## Relationship to the Paper Artifact

The workflow in this directory has been refactored for self-hosted and
general-purpose use. Its prompts, graph-construction pipeline, retrieval
stages, model assignments, configuration defaults, and evaluation process may
differ from the frozen paper artifact.

For comparisons against the published paper tables, use the corresponding
workflow in the [`paper-repro`](https://github.com/AgentCombo/Mandol/tree/paper-repro)
branch:

- [LoCoMo paper reproduction](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_locomo/REPRODUCE.md)
- [LongMemEval paper reproduction](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_longmemeval/REPRODUCE.md)
