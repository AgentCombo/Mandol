# LoCoMo10 Self-Host Reproduction Scripts

Run from the repository root or from this directory. Every script resolves the repository root and exports `PYTHONPATH=<repo>/src:<repo>` automatically.

## One-Click Run

```bash
GENERATION_MODEL=gpt-4o-mini-closeai \
JUDGE_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/locomo10/scripts/run_all.sh
```

Supported generation models: `gpt-4o-mini-closeai`, `gpt-4o-mini-openrouter`, `gpt-4.1-mini-closeai`, `gpt-4.1-mini-openrouter`.

Supported judge models: `gpt-4o-mini-closeai`, `gpt-4o-mini-openrouter`.

By default, outputs go to:

```text
benchmark_self_host/locomo10/test_runs/<timestamp>__gen-<generation-model>__judge-<judge-model>/
```

## Resume By Stage

Set the same `RUN_ROOT` and run the stage you need:

```bash
RUN_ROOT=benchmark_self_host/locomo10/test_runs/20260519_120000__gen-gpt-4o-mini-closeai__judge-gpt-4o-mini-closeai \
bash benchmark_self_host/locomo10/scripts/run_retrieve.sh

RUN_ROOT=benchmark_self_host/locomo10/test_runs/20260519_120000__gen-gpt-4o-mini-closeai__judge-gpt-4o-mini-closeai \
GENERATION_MODEL=gpt-4.1-mini-openrouter \
bash benchmark_self_host/locomo10/scripts/run_generate.sh

RUN_ROOT=benchmark_self_host/locomo10/test_runs/20260519_120000__gen-gpt-4o-mini-closeai__judge-gpt-4o-mini-closeai \
JUDGE_MODEL=gpt-4o-mini-openrouter \
bash benchmark_self_host/locomo10/scripts/run_score.sh
```

## Useful Variables

- `SAMPLE_IDS="conv-26 conv-30"`: limit to specific samples.
- `LIMIT=10`: build at most N samples.
- `MAX_QUESTIONS=20`: limit retrieve/generate/score questions.
- `FORCE=1`, `SKIP_EXISTING=1`, `NO_RESUME=1`: pass common build controls.
- `BUILD_EXTRA_ARGS`, `RETRIEVE_EXTRA_ARGS`, `GENERATE_EXTRA_ARGS`, `SCORE_EXTRA_ARGS`: append advanced CLI flags without editing scripts.
- `DRY_RUN=1`: dry-run generate/score stages.
- `JUDGE_WORKERS=4`: concurrent judge calls.