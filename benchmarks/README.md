# Benchmarks

## Exact paper reproduction

The LoCoMo and LongMemEval results reported in the Mandol paper were produced
with the frozen
[`paper-repro`](https://github.com/AgentCombo/Mandol/tree/paper-repro)
artifact.

Please use the benchmark-specific instructions in that branch:

- [LoCoMo paper reproduction](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_locomo/REPRODUCE.md)
- [LongMemEval paper reproduction](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_longmemeval/REPRODUCE.md)

## Main-branch development workflows

The refactored self-host evaluation workflows previously located in this
directory have moved to:

- [`experimental/self_host_benchmarks`](../experimental/self_host_benchmarks/)

These development workflows are intended for smoke testing, integration
validation, debugging, and continued workflow development. They are not the
frozen pipelines used to produce the paper tables.
