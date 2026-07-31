# LongMemEval Dataset

The LongMemEval benchmark scripts expect cleaned LongMemEval files under:

```text
benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json
benchmark_longmemeval/dataset/LongMemEval/longmemeval_m_cleaned.json
```

The cleaned dataset is distributed by the LongMemEval authors:

- GitHub: https://github.com/xiaowu0162/LongMemEval
- Hugging Face: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Download the small split used by the paper reproduction commands:

```bash
mkdir -p benchmark_longmemeval/dataset/LongMemEval
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json
```

Download the medium split only if you plan to run `--dataset-size m`:

```bash
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_m_cleaned.json
```

Generated tower graphs are also written below
`benchmark_longmemeval/dataset/LongMemEval/` by the dataset-maker scripts. The
JSON dataset and generated graph artifacts are intentionally ignored by Git; this
README is the only file expected to be tracked in this directory.
