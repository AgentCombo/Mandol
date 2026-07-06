# LongMemEval Dataset For Self-Host Runs

The self-host LongMemEval scripts expect the cleaned small split under:

```text
benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
```

The cleaned dataset is distributed by the LongMemEval authors:

- GitHub: https://github.com/xiaowu0162/LongMemEval
- Hugging Face: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Download it from the repository root:

```bash
mkdir -p benchmark_self_host/longmemeval/dataset
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
```

If you already downloaded the paper benchmark copy, you can reuse it:

```bash
cp benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json \
  benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
```

The downloaded JSON file is intentionally ignored by Git.
