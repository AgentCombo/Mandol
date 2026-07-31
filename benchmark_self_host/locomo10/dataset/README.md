# LoCoMo10 Dataset For Self-Host Runs

The self-host LoCoMo10 scripts expect the public LoCoMo10 JSON file under:

```text
benchmark_self_host/locomo10/dataset/locomo10.json
```

The raw dataset is maintained by the LoCoMo authors:

- GitHub: https://github.com/snap-research/locomo
- Direct JSON: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

Download it from the repository root:

```bash
mkdir -p benchmark_self_host/locomo10/dataset
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o benchmark_self_host/locomo10/dataset/locomo10.json
```

If you already downloaded the paper benchmark copy, you can reuse it:

```bash
cp benchmark_locomo/dataset/locomo/locomo10.json \
  benchmark_self_host/locomo10/dataset/locomo10.json
```

The downloaded JSON file is intentionally ignored by Git.
