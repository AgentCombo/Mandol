# LoCoMo Dataset

The LoCoMo benchmark scripts expect the public LoCoMo10 JSON file under:

```text
benchmark_locomo/dataset/locomo/locomo10.json
```

The raw dataset is maintained by the LoCoMo authors:

- GitHub: https://github.com/snap-research/locomo
- Direct JSON: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

Download it from the repository root:

```bash
mkdir -p benchmark_locomo/dataset/locomo
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o benchmark_locomo/dataset/locomo/locomo10.json
```

Generated tower graphs are also written below `benchmark_locomo/dataset/locomo/`
by the dataset-maker scripts. The JSON dataset and generated graph artifacts are
intentionally ignored by Git; this README is the only file expected to be
tracked in this directory.
