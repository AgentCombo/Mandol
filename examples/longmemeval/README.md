# LongMemEval Example — Long-Text Memory Evaluation Demonstration

This example is based on the LongMemEval benchmark dataset and demonstrates the Mandol memory system's capabilities in long-document information retention and precise retrieval.

## Overview

| Item | Description |
|------|------|
| Data Source | LongMemEval benchmark (HuggingFace: `xiaowu0162/longmemeval-cleaned`) |
| Selected Sample | A complete article on ML in healthcare: history & future |
| Text Length | 468 words (~3,200 characters) |
| QA Pairs | 12 questions covering all 6 query categories |
| Data Size | ~8 KB |

## Dataset Structure

Each LongMemEval record contains a long article with corresponding QA pairs:

```json
{
  "sample_id": "lme-001",
  "title": "Article Title",
  "passage": "Full article body (hundreds to thousands of words)...",
  "qa": [
    {
      "question": "...",
      "answer": "...",
      "evidence": ["Supporting text snippet from the passage"],
      "category": "SS-Pref"
    }
  ],
  "metadata": {
    "word_count": 468,
    "source_type": "...",
    "domain": "..."
  }
}
```

### Six Query Categories

LongMemEval defines the following six categories to evaluate different long-context memory skills:

| Category | Full Name | Description | Count in This Example |
|------|------|------|-----------|
| SS-Pref | Single-Session Preference | Fact retrieval based on user preference | 2 |
| SS-Asst | Single-Session Assistant | Precise information extraction | 2 |
| Temporal | Temporal Reasoning | Questions involving time-based relationships | 2 |
| Multi-S | Multi-Session | Cross-session information synthesis | 2 |
| Know.Upd. | Knowledge Update | Tracking value changes and knowledge updates | 2 |
| SS-User | Single-Session User | User-specific detail retention | 2 |

## Directory Structure

```
longmemeval/
├── README.md                           # This file
├── data/
│   └── longmemeval_example.json        # Example data (synthetic article)
├── config.yaml                         # System configuration file
├── .env.template                       # Environment variable template
├── run_example.py                      # Main execution script
└── download_data.py                    # HuggingFace data download script
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install -e .
# To download the full dataset from HuggingFace:
pip install huggingface_hub
```

### 2. Configure API Keys

```bash
cp .env.template .env
# Edit the .env file to fill in your API keys
```

Required variables:

| Variable | Description | Example |
|------|------|------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-xxx` |
| `LLM_BASE_URL` | LLM API address | `https://api.openai.com/v1` |
| `LLM_MODEL` | Model to use | `gpt-4o-mini` |

### 3. Data Preparation

This example includes a built-in synthetic data file `data/longmemeval_example.json` that works out of the box.

To use the full dataset:

```bash
python download_data.py
```

This script downloads `xiaowu0162/longmemeval-cleaned` from HuggingFace into the `data/` directory.

## Running the Example

### Method 1: Use Built-in Synthetic Data (Ready to Run)

```bash
cd examples/longmemeval
python run_example.py
```

### Method 2: Use Full HuggingFace Data

```bash
python download_data.py                   # Download data first
python run_example.py --data-dir data/    # Run with downloaded data
```

### Method 3: Custom Query

```bash
python run_example.py --query "How many AI tools were deployed during the pandemic?"
```

## Parameter Reference

| Parameter | Description | Default |
|------|------|--------|
| `--data` | Data file or directory path | `data/longmemeval_example.json` |
| `--query` | Custom query text | Uses preset queries |
| `--top-k` | Number of results returned | `5` |
| `--no-rerank` | Disable reranking | `False` |

## System Configuration Guide

### config.yaml Key Parameters

```yaml
llm:
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"

embedder:
  model: "Qwen/Qwen3-Embedding-4B"
  device: "cpu"
  dimension: 2560

system:
  chunk_max_tokens: 512       # Text chunk size
  session_time_gap_seconds: 1800
  similarity_top_k: 5
  similarity_threshold: 0.7
  bfs_expansion_per_seed: 3
  bfs_expansion_hops: 1
```

## Data Format Adaptation Guide

### LongMemEval Article → Mandol MemoryUnit Conversion

The original article is a complete long text and requires the following adaptation steps:

**Step 1: Text Chunking**

Mandol uses `DocumentChunker` to split the long article into multiple chunks. Each chunk is stored as an independent `MemoryUnit`.

Configuration: `chunk_max_tokens: 512` (up to 512 tokens per chunk)

**Step 2: Building MemoryUnits**

```python
# Raw data
passage = "Machine learning has fundamentally transformed..."

# Chunk and build MemoryUnits
for chunk in chunker.chunk(passage):
    unit = MemoryUnit(
        uid=Uid(f"lme-001_chunk_{chunk_index}"),
        raw_data={
            "text_content": chunk.text,
            "chunk_index": chunk_index,
            "title": "The History and Future of ML in Healthcare",
        },
        metadata={
            "unit_type": "document_chunk",
            "sample_id": "lme-001",
            "source": "longmemeval",
        },
    )
    system.add(unit)
```

**Step 3: Building High-Level Memories**

Call `system.build_high_level(mode="auto")` to automatically extract:
- Entities (e.g., OsteoDetect, NHS, MedBERT, WHO)
- Events (e.g., FDA clearance process, AI Lab launch)
- Causal relationships (e.g., COVID-19 → accelerated AI adoption)
- Summaries (article topic overview)

**Step 4: Retrieval Queries**

Use `system.holistic_retrieve(query, top_k=5)` to perform semantic retrieval, returning the memory units most relevant to the query.

## Expected Output

### Retrieval Result Example

```
Query: What was the first FDA-approved AI diagnostic tool?
───────────────────────────────────────────────────────
[0.947] lme-001_chunk_3 | The first FDA-approved AI diagnostic tool, OsteoDetect...
[0.892] lme-001_entity_OsteoDetect | Entity: OsteoDetect - First FDA-approved AI...
[0.834] lme-001_summary_overview | Summary: The passage describes the evolution of...
```

### Memory Statistics Example

```
Memory Stats:
  Total units: 8 (chunks + entities + events)
  Processed sample IDs: ['lme-001']
  Document chunks: 4
  Extracted entities: 6
  Extracted events: 3
```

### Per-Category Evaluation Example

```
Category Breakdown:
  SS-Pref:      2/2 correct (100.0%)
  SS-Asst:      2/2 correct (100.0%)
  Temporal:     2/2 correct (100.0%)
  Multi-S:      2/2 correct (100.0%)
  Know.Upd.:    2/2 correct (100.0%)
  SS-User:      2/2 correct (100.0%)
  ─────────────────────────────
  Total:       12/12 correct (100.0%)
```

## Typical Application Scenarios

| Scenario | Text Type | Core Challenge | Demonstrated By This Example |
|------|----------|---------|-----------|
| Medical literature understanding | Long papers/reports | Domain terms + long-range dependencies | Numerical accuracy tracking |
| Legal document analysis | Contracts/judgments | Precise information extraction | Detail vs. overview |
| Customer service knowledge base | Product documentation | Fragment information synthesis | Cross-paragraph linking |
| News summarization | Long reports | Timeline reasoning | Temporal category |

## Dependencies

This example depends on the following components:

- **Mandol Core Library**: `pip install -e .`
- **Python ≥ 3.10**
- **LLM API Access**: Requires configuration of an OpenAI-compatible API key
- **Local Models** (optional): `Qwen3-Embedding-4B` (~8 GB), `Qwen3-Reranker-4B` (~8 GB)
- **Full Data Download** (optional): `huggingface_hub` package
- **Recommended Hardware**: 8 GB RAM, 16 GB+ for a better experience

## Full Dataset Download Guide

The synthetic data included in this example is for format and workflow demonstration only. To evaluate with the complete dataset:

### Download Command

```bash
python download_data.py
```

This script uses `huggingface-cli` or `git lfs` to download the complete dataset from HuggingFace.

### Dataset Information

| Property | Value |
|------|-----|
| Repository | `xiaowu0162/longmemeval-cleaned` |
| Format | JSON / CSV |
| Total articles | ~500 |
| Total QA pairs | ~6,000 |
| Average article length | 800–2,000 words |
| License | See HuggingFace repository |

## Troubleshooting

| Issue | Likely Cause | Solution |
|------|---------|---------|
| Data file not found | Full dataset not downloaded | Use built-in synthetic data or run `python download_data.py` |
| API call failure | Incorrect key configuration | Verify the API key in the `.env` file |
| Poor chunking results | chunk_max_tokens mismatch | Adjust `chunk_max_tokens` based on article length |
| Low retrieval precision | Model does not support domain terms | Consider fine-tuning the embedder or using a domain-specific model |
