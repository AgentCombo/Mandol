# LoCoMo Example — Long Conversational Memory Demonstration

This example is based on the LoCoMo (Long Conversational Memory) benchmark dataset and demonstrates the core capabilities of the Mandol memory system in multi-turn long conversation scenarios.

## Overview

| Item | Description |
|------|------|
| Data Source | LoCoMo benchmark dataset `locomo10.json` |
| Selected Sample | `conv-26`: 19 cross-time dialogues between Caroline & Melanie |
| Dialogue Scale | 19 sessions, 419 dialogue turns |
| QA Pairs | 199 questions covering all 5 query categories |
| Data Size | ~213 KB |

## Dataset Structure

The conv-26 sample has the following structure:

```
{
  "sample_id": "conv-26",
  "conversation": {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      {"speaker": "Caroline", "dia_id": "D1:1", "text": "..."},
      {"speaker": "Melanie", "dia_id": "D1:2", "text": "..."},
      ...
    ],
    "session_1_summary": "...",
    ...
  },
  "qa": [
    {"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 2},
    ...
  ]
}
```

### Five Query Categories

| Category # | Name | Description | Count in This Example |
|----------|----------|------|-----------|
| 1 | Single-hop | Direct fact retrieval from a single dialogue turn | 32 |
| 2 | Temporal | Questions involving time-based ordering or recency | 37 |
| 3 | Multi-hop | Reasoning across multiple dialogue turns or sessions | 13 |
| 4 | Open-domain | Broad questions requiring comprehensive memory synthesis | 70 |
| 5 | Adversarial | Questions designed to test anti-misleading capability | 47 |

## Directory Structure

```
locomo/
├── README.md                  # This file
├── data/
│   └── locomo_example_conv26.json  # Extracted subset data
├── config.yaml                # System configuration file
├── .env.template              # Environment variable template
├── config.py                  # Python configuration module
├── locomo_memory_system.py    # Memory system wrapper
├── __init__.py                # Package initialization
├── run_example.py             # Main execution script (recommended)
└── run_tests.py               # Test runner
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install -e .
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

### 3. Data File Path

Ensure the data file exists:

```bash
ls data/locomo_example_conv26.json
```

## Running the Example

### Method 1: Main Example Script (Recommended)

```bash
cd examples/locomo
python run_example.py
```

This script performs the following steps:
1. Initialize the memory system (load Embedder, Reranker, LLM)
2. Load conv-26 dialogue data
3. Add dialogues to the memory system one by one
4. Build high-level memories (entities, events, summaries, insights)
5. Execute retrieval queries and display results

### Method 2: Custom Run

```bash
python run_example.py --mode demo      # Quick demo (process first 3 sessions)
python run_example.py --mode full      # Full run (process all 19 sessions)
python run_example.py --query "What is Caroline's identity?"  # Custom query
```

### Method 3: Run Tests

```bash
python run_tests.py --sample-count 1
```

## Parameter Reference

| Parameter | Description | Default |
|------|------|--------|
| `--mode` | Run mode: `demo` / `full` | `demo` |
| `--query` | Custom query text | Uses preset queries |
| `--top-k` | Number of results returned | `5` |
| `--no-rerank` | Disable reranking | `False` |

## System Configuration Guide

### config.yaml Key Parameters

```yaml
llm:
  model: "gpt-4o-mini"        # LLM model
  base_url: "https://api.openai.com/v1"

embedder:
  model: "Qwen/Qwen3-Embedding-4B"  # Embedding model
  device: "cpu"                      # Execution device
  dimension: 2560

system:
  chunk_max_tokens: 512       # Maximum tokens per text chunk
  session_time_gap_seconds: 1800  # Session gap threshold (30 minutes)
  similarity_top_k: 5         # Similarity retrieval top-k
  similarity_threshold: 0.7   # Similarity edge threshold
  bfs_expansion_per_seed: 3   # BFS expansion seeds per node
  bfs_expansion_hops: 1       # BFS expansion hop count
```

## Data Format Adaptation Guide

### LoCoMo Raw Data → Mandol MemoryUnit Conversion

Original dialogue entry format:
```json
{
  "speaker": "Caroline",
  "dia_id": "D1:3",
  "text": "I went to a LGBTQ support group yesterday..."
}
```

Adapted MemoryUnit format:
```python
MemoryUnit(
    uid=Uid("conv-26_dialogue_D1:3"),
    raw_data={
        "type": "dialogue",
        "dia_id": "D1:3",
        "speaker": "Caroline",
        "text": "I went to a LGBTQ support group yesterday...",
        "text_content": "Dialogue D1:3 [Time 1:56 pm on 8 May, 2023]: Caroline said: I went to...",
        "session_number": 1,
        "session_datetime": "1:56 pm on 8 May, 2023",
    },
    metadata={
        "unit_type": "dialogue",
        "sample_id": "conv-26",
        "session_number": 1,
    },
)
```

The adaptation is performed by the `_process_sample()` method in `locomo_memory_system.py`. Core logic:
1. Iterate over each `session_N` in `conversation`
2. Extract `speaker`, `dia_id`, `text` fields from each dialogue entry
3. Assemble `text_content` (includes time, speaker, content)
4. Create `MemoryUnit` and set metadata
5. Build `PRECEDES`/`FOLLOWS` temporal relationship edges in order

## Expected Output

### Retrieval Result Example

```
Query: What is Caroline's identity?
───────────────────────────────────
[0.952] conv-26_dialogue_D1:5 | Caroline: The transgender stories were so inspiring!
[0.891] conv-26_dialogue_D1:7 | Caroline: The support group has made me feel accepted...
[0.845] conv-26_entity_Caroline | Entity: Caroline - Transgender woman, live in...
```

### Memory Statistics Example

```
Memory Stats:
  Total units: 419
  Processed sample IDs: ['conv-26']
  Spaces:
    conv-26_session_1: 22 units
    conv-26_session_2: 24 units
    ...
```

## Dependencies

This example depends on the following components:

- **Mandol Core Library**: `pip install -e .` (install from project root)
- **Python ≥ 3.10**
- **LLM API Access**: Requires configuration of an OpenAI-compatible API key
- **Local Models** (optional): If not using remote APIs, download Qwen3-Embedding-4B and Qwen3-Reranker-4B models
- **Recommended Hardware**: At least 8 GB RAM, 16 GB+ for a better experience

## Troubleshooting

| Issue | Likely Cause | Solution |
|------|---------|---------|
| `FileNotFoundError` | Incorrect data file path | Check the JSON file in the `data/` directory |
| API call failure | API key not correctly configured | Verify `OPENAI_API_KEY` in the `.env` file |
| Model download failure | Network issues | Set HuggingFace mirror or switch to remote API |
| Insufficient memory | Processing all 19 sessions | Use `--mode demo` to reduce the processing load |
