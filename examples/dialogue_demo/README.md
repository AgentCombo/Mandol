# Dialogue Demo

A slightly more complete example demonstrating Mandol's memory system with a small multi-session dialogue dataset.

## What This Demo Shows

- Loading dialogue data from a JSON file
- Adding dialogues across multiple sessions
- Building high-level memories (entities, events, summaries)
- Performing holistic retrieval queries

## Data

The demo uses `demo_data.json`, which contains 10 dialogue turns across 2 sessions between Alice and Bob, covering topics like:
- Alice's move to Shanghai and new job
- Alice's visit to the Bund
- Alice's work progress at DataFlow AI

## Usage

```bash
python run_demo.py
```

## Requirements

- Mandol installed (`pip install -e .`)
- LLM API key configured (set `OPENAI_API_KEY` in `.env` or environment)
