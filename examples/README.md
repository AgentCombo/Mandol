# Mandol Examples

This directory contains examples for the Mandol memory system, helping users get started quickly and understand the system's core capabilities.

## Directory Overview

```
examples/
├── README.md                          # This file — overview and navigation
│
├── quick_start.py                     # [Getting Started] Quick start example
│
├── dialogue_demo/                     # [Scenario] Multi-session dialogue demo
│   ├── README.md
│   ├── run_demo.py
│   └── demo_data.json
│
├── knowledge_base/                    # [Scenario] Enterprise knowledge base Q&A
│   └── run_knowledge_base.py
│
├── personal_assistant/                # [Scenario] Personal assistant long-term memory
│   └── run_personal_assistant.py
│
├── customer_support/                  # [Scenario] Customer support memory system
│   └── run_customer_support.py
│
├── custom_provider/                   # [Advanced] Custom provider integration
│   └── .gitkeep
│
├── locomo/                            # [Benchmark] LoCoMo dataset example
│   ├── README.md                      #    Detailed usage guide
│   ├── data/
│   │   └── locomo_example_conv26.json #    Subset data (1 complete conversation group)
│   ├── config.yaml                    #    System configuration
│   ├── .env.template                  #    Environment variable template
│   ├── config.py                      #    Python configuration module
│   ├── locomo_memory_system.py        #    Memory system wrapper layer
│   ├── __init__.py                    #    Package initialization
│   ├── run_example.py                 #    Main execution script
│   └── run_tests.py                   #    Test runner
│
└── longmemeval/                       # [Benchmark] LongMemEval dataset example
    ├── README.md                      #    Detailed usage guide
    ├── data/
    │   └── longmemeval_example.json   #    Example data (1 complete article)
    ├── config.yaml                    #    System configuration
    ├── .env.template                  #    Environment variable template
    ├── run_example.py                 #    Main execution script
    └── download_data.py               #    Full dataset download tool
```

## Quick Navigation

### Getting Started

| Example | File | Target Audience | Estimated Time |
|------|------|---------|---------|
| Quick Start | `quick_start.py` | First-time Mandol users | < 1 min |
| Dialogue Demo | `dialogue_demo/` | Understanding multi-session processing | 2 min |

```bash
# One-click experience
python quick_start.py
cd dialogue_demo && python run_demo.py
```

### Scenario Applications

| Example | File | Core Capability |
|------|------|---------|
| Knowledge Base Q&A | `knowledge_base/` | Document import → semantic retrieval |
| Personal Assistant | `personal_assistant/` | Cross-session memory → comprehensive query |
| Customer Support | `customer_support/` | User info tracking → context awareness |

### Benchmark Evaluations

| Example | File | Dataset |
|------|------|--------|
| LoCoMo Long Conversations | `locomo/` | conv-26 (19 sessions, 199 QA pairs) |
| LongMemEval Long Text | `longmemeval/` | 12 QA pairs (with full evaluation) |

## Example Details

### quick_start.py — Quick Start

A minimal example demonstrating the three core Mandol operations:

```python
system = MemorySystem()                          # 1. Initialize
system.add(MemoryUnit(...))                     # 2. Add memory
hits = system.holistic_retrieve("query...")     # 3. Retrieve
```

### dialogue_demo — Multi-Session Dialogue

Uses a small dialogue dataset (10 turns, 2 sessions) to demonstrate:
- Loading dialogue data from a JSON file
- Adding memory units across sessions
- Building high-level memories
- Performing semantic retrieval

### knowledge_base — Knowledge Base Q&A

Imports enterprise documents into the memory system, demonstrating:
- Batch import via `MemoryUnit`
- Using the `add_many()` method
- Handling synonym/near-synonym retrieval (e.g., "work from home" → "remote work policy")

### personal_assistant — Personal Assistant

Simulates cross-time user interaction memory, demonstrating:
- Automatic session segmentation (based on time intervals)
- Cross-session memory retrieval
- Multi-view retrieval (knowledge view, event view, etc.)

### customer_support — Customer Support

Showcases user information memory and tracking in customer service scenarios, demonstrating:
- User context maintenance
- Historical issue tracing
- Preference memory and application

### locomo — LoCoMo Benchmark Example

A complete demonstration based on the LoCoMo benchmark dataset:

| Property | Description |
|------|------|
| Selected Sample | conv-26: Caroline & Melanie |
| Scale | 19 sessions, 419 dialogue turns, 199 QA pairs |
| Covered Categories | Single-hop / Multi-hop / Temporal / Open-domain / Adversarial |
| Data Size | ~213 KB |

Detailed documentation: [locomo/README.md](locomo/README.md)

### longmemeval — LongMemEval Benchmark Example

A complete demonstration based on the LongMemEval benchmark:

| Property | Description |
|------|------|
| Selected Sample | Article on ML in healthcare history & future |
| Scale | 468-word article, 12 QA pairs |
| Covered Categories | SS-Pref / SS-Asst / Temporal / Multi-S / Know.Upd. / SS-User |
| Data Size | ~8 KB |

Detailed documentation: [longmemeval/README.md](longmemeval/README.md)

## Requirements

### General Dependencies

All examples require the following base environment:

```bash
# Install Mandol
pip install -e .

# Python version requirement
python --version  # >= 3.10
```

### API Key Configuration

Most examples require LLM API access. In each example directory:

```bash
cp .env.template .env
# Edit .env to fill in OPENAI_API_KEY
```

### Hardware Recommendations

| Mode | Minimum | Recommended |
|------|---------|---------|
| CPU-only (local models) | 8 GB RAM | 16 GB RAM |
| Remote API mode | 4 GB RAM | 8 GB RAM |
| GPU-accelerated (local models) | 12 GB VRAM | 24 GB VRAM |

## Example Selection Guide

Choose the appropriate example based on your needs:

| Need | Recommended Example |
|------|---------|
| First contact | `quick_start.py` |
| Understanding the full memory pipeline | `dialogue_demo/` |
| Testing system performance on long conversations | `locomo/` |
| Testing system performance on long text | `longmemeval/` |
| Integrating into a knowledge base product | `knowledge_base/` |
| Integrating into a personal assistant product | `personal_assistant/` |
| Integrating into a customer support system | `customer_support/` |
| Custom hardware/backend integration | `custom_provider/` |
