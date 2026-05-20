# Baseline Methods

This directory documents the baseline memory systems compared in the LoCoMo benchmark paper. Implementation code for these external systems is maintained separately.

## Compared Systems

| System | Description | Avg. Tok. | Reference |
|--------|-------------|-----------|-----------|
| **EverMemOS** | Conversational memory OS with hierarchical episodic summarization | 2.3–2.5k | [Paper](https://arxiv.org/abs/2501.01009) |
| **Zep** | Production memory platform with knowledge graph and entity extraction | 1.4k | [GitHub](https://github.com/getzep/zep) |
| **MemOS** | Memory-augmented OS framework with dual-tower retrieval | 2.5k | [Paper](https://arxiv.org/abs/2405.16407) |
| **Mem0** | Lightweight memory layer with vector store and graph store | 1.0k | [GitHub](https://github.com/mem0ai/mem0) |
| **MemU** | Memory unification framework with multimodal fusion | 4.0k | [Paper](https://arxiv.org/abs/2409.10542) |

## Reproduction

To reproduce results for these baselines, refer to each system's official repository and documentation. The evaluation prompt template and judge prompt used in our comparison are available in [pipeline_utils.py](../pipeline_utils.py) (`GENERATION_PROMPT_TEMPLATE` and `EVALUATION_PROMPT_TEMPLATE`).

The LoCoMo dataset is available at `../data/locomo10.json`.

## Expected Results

See the main [README.md](../README.md) for full comparison tables across GPT-4o-mini and GPT-4.1-mini backbones.
