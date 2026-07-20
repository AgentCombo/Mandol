#!/usr/bin/env bash

set -euo pipefail

# Purpose: start the vLLM-compatible BGE reranker used by async retrieval.
CUDA_VISIBLE_DEVICES=0 vllm serve BAAI/bge-reranker-v2-m3 \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
