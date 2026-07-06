# vllm启动命令
CUDA_VISIBLE_DEVICES=0 vllm serve BAAI/bge-reranker-v2-m3 \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.5