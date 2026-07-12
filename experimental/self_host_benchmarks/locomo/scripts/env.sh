#!/bin/bash

_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_WORKFLOW_DIR="$(cd -- "${_SCRIPT_DIR}/.." && pwd)"

# 保存生成的输出到此位置
export OUT_DIR="${_WORKFLOW_DIR}/output"

# 保存嵌入到此位置
export EMB_DIR="${_WORKFLOW_DIR}/output"

# LoCoMo数据文件路径
export DATA_FILE_PATH="${_WORKFLOW_DIR}/data/locomo10.json"

# 不同输出的文件名
export QA_OUTPUT_FILE=locomo10_qa.json
export OBS_OUTPUT_FILE=locomo10_observation.json
export SESS_SUMM_OUTPUT_FILE=locomo10_session_summary.json

# 包含提示词和上下文示例的文件夹路径
export PROMPT_DIR="${_WORKFLOW_DIR}/prompt_examples"

# API Keys
export DEEPSEEK_API_KEY="your deepseek api key here"
export OPENAI_API_KEY="your openai api key here"

# 模型配置
export DEFAULT_LLM_MODEL="deepseek-chat"
export EVALUATION_MODEL="all-MiniLM-L6-v2"

echo "LoCoMo development environment variables configured"

# # save generated outputs to this location
# OUT_DIR=./output

# # save embeddings to this location
# EMB_DIR=./output

# # path to LoCoMo data file
# DATA_FILE_PATH=./data/locomo10.json

# # filenames for different outputs
# QA_OUTPUT_FILE=locomo10_qa.json
# OBS_OUTPUT_FILE=locomo10_observation.json
# SESS_SUMM_OUTPUT_FILE=locomo10_session_summary.json

# # path to folder containing prompts and in-context examples
# PROMPT_DIR=./prompt_examples

# # DEEPSEEK API Key
# export DEEPSEEK_API_KEY=""

# # 设置环境变量
# source experimental/self_host_benchmarks/locomo/scripts/env.sh

# # 运行开发评测
# cd experimental/self_host_benchmarks/locomo
# python run.py --smoke --config configs/base.yaml
