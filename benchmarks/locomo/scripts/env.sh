#!/bin/bash

# 保存生成的输出到此位置
export OUT_DIR=./outputs

# 保存嵌入到此位置  
export EMB_DIR=./outputs

# LoCoMo数据文件路径
export DATA_FILE_PATH=./dataset/locomo/locomo10.json

# 不同输出的文件名
export QA_OUTPUT_FILE=locomo10_qa.json
export OBS_OUTPUT_FILE=locomo10_observation.json
export SESS_SUMM_OUTPUT_FILE=locomo10_session_summary.json

# 包含提示词和上下文示例的文件夹路径
export PROMPT_DIR=./prompt_examples

# API Keys
export DEEPSEEK_API_KEY="your deepseek api key here"
export OPENAI_API_KEY="your openai api key here"

# 模型配置
export DEFAULT_LLM_MODEL="deepseek-chat"
export EVALUATION_MODEL="all-MiniLM-L6-v2"

echo "环境变量已设置完成"

# # save generated outputs to this location
# OUT_DIR=./outputs

# # save embeddings to this location
# EMB_DIR=./outputs

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
# source benchmark/scripts/env.sh

# # 运行评估
# cd benchmark/task_eval
# python locomo_test.py