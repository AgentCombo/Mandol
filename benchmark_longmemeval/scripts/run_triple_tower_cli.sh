#!/bin/bash
# Purpose: run one configurable LongMemEval triple-tower benchmark from CLI
# arguments instead of launching the full ablation matrix.
# Runs: benchmark_longmemeval.task_eval.benchmark_triple.

# Resolve repository root for the src/ package layout.
AMS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMS_REPO_ROOT="$AMS_SCRIPT_DIR"
while [[ "$AMS_REPO_ROOT" != "/" && ! -d "$AMS_REPO_ROOT/src/mandol" ]]; do
    AMS_REPO_ROOT="$(dirname "$AMS_REPO_ROOT")"
done
if [[ ! -d "$AMS_REPO_ROOT/src/mandol" ]]; then
    echo "Could not locate Mandol repo root from $AMS_SCRIPT_DIR" >&2
    exit 1
fi
cd "$AMS_REPO_ROOT"
if [[ -d "$AMS_REPO_ROOT/.venv/bin" ]]; then
    export PATH="$AMS_REPO_ROOT/.venv/bin:$PATH"
fi
export PYTHONPATH="$AMS_REPO_ROOT/src:$AMS_REPO_ROOT:${PYTHONPATH:-}"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to run this benchmark script" >&2
    exit 1
fi

# ============================================================================
# LongMemEval 三重检索融合 Benchmark 运行脚本
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$AMS_REPO_ROOT"

cd "$PROJECT_ROOT"

echo "============================================================================"
echo "🔗 LongMemEval 三重检索融合 Benchmark"
echo "============================================================================"

# 默认参数
DATASET_SIZE="s"
SENTENCE_TOP_K=5
EPISODIC_TOP_K=5
ENTITY_TOP_K=5
RERANK_METHOD="baai"
MAX_TESTS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-size)
            DATASET_SIZE="$2"
            shift 2
            ;;
        --sentence-top-k)
            SENTENCE_TOP_K="$2"
            shift 2
            ;;
        --episodic-top-k)
            EPISODIC_TOP_K="$2"
            shift 2
            ;;
        --entity-top-k)
            ENTITY_TOP_K="$2"
            shift 2
            ;;
        --rerank-method)
            RERANK_METHOD="$2"
            shift 2
            ;;
        --max-tests)
            MAX_TESTS="--max-tests $2"
            shift 2
            ;;
        --disable-sentence)
            DISABLE_SENTENCE="--disable-sentence"
            shift
            ;;
        --disable-episodic)
            DISABLE_EPISODIC="--disable-episodic"
            shift
            ;;
        --disable-entity)
            DISABLE_ENTITY="--disable-entity"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "📊 配置:"
echo "   数据集大小: $DATASET_SIZE"
echo "   Sentence Top-K: $SENTENCE_TOP_K"
echo "   Episodic Top-K: $EPISODIC_TOP_K"
echo "   Entity Top-K: $ENTITY_TOP_K"
echo "   重排序方法: $RERANK_METHOD"
echo ""

uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size "$DATASET_SIZE" \
    --sentence-top-k "$SENTENCE_TOP_K" \
    --episodic-top-k "$EPISODIC_TOP_K" \
    --entity-top-k "$ENTITY_TOP_K" \
    --rerank-method "$RERANK_METHOD" \
    $MAX_TESTS \
    $DISABLE_SENTENCE \
    $DISABLE_EPISODIC \
    $DISABLE_ENTITY \
    "$@"

echo ""
echo "============================================================================"
echo "✅ 测试完成"
echo "============================================================================"

