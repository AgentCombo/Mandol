# 情景记忆塔 (Episodic Memory Tower)

## 概述

情景记忆塔是LoCoMo记忆系统三塔架构的核心组件之一，专门用于处理事件级别的记忆检索。

## 三塔协同设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     LoCoMo 三塔记忆系统                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   实体关系塔     │    分层塔        │      情景记忆塔              │
│  (Entity Tower) │ (Hierarchy Tower)│   (Episodic Tower)          │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ • 实体抽取      │ • L0: 对话节点   │ • 事实抽取                   │
│ • 关系生成      │ • L1: 会话摘要   │ • 时间锚定                   │
│ • 图谱构建      │ • L2: 样本洞见   │ • 多维索引                   │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ 解决问题类型:    │ 解决问题类型:    │ 解决问题类型:                 │
│ - Who is X?     │ - Overall?      │ - When did X happen?        │
│ - X和Y的关系?   │ - Summary?      │ - What did X do?            │
│ - X认识谁?      │ - Theme?        │ - How many times?           │
│                 │ - Key events?    │ - What specifically?        │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 针对失败QA的优化设计

### 失败QA类型分析

| 问题类型 | 示例 | 失败原因 | 解决方案 |
|---------|------|---------|---------|
| 时间精确度 | "Thursday before December 17, 2023" | 时间计算错误 | 双重时间保存 (原文+绝对) |
| 多跳聚合 | "How many charity tournaments?" | 信息分散 | 累积事实自动生成 |
| 细节遗漏 | "What food did X recommend?" | 具体名称丢失 | 细节字段专门保存 |
| 推理判断 | "Was X feeling lonely?" | 缺少状态追踪 | 状态变化时间线 |

### 事实类型设计

```python
class FactType:
    # 核心事件
    EVENT = "EVENT"              # 离散事件
    STATE_CHANGE = "STATE_CHANGE" # 状态变化
    ACTIVITY = "ACTIVITY"        # 活动
    PLAN = "PLAN"               # 计划/意图
    ACHIEVEMENT = "ACHIEVEMENT"  # 成就
    
    # 信息类型
    RECOMMENDATION = "RECOMMENDATION"  # 推荐/建议
    OPINION = "OPINION"          # 观点
    PREFERENCE = "PREFERENCE"    # 偏好
    
    # 关系类型
    RELATIONSHIP = "RELATIONSHIP" # 关系
    INTERACTION = "INTERACTION"   # 互动
    
    # 细节类型
    POSSESSION = "POSSESSION"    # 拥有
    ATTRIBUTE = "ATTRIBUTE"      # 属性
    NUMERICAL = "NUMERICAL"      # 数值
```

## 处理流程

### Step 1: 情景事实抽取 (step1_extract_episodic_facts.py)

从对话中抽取可回答问题的事实单元。

**核心功能:**
- 逐Session处理对话
- LLM驱动的事实抽取
- 时间归一化 (相对时间→绝对日期)
- 多维检索关键词生成

**输出格式:**
```json
{
    "fact_id": "conv-49_session_1_f0",
    "content": "Sam fell in love with a Canadian woman towards the end of summer 2023.",
    "fact_type": "STATE_CHANGE",
    "participants": ["Sam", "Canadian woman"],
    "time": {
        "original_text": "towards the end of summer",
        "absolute_start": "2023-08-15",
        "absolute_end": "2023-09-01",
        "reference_date": "2023-09-08"
    },
    "details": {
        "what": "fell in love",
        "with_whom": "Canadian woman"
    },
    "retrieval_keys": ["Sam love", "Canadian woman", "summer 2023"]
}
```

### Step 2: 事实去重与增强 (step2_deduplicate_and_enhance.py)

合并重复事实，生成聚合信息。

**核心功能:**
- 语义相似度聚类
- 跨Session事实合并
- 累积事实生成 (解决"How many times"问题)
- 时间线构建 (追踪状态变化)

**累积事实示例:**
```json
{
    "accumulation_id": "conv-47_acc_john_organized",
    "description": "John has organized 2 charity tournaments",
    "count": 2,
    "subject": "John",
    "action": "organized",
    "component_fact_ids": ["conv-47_s10_f2", "conv-47_s29_f1"]
}
```

### Step 3: 加载到检索系统 (step3_load_to_retrieval.py)

构建多维索引，准备检索。

**索引类型:**
- 时间索引: `by_time["2023-08"]` → [fact_ids]
- 参与者索引: `by_participant["sam"]` → [fact_ids]
- 类型索引: `by_type["STATE_CHANGE"]` → [fact_ids]

## 使用方法

### 运行完整Pipeline

```bash
# 方式1: Shell脚本
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/pipeline.sh

# 方式2: Python脚本
python benchmark_locomo/dataset_maker/locomo_episodic_memory/pipeline.py

# 方式3: 分步执行
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/step1.sh
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/step2.sh
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/step3.sh
```

### 命令行参数

```bash
# Step 1: 事实抽取
python benchmark_locomo/dataset_maker/locomo_episodic_memory/step1_extract_episodic_facts.py \
    --input-file "benchmark_locomo/dataset/locomo/locomo10.json" \
    --output-dir "benchmark_locomo/dataset/locomo/episodic_memory/step1_facts" \
    --extract-model "qwen-3.5-plus-thinking" \
    --max-workers 6 \
    --sample-ids conv-49 conv-47  # 可选：指定处理的样本

# Step 2: 去重增强
python benchmark_locomo/dataset_maker/locomo_episodic_memory/step2_deduplicate_and_enhance.py \
    --input-dir "step1_facts" \
    --output-dir "step2_enhanced" \
    --dedup-model "deepseek-v3.2-dashscope"

# Step 3: 加载索引
python benchmark_locomo/dataset_maker/locomo_episodic_memory/step3_load_to_retrieval_batch.py \
    --input-dir "step2_enhanced" \
    --output-dir "step3_loaded"
```

## 三塔协同检索策略

```python
def retrieve_for_question(question: str):
    """根据问题类型选择合适的塔进行检索"""
    
    if is_when_question(question):
        # When did X happen? → 情景记忆塔 (时间索引)
        return episodic_tower.retrieve_by_time(question)
    
    elif is_relationship_question(question):
        # What is X's relationship with Y? → 实体关系塔
        return entity_tower.retrieve_relationship(question)
    
    elif is_summary_question(question):
        # Overall/Summary? → 分层塔
        return hierarchy_tower.retrieve_summary(question)
    
    elif is_count_question(question):
        # How many times? → 情景记忆塔 (累积事实)
        return episodic_tower.retrieve_accumulation(question)
    
    else:
        # 默认：多塔融合检索
        results = []
        results.extend(episodic_tower.retrieve(question))
        results.extend(entity_tower.retrieve(question))
        results.extend(hierarchy_tower.retrieve(question))
        return merge_and_rank(results)
```

## 输出目录结构

```
benchmark_locomo/dataset/locomo/episodic_memory/
├── step1_facts/
│   ├── conv-26_episodic_facts.json
│   ├── conv-30_episodic_facts.json
│   └── ...
├── step2_enhanced/
│   ├── conv-26_enhanced.json
│   ├── conv-30_enhanced.json
│   └── enhancement_stats.json
└── step3_loaded/
    ├── unified_index.json
    ├── all_episodic_facts.json
    └── load_stats.json
```

## 与其他塔的数据关联

### 与实体关系塔关联

```json
{
    "fact_id": "conv-49_s1_f0",
    "participants": ["Sam", "Canadian woman"],
    "linked_entities": [
        {"entity_id": "E_SAM", "type": "PERSON"},
        {"entity_id": "E_CANADIAN_WOMAN", "type": "PERSON"}
    ]
}
```

### 与分层塔关联

```json
{
    "fact_id": "conv-49_s1_f0",
    "source_session_id": "session_1",
    "linked_l1_summary_id": "conv-49_L1_session_1",
    "linked_l2_insight_id": "conv-49_L2_relationship_theme"
}
```

## 常见问题

### Q: 如何处理时间模糊的事实?
A: 保留原文时间表述(`original_text`)，同时尽可能计算绝对时间。检索时支持两种方式匹配。

### Q: 累积事实的阈值是多少?
A: 默认阈值为2次以上的同类事件才生成累积事实。可通过配置调整。

### Q: 如何扩展事实类型?
A: 修改`FactType`类，并相应更新抽取Prompt中的类型描述。
