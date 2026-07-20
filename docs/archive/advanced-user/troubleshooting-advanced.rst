高级故障排除
==============

检索质量问题
------------

**检索结果不相关**

1. 检查查询语言和记忆语言一致
2. 调低 ``similarity_threshold`` 到 0.5
3. 确认已调用 ``build_high_level``
4. 检查 metadata timestamp 是否正确

**某条明显相关的记忆没被召回**

1. 确认该记忆的 ``text_content`` 字段有值
2. 检查 raw_data 字段名是否正确
3. 用 ``retrieve_in_space`` 在该记忆所在空间检索
4. 检查该记忆是否被添加到了正确的空间

性能问题
--------

**build_high_level 超时**

1. 减少会话大小（降低 session_max_pending）
2. 切换到更快的 LLM 模型
3. 用 mode="auto" 增量而非 mode="force" 全量

**检索延迟 > 1 秒**

1. 关闭 Rerank + BFS 扩展
2. 减少 similarity_top_k
3. 确认 Embedding 模型在 GPU 上运行

构建问题
--------

**会话分割不对**

1. 系统通过 LLM 语义分析检测话题边界，时间间隔仅作参考
2. 在 metadata 中手动标记 session_id 可覆盖自动分割
3. 使用 mode="force" 重建

**实体/事件去重不准**

1. 增大 max_entities_per_llm / max_events_per_llm
2. 使用更强的 LLM 模型
3. 手动检查去重日志

数据持久化问题
--------------

**加载后检索结果不对**

1. 确认 save/load 使用相同的 Embedding 模型
2. 检查 save 目录完整
3. load 后不需要重新 build_high_level（已保存了高阶结构）
