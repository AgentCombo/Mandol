常见开发陷阱
==============

1. 忘记调用 build_high_level → 检索空结果
2. raw_data 中使用错误字段名（应为 text_content 而非 content）
3. 多用户混淆：未用 space 隔离用户数据
4. metadata timestamp 格式不统一（ISO 8601 推荐）
5. Embedding 维度不匹配：更换 Embedder 后需重建索引
6. flush 时机不当：大量数据时未及时 flush 导致内存飙升
7. YAML 配置中的 LLM model 名称拼写错误
8. 忽略 Cross-Encoder Reranker 的 GPU 内存占用（~2-4GB）
