CI/CD
===========

每次 Push 自动触发：

1. ``ruff check`` — 代码规范检查
2. ``make test-unit`` — 单元测试
3. ``make test-integration`` — 集成测试（如有 API Key）

通过后自动生成文档并部署。
