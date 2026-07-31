SessionManager 参考
=========================

负责会话分割和生命周期管理。

构造
----

通过 MemorySystem 内部创建，通过 ``session_time_gap_seconds`` 等配置控制。

主要方法
--------

- ``add_unit(unit: MemoryUnit) -> None`` — 将单元加入待处理队列
- ``process_pending_sessions() -> list[Session]`` — 处理队列中的会话分割
- ``get_sessions() -> list[Session]`` — 获取全部已分割会话

Session 对象包含：

- ``id`` — 会话 ID
- ``units`` — MemoryUnit 列表
- ``start_time`` / ``end_time`` — 时间范围
- ``summary`` — 会话摘要

使用示例
--------

.. code-block:: python

   # SessionManager 是 MemorySystem 内部的组件
   # 通过 build_high_level() 间接使用
   report = system.build_high_level(mode="auto")
   print(f"处理了 {report.sessions_processed} 个会话")
